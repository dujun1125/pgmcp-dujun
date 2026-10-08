"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, rate limiting,
circuit breaker pattern for fault tolerance, metrics collection, and request
tracing via request IDs.
"""

import asyncio
import logging
import time
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector, metrics
from pg_mcp.observability.tracing import generate_request_id, request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation across multiple databases. It implements retry logic with
    exponential backoff, per-resource rate limiting (queries and LLM calls),
    a circuit breaker for LLM fault tolerance, Prometheus metrics collection,
    and request-ID tracing.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"mydb": executor},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ...     rate_limiter=rate_limiter,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        rate_limiter: MultiRateLimiter | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executors: SQL execution services keyed by database name.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, backoff,
                circuit breaker, and rate limits.
            validation_config: Validation configuration including thresholds.
            rate_limiter: Optional rate limiter guarding query and LLM
                concurrency. When None, no rate limiting is applied.
            metrics_collector: Optional metrics collector. When None, the
                global Prometheus ``metrics`` singleton is used.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics_collector if metrics_collector is not None else metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Acquire a query slot from the rate limiter (if configured)
        2. Generate request_id for tracing
        3. Resolve and validate database name
        4. Load schema from cache
        5. Generate and validate SQL with retry and exponential backoff
        6. Execute SQL on the selected database (if return_type == RESULT)
        7. Validate results (optional)
        8. Record metrics and return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Raises:
            RateLimitExceededError: If no query slot could be acquired within
                the configured rate limit timeout. This is raised before any
                processing starts and is not converted into a QueryResponse.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        # Rate limit the entire request processing flow
        if self.rate_limiter is not None:
            try:
                async with self.rate_limiter.for_queries(
                    timeout=self.resilience_config.rate_limit_timeout
                ):
                    return await self._execute_query_locked(request)
            except TimeoutError as e:
                stats = self.rate_limiter.query_limiter.get_stats()
                logger.warning(
                    "Query rate limit exceeded",
                    extra={"active": stats["active_count"], "max": stats["max_concurrent"]},
                )
                raise RateLimitExceededError(
                    message="Query rate limit exceeded, please retry later",
                    details={
                        "max_concurrent_queries": stats["max_concurrent"],
                        "timeout_seconds": self.resilience_config.rate_limit_timeout,
                    },
                ) from e

        return await self._execute_query_locked(request)

    async def _execute_query_locked(self, request: QueryRequest) -> QueryResponse:
        """Run the query pipeline while holding a rate limiter slot.

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        # Generate request_id and establish tracing context
        request_id = generate_request_id()
        async with request_context(request_id):
            return await self._process_query(request, request_id)

    async def _process_query(self, request: QueryRequest, request_id: str) -> QueryResponse:
        """Run the actual query pipeline within a tracing context.

        Args:
            request: Query request containing question and parameters.
            request_id: Request ID for tracing.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        start_time = self._get_current_time_ms()
        database_name: str | None = None

        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 3: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 4: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                self._record_request_metrics(
                    status="success", database=database_name, start_time_ms=start_time
                )
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 5: Execute SQL on the selected database
            executor = self.sql_executors.get(database_name)
            if executor is None:
                raise DatabaseError(
                    message=f"No SQL executor available for database '{database_name}'",
                    details={
                        "database": database_name,
                        "available_executors": list(self.sql_executors.keys()),
                    },
                )

            logger.debug(
                "Executing SQL",
                extra={"request_id": request_id, "database": database_name},
            )
            db_start_time = self._get_current_time_ms()

            results, total_count = await executor.execute(generated_sql)

            execution_time_ms = self._get_current_time_ms() - db_start_time
            self.metrics.observe_db_query_duration(execution_time_ms / 1000.0)
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 6: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 7: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            self._record_request_metrics(
                status="success", database=database_name, start_time_ms=start_time
            )
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            if isinstance(e, SecurityViolationError):
                self.metrics.increment_sql_rejected(reason="security_violation")
            elif isinstance(e, SQLParseError):
                self.metrics.increment_sql_rejected(reason="sql_parse_error")
            self._record_request_metrics(
                status="error",
                database=database_name or "unknown",
                start_time_ms=start_time,
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            self._record_request_metrics(
                status="error",
                database=database_name or "unknown",
                start_time_ms=start_time,
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )

    def _record_request_metrics(
        self,
        status: str,
        database: str,
        start_time_ms: float,
    ) -> None:
        """Record request count and total duration metrics.

        Args:
            status: Request outcome ("success" or "error").
            database: Target database name (or "unknown" before resolution).
            start_time_ms: Pipeline start time in milliseconds.
        """
        self.metrics.increment_query_request(status=status, database=database)
        self.metrics.query_duration.observe((self._get_current_time_ms() - start_time_ms) / 1000.0)

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Acquires an LLM slot from the rate limiter (if configured)
        3. Generates SQL using LLM
        4. Validates the generated SQL
        5. On validation failure, waits with exponential backoff and retries
           with error feedback
        6. Records success/failure to circuit breaker and metrics

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            RateLimitExceededError: If no LLM slot could be acquired in time.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL under the LLM rate limiter
                llm_start = self._get_current_time_ms()
                generated_sql = await self._generate_with_rate_limit(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                    request_id=request_id,
                )
                self.metrics.increment_llm_call(operation="generate_sql")
                self.metrics.observe_llm_latency(
                    operation="generate_sql",
                    duration=(self._get_current_time_ms() - llm_start) / 1000.0,
                )

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if attempt < max_retries:
                        # Record as failure and retry with feedback after backoff
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        await self._wait_backoff(attempt, request_id)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, RateLimitExceededError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _generate_with_rate_limit(
        self,
        question: str,
        schema: Any,
        previous_attempt: str | None,
        error_feedback: str | None,
        request_id: str,
    ) -> str:
        """Call the SQL generator under the LLM rate limiter.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_attempt: Previously generated SQL that failed (for retry).
            error_feedback: Error message from previous attempt (for retry).
            request_id: Request ID for tracking.

        Returns:
            str: Generated SQL query.

        Raises:
            RateLimitExceededError: If no LLM slot could be acquired in time.
        """
        if self.rate_limiter is None:
            return await self.sql_generator.generate(
                question=question,
                schema=schema,
                previous_attempt=previous_attempt,
                error_feedback=error_feedback,
            )

        try:
            async with self.rate_limiter.for_llm(timeout=self.resilience_config.rate_limit_timeout):
                return await self.sql_generator.generate(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_attempt,
                    error_feedback=error_feedback,
                )
        except TimeoutError as e:
            stats = self.rate_limiter.llm_limiter.get_stats()
            logger.warning(
                "LLM rate limit exceeded",
                extra={
                    "request_id": request_id,
                    "active": stats["active_count"],
                    "max": stats["max_concurrent"],
                },
            )
            raise RateLimitExceededError(
                message="LLM rate limit exceeded, please retry later",
                details={
                    "max_concurrent_llm_calls": stats["max_concurrent"],
                    "timeout_seconds": self.resilience_config.rate_limit_timeout,
                },
            ) from e

    async def _wait_backoff(self, attempt: int, request_id: str) -> None:
        """Wait before the next retry attempt using exponential backoff.

        The delay grows as ``retry_delay * backoff_factor ** attempt``.

        Args:
            attempt: Zero-based index of the attempt that just failed.
            request_id: Request ID for tracking.
        """
        delay = self.resilience_config.retry_delay * (
            self.resilience_config.backoff_factor**attempt
        )
        logger.debug(
            "Waiting before next retry attempt",
            extra={"request_id": request_id, "attempt": attempt + 1, "delay_seconds": delay},
        )
        await asyncio.sleep(delay)

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
