"""Unit tests for QueryOrchestrator.

This module tests the orchestrator's coordination of the query pipeline,
including retry logic, error handling, and integration with all components.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    LLMError,
    RateLimitExceededError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    QueryRequest,
    ResultValidationResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.observability.tracing import get_request_id
from pg_mcp.resilience.circuit_breaker import CircuitState
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


class TestDatabaseResolution:
    """Test database name resolution logic."""

    @pytest.fixture
    def mock_pools(self) -> dict[str, MagicMock]:
        """Create mock connection pools."""
        return {
            "db1": MagicMock(),
            "db2": MagicMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_pools: dict[str, MagicMock]) -> QueryOrchestrator:
        """Create orchestrator with mocked components."""
        return QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"db1": MagicMock(), "db2": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools=mock_pools,
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

    def test_resolve_database_specified_valid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified valid database."""
        result = orchestrator._resolve_database("db1")
        assert result == "db1"

    def test_resolve_database_specified_invalid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified but invalid database."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database("nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]
        assert "db2" in exc_info.value.details["available_databases"]

    def test_resolve_database_auto_select_single(self) -> None:
        """Test auto-selecting when only one database available."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"only_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        result = orchestrator._resolve_database(None)
        assert result == "only_db"

    def test_resolve_database_auto_select_multiple_fails(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        """Test that auto-select fails when multiple databases available."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "multiple databases" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]

    def test_resolve_database_no_databases(self) -> None:
        """Test error when no databases configured."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "no databases configured" in str(exc_info.value).lower()


class TestSQLGenerationWithRetry:
    """Test SQL generation with retry logic."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_generate_sql_success_first_attempt(self, mock_schema: DatabaseSchema) -> None:
        """Test successful SQL generation on first attempt."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None  # No exception = valid

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert validation_result.is_select is True
        mock_generator.generate.assert_called_once()
        mock_validator.validate_or_raise.assert_called_once_with("SELECT * FROM users;")

    @pytest.mark.asyncio
    async def test_generate_sql_retry_on_validation_failure(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test retry logic when validation fails."""
        # Setup mocks - first attempt fails validation, second succeeds
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = [
            "SELECT * FROM user;",  # First attempt (wrong table name)
            "SELECT * FROM users;",  # Second attempt (correct)
        ]

        mock_validator = MagicMock()
        # First call raises error, second call succeeds
        mock_validator.validate_or_raise.side_effect = [
            SQLParseError('relation "user" does not exist'),
            None,  # Success on second attempt
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert mock_generator.generate.call_count == 2
        assert mock_validator.validate_or_raise.call_count == 2

        # Verify retry included error feedback
        second_call = mock_generator.generate.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "SELECT * FROM user;"
        assert 'relation "user" does not exist' in second_call.kwargs["error_feedback"]

    @pytest.mark.asyncio
    async def test_generate_sql_fails_after_max_retries(self, mock_schema: DatabaseSchema) -> None:
        """Test failure after exhausting all retries."""
        # Setup mocks - all attempts fail validation
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2),
            validation_config=ValidationConfig(),
        )

        # Execute and verify exception
        with pytest.raises(SecurityViolationError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "DELETE statements are not allowed" in str(exc_info.value)
        # Should attempt max_retries + 1 times (initial + retries)
        assert mock_generator.generate.call_count == 3
        assert orchestrator.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_generate_sql_circuit_breaker_open(self, mock_schema: DatabaseSchema) -> None:
        """Test that open circuit breaker prevents SQL generation."""
        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(circuit_breaker_threshold=1),
            validation_config=ValidationConfig(),
        )

        # Manually open the circuit breaker
        orchestrator.circuit_breaker._state = CircuitState.OPEN
        orchestrator.circuit_breaker._failure_count = 5

        # Attempt should fail immediately
        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_sql_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors during generation."""
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "unexpectedly" in str(exc_info.value).lower()
        assert orchestrator.circuit_breaker.failure_count == 1


class TestResultValidation:
    """Test result validation logic."""

    @pytest.mark.asyncio
    async def test_validate_results_success(self) -> None:
        """Test successful result validation."""
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = ResultValidationResult(
            confidence=85,
            explanation="Results match the question well",
            suggestion=None,
            is_acceptable=True,
        )

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 85
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_results_disabled(self) -> None:
        """Test that validation is skipped when disabled."""
        mock_validator = AsyncMock()

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100
        mock_validator.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_results_failure_does_not_raise(self) -> None:
        """Test that validation failures don't raise exceptions."""
        mock_validator = AsyncMock()
        mock_validator.validate.side_effect = Exception("Validation failed")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Should not raise, returns default confidence
        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100


class TestExecuteQueryFlow:
    """Test complete query execution flow."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_execute_query_sql_only(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=SQL."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT * FROM users;"
        assert response.validation is not None
        assert response.validation.is_valid is True
        assert response.data is None  # No execution for SQL-only
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_with_results(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=RESULT."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT id, name FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = (
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            2,  # total count
        )

        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=90,
            explanation="Good results",
            suggestion=None,
            is_acceptable=True,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            result_validator=mock_result_validator,
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT id, name FROM users;"
        assert response.data is not None
        assert response.data.row_count == 2
        assert len(response.data.rows) == 2
        assert response.data.columns == ["id", "name"]
        assert response.confidence == 90
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_schema_not_cached(self) -> None:
        """Test loading schema when not in cache."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Not in cache
        mock_cache.load = AsyncMock(return_value=mock_schema)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_pool = MagicMock()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": mock_pool},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify schema was loaded
        mock_cache.load.assert_called_once_with("test_db", mock_pool)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_execute_query_schema_load_fails(self) -> None:
        """Test handling of schema load failure."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.load = AsyncMock(side_effect=Exception("DB connection failed"))

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "schema" in response.error.message.lower()
        assert response.generated_sql is None

    @pytest.mark.asyncio
    async def test_execute_query_validation_error(self) -> None:
        """Test handling of SQL validation errors."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "DELETE not allowed" in response.error.message
        assert response.error.code == "security_violation"

    @pytest.mark.asyncio
    async def test_execute_query_execution_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of SQL execution errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.side_effect = DatabaseError("Query execution failed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "execution failed" in response.error.message.lower()
        assert response.error.code == "database_error"

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "internal_error"
        assert "internal server error" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_execute_query_auto_select_database(self, mock_schema: DatabaseSchema) -> None:
        """Test auto-selecting database when only one available."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"only_db": MagicMock()},  # Only one database
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute without specifying database
        request = QueryRequest(
            question="Test query",
            database=None,  # No database specified
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        # Verify schema was fetched for auto-selected database
        mock_cache.get.assert_called_once_with("only_db")


class TestExecutorSelection:
    """Test per-database SQL executor selection."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(database_name="test_db", tables=[], version="15.0")

    @pytest.mark.asyncio
    async def test_executor_selected_per_database(self, mock_schema: DatabaseSchema) -> None:
        """Test that the executor matching the requested database is used."""
        executor_db1 = AsyncMock()
        executor_db1.execute.return_value = ([{"id": 1}], 1)
        executor_db2 = AsyncMock()
        executor_db2.execute.return_value = ([{"id": 2}], 1)

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"db1": executor_db1, "db2": executor_db2},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"db1": MagicMock(), "db2": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        # Query db2 - only db2's executor should be used
        request = QueryRequest(question="q", database="db2", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        executor_db2.execute.assert_awaited_once()
        executor_db1.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_executor_returns_error(self, mock_schema: DatabaseSchema) -> None:
        """Test that a missing executor for a valid database returns an error."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={},  # No executors at all
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"db1": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        request = QueryRequest(question="q", database="db1", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "no sql executor" in response.error.message.lower()


class TestRateLimiting:
    """Test rate limiter integration in the request pipeline."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(database_name="test_db", tables=[], version="15.0")

    def _build_orchestrator(
        self,
        mock_schema: DatabaseSchema,
        rate_limiter: MultiRateLimiter,
        generator: AsyncMock | None = None,
    ) -> QueryOrchestrator:
        """Create an orchestrator with the given rate limiter."""
        mock_generator = generator or AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        return QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(rate_limit_timeout=1.0),
            validation_config=ValidationConfig(),
            rate_limiter=rate_limiter,
        )

    @pytest.mark.asyncio
    async def test_query_rate_limit_rejection(self, mock_schema: DatabaseSchema) -> None:
        """Test that exceeding the query limit raises RateLimitExceededError."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=5)

        # Exhaust the only query slot
        async with limiter.for_queries():
            orchestrator = self._build_orchestrator(mock_schema, limiter)
            request = QueryRequest(question="q", database="test_db")

            with pytest.raises(RateLimitExceededError) as exc_info:
                await orchestrator.execute_query(request)

            assert "query rate limit" in str(exc_info.value).lower()
            assert exc_info.value.details["max_concurrent_queries"] == 1

    @pytest.mark.asyncio
    async def test_query_allowed_within_limit(self, mock_schema: DatabaseSchema) -> None:
        """Test that requests within the rate limit proceed normally."""
        limiter = MultiRateLimiter(query_limit=2, llm_limit=2)
        orchestrator = self._build_orchestrator(mock_schema, limiter)

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_llm_rate_limit_rejection(self, mock_schema: DatabaseSchema) -> None:
        """Test that exceeding the LLM limit raises RateLimitExceededError."""
        limiter = MultiRateLimiter(query_limit=5, llm_limit=1)
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        # Exhaust the only LLM slot
        async with limiter.for_llm():
            orchestrator = self._build_orchestrator(mock_schema, limiter, mock_generator)
            request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)

            response = await orchestrator.execute_query(request)

            # execute_query converts the rejection into an error response
            assert response.success is False
            assert response.error is not None
            assert response.error.code == "rate_limit_exceeded"
            mock_generator.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_rate_limiter_runs_freely(self, mock_schema: DatabaseSchema) -> None:
        """Test that a None rate limiter does not block requests."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=MagicMock(),
            sql_executors={"test_db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            rate_limiter=None,
        )

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True


class TestExponentialBackoff:
    """Test exponential backoff between retry attempts."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(database_name="test_db", tables=[], version="15.0")

    @pytest.mark.asyncio
    async def test_backoff_delays_applied(self, mock_schema: DatabaseSchema) -> None:
        """Test that retries wait retry_delay * backoff_factor**attempt seconds."""
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = [
            "SELECT * FROM user;",
            "SELECT * FROM user;",
            "SELECT * FROM users;",
        ]

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = [
            SQLParseError("relation does not exist"),
            SQLParseError("relation does not exist"),
            None,
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2, retry_delay=0.5, backoff_factor=2.0),
            validation_config=ValidationConfig(),
        )

        with patch(
            "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            sql, _, _ = await orchestrator._generate_sql_with_retry(
                question="Get users", schema=mock_schema, request_id="test-123"
            )

        assert sql == "SELECT * FROM users;"
        # First retry waits 0.5 * 2^0 = 0.5s, second waits 0.5 * 2^1 = 1.0s
        assert mock_sleep.await_count == 2
        mock_sleep.assert_any_await(0.5)
        mock_sleep.assert_any_await(1.0)

    @pytest.mark.asyncio
    async def test_no_backoff_on_first_attempt_success(self, mock_schema: DatabaseSchema) -> None:
        """Test that successful first attempts never sleep."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        with patch(
            "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await orchestrator._generate_sql_with_retry(
                question="q", schema=mock_schema, request_id="test-123"
            )

        mock_sleep.assert_not_awaited()


class TestMetricsIntegration:
    """Test metrics collection in the request pipeline."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(database_name="test_db", tables=[], version="15.0")

    @pytest.mark.asyncio
    async def test_success_metrics_recorded(self, mock_schema: DatabaseSchema) -> None:
        """Test that success requests record request and duration metrics."""
        mock_metrics = MagicMock()

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            metrics_collector=mock_metrics,
        )

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        mock_metrics.increment_query_request.assert_called_once_with(
            status="success", database="test_db"
        )
        mock_metrics.query_duration.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_metrics_recorded(self, mock_schema: DatabaseSchema) -> None:
        """Test that failures record error and rejection metrics."""
        mock_metrics = MagicMock()

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError("nope")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(),
            metrics_collector=mock_metrics,
        )

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        mock_metrics.increment_query_request.assert_called_once_with(
            status="error", database="test_db"
        )
        mock_metrics.increment_sql_rejected.assert_called_once_with(reason="security_violation")

    @pytest.mark.asyncio
    async def test_llm_metrics_recorded(self, mock_schema: DatabaseSchema) -> None:
        """Test that LLM calls record call count and latency."""
        mock_metrics = MagicMock()

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            metrics_collector=mock_metrics,
        )

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        await orchestrator.execute_query(request)

        mock_metrics.increment_llm_call.assert_called_once_with(operation="generate_sql")
        mock_metrics.observe_llm_latency.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_duration_metric_recorded(self, mock_schema: DatabaseSchema) -> None:
        """Test that SQL execution records database query duration."""
        mock_metrics = MagicMock()

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ([{"id": 1}], 1)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            metrics_collector=mock_metrics,
        )

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        mock_metrics.observe_db_query_duration.assert_called_once()


class TestTracingIntegration:
    """Test request tracing context propagation."""

    @pytest.mark.asyncio
    async def test_request_context_active_during_execution(self) -> None:
        """Test that a request ID context is set during query processing."""
        captured_request_id: list[str | None] = []

        async def fake_generate(**kwargs: object) -> str:
            captured_request_id.append(get_request_id())
            return "SELECT 1;"

        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = fake_generate

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        request = QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert len(captured_request_id) == 1
        assert captured_request_id[0] is not None
        # Context is reset after the request completes
        assert get_request_id() is None
