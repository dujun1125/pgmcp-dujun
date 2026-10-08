"""Unit tests for request tracing utilities.

Tests for request context propagation, tracing decorators, and TracingLogger.
"""

import logging

import pytest

from pg_mcp.observability.tracing import (
    TraceContext,
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestId:
    """Tests for request ID generation and context management."""

    def test_generate_request_id_unique(self) -> None:
        """Test generated request IDs are unique."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_get_request_id_default_none(self) -> None:
        """Test request ID is None outside a context."""
        clear_request_id()
        assert get_request_id() is None

    def test_set_and_get_request_id(self) -> None:
        """Test setting and reading a request ID."""
        set_request_id("custom-id")
        assert get_request_id() == "custom-id"
        clear_request_id()
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_sets_and_resets(self) -> None:
        """Test context manager sets ID inside and resets after."""
        async with request_context("ctx-1") as request_id:
            assert request_id == "ctx-1"
            assert get_request_id() == "ctx-1"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_generates_id(self) -> None:
        """Test context manager generates an ID when none provided."""
        async with request_context() as request_id:
            assert request_id is not None
            assert get_request_id() == request_id
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_context_isolated_across_tasks(self) -> None:
        """Test request contexts do not leak between concurrent tasks."""
        import asyncio

        seen: dict[str, str | None] = {}

        async def worker(name: str, delay: float) -> None:
            async with request_context(name):
                await asyncio.sleep(delay)
                seen[name] = get_request_id()

        await asyncio.gather(worker("a", 0.02), worker("b", 0.01))
        assert seen["a"] == "a"
        assert seen["b"] == "b"


class TestTraceDecorators:
    """Tests for trace_async and trace_sync decorators."""

    @pytest.mark.asyncio
    async def test_trace_async_returns_result(self) -> None:
        """Test traced async function preserves return value."""

        @trace_async(operation="gen")
        async def work(value: str) -> str:
            return value.upper()

        assert await work("abc") == "ABC"

    @pytest.mark.asyncio
    async def test_trace_async_records_operation(self) -> None:
        """Test traced async function annotates records with operation."""
        captured: list[logging.LogRecord] = []

        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[method-assign]

        logger = logging.getLogger("trace_async_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        @trace_async(operation="op-name")
        async def work() -> None:
            logger.info("inside")

        # The decorator only annotates records when a request context exists.
        async with request_context("req-op"):
            await work()

        assert len(captured) == 1
        assert captured[0].operation == "op-name"  # type: ignore[attr-defined]
        assert captured[0].request_id == "req-op"  # type: ignore[attr-defined]
        logger.removeHandler(handler)

    def test_trace_sync_returns_result(self) -> None:
        """Test traced sync function preserves return value."""

        @trace_sync(operation="sync-op")
        def work(value: int) -> int:
            return value * 2

        assert work(3) == 6

    def test_trace_sync_without_context(self) -> None:
        """Test traced sync function works without request context."""
        clear_request_id()

        @trace_sync()
        def work() -> str:
            return "ok"

        assert work() == "ok"


class TestTracingLogger:
    """Tests for TracingLogger."""

    def test_forwards_to_standard_logger(self) -> None:
        """Test all levels forward to the wrapped logger."""
        logger = get_tracing_logger("tracing_logger_test")
        # Should not raise for any level
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        logger.critical("c")
        logger.exception("x")

    @pytest.mark.asyncio
    async def test_adds_request_id_to_extra(self) -> None:
        """Test request ID injected into extra when in context."""
        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[method-assign]

        logger_name = "tracing_logger_extra_test"
        std_logger = logging.getLogger(logger_name)
        std_logger.addHandler(handler)
        std_logger.setLevel(logging.INFO)

        tracing_logger = TracingLogger(logger_name)
        async with request_context("req-extra"):
            tracing_logger.info("hello", extra={"database": "db1"})

        assert len(captured) == 1
        record = captured[0]
        assert record.request_id == "req-extra"  # type: ignore[attr-defined]
        assert record.database == "db1"  # type: ignore[attr-defined]
        std_logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_existing_request_id_not_overwritten(self) -> None:
        """Test explicit request_id in extra is preserved."""
        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[method-assign]

        logger_name = "tracing_logger_no_overwrite_test"
        std_logger = logging.getLogger(logger_name)
        std_logger.addHandler(handler)
        std_logger.setLevel(logging.INFO)

        tracing_logger = TracingLogger(logger_name)
        async with request_context("ctx-id"):
            tracing_logger.info("hello", extra={"request_id": "explicit"})

        assert captured[0].request_id == "explicit"  # type: ignore[attr-defined]
        std_logger.removeHandler(handler)


class TestTraceContext:
    """Tests for TraceContext model."""

    def test_trace_context_creation(self) -> None:
        """Test creating a trace context with metadata."""
        ctx = TraceContext(
            request_id="req-1",
            parent_id="req-0",
            operation="query",
            metadata={"user": "test"},
        )
        assert ctx.request_id == "req-1"
        assert ctx.parent_id == "req-0"
        assert ctx.operation == "query"
        assert ctx.metadata == {"user": "test"}

    def test_trace_context_defaults(self) -> None:
        """Test trace context optional fields default correctly."""
        ctx = TraceContext(request_id="req-2")
        assert ctx.parent_id is None
        assert ctx.operation is None
        assert ctx.metadata is None
