"""Unit tests for structured logging utilities.

Tests for JSONFormatter, TextFormatter, SensitiveDataFilter, and
configure_logging behavior.
"""

import json
import logging

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)


@pytest.fixture
def record() -> logging.LogRecord:
    """Create a basic log record for formatter tests."""
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test_module.py",
        lineno=42,
        msg="Hello %s",
        args=("world",),
        exc_info=None,
    )


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self, record: logging.LogRecord) -> None:
        """Test basic JSON output structure."""
        output = JSONFormatter().format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Hello world"
        assert data["module"] == "test_module"
        assert data["line"] == 42
        assert "timestamp" in data

    def test_includes_request_id(self, record: logging.LogRecord) -> None:
        """Test that request_id is included when present."""
        record.request_id = "req-123"
        data = json.loads(JSONFormatter().format(record))
        assert data["request_id"] == "req-123"

    def test_extra_fields_nested(self, record: logging.LogRecord) -> None:
        """Test that custom extra fields are nested under 'extra'."""
        record.database = "test_db"
        data = json.loads(JSONFormatter().format(record))
        assert data["extra"]["database"] == "test_db"

    def test_exception_included(self, record: logging.LogRecord) -> None:
        """Test that exception info is captured."""
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record.exc_info = sys.exc_info()
        data = json.loads(JSONFormatter().format(record))
        assert "boom" in data["exception"]

    def test_no_extra_key_when_absent(self, record: logging.LogRecord) -> None:
        """Test that 'extra' key is omitted when no custom fields exist."""
        data = json.loads(JSONFormatter().format(record))
        assert "extra" not in data


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_basic_format(self, record: logging.LogRecord) -> None:
        """Test human-readable output."""
        output = TextFormatter().format(record)
        assert "[INFO]" in output
        assert "test.logger - Hello world" in output

    def test_includes_request_id(self, record: logging.LogRecord) -> None:
        """Test request_id appended when present."""
        record.request_id = "req-456"
        output = TextFormatter().format(record)
        assert "[request_id=req-456]" in output

    def test_exception_included(self, record: logging.LogRecord) -> None:
        """Test exception appended when present."""
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record.exc_info = sys.exc_info()
        output = TextFormatter().format(record)
        assert "boom" in output


class TestSensitiveDataFilter:
    """Tests for SensitiveDataFilter."""

    @pytest.fixture
    def log_filter(self) -> SensitiveDataFilter:
        """Create filter instance."""
        return SensitiveDataFilter()

    def test_sensitive_keys_in_args_redacted(self, log_filter: SensitiveDataFilter) -> None:
        """Test sensitive keys in args dict are redacted."""
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="p",
            lineno=1,
            msg="connect",
            args=({"password": "hunter2"},),
            exc_info=None,
        )
        assert log_filter.filter(record) is True
        # A single mapping arg is stored directly as record.args (not a tuple)
        args_dict = record.args
        assert isinstance(args_dict, dict)
        assert args_dict["password"] == "***REDACTED***"

    def test_sensitive_extra_attributes_redacted(self, log_filter: SensitiveDataFilter) -> None:
        """Test sensitive extra attributes on the record are redacted."""
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="p",
            lineno=1,
            msg="msg",
            args=None,
            exc_info=None,
        )
        record.api_key = "sk-secret"
        log_filter.filter(record)
        assert record.api_key == "***REDACTED***"  # type: ignore[attr-defined]

    def test_nested_dicts_sanitized(self, log_filter: SensitiveDataFilter) -> None:
        """Test nested dict values are recursively sanitized."""
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="p",
            lineno=1,
            msg="msg",
            args=None,
            exc_info=None,
        )
        record.details = {"user": "bob", "token": "abc", "nested": {"secret": "x"}}
        log_filter.filter(record)
        assert record.details["token"] == "***REDACTED***"
        assert record.details["nested"]["secret"] == "***REDACTED***"
        assert record.details["user"] == "bob"

    def test_case_insensitive_matching(self, log_filter: SensitiveDataFilter) -> None:
        """Test key matching is case-insensitive."""
        sanitized = log_filter._sanitize_dict({"API_KEY": "abc", "Token": "xyz"})
        assert sanitized["API_KEY"] == "***REDACTED***"
        assert sanitized["Token"] == "***REDACTED***"

    def test_lists_and_tuples_preserved_type(self, log_filter: SensitiveDataFilter) -> None:
        """Test list/tuple types are preserved after sanitization."""
        sanitized = log_filter._sanitize_dict({"items": [{"password": "x"}, "ok"]})
        assert isinstance(sanitized["items"], list)
        assert sanitized["items"][0]["password"] == "***REDACTED***"


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_json_format(self) -> None:
        """Test configuring JSON logging."""
        configure_logging(level="INFO", log_format="json")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)
        assert root.level == logging.INFO

    def test_text_format(self) -> None:
        """Test configuring text logging."""
        configure_logging(level="DEBUG", log_format="text")
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, TextFormatter)
        assert root.level == logging.DEBUG

    def test_sensitive_filter_attached(self) -> None:
        """Test sensitive filter is attached when enabled."""
        configure_logging(enable_sensitive_filter=True)
        filters = [
            f for f in root_filters(logging.getLogger()) if isinstance(f, SensitiveDataFilter)
        ]
        assert len(filters) == 1

    def test_sensitive_filter_skipped(self) -> None:
        """Test sensitive filter is not attached when disabled."""
        configure_logging(enable_sensitive_filter=False)
        filters = [
            f for f in root_filters(logging.getLogger()) if isinstance(f, SensitiveDataFilter)
        ]
        assert len(filters) == 0

    def test_third_party_loggers_quieted(self) -> None:
        """Test third-party library loggers are set to WARNING."""
        configure_logging()
        assert logging.getLogger("asyncpg").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING

    def test_get_logger(self) -> None:
        """Test get_logger returns a named logger."""
        logger = get_logger("test.named")
        assert logger.name == "test.named"


def root_filters(logger: logging.Logger) -> list[logging.Filter]:
    """Collect filters from all handlers of a logger."""
    return [f for handler in logger.handlers for f in handler.filters]
