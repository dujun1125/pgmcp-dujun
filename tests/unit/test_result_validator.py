"""Unit tests for the result validation service.

Tests for ResultValidator LLM interaction using mocked OpenAI clients.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.models.query import ResultValidationResult
from pg_mcp.services.result_validator import ResultValidator


class FakeCompletion:
    """Fake ChatCompletion-like response object."""

    def __init__(self, content: str | None) -> None:
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.model_dump = MagicMock(return_value={"content": content})


def build_validator(**validation_kwargs: Any) -> tuple[ResultValidator, AsyncMock]:
    """Build a ResultValidator with a mocked OpenAI client.

    Args:
        **validation_kwargs: Passed to ValidationConfig.

    Returns:
        tuple: (validator, mocked create method).
    """
    with patch("pg_mcp.services.result_validator.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_create = mock_client.chat.completions.create
        mock_create = AsyncMock()
        mock_client.chat.completions.create = mock_create

        validator = ResultValidator(
            openai_config=OpenAIConfig(api_key="sk-test"),
            validation_config=ValidationConfig(**validation_kwargs),
        )
    return validator, mock_create


class TestResultValidatorDisabled:
    """Tests for disabled validation path."""

    @pytest.mark.asyncio
    async def test_disabled_returns_full_confidence(self) -> None:
        """Test disabled validation skips the LLM entirely."""
        validator, mock_create = build_validator(enabled=False)
        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"a": 1}], row_count=1
        )
        assert result.confidence == 100
        assert result.is_acceptable is True
        mock_create.assert_not_awaited()


class TestResultValidatorSuccess:
    """Tests for successful validation flows."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self) -> None:
        """Test parsing a valid JSON validation response."""
        validator, mock_create = build_validator()
        mock_create.return_value = FakeCompletion(
            '{"confidence": 85, "explanation": "matches", "suggestion": null}'
        )

        result = await validator.validate(
            question="count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 5}],
            row_count=1,
        )

        assert isinstance(result, ResultValidationResult)
        assert result.confidence == 85
        assert result.explanation == "matches"
        assert result.is_acceptable is True  # >= default threshold 70

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_not_acceptable(self) -> None:
        """Test results below threshold are flagged not acceptable."""
        validator, mock_create = build_validator(confidence_threshold=80)
        mock_create.return_value = FakeCompletion(
            '{"confidence": 60, "explanation": "weak match"}'
        )

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[], row_count=0
        )
        assert result.confidence == 60
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_sample_rows_limit(self) -> None:
        """Test only sample_rows rows are sent to the LLM."""
        validator, mock_create = build_validator(sample_rows=2)
        mock_create.return_value = FakeCompletion('{"confidence": 90}')

        rows = [{"id": i} for i in range(10)]
        await validator.validate(question="q", sql="SELECT 1", results=rows, row_count=10)

        prompt = mock_create.call_args.kwargs["messages"][1]["content"]
        assert '"id": 0' in prompt
        assert '"id": 9' not in prompt

    @pytest.mark.asyncio
    async def test_missing_fields_use_defaults(self) -> None:
        """Test missing JSON fields fall back to defaults."""
        validator, mock_create = build_validator()
        mock_create.return_value = FakeCompletion("{}")

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[], row_count=0
        )
        assert result.confidence == 50
        assert result.explanation == "No explanation provided"

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_clamped(self) -> None:
        """Test numeric confidence outside 0-100 is clamped."""
        validator, mock_create = build_validator()
        mock_create.return_value = FakeCompletion(
            '{"confidence": 250, "explanation": "x"}'
        )
        result = await validator.validate(
            question="q", sql="SELECT 1", results=[], row_count=0
        )
        assert result.confidence == 100


class TestResultValidatorErrorPaths:
    """Tests for error handling in result validation."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        """Test invalid JSON yields confidence 60 and not acceptable."""
        validator, mock_create = build_validator()
        mock_create.return_value = FakeCompletion("not json at all")

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[], row_count=0
        )
        assert result.confidence == 60
        assert result.is_acceptable is False
        assert "parsing failed" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_empty_response_raises_llm_error(self) -> None:
        """Test empty choices raise LLMError."""
        validator, mock_create = build_validator()
        response = FakeCompletion("x")
        response.choices = []
        mock_create.return_value = response

        with pytest.raises(LLMError, match="empty response"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_empty_content_raises_llm_error(self) -> None:
        """Test empty message content raises LLMError."""
        validator, mock_create = build_validator()
        mock_create.return_value = FakeCompletion(None)

        with pytest.raises(LLMError, match="empty message"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout_error(self) -> None:
        """Test TimeoutError is converted to LLMTimeoutError."""
        validator, mock_create = build_validator()
        mock_create.side_effect = TimeoutError()

        with pytest.raises(LLMTimeoutError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_auth_error_raises_llm_unavailable(self) -> None:
        """Test authentication failures map to LLMUnavailableError."""
        validator, mock_create = build_validator()
        mock_create.side_effect = Exception("invalid api_key provided")

        with pytest.raises(LLMUnavailableError, match="authentication"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_rate_limit_error_raises_llm_unavailable(self) -> None:
        """Test API rate limits map to LLMUnavailableError."""
        validator, mock_create = build_validator()
        mock_create.side_effect = Exception("rate_limit exceeded")

        with pytest.raises(LLMUnavailableError, match="rate limit"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_llm_errors_reraised_as_is(self) -> None:
        """Test existing LLMError instances are not re-wrapped."""
        validator, mock_create = build_validator()
        original = LLMError("original error")
        mock_create.side_effect = original

        with pytest.raises(LLMError) as exc_info:
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)
        assert exc_info.value is original

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_llm_error(self) -> None:
        """Test unexpected exceptions map to LLMError."""
        validator, mock_create = build_validator()
        mock_create.side_effect = RuntimeError("connection reset")

        with pytest.raises(LLMError, match="Result validation failed"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)
