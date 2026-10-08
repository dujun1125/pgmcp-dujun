"""Configuration management for PostgreSQL MCP Server.

This module defines all configuration settings using Pydantic for validation
and type safety. Configuration is loaded from environment variables with
sensible defaults.
"""

import os
import re
from typing import Literal

from pydantic import Field, PrivateAttr, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")

    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=100, description="Minimum pool size")
    max_pool_size: int = Field(default=20, ge=1, le=100, description="Maximum pool size")
    pool_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Pool acquire timeout in seconds"
    )
    command_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Command execution timeout in seconds"
    )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """Build DSN with masked password for logging."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.name}"


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Model to use for SQL generation")
    max_tokens: int = Field(default=2000, ge=100, le=4096, description="Maximum tokens in response")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for response randomness"
    )
    timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="API request timeout in seconds"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate API key is not empty and has correct format."""
        api_key_str = v.get_secret_value()
        if not api_key_str or not api_key_str.strip():
            raise ValueError("OpenAI API key must not be empty")
        if not api_key_str.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class SecurityConfig(BaseSettings):
    """Security and access control configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    allow_write_operations: bool = Field(
        default=False, description="Allow write operations (INSERT, UPDATE, DELETE)"
    )
    blocked_functions: list[str] = Field(
        default_factory=lambda: [
            "pg_sleep",
            "pg_read_file",
            "pg_write_file",
            "lo_import",
            "lo_export",
        ],
        description="List of blocked PostgreSQL functions",
    )
    max_rows: int = Field(default=10000, ge=1, le=100000, description="Maximum rows to return")
    max_execution_time: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Maximum query execution time in seconds"
    )
    readonly_role: str | None = Field(
        default=None, description="PostgreSQL role to switch to for read-only access"
    )
    safe_search_path: str = Field(
        default="public", description="Safe search_path to set during query execution"
    )
    blocked_tables: list[str] = Field(
        default_factory=list,
        description="Table names that queries must not access (e.g. secrets, passwords)",
    )
    blocked_columns: list[str] = Field(
        default_factory=list,
        description="Column names that queries must not access (e.g. password, ssn)",
    )
    allow_explain: bool = Field(
        default=False, description="Whether EXPLAIN statements are permitted"
    )

    @field_validator("blocked_functions", mode="before")
    @classmethod
    def parse_blocked_functions(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v

    @field_validator("blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_blocked_names(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list for table/column blocklists."""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v


class ValidationConfig(BaseSettings):
    """Query validation configuration."""

    model_config = SettingsConfigDict(env_prefix="VALIDATION_")

    max_question_length: int = Field(
        default=10000, ge=1, le=50000, description="Maximum question length in characters"
    )

    # Result validation settings
    enabled: bool = Field(default=True, description="Enable result validation using LLM")
    sample_rows: int = Field(
        default=5, ge=1, le=100, description="Number of sample rows to include in validation"
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Result validation timeout in seconds"
    )
    confidence_threshold: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence for acceptable results"
    )


class CacheConfig(BaseSettings):
    """Schema cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    schema_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Schema cache TTL in seconds"
    )
    max_size: int = Field(default=100, ge=1, le=1000, description="Maximum cache entries")
    enabled: bool = Field(default=True, description="Enable schema caching")


class ResilienceConfig(BaseSettings):
    """Resilience and fault tolerance configuration."""

    model_config = SettingsConfigDict(env_prefix="RESILIENCE_")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Initial retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=100, description="Failures before circuit opens"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Circuit breaker timeout in seconds"
    )
    rate_limit_queries: int = Field(
        default=10, ge=1, le=1000, description="Maximum concurrent database queries"
    )
    rate_limit_llm: int = Field(
        default=5, ge=1, le=1000, description="Maximum concurrent LLM API calls"
    )
    rate_limit_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Seconds to wait for a rate limiter slot before rejecting",
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")

    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Metrics HTTP server port"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="text", description="Log format")


# Environment variables matching <PREFIX>_DATABASE_<FIELD> define extra databases
_EXTRA_DB_PATTERN = re.compile(
    r"^([A-Z][A-Z0-9]*)_DATABASE_("
    r"HOST|PORT|NAME|USER|PASSWORD|"
    r"MIN_POOL_SIZE|MAX_POOL_SIZE|POOL_TIMEOUT|COMMAND_TIMEOUT)$"
)


def _scan_extra_databases() -> dict[str, DatabaseConfig]:
    """Scan environment variables for extra database configurations.

    Any environment variable matching ``<PREFIX>_DATABASE_<FIELD>`` (where
    PREFIX is an uppercase identifier and FIELD is one of the DatabaseConfig
    fields in upper snake case) contributes to an extra database config.
    The resulting configs are keyed by their ``name`` field, e.g.::

        SECONDARY_DATABASE_HOST=localhost
        SECONDARY_DATABASE_NAME=reports
        SECONDARY_DATABASE_USER=reader

    creates ``extra_databases["reports"]``.

    Returns:
        dict[str, DatabaseConfig]: Extra database configs keyed by name.

    Raises:
        ValueError: If a variable group is missing ``<PREFIX>_DATABASE_NAME``
            or contains invalid values.
    """
    grouped: dict[str, dict[str, str]] = {}
    for env_key, env_value in os.environ.items():
        match = _EXTRA_DB_PATTERN.match(env_key)
        if match:
            prefix, field = match.groups()
            grouped.setdefault(prefix, {})[field.lower()] = env_value

    extras: dict[str, DatabaseConfig] = {}
    for prefix, fields in grouped.items():
        if "name" not in fields:
            raise ValueError(
                f"Extra database config '{prefix}_DATABASE_*' is missing '{prefix}_DATABASE_NAME'"
            )
        try:
            config = DatabaseConfig(**fields)
        except ValueError as e:
            raise ValueError(f"Invalid extra database config '{prefix}_DATABASE_*': {e}") from e
        extras[config.name] = config
    return extras


class Settings(BaseSettings):
    """Main application settings aggregating all config sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # Nested configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Additional databases keyed by database name, discovered from
    # <PREFIX>_DATABASE_* environment variables (e.g. SECONDARY_DATABASE_HOST).
    # Stored as a private attribute so pydantic-settings does not treat the
    # dict values as nested env-configured models.
    _extra_databases: dict[str, DatabaseConfig] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        """Discover extra database configs after the model is initialized."""
        self._extra_databases = _scan_extra_databases()

    @property
    def extra_databases(self) -> dict[str, DatabaseConfig]:
        """Additional databases keyed by database name."""
        return self._extra_databases

    @property
    def all_databases(self) -> list[DatabaseConfig]:
        """All configured databases (primary first, then extras)."""
        return [self.database, *self._extra_databases.values()]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings instance. Useful for testing."""
    global _settings
    _settings = None
