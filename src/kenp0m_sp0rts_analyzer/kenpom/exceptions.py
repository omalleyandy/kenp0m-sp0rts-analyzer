"""Custom exceptions for KenPom data operations.

This module defines a hierarchy of exceptions for handling errors
in the KenPom data pipeline.
"""


class KenPomError(Exception):
    """Base exception for all KenPom-related errors."""

    pass


class DataValidationError(KenPomError):
    """Raised when data fails validation checks."""

    def __init__(self, message: str, field: str | None = None, value: any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class DatabaseError(KenPomError):
    """Raised for database operation failures."""

    pass


class SyncError(KenPomError):
    """Raised when data synchronization fails."""

    def __init__(self, message: str, partial_count: int = 0):
        self.partial_count = partial_count
        super().__init__(message)


class RateLimitError(KenPomError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        self.retry_after = retry_after
        super().__init__(message)


class ArchiveNotAvailableError(KenPomError):
    """Raised when requested archive data is not available."""

    def __init__(self, message: str, requested_date: str | None = None):
        self.requested_date = requested_date
        super().__init__(message)


class TeamNotFoundError(KenPomError):
    """Raised when a team cannot be found."""

    def __init__(self, team_identifier: str | int):
        self.team_identifier = team_identifier
        super().__init__(f"Team not found: {team_identifier}")


class ConfigurationError(KenPomError):
    """Raised for configuration-related errors."""

    pass
