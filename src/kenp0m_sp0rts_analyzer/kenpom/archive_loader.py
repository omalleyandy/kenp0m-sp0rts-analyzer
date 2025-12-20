"""Archive data loader for historical KenPom data (2023+).

This module handles loading and backfilling historical KenPom data
for training ML models and analyzing trends over time.

The KenPom archive API provides historical snapshots from 2023 onwards,
including:
- Daily team ratings at any past date
- Preseason ratings for any season
"""

import logging
import time
from collections.abc import Generator
from datetime import date, timedelta
from typing import Any

from .exceptions import ArchiveNotAvailableError, SyncError
from .models import BackfillResult
from .repository import KenPomRepository
from .validators import DataValidator

logger = logging.getLogger(__name__)


class ArchiveLoader:
    """Loads historical KenPom data from the archive API.

    The archive API supports dates from 2023 onwards. Use this class to:
    - Backfill historical ratings for ML training
    - Load preseason ratings for comparison
    - Fill gaps in daily data

    Example:
        >>> from kenp0m_sp0rts_analyzer.kenpom import ArchiveLoader
        >>>
        >>> loader = ArchiveLoader()
        >>>
        >>> # Load ratings for a specific date
        >>> ratings = loader.load_date("2024-02-15")
        >>>
        >>> # Backfill a date range
        >>> result = loader.backfill(
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 3, 1)
        ... )
        >>>
        >>> # Load preseason ratings
        >>> preseason = loader.load_preseason(year=2025)
    """

    # KenPom archive API limitations
    MIN_ARCHIVE_YEAR = 2023
    REQUESTS_PER_MINUTE = 10  # Conservative rate limit

    def __init__(
        self,
        db_path: str = "data/kenpom.db",
        api_client: Any = None,
    ):
        """Initialize the archive loader.

        Args:
            db_path: Path to SQLite database.
            api_client: Optional KenPomAPI instance.
        """
        self.repository = KenPomRepository(db_path)
        self.validator = DataValidator()
        self._api_client = api_client
        self._api_initialized = False
        self._request_count = 0
        self._last_request_time = 0.0

    @property
    def api(self):
        """Lazy-load the API client."""
        if not self._api_initialized:
            if self._api_client is None:
                try:
                    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

                    self._api_client = KenPomAPI()
                except Exception as e:
                    raise SyncError(f"Failed to initialize API: {e}") from e
            self._api_initialized = True
        return self._api_client

    def _rate_limit(self) -> None:
        """Enforce rate limiting to avoid API throttling."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        # Reset counter every minute
        if elapsed >= 60:
            self._request_count = 0

        # Throttle if approaching limit
        if self._request_count >= self.REQUESTS_PER_MINUTE:
            sleep_time = 60 - elapsed
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self._request_count = 0

        self._request_count += 1
        self._last_request_time = time.time()

    def load_date(
        self,
        archive_date: date | str,
        store: bool = True,
    ) -> list[dict[str, Any]]:
        """Load ratings for a specific historical date.

        Args:
            archive_date: Date to load (YYYY-MM-DD string or date object).
            store: Whether to save to database.

        Returns:
            List of rating dictionaries.

        Raises:
            ArchiveNotAvailableError: If date is before 2023 or in the future.
        """
        # Normalize date
        if isinstance(archive_date, str):
            archive_date = date.fromisoformat(archive_date)

        # Validate date range
        result = self.validator.validate_date_range(
            start_date=archive_date,
            end_date=archive_date,
            min_year=self.MIN_ARCHIVE_YEAR,
        )
        if not result.valid:
            raise ArchiveNotAvailableError(
                f"Invalid archive date: {result.errors}",
                requested_date=str(archive_date),
            )

        self._rate_limit()

        logger.info(f"Loading archive data for {archive_date}")

        # Fetch from API
        response = self.api.get_archive(archive_date=str(archive_date))
        data = list(response.data)

        # Sanitize and validate
        sanitized = self.validator.sanitize_response(data)

        if store:
            # Determine season from date (Nov-Dec = next year's season)
            season = (
                archive_date.year + 1
                if archive_date.month >= 11
                else archive_date.year
            )

            count = self.repository.save_archive_ratings(
                archive_date=archive_date,
                season=season,
                is_preseason=False,
                data=sanitized,
            )
            logger.debug(f"Stored {count} archive ratings for {archive_date}")

        return sanitized

    def load_preseason(
        self,
        year: int,
        store: bool = True,
    ) -> list[dict[str, Any]]:
        """Load preseason ratings for a season.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season).
            store: Whether to save to database.

        Returns:
            List of preseason rating dictionaries.

        Raises:
            ArchiveNotAvailableError: If year is before 2023.
        """
        if year < self.MIN_ARCHIVE_YEAR:
            raise ArchiveNotAvailableError(
                f"Preseason archive not available before {self.MIN_ARCHIVE_YEAR}",
                requested_date=str(year),
            )

        self._rate_limit()

        logger.info(f"Loading preseason ratings for {year}")

        response = self.api.get_archive(preseason=True, year=year)
        data = list(response.data)

        sanitized = self.validator.sanitize_response(data)

        if store:
            # Store with a special date (Nov 1 of the season start)
            preseason_date = date(year - 1, 11, 1)

            count = self.repository.save_archive_ratings(
                archive_date=preseason_date,
                season=year,
                is_preseason=True,
                data=sanitized,
            )
            logger.debug(
                f"Stored {count} preseason archive ratings for {year}"
            )

        return sanitized

    def get_available_dates(
        self,
        season: int | None = None,
    ) -> list[date]:
        """Get list of dates already in the database.

        Args:
            season: Optional season filter.

        Returns:
            List of dates with stored data.
        """
        return self.repository.get_available_dates(season)

    def get_missing_dates(
        self,
        start_date: date,
        end_date: date,
        exclude_offseason: bool = True,
    ) -> list[date]:
        """Find dates in a range that are missing from the database.

        Args:
            start_date: Start of range.
            end_date: End of range.
            exclude_offseason: Skip April-October (no meaningful data).

        Returns:
            List of dates needing backfill.
        """
        existing = set(self.get_available_dates())
        missing = []

        current = start_date
        while current <= end_date:
            # Skip offseason months if requested
            if exclude_offseason and current.month in range(5, 11):
                current += timedelta(days=1)
                continue

            if current not in existing:
                missing.append(current)

            current += timedelta(days=1)

        return missing

    def backfill(
        self,
        start_date: date,
        end_date: date,
        skip_existing: bool = True,
        progress_callback: Any = None,
    ) -> BackfillResult:
        """Backfill historical data for a date range.

        Args:
            start_date: Start of backfill range.
            end_date: End of backfill range.
            skip_existing: Skip dates already in database.
            progress_callback: Optional callable(current_date, total, completed).

        Returns:
            BackfillResult with operation summary.

        Example:
            >>> result = loader.backfill(
            ...     start_date=date(2024, 1, 1),
            ...     end_date=date(2024, 3, 1),
            ...     progress_callback=lambda d, t, c: print(f"{c}/{t}: {d}")
            ... )
            >>> print(f"Filled {result.dates_filled} dates")
        """
        from datetime import datetime

        start_time = datetime.now()
        errors = []

        # Validate range
        validation = self.validator.validate_date_range(
            start_date, end_date, self.MIN_ARCHIVE_YEAR
        )
        if not validation.valid:
            return BackfillResult(
                success=False,
                start_date=start_date,
                end_date=end_date,
                dates_requested=0,
                dates_filled=0,
                dates_skipped=0,
                errors=validation.errors,
                duration_seconds=0.0,
            )

        # Get dates to fill
        if skip_existing:
            dates_to_fill = self.get_missing_dates(start_date, end_date)
        else:
            dates_to_fill = []
            current = start_date
            while current <= end_date:
                if current.month not in range(5, 11):  # Skip offseason
                    dates_to_fill.append(current)
                current += timedelta(days=1)

        total_dates = len(dates_to_fill)
        filled = 0
        skipped = 0

        logger.info(
            f"Starting backfill: {total_dates} dates from {start_date} to {end_date}"
        )

        for i, archive_date in enumerate(dates_to_fill):
            try:
                if progress_callback:
                    progress_callback(archive_date, total_dates, i)

                self.load_date(archive_date, store=True)
                filled += 1

                # Log progress periodically
                if filled % 10 == 0:
                    logger.info(f"Backfill progress: {filled}/{total_dates}")

            except ArchiveNotAvailableError as e:
                logger.warning(f"Skipping {archive_date}: {e}")
                skipped += 1

            except Exception as e:
                logger.error(f"Error loading {archive_date}: {e}")
                errors.append(f"{archive_date}: {str(e)}")
                # Continue with next date

        duration = (datetime.now() - start_time).total_seconds()

        result = BackfillResult(
            success=len(errors) == 0,
            start_date=start_date,
            end_date=end_date,
            dates_requested=total_dates,
            dates_filled=filled,
            dates_skipped=skipped,
            errors=errors,
            duration_seconds=duration,
        )

        logger.info(
            f"Backfill complete: {filled} filled, {skipped} skipped, "
            f"{len(errors)} errors in {duration:.1f}s"
        )

        return result

    def backfill_season(
        self,
        season: int,
        skip_existing: bool = True,
    ) -> BackfillResult:
        """Backfill an entire season's data.

        Args:
            season: Season year (e.g., 2025 for 2024-25 season).
            skip_existing: Skip dates already in database.

        Returns:
            BackfillResult with operation summary.
        """
        # Season typically runs Nov 1 to April 15
        start_date = date(season - 1, 11, 1)
        end_date = date(season, 4, 15)

        # Don't go beyond today
        if end_date > date.today():
            end_date = date.today()

        return self.backfill(start_date, end_date, skip_existing)

    def iter_date_range(
        self,
        start_date: date,
        end_date: date,
        skip_existing: bool = True,
    ) -> Generator[tuple[date, list[dict]], None, None]:
        """Iterate through a date range, yielding ratings for each date.

        This is useful for processing historical data without storing it,
        or for custom processing pipelines.

        Args:
            start_date: Start of range.
            end_date: End of range.
            skip_existing: Skip dates already in database.

        Yields:
            Tuple of (date, list of ratings) for each date.

        Example:
            >>> for dt, ratings in loader.iter_date_range(start, end):
            ...     process_ratings(dt, ratings)
        """
        if skip_existing:
            dates = self.get_missing_dates(start_date, end_date)
        else:
            dates = []
            current = start_date
            while current <= end_date:
                if current.month not in range(5, 11):
                    dates.append(current)
                current += timedelta(days=1)

        for archive_date in dates:
            try:
                ratings = self.load_date(archive_date, store=False)
                yield archive_date, ratings
            except ArchiveNotAvailableError:
                continue

    def get_rating_changes(
        self,
        team_id: int,
        start_date: date,
        end_date: date,
        metric: str = "adj_em",
    ) -> list[dict]:
        """Calculate day-over-day rating changes for a team.

        Useful for identifying momentum and trend analysis.

        Args:
            team_id: Team to analyze.
            start_date: Start of range.
            end_date: End of range.
            metric: Metric to track ('adj_em', 'adj_oe', 'adj_de', etc).

        Returns:
            List of {date, value, change} dictionaries.
        """
        history = self.repository.get_team_rating_history(
            team_id, days=(end_date - start_date).days
        )

        if not history:
            return []

        # Sort by date ascending
        history.sort(key=lambda r: r.snapshot_date)

        changes = []
        prev_value = None

        for rating in history:
            value = getattr(rating, metric, None)
            if value is not None:
                change = value - prev_value if prev_value is not None else 0.0
                changes.append(
                    {
                        "date": rating.snapshot_date,
                        "value": value,
                        "change": round(change, 2),
                    }
                )
                prev_value = value

        return changes
