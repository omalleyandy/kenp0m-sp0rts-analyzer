"""Real-time monitoring for KenPom rating changes.

This module provides infrastructure for monitoring KenPom data changes
and triggering callbacks when significant updates are detected.

Use cases:
- Alert when team ratings change significantly
- Trigger model retraining when data updates
- Log rating movements for analysis
"""

import logging
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from .database import DatabaseManager
from .exceptions import KenPomError
from .models import RatingChange, TeamRating
from .repository import KenPomRepository
from .validators import DataValidator

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of rating change detected."""

    RATING_UPDATE = "rating_update"
    SIGNIFICANT_MOVE = "significant_move"
    NEW_DATA = "new_data"
    RANK_CHANGE = "rank_change"


@dataclass
class ChangeEvent:
    """A detected change event."""

    change_type: ChangeType
    team_id: int
    team_name: str
    field: str
    old_value: float | None
    new_value: float
    change_amount: float
    detected_at: datetime
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RealtimeMonitor:
    """Monitors KenPom data for changes and triggers callbacks.

    Provides a decorator-based callback system for reacting to data changes.
    Can run in the background or be polled manually.

    Example:
        >>> from kenp0m_sp0rts_analyzer.kenpom import RealtimeMonitor
        >>>
        >>> monitor = RealtimeMonitor()
        >>>
        >>> @monitor.on_rating_change
        ... def handle_change(event: ChangeEvent):
        ...     print(f"{event.team_name}: {event.field} changed by {event.change_amount}")
        >>>
        >>> @monitor.on_significant_move(threshold=2.0)
        ... def handle_big_move(event: ChangeEvent):
        ...     print(f"Big move! {event.team_name} AdjEM: {event.change_amount:+.1f}")
        >>>
        >>> # Start monitoring (runs in background)
        >>> monitor.start(poll_interval=300)  # Check every 5 minutes
        >>>
        >>> # Or poll manually
        >>> changes = monitor.poll()
        >>> for change in changes:
        ...     print(change)
    """

    # Default thresholds for significant changes
    DEFAULT_THRESHOLDS = {
        "adj_em": 1.5,  # +/- 1.5 points is notable
        "adj_oe": 2.0,
        "adj_de": 2.0,
        "adj_tempo": 1.0,
        "rank_adj_em": 5,  # +/- 5 ranking spots
        "pythag": 0.02,  # +/- 2%
    }

    def __init__(
        self,
        db_path: str = "data/kenpom.db",
        api_client: Any = None,
        thresholds: dict[str, float] | None = None,
    ):
        """Initialize the real-time monitor.

        Args:
            db_path: Path to SQLite database.
            api_client: Optional KenPomAPI instance.
            thresholds: Custom thresholds for significant changes.
        """
        self.repository = KenPomRepository(db_path)
        self.validator = DataValidator()
        self._api_client = api_client
        self._api_initialized = False

        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

        # Callback registries
        self._on_change_callbacks: list[Callable[[ChangeEvent], None]] = []
        self._on_significant_callbacks: list[tuple[float, Callable[[ChangeEvent], None]]] = []
        self._on_new_data_callbacks: list[Callable[[list[TeamRating]], None]] = []

        # State tracking
        self._last_ratings: dict[int, TeamRating] = {}
        self._last_poll: datetime | None = None
        self._is_running = False
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def api(self):
        """Lazy-load the API client."""
        if not self._api_initialized:
            if self._api_client is None:
                try:
                    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

                    self._api_client = KenPomAPI()
                except Exception as e:
                    raise KenPomError(f"Failed to initialize API: {e}") from e
            self._api_initialized = True
        return self._api_client

    def on_rating_change(self, func: Callable[[ChangeEvent], None]) -> Callable:
        """Decorator to register a callback for any rating change.

        Args:
            func: Callback function that receives a ChangeEvent.

        Returns:
            The decorated function.

        Example:
            >>> @monitor.on_rating_change
            ... def handle_change(event):
            ...     print(f"{event.team_name}: {event.field} changed")
        """
        self._on_change_callbacks.append(func)
        return func

    def on_significant_move(
        self,
        threshold: float = 2.0,
        field: str = "adj_em",
    ) -> Callable:
        """Decorator to register a callback for significant rating moves.

        Args:
            threshold: Minimum change amount to trigger callback.
            field: Field to monitor (default: adj_em).

        Returns:
            Decorator function.

        Example:
            >>> @monitor.on_significant_move(threshold=3.0)
            ... def handle_big_move(event):
            ...     print(f"Big move! {event.team_name}: {event.change_amount:+.1f}")
        """
        def decorator(func: Callable[[ChangeEvent], None]) -> Callable:
            self._on_significant_callbacks.append((threshold, func))
            return func
        return decorator

    def on_new_data(self, func: Callable[[list[TeamRating]], None]) -> Callable:
        """Decorator to register a callback for new data availability.

        Triggered when fresh ratings are fetched from the API.

        Args:
            func: Callback receiving list of new TeamRating objects.

        Returns:
            The decorated function.

        Example:
            >>> @monitor.on_new_data
            ... def handle_new_data(ratings):
            ...     print(f"Received {len(ratings)} new ratings")
        """
        self._on_new_data_callbacks.append(func)
        return func

    def register_callback(
        self,
        callback: Callable[[ChangeEvent], None],
        change_type: ChangeType | None = None,
        threshold: float | None = None,
    ) -> None:
        """Programmatically register a callback.

        Args:
            callback: Function to call on changes.
            change_type: Optional filter by change type.
            threshold: Optional minimum change threshold.
        """
        if threshold is not None:
            self._on_significant_callbacks.append((threshold, callback))
        else:
            self._on_change_callbacks.append(callback)

    def _load_baseline(self) -> None:
        """Load current ratings as baseline for change detection."""
        ratings = self.repository.get_latest_ratings()
        self._last_ratings = {r.team_id: r for r in ratings}
        logger.debug(f"Loaded baseline: {len(self._last_ratings)} teams")

    def _detect_changes(
        self,
        new_ratings: list[TeamRating],
    ) -> list[ChangeEvent]:
        """Detect changes between new ratings and baseline.

        Args:
            new_ratings: Fresh ratings from API.

        Returns:
            List of detected change events.
        """
        changes = []
        now = datetime.now()

        for rating in new_ratings:
            old_rating = self._last_ratings.get(rating.team_id)

            if old_rating is None:
                # New team (rare, but possible)
                changes.append(ChangeEvent(
                    change_type=ChangeType.NEW_DATA,
                    team_id=rating.team_id,
                    team_name=rating.team_name,
                    field="adj_em",
                    old_value=None,
                    new_value=rating.adj_em,
                    change_amount=0.0,
                    detected_at=now,
                ))
                continue

            # Check each monitored field
            for field in ["adj_em", "adj_oe", "adj_de", "adj_tempo", "pythag"]:
                old_value = getattr(old_rating, field, None)
                new_value = getattr(rating, field, None)

                if old_value is None or new_value is None:
                    continue

                change_amount = new_value - old_value
                threshold = self.thresholds.get(field, 1.0)

                if abs(change_amount) > 0.001:  # Any change
                    change_type = (
                        ChangeType.SIGNIFICANT_MOVE
                        if abs(change_amount) >= threshold
                        else ChangeType.RATING_UPDATE
                    )

                    changes.append(ChangeEvent(
                        change_type=change_type,
                        team_id=rating.team_id,
                        team_name=rating.team_name,
                        field=field,
                        old_value=old_value,
                        new_value=new_value,
                        change_amount=round(change_amount, 2),
                        detected_at=now,
                    ))

            # Check rank changes
            old_rank = old_rating.rank_adj_em
            new_rank = rating.rank_adj_em

            if old_rank and new_rank and old_rank != new_rank:
                rank_change = old_rank - new_rank  # Positive = improved
                threshold = self.thresholds.get("rank_adj_em", 5)

                if abs(rank_change) >= threshold:
                    changes.append(ChangeEvent(
                        change_type=ChangeType.RANK_CHANGE,
                        team_id=rating.team_id,
                        team_name=rating.team_name,
                        field="rank_adj_em",
                        old_value=old_rank,
                        new_value=new_rank,
                        change_amount=rank_change,
                        detected_at=now,
                        metadata={"direction": "up" if rank_change > 0 else "down"},
                    ))

        return changes

    def _trigger_callbacks(self, changes: list[ChangeEvent]) -> None:
        """Trigger registered callbacks for detected changes.

        Args:
            changes: List of change events.
        """
        for change in changes:
            # Trigger general change callbacks
            for callback in self._on_change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Trigger significant move callbacks if applicable
            if change.change_type == ChangeType.SIGNIFICANT_MOVE:
                for threshold, callback in self._on_significant_callbacks:
                    if abs(change.change_amount) >= threshold:
                        try:
                            callback(change)
                        except Exception as e:
                            logger.error(f"Significant callback error: {e}")

    def poll(self, year: int | None = None) -> list[ChangeEvent]:
        """Poll for rating changes.

        Fetches current ratings from API, compares to baseline,
        triggers callbacks for any changes, and updates baseline.

        Args:
            year: Season year (defaults to current).

        Returns:
            List of detected change events.
        """
        from datetime import date

        year = year or (date.today().year if date.today().month >= 11 else date.today().year)

        # Load baseline if not present
        if not self._last_ratings:
            self._load_baseline()

        logger.debug("Polling for rating changes...")

        try:
            # Fetch current ratings
            response = self.api.get_ratings(year=year)
            new_ratings = [
                TeamRating(
                    team_id=r.get("TeamID", 0),
                    team_name=r.get("TeamName", "Unknown"),
                    snapshot_date=date.today(),
                    season=year,
                    adj_em=r.get("AdjEM", 0.0),
                    adj_oe=r.get("AdjOE", 100.0),
                    adj_de=r.get("AdjDE", 100.0),
                    adj_tempo=r.get("AdjTempo", 67.0),
                    luck=r.get("Luck", 0.0),
                    sos=r.get("SOS", 0.0),
                    pythag=r.get("Pythag", 0.5),
                    rank_adj_em=r.get("RankAdjEM"),
                    rank_adj_oe=r.get("RankAdjOE"),
                    rank_adj_de=r.get("RankAdjDE"),
                    rank_tempo=r.get("RankAdjTempo"),
                    wins=r.get("Wins", 0),
                    losses=r.get("Losses", 0),
                    conference=r.get("ConfShort"),
                )
                for r in response.data
            ]

            # Trigger new data callbacks
            for callback in self._on_new_data_callbacks:
                try:
                    callback(new_ratings)
                except Exception as e:
                    logger.error(f"New data callback error: {e}")

            # Detect changes
            changes = self._detect_changes(new_ratings)

            # Trigger change callbacks
            if changes:
                logger.info(f"Detected {len(changes)} rating changes")
                self._trigger_callbacks(changes)

            # Update baseline
            self._last_ratings = {r.team_id: r for r in new_ratings}
            self._last_poll = datetime.now()

            return changes

        except Exception as e:
            logger.error(f"Poll failed: {e}")
            raise

    def _poll_loop(self, interval: int) -> None:
        """Background polling loop.

        Args:
            interval: Seconds between polls.
        """
        logger.info(f"Starting monitor loop (interval: {interval}s)")

        while not self._stop_event.is_set():
            try:
                self.poll()
            except Exception as e:
                logger.error(f"Poll loop error: {e}")

            # Wait for interval or stop signal
            self._stop_event.wait(timeout=interval)

        logger.info("Monitor loop stopped")

    def start(self, poll_interval: int = 300) -> None:
        """Start background monitoring.

        Args:
            poll_interval: Seconds between polls (default 5 minutes).
        """
        if self._is_running:
            logger.warning("Monitor already running")
            return

        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(poll_interval,),
            daemon=True,
        )
        self._poll_thread.start()
        self._is_running = True

        logger.info("Real-time monitor started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop background monitoring.

        Args:
            timeout: Seconds to wait for thread to stop.
        """
        if not self._is_running:
            return

        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=timeout)

        self._is_running = False
        logger.info("Real-time monitor stopped")

    def get_movers(
        self,
        hours: int = 24,
        min_change: float = 1.0,
        limit: int = 20,
    ) -> list[RatingChange]:
        """Get teams with the biggest rating moves in the last N hours.

        Args:
            hours: Hours to look back.
            min_change: Minimum change to include.
            limit: Maximum results.

        Returns:
            List of RatingChange objects sorted by change magnitude.
        """
        from datetime import date

        cutoff_date = date.today() - timedelta(days=max(1, hours // 24))

        movers = []
        current_ratings = self.repository.get_latest_ratings()

        for rating in current_ratings:
            history = self.repository.get_team_rating_history(
                rating.team_id, days=max(1, hours // 24 + 1)
            )

            if len(history) < 2:
                continue

            old = history[-1]  # Oldest in range
            change = rating.adj_em - old.adj_em

            if abs(change) >= min_change:
                movers.append(RatingChange(
                    team_id=rating.team_id,
                    team_name=rating.team_name,
                    old_rating=old.adj_em,
                    new_rating=rating.adj_em,
                    change=round(change, 2),
                    field="adj_em",
                ))

        # Sort by absolute change
        movers.sort(key=lambda m: abs(m.change), reverse=True)

        return movers[:limit]

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._is_running

    @property
    def last_poll_time(self) -> datetime | None:
        """Get time of last poll."""
        return self._last_poll

    @property
    def teams_tracked(self) -> int:
        """Number of teams being tracked."""
        return len(self._last_ratings)
