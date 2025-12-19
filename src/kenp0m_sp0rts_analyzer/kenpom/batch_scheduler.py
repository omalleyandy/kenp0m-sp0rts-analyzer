"""Batch scheduler for daily KenPom data synchronization.

This module provides scheduling infrastructure for automated daily
data collection, validation, and storage operations.

Designed to integrate with:
- cron/systemd on Linux
- Task Scheduler on Windows
- Any Python scheduler (APScheduler, schedule, etc.)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Callable

from .database import DatabaseManager
from .exceptions import SyncError
from .models import SyncResult
from .repository import KenPomRepository
from .validators import DataValidator

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Status of a sync operation."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScheduledTask:
    """A scheduled sync task."""

    name: str
    endpoint: str
    sync_func: Callable
    priority: int = 1
    enabled: bool = True
    last_run: datetime | None = None
    last_status: SyncStatus = SyncStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class DailyWorkflowResult:
    """Result of a complete daily workflow run."""

    run_date: date
    started_at: datetime
    completed_at: datetime
    tasks_run: int
    tasks_succeeded: int
    tasks_failed: int
    total_records: int
    results: dict[str, SyncResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.tasks_failed == 0

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


class BatchScheduler:
    """Orchestrates daily KenPom data synchronization.

    Manages the complete daily workflow:
    1. Sync team ratings
    2. Sync Four Factors
    3. Sync Point Distribution
    4. Validate all data
    5. Update accuracy metrics

    Can be run manually or integrated with system schedulers.

    Example:
        >>> from kenp0m_sp0rts_analyzer.kenpom import BatchScheduler
        >>>
        >>> scheduler = BatchScheduler()
        >>>
        >>> # Run complete daily workflow
        >>> result = scheduler.run_daily_workflow()
        >>> print(f"Synced {result.total_records} records")
        >>>
        >>> # Run with custom callbacks
        >>> def on_task_complete(task, result):
        ...     print(f"{task.name}: {result.records_synced} records")
        >>>
        >>> scheduler.on_task_complete = on_task_complete
        >>> scheduler.run_daily_workflow()
    """

    # Default schedule times (can be overridden)
    MORNING_SYNC_HOUR = 6  # 6 AM - before games
    EVENING_SYNC_HOUR = 23  # 11 PM - after most games

    def __init__(
        self,
        db_path: str = "data/kenpom.db",
        api_client: Any = None,
    ):
        """Initialize the batch scheduler.

        Args:
            db_path: Path to SQLite database.
            api_client: Optional KenPomAPI instance.
        """
        self.db_path = db_path
        self.repository = KenPomRepository(db_path)
        self.validator = DataValidator()
        self._api_client = api_client
        self._api_initialized = False

        # Task registry
        self.tasks: list[ScheduledTask] = []
        self._register_default_tasks()

        # Callbacks
        self.on_task_start: Callable[[ScheduledTask], None] | None = None
        self.on_task_complete: Callable[[ScheduledTask, SyncResult], None] | None = None
        self.on_workflow_complete: Callable[[DailyWorkflowResult], None] | None = None

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

    def _register_default_tasks(self) -> None:
        """Register the default sync tasks."""
        self.tasks = [
            ScheduledTask(
                name="ratings",
                endpoint="ratings",
                sync_func=self._sync_ratings,
                priority=1,
            ),
            ScheduledTask(
                name="four_factors",
                endpoint="four-factors",
                sync_func=self._sync_four_factors,
                priority=2,
            ),
            ScheduledTask(
                name="point_distribution",
                endpoint="pointdist",
                sync_func=self._sync_point_distribution,
                priority=3,
            ),
        ]

    def _get_current_season(self) -> int:
        """Get the current season year."""
        today = date.today()
        return today.year if today.month >= 11 else today.year

    def _sync_ratings(self, year: int, snapshot_date: date) -> SyncResult:
        """Sync team ratings."""
        start_time = datetime.now()
        errors = []

        try:
            response = self.api.get_ratings(year=year)
            data = list(response.data)

            sanitized = self.validator.sanitize_response(data)
            count = self.repository.save_ratings_snapshot(
                snapshot_date=snapshot_date,
                season=year,
                ratings=sanitized,
            )
            self.repository.upsert_teams(sanitized)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Synced {count} ratings in {duration:.2f}s")

            return SyncResult(
                success=True,
                endpoint="ratings",
                records_synced=count,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Ratings sync failed: {e}")
            errors.append(str(e))
            return SyncResult(
                success=False,
                endpoint="ratings",
                records_synced=0,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _sync_four_factors(self, year: int, snapshot_date: date) -> SyncResult:
        """Sync Four Factors data."""
        start_time = datetime.now()
        errors = []

        try:
            response = self.api.get_four_factors(year=year)
            data = list(response.data)

            count = self.repository.save_four_factors(snapshot_date, data)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Synced {count} four factors in {duration:.2f}s")

            return SyncResult(
                success=True,
                endpoint="four-factors",
                records_synced=count,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Four factors sync failed: {e}")
            errors.append(str(e))
            return SyncResult(
                success=False,
                endpoint="four-factors",
                records_synced=0,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _sync_point_distribution(self, year: int, snapshot_date: date) -> SyncResult:
        """Sync point distribution data."""
        start_time = datetime.now()
        errors = []

        try:
            response = self.api.get_pointdist(year=year)
            data = list(response.data)

            count = self.repository.save_point_distribution(snapshot_date, data)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Synced {count} point distributions in {duration:.2f}s")

            return SyncResult(
                success=True,
                endpoint="pointdist",
                records_synced=count,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Point distribution sync failed: {e}")
            errors.append(str(e))
            return SyncResult(
                success=False,
                endpoint="pointdist",
                records_synced=0,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def run_task(
        self,
        task: ScheduledTask,
        year: int | None = None,
        snapshot_date: date | None = None,
    ) -> SyncResult:
        """Run a single sync task.

        Args:
            task: The task to run.
            year: Season year (defaults to current).
            snapshot_date: Date for snapshot (defaults to today).

        Returns:
            SyncResult with operation details.
        """
        year = year or self._get_current_season()
        snapshot_date = snapshot_date or date.today()

        if self.on_task_start:
            self.on_task_start(task)

        task.last_run = datetime.now()
        task.last_status = SyncStatus.RUNNING

        result = task.sync_func(year, snapshot_date)

        if result.success:
            task.last_status = SyncStatus.SUCCESS
            task.retry_count = 0
        else:
            task.retry_count += 1
            if task.retry_count >= task.max_retries:
                task.last_status = SyncStatus.FAILED
            else:
                task.last_status = SyncStatus.PARTIAL

        if self.on_task_complete:
            self.on_task_complete(task, result)

        # Record in database
        self.repository.db.record_sync(
            endpoint=task.endpoint,
            sync_type="daily",
            status=task.last_status.value,
            records_synced=result.records_synced,
            records_skipped=result.records_skipped,
            error_message="; ".join(result.errors) if result.errors else None,
            started_at=task.last_run,
            completed_at=datetime.now(),
        )

        return result

    def run_daily_workflow(
        self,
        year: int | None = None,
        snapshot_date: date | None = None,
        retry_failed: bool = True,
    ) -> DailyWorkflowResult:
        """Run the complete daily sync workflow.

        Executes all registered tasks in priority order, handling errors
        and retries automatically.

        Args:
            year: Season year (defaults to current).
            snapshot_date: Date for snapshot (defaults to today).
            retry_failed: Whether to retry failed tasks.

        Returns:
            DailyWorkflowResult with complete summary.

        Example:
            >>> scheduler = BatchScheduler()
            >>> result = scheduler.run_daily_workflow()
            >>> if result.success:
            ...     print(f"All tasks succeeded: {result.total_records} records")
            >>> else:
            ...     print(f"Errors: {result.errors}")
        """
        year = year or self._get_current_season()
        snapshot_date = snapshot_date or date.today()

        started_at = datetime.now()
        results: dict[str, SyncResult] = {}
        all_errors: list[str] = []

        logger.info(f"Starting daily workflow for {snapshot_date} (season {year})")

        # Sort tasks by priority
        sorted_tasks = sorted(
            [t for t in self.tasks if t.enabled],
            key=lambda t: t.priority,
        )

        for task in sorted_tasks:
            try:
                result = self.run_task(task, year, snapshot_date)
                results[task.name] = result

                if not result.success:
                    all_errors.extend(result.errors)

                    # Retry if enabled
                    if retry_failed and task.retry_count < task.max_retries:
                        logger.info(f"Retrying {task.name} ({task.retry_count}/{task.max_retries})")
                        time.sleep(2)  # Brief pause before retry
                        result = self.run_task(task, year, snapshot_date)
                        results[task.name] = result

            except Exception as e:
                logger.error(f"Task {task.name} raised exception: {e}")
                all_errors.append(f"{task.name}: {str(e)}")
                results[task.name] = SyncResult(
                    success=False,
                    endpoint=task.endpoint,
                    records_synced=0,
                    errors=[str(e)],
                    duration_seconds=0.0,
                )

        completed_at = datetime.now()

        workflow_result = DailyWorkflowResult(
            run_date=snapshot_date,
            started_at=started_at,
            completed_at=completed_at,
            tasks_run=len(results),
            tasks_succeeded=sum(1 for r in results.values() if r.success),
            tasks_failed=sum(1 for r in results.values() if not r.success),
            total_records=sum(r.records_synced for r in results.values()),
            results=results,
            errors=all_errors,
        )

        if self.on_workflow_complete:
            self.on_workflow_complete(workflow_result)

        logger.info(
            f"Daily workflow complete: {workflow_result.tasks_succeeded}/{workflow_result.tasks_run} "
            f"tasks succeeded, {workflow_result.total_records} records in "
            f"{workflow_result.duration_seconds:.1f}s"
        )

        return workflow_result

    def check_should_run(self) -> tuple[bool, str]:
        """Check if the daily workflow should run.

        Checks if data is stale and needs refresh.

        Returns:
            Tuple of (should_run, reason).
        """
        last_sync = self.repository.db.get_latest_sync("ratings")

        if last_sync is None:
            return True, "No previous sync found"

        last_date = last_sync.get("completed_at")
        if last_date is None:
            return True, "No completion time recorded"

        # Check if last sync was today
        if isinstance(last_date, datetime):
            last_date = last_date.date()

        if last_date >= date.today():
            return False, f"Already synced today ({last_date})"

        # Check if it's been too long
        days_since = (date.today() - last_date).days
        if days_since > 1:
            return True, f"Last sync was {days_since} days ago"

        return True, "Daily sync needed"

    def get_status(self) -> dict:
        """Get current scheduler status.

        Returns:
            Dictionary with task statuses and sync history.
        """
        status = {
            "tasks": [],
            "last_workflow": None,
            "database_stats": self.repository.db.get_stats(),
        }

        for task in self.tasks:
            status["tasks"].append({
                "name": task.name,
                "endpoint": task.endpoint,
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "last_status": task.last_status.value,
                "retry_count": task.retry_count,
            })

        # Get most recent sync for each endpoint
        for task in self.tasks:
            sync = self.repository.db.get_latest_sync(task.endpoint)
            if sync:
                if status["last_workflow"] is None:
                    status["last_workflow"] = sync
                elif sync.get("completed_at", datetime.min) > status["last_workflow"].get(
                    "completed_at", datetime.min
                ):
                    status["last_workflow"] = sync

        return status

    def enable_task(self, task_name: str) -> None:
        """Enable a task by name."""
        for task in self.tasks:
            if task.name == task_name:
                task.enabled = True
                logger.info(f"Enabled task: {task_name}")
                return
        raise ValueError(f"Task not found: {task_name}")

    def disable_task(self, task_name: str) -> None:
        """Disable a task by name."""
        for task in self.tasks:
            if task.name == task_name:
                task.enabled = False
                logger.info(f"Disabled task: {task_name}")
                return
        raise ValueError(f"Task not found: {task_name}")

    def register_task(
        self,
        name: str,
        endpoint: str,
        sync_func: Callable,
        priority: int = 10,
    ) -> None:
        """Register a custom sync task.

        Args:
            name: Task name.
            endpoint: API endpoint.
            sync_func: Function to call (receives year, snapshot_date).
            priority: Priority (lower runs first).
        """
        task = ScheduledTask(
            name=name,
            endpoint=endpoint,
            sync_func=sync_func,
            priority=priority,
        )
        self.tasks.append(task)
        logger.info(f"Registered task: {name} (priority {priority})")


def create_cron_command() -> str:
    """Generate a cron command for daily scheduling.

    Returns:
        String suitable for crontab.

    Example:
        >>> print(create_cron_command())
        0 6 * * * cd /path/to/project && python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler
    """
    import sys
    return (
        f"0 6 * * * cd {sys.path[0]} && "
        f"python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler"
    )


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="KenPom Daily Sync Scheduler")
    parser.add_argument("--check", action="store_true", help="Check if sync is needed")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--force", action="store_true", help="Force sync even if recent")
    parser.add_argument("--year", type=int, help="Season year to sync")
    parser.add_argument("--db-path", default="data/kenpom.db", help="Database path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    scheduler = BatchScheduler(db_path=args.db_path)

    if args.status:
        import json
        status = scheduler.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.check:
        should_run, reason = scheduler.check_should_run()
        print(f"Should run: {should_run}")
        print(f"Reason: {reason}")
        return

    # Run the workflow
    should_run, reason = scheduler.check_should_run()

    if not should_run and not args.force:
        print(f"Skipping sync: {reason}")
        return

    result = scheduler.run_daily_workflow(year=args.year)

    if result.success:
        print(f"✓ Sync complete: {result.total_records} records in {result.duration_seconds:.1f}s")
    else:
        print(f"✗ Sync failed: {result.errors}")
        exit(1)


if __name__ == "__main__":
    main()
