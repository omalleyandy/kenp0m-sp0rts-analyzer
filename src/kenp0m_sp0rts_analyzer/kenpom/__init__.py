"""KenPom data integration module for NCAA basketball analytics.

This package provides a complete solution for:
- API integration with KenPom.com
- Historical data archival (2023+)
- Daily batch synchronization
- Real-time change monitoring
- ML feature generation
- Prediction accuracy tracking

Architecture Overview
---------------------
┌─────────────────────────────────────────────────────────────┐
│                    Kenpom API                               │
│                 (api_client.py)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼────┐      ┌───▼────┐      ┌───▼────┐
    │ Batch  │      │ Real-  │      │Archive │
    │ Jobs   │      │ time   │      │Loader  │
    │(Daily) │      │Monitor │      │(2023+) │
    └───┬────┘      └───┬────┘      └───┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      Data Validation &          │
        │    Error Handling Layer         │
        │   (validators.py, exceptions.py)│
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      SQLite3 Database           │
        │  (Teams, Ratings, Historical,   │
        │   Predictions, Game Stats)      │
        │   (database.py, repository.py)  │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    ML Feature Engineering       │
        │   & Predictive Modeling         │
        │      (models.py, api.py)        │
        └─────────────────────────────────┘

Quick Start
-----------
>>> from kenp0m_sp0rts_analyzer.kenpom import KenPomService
>>>
>>> # Initialize the service
>>> service = KenPomService()
>>>
>>> # Get current ratings
>>> ratings = service.get_latest_ratings()
>>> print(f"Top team: {ratings[0].team_name} ({ratings[0].adj_em})")
>>>
>>> # Get matchup data for ML
>>> features = service.get_features_for_game(team1_id=73, team2_id=152)
>>>
>>> # Sync from API (requires API key)
>>> result = service.sync_all()
>>> print(f"Synced {result['ratings'].records_synced} ratings")

Daily Workflow
--------------
>>> from kenp0m_sp0rts_analyzer.kenpom import BatchScheduler
>>>
>>> scheduler = BatchScheduler()
>>>
>>> # Run complete daily workflow
>>> result = scheduler.run_daily_workflow()
>>> if result.success:
...     print(f"All tasks succeeded: {result.total_records} records")

Historical Data
---------------
>>> from kenp0m_sp0rts_analyzer.kenpom import ArchiveLoader
>>> from datetime import date
>>>
>>> loader = ArchiveLoader()
>>>
>>> # Backfill a season
>>> result = loader.backfill_season(season=2024)
>>> print(f"Filled {result.dates_filled} dates")
>>>
>>> # Load specific date
>>> ratings = loader.load_date("2024-02-15")

Real-time Monitoring
--------------------
>>> from kenp0m_sp0rts_analyzer.kenpom import RealtimeMonitor
>>>
>>> monitor = RealtimeMonitor()
>>>
>>> @monitor.on_significant_move(threshold=2.0)
... def handle_big_move(event):
...     print(f"Big move! {event.team_name}: {event.change_amount:+.1f}")
>>>
>>> # Start background monitoring
>>> monitor.start(poll_interval=300)

Database Schema
---------------
Tables:
- teams: Team information (id, name, conference, coach)
- ratings_snapshots: Daily rating snapshots with UNIQUE(date, team_id)
- four_factors: Dean Oliver's Four Factors data
- point_distribution: Point source distribution (FT, 2P, 3P)
- height_experience: Height and experience metrics
- game_predictions: Predictions for accuracy tracking
- accuracy_metrics: Daily accuracy summaries
- sync_history: Data sync operation logs

Environment Variables
---------------------
- KENPOM_API_KEY: Required for API access

See Also
--------
- KENPOM_API.md: Full API documentation
- prediction.py: XGBoost prediction model
- api_client.py: Low-level API client
"""

# Core service
from .api import KenPomService

# Data collection components
from .archive_loader import ArchiveLoader
from .batch_scheduler import BatchScheduler, DailyWorkflowResult, ScheduledTask
from .realtime_monitor import ChangeEvent, ChangeType, RealtimeMonitor

# Database layer
from .database import DatabaseManager
from .repository import KenPomRepository

# Data models
from .models import (
    AccuracyReport,
    BackfillResult,
    DailySnapshot,
    FourFactors,
    GamePrediction,
    GameResult,
    HeightExperience,
    MatchupData,
    PointDistribution,
    RatingChange,
    SeasonData,
    SyncResult,
    Team,
    TeamRating,
)

# Validation
from .validators import DataValidator, ValidationResult, Anomaly

# Exceptions
from .exceptions import (
    ArchiveNotAvailableError,
    ConfigurationError,
    DatabaseError,
    DataValidationError,
    KenPomError,
    RateLimitError,
    SyncError,
    TeamNotFoundError,
)

__version__ = "1.0.0"
__author__ = "kenp0m-sp0rts-analyzer"

__all__ = [
    # Core service
    "KenPomService",
    # Data collection
    "ArchiveLoader",
    "BatchScheduler",
    "DailyWorkflowResult",
    "ScheduledTask",
    "RealtimeMonitor",
    "ChangeEvent",
    "ChangeType",
    # Database
    "DatabaseManager",
    "KenPomRepository",
    # Models
    "AccuracyReport",
    "BackfillResult",
    "DailySnapshot",
    "FourFactors",
    "GamePrediction",
    "GameResult",
    "HeightExperience",
    "MatchupData",
    "PointDistribution",
    "RatingChange",
    "SeasonData",
    "SyncResult",
    "Team",
    "TeamRating",
    # Validation
    "DataValidator",
    "ValidationResult",
    "Anomaly",
    # Exceptions
    "ArchiveNotAvailableError",
    "ConfigurationError",
    "DatabaseError",
    "DataValidationError",
    "KenPomError",
    "RateLimitError",
    "SyncError",
    "TeamNotFoundError",
]


def get_version() -> str:
    """Get the module version."""
    return __version__


def create_service(db_path: str = "data/kenpom.db") -> KenPomService:
    """Factory function to create a KenPomService instance.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Configured KenPomService instance.

    Example:
        >>> service = create_service()
        >>> ratings = service.get_latest_ratings()
    """
    return KenPomService(db_path=db_path)


def create_scheduler(db_path: str = "data/kenpom.db") -> BatchScheduler:
    """Factory function to create a BatchScheduler instance.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Configured BatchScheduler instance.

    Example:
        >>> scheduler = create_scheduler()
        >>> result = scheduler.run_daily_workflow()
    """
    return BatchScheduler(db_path=db_path)


def create_loader(db_path: str = "data/kenpom.db") -> ArchiveLoader:
    """Factory function to create an ArchiveLoader instance.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Configured ArchiveLoader instance.

    Example:
        >>> loader = create_loader()
        >>> result = loader.backfill_season(2024)
    """
    return ArchiveLoader(db_path=db_path)


def create_monitor(db_path: str = "data/kenpom.db") -> RealtimeMonitor:
    """Factory function to create a RealtimeMonitor instance.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Configured RealtimeMonitor instance.

    Example:
        >>> monitor = create_monitor()
        >>> @monitor.on_rating_change
        ... def handle(event): print(event)
        >>> monitor.start()
    """
    return RealtimeMonitor(db_path=db_path)
