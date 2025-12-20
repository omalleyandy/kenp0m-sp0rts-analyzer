"""Overtime.ag timing analysis and monitoring utilities.

This module provides:
- TimingDatabase: SQLite storage for API snapshots and game tracking
- TimingAnalyzer: Analysis of odds release patterns
- Utility functions for game ID generation and timestamp field discovery

Usage:
    from kenp0m_sp0rts_analyzer.overtime_timing import (
        TimingDatabase,
        TimingAnalyzer,
        get_monitoring_dir,
    )

    # Store snapshots
    db = TimingDatabase()
    db.save_api_snapshot(endpoint, response_data, game_count)

    # Analyze patterns
    analyzer = TimingAnalyzer()
    report = analyzer.generate_report()
"""

import json
import logging
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default monitoring directory
MONITORING_DIR = (
    Path(__file__).parent.parent.parent / "data" / "overtime_monitoring"
)


def get_monitoring_dir() -> Path:
    """Get the monitoring data directory, creating if needed."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    return MONITORING_DIR


def generate_game_id(
    home_team: str,
    away_team: str,
    game_date: date | str | None = None,
) -> str:
    """Generate a unique game ID from team names and date.

    Args:
        home_team: Home team name
        away_team: Away team name
        game_date: Game date (defaults to today)

    Returns:
        Unique game ID string
    """
    if game_date is None:
        game_date = date.today()
    elif isinstance(game_date, str):
        game_date = datetime.strptime(game_date, "%Y-%m-%d").date()

    # Normalize team names
    home_norm = re.sub(r"[^\w]", "", home_team.lower())[:20]
    away_norm = re.sub(r"[^\w]", "", away_team.lower())[:20]

    return f"cbb_{game_date.isoformat()}_{away_norm}_{home_norm}"


def find_timestamp_fields(
    data: dict[str, Any],
    prefix: str = "",
    db: "TimingDatabase | None" = None,
) -> dict[str, str]:
    """Recursively find timestamp-like fields in JSON data.

    Args:
        data: Dictionary to search
        prefix: Current key path prefix
        db: Optional TimingDatabase to record discoveries

    Returns:
        Dictionary of field_path -> sample_value
    """
    timestamps = {}

    timestamp_patterns = [
        r"^\d{4}-\d{2}-\d{2}",  # ISO date
        r"^\d{4}/\d{2}/\d{2}",  # Slash date
        r"^\d{10,13}$",  # Unix timestamp
        r"^\d{2}:\d{2}",  # Time
        r"T\d{2}:\d{2}",  # ISO datetime T portion
    ]

    timestamp_keywords = [
        "time",
        "date",
        "created",
        "updated",
        "posted",
        "modified",
        "start",
        "end",
        "open",
        "close",
    ]

    for key, value in data.items():
        field_path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recurse into nested objects
            nested = find_timestamp_fields(value, field_path, db)
            timestamps.update(nested)

        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Recurse into first item of list
            nested = find_timestamp_fields(value[0], f"{field_path}[0]", db)
            timestamps.update(nested)

        elif isinstance(value, str):
            # Check if field name suggests timestamp
            key_lower = key.lower()
            is_keyword = any(kw in key_lower for kw in timestamp_keywords)

            # Check if value matches timestamp pattern
            is_pattern = any(re.search(p, value) for p in timestamp_patterns)

            if is_keyword or is_pattern:
                timestamps[field_path] = value

                if db:
                    db.record_timestamp_field(field_path, value)

        elif isinstance(value, (int, float)):
            # Check for Unix timestamps (10-13 digit numbers)
            if 1_000_000_000 < value < 10_000_000_000_000:
                timestamps[field_path] = str(value)
                if db:
                    db.record_timestamp_field(field_path, str(value))

    return timestamps


class TimingDatabase:
    """SQLite database for overtime.ag timing analysis.

    Stores:
    - API response snapshots
    - Game tracking (first appearance times)
    - Discovered timestamp fields
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the timing database.

        Args:
            db_path: Path to SQLite database (default: monitoring dir)
        """
        if db_path is None:
            db_path = get_monitoring_dir() / "overtime_odds.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = None
        self._initialize()

    def _initialize(self) -> None:
        """Create database tables."""
        conn = self._get_connection()

        conn.executescript(
            """
            -- API response snapshots
            CREATE TABLE IF NOT EXISTS api_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                response_size INTEGER,
                game_count INTEGER,
                raw_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_time
                ON api_snapshots(captured_at);
            CREATE INDEX IF NOT EXISTS idx_snapshots_endpoint
                ON api_snapshots(endpoint);

            -- Game tracking
            CREATE TABLE IF NOT EXISTS game_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE NOT NULL,
                first_seen_at TEXT NOT NULL,
                game_date TEXT,
                game_time TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                board_type TEXT,
                json_timestamps TEXT,
                raw_game_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_games_first_seen
                ON game_tracking(first_seen_at);
            CREATE INDEX IF NOT EXISTS idx_games_date
                ON game_tracking(game_date);

            -- Timestamp field discovery
            CREATE TABLE IF NOT EXISTS timestamp_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_path TEXT UNIQUE NOT NULL,
                sample_value TEXT,
                first_seen_at TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_ts_fields_path
                ON timestamp_fields(field_path);

            -- Odds snapshots for line movement tracking
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                snapshot_at TEXT NOT NULL,
                spread REAL,
                total REAL,
                away_ml INTEGER,
                home_ml INTEGER,
                spread_away_odds INTEGER,
                spread_home_odds INTEGER,
                over_odds INTEGER,
                under_odds INTEGER,
                source TEXT DEFAULT 'overtime.ag',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES game_tracking(game_id)
            );

            CREATE INDEX IF NOT EXISTS idx_odds_game
                ON odds_snapshots(game_id);
            CREATE INDEX IF NOT EXISTS idx_odds_time
                ON odds_snapshots(snapshot_at);
        """
        )

        conn.commit()
        logger.debug(f"TimingDatabase initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def save_api_snapshot(
        self,
        endpoint: str,
        response_data: dict | list,
        game_count: int = 0,
        captured_at: datetime | None = None,
    ) -> int:
        """Save an API response snapshot.

        Args:
            endpoint: API endpoint URL
            response_data: Parsed JSON response
            game_count: Number of games in response
            captured_at: Capture timestamp (default: now)

        Returns:
            Snapshot ID
        """
        conn = self._get_connection()
        captured_at = captured_at or datetime.now()

        raw_json = json.dumps(response_data)

        cursor = conn.execute(
            """
            INSERT INTO api_snapshots (captured_at, endpoint, response_size, game_count, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                captured_at.isoformat(),
                endpoint,
                len(raw_json),
                game_count,
                raw_json,
            ),
        )

        conn.commit()
        return cursor.lastrowid or 0

    def track_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        game_time: str = "",
        game_date: str | None = None,
        timestamps: dict[str, str] | None = None,
        raw_data: dict | None = None,
        first_seen_at: datetime | None = None,
        board_type: str = "college_basketball",
    ) -> bool:
        """Track a game's first appearance.

        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            game_time: Game time string
            game_date: Game date string
            timestamps: Discovered timestamp fields
            raw_data: Raw game JSON
            first_seen_at: First seen timestamp
            board_type: Board/category type

        Returns:
            True if this is a new game, False if already tracked
        """
        conn = self._get_connection()
        first_seen_at = first_seen_at or datetime.now()
        game_date = game_date or date.today().isoformat()

        try:
            conn.execute(
                """
                INSERT INTO game_tracking (
                    game_id, first_seen_at, game_date, game_time,
                    home_team, away_team, board_type, json_timestamps, raw_game_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    first_seen_at.isoformat(),
                    game_date,
                    game_time,
                    home_team,
                    away_team,
                    board_type,
                    json.dumps(timestamps) if timestamps else None,
                    json.dumps(raw_data) if raw_data else None,
                ),
            )
            conn.commit()
            logger.info(f"New game tracked: {game_id}")
            return True

        except sqlite3.IntegrityError:
            # Game already exists
            return False

    def save_odds_snapshot(
        self,
        game_id: str,
        spread: float | None = None,
        total: float | None = None,
        away_ml: int | None = None,
        home_ml: int | None = None,
        spread_away_odds: int | None = None,
        spread_home_odds: int | None = None,
        over_odds: int | None = None,
        under_odds: int | None = None,
        snapshot_at: datetime | None = None,
    ) -> int:
        """Save an odds snapshot for line movement tracking.

        Args:
            game_id: Game identifier
            spread: Point spread (home team perspective)
            total: Over/under total
            away_ml: Away team moneyline
            home_ml: Home team moneyline
            spread_away_odds: Away spread odds
            spread_home_odds: Home spread odds
            over_odds: Over odds
            under_odds: Under odds
            snapshot_at: Snapshot timestamp

        Returns:
            Snapshot ID
        """
        conn = self._get_connection()
        snapshot_at = snapshot_at or datetime.now()

        cursor = conn.execute(
            """
            INSERT INTO odds_snapshots (
                game_id, snapshot_at, spread, total, away_ml, home_ml,
                spread_away_odds, spread_home_odds, over_odds, under_odds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                snapshot_at.isoformat(),
                spread,
                total,
                away_ml,
                home_ml,
                spread_away_odds,
                spread_home_odds,
                over_odds,
                under_odds,
            ),
        )

        conn.commit()
        return cursor.lastrowid or 0

    def record_timestamp_field(
        self,
        field_path: str,
        sample_value: str,
    ) -> None:
        """Record a discovered timestamp field.

        Args:
            field_path: JSON path to the field
            sample_value: Example value found
        """
        conn = self._get_connection()

        try:
            conn.execute(
                """
                INSERT INTO timestamp_fields (field_path, sample_value, first_seen_at)
                VALUES (?, ?, ?)
                """,
                (field_path, sample_value, datetime.now().isoformat()),
            )
        except sqlite3.IntegrityError:
            # Field exists, increment count
            conn.execute(
                """
                UPDATE timestamp_fields
                SET occurrence_count = occurrence_count + 1
                WHERE field_path = ?
                """,
                (field_path,),
            )

        conn.commit()

    def get_games_for_date(self, game_date: str | date) -> list[dict]:
        """Get all tracked games for a specific date.

        Args:
            game_date: Date to query

        Returns:
            List of game records
        """
        conn = self._get_connection()

        if isinstance(game_date, date):
            game_date = game_date.isoformat()

        cursor = conn.execute(
            """
            SELECT * FROM game_tracking
            WHERE game_date = ?
            ORDER BY game_time
            """,
            (game_date,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_latest_odds(self, game_id: str) -> dict | None:
        """Get the most recent odds snapshot for a game.

        Args:
            game_id: Game identifier

        Returns:
            Latest odds record or None
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT * FROM odds_snapshots
            WHERE game_id = ?
            ORDER BY snapshot_at DESC
            LIMIT 1
            """,
            (game_id,),
        )

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_line_movement(self, game_id: str) -> list[dict]:
        """Get line movement history for a game.

        Args:
            game_id: Game identifier

        Returns:
            List of odds snapshots in chronological order
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT * FROM odds_snapshots
            WHERE game_id = ?
            ORDER BY snapshot_at
            """,
            (game_id,),
        )

        return [dict(row) for row in cursor.fetchall()]


class TimingAnalyzer:
    """Analyze odds release timing patterns."""

    def __init__(self, db: TimingDatabase | None = None):
        """Initialize analyzer with database.

        Args:
            db: TimingDatabase instance (creates default if None)
        """
        self.db = db or TimingDatabase()

    def get_timestamp_field_stats(self) -> list[dict]:
        """Get statistics on discovered timestamp fields.

        Returns:
            List of field statistics sorted by occurrence
        """
        conn = self.db._get_connection()

        cursor = conn.execute(
            """
            SELECT field_path, sample_value, occurrence_count, first_seen_at
            FROM timestamp_fields
            ORDER BY occurrence_count DESC
            """
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_snapshot_summary(self, hours: int = 24) -> dict:
        """Get summary of recent API snapshots.

        Args:
            hours: Number of hours to include

        Returns:
            Summary statistics
        """
        conn = self.db._get_connection()

        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total_snapshots,
                COUNT(DISTINCT endpoint) as unique_endpoints,
                SUM(game_count) as total_games_seen,
                MIN(captured_at) as first_capture,
                MAX(captured_at) as last_capture
            FROM api_snapshots
            WHERE captured_at >= datetime('now', ?)
            """,
            (f"-{hours} hours",),
        )

        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_game_appearance_times(self, days: int = 7) -> list[dict]:
        """Get first appearance times for recent games.

        Args:
            days: Number of days to include

        Returns:
            List of games with first seen timestamps
        """
        conn = self.db._get_connection()

        cursor = conn.execute(
            """
            SELECT game_id, home_team, away_team, game_date, game_time, first_seen_at
            FROM game_tracking
            WHERE first_seen_at >= datetime('now', ?)
            ORDER BY first_seen_at DESC
            """,
            (f"-{days} days",),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_lead_time_stats(self) -> dict:
        """Calculate statistics on how early games appear.

        Returns:
            Lead time statistics
        """
        conn = self.db._get_connection()

        # Get games with parseable times
        cursor = conn.execute(
            """
            SELECT game_date, game_time, first_seen_at
            FROM game_tracking
            WHERE game_date IS NOT NULL AND game_time IS NOT NULL
            """
        )

        lead_times = []
        for row in cursor.fetchall():
            try:
                # Parse game datetime
                game_dt_str = f"{row['game_date']} {row['game_time']}"
                # Try common formats
                for fmt in ["%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M"]:
                    try:
                        game_dt = datetime.strptime(game_dt_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                first_seen = datetime.fromisoformat(row["first_seen_at"])
                lead_hours = (game_dt - first_seen).total_seconds() / 3600

                if lead_hours > 0:
                    lead_times.append(lead_hours)

            except (ValueError, TypeError):
                continue

        if not lead_times:
            return {"error": "No parseable lead times found"}

        return {
            "count": len(lead_times),
            "min_hours": round(min(lead_times), 1),
            "max_hours": round(max(lead_times), 1),
            "avg_hours": round(sum(lead_times) / len(lead_times), 1),
            "median_hours": round(sorted(lead_times)[len(lead_times) // 2], 1),
        }

    def generate_report(self) -> str:
        """Generate a text report of timing analysis.

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "OVERTIME.AG TIMING ANALYSIS REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        # Snapshot summary
        summary = self.get_snapshot_summary(hours=48)
        lines.extend(
            [
                "API SNAPSHOT SUMMARY (Last 48 Hours)",
                "-" * 40,
                f"Total Snapshots: {summary.get('total_snapshots', 0)}",
                f"Unique Endpoints: {summary.get('unique_endpoints', 0)}",
                f"Total Games Seen: {summary.get('total_games_seen', 0)}",
                f"First Capture: {summary.get('first_capture', 'N/A')}",
                f"Last Capture: {summary.get('last_capture', 'N/A')}",
                "",
            ]
        )

        # Lead time stats
        lead_stats = self.get_lead_time_stats()
        if "error" not in lead_stats:
            lines.extend(
                [
                    "LEAD TIME STATISTICS",
                    "-" * 40,
                    f"Games Analyzed: {lead_stats.get('count', 0)}",
                    f"Min Lead Time: {lead_stats.get('min_hours', 0)} hours",
                    f"Max Lead Time: {lead_stats.get('max_hours', 0)} hours",
                    f"Avg Lead Time: {lead_stats.get('avg_hours', 0)} hours",
                    f"Median Lead Time: {lead_stats.get('median_hours', 0)} hours",
                    "",
                ]
            )

        # Timestamp fields
        ts_fields = self.get_timestamp_field_stats()
        if ts_fields:
            lines.extend(
                [
                    "DISCOVERED TIMESTAMP FIELDS",
                    "-" * 40,
                ]
            )
            for field in ts_fields[:10]:  # Top 10
                lines.append(
                    f"  {field['field_path']}: "
                    f"{field['occurrence_count']} occurrences"
                )
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    # Run timing analysis
    logging.basicConfig(level=logging.INFO)

    analyzer = TimingAnalyzer()
    print(analyzer.generate_report())
