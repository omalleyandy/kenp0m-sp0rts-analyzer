"""Overtime.ag Odds Release Timing Analysis.

This module provides tools for discovering when overtime.ag releases
college basketball odds by capturing API responses and analyzing
timestamp patterns.

Integrated with OvertimeScraper for data collection.

Example:
    ```python
    from kenp0m_sp0rts_analyzer.overtime_timing import (
        TimingDatabase,
        TimingAnalyzer,
    )

    # Initialize database
    db = TimingDatabase()

    # After collecting data with OvertimeScraper, analyze patterns
    analyzer = TimingAnalyzer(db)
    report = analyzer.generate_report()
    print(report)
    ```
"""

import json
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default data directory (relative to project root)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_MONITORING_DIR = DEFAULT_DATA_DIR / "overtime_monitoring"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_monitoring_dir() -> Path:
    """Get the overtime monitoring data directory."""
    monitoring_dir = DEFAULT_MONITORING_DIR
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    return monitoring_dir


# Timestamp field patterns for discovery
TIMESTAMP_PATTERNS = [
    'time', 'date', 'at', 'timestamp', 'created', 'updated',
    'opened', 'posted', 'modified', 'when', 'start', 'commence'
]

# Keywords for categorizing timestamp fields
OPENING_KEYWORDS = ['open', 'create', 'post', 'first', 'initial']
UPDATE_KEYWORDS = ['update', 'modify', 'change', 'last', 'recent']
GAMETIME_KEYWORDS = ['start', 'commence', 'begin', 'game_time', 'event_time']


class TimingDatabase:
    """SQLite database for tracking odds release timing patterns.

    Stores:
    - API response snapshots
    - Game tracking (when games first appear)
    - Discovered timestamp fields

    Example:
        ```python
        db = TimingDatabase()

        # Record a captured response
        db.save_api_snapshot(
            endpoint="https://overtime.ag/api/odds",
            response_data={"games": [...]},
            game_count=5,
        )

        # Track a new game
        db.track_game(
            game_id="duke@unc",
            game_time="2025-12-20T19:00:00",
            home_team="UNC",
            away_team="Duke",
            timestamps={"created_at": "2025-12-19T08:00:00"},
        )
        ```
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the timing database.

        Args:
            db_path: Path to SQLite database. Defaults to
                     data/overtime_monitoring/overtime_odds.db
        """
        if db_path is None:
            db_path = get_monitoring_dir() / "overtime_odds.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # API response snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                endpoint TEXT,
                response_size INTEGER,
                game_count INTEGER,
                raw_json TEXT,
                UNIQUE(captured_at, endpoint)
            )
        """)

        # Game tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                game_time TEXT,
                home_team TEXT,
                away_team TEXT,
                board_type TEXT,
                opening_spread TEXT,
                opening_total TEXT,
                opening_ml_home TEXT,
                opening_ml_away TEXT,
                json_timestamps TEXT,
                raw_game_json TEXT,
                UNIQUE(game_id)
            )
        """)

        # Timestamp field discoveries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timestamp_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_path TEXT NOT NULL,
                sample_value TEXT,
                first_seen_at TEXT,
                occurrence_count INTEGER DEFAULT 1,
                UNIQUE(field_path)
            )
        """)

        conn.commit()
        conn.close()
        logger.debug(f"Database initialized: {self.db_path}")

    def save_api_snapshot(
        self,
        endpoint: str,
        response_data: dict | list,
        game_count: int = 0,
        captured_at: datetime | None = None,
    ) -> None:
        """Save an API response snapshot.

        Args:
            endpoint: API endpoint URL
            response_data: Parsed JSON response
            game_count: Number of games in response
            captured_at: Capture timestamp (defaults to now)
        """
        if captured_at is None:
            captured_at = datetime.now()

        raw_json = json.dumps(response_data)
        # Limit stored size
        if len(raw_json) > 100000:
            raw_json = raw_json[:100000]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO api_snapshots
                (captured_at, endpoint, response_size, game_count, raw_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                captured_at.isoformat(),
                endpoint,
                len(raw_json),
                game_count,
                raw_json,
            ))
            conn.commit()
        except Exception as e:
            logger.warning(f"Error saving snapshot: {e}")
        finally:
            conn.close()

    def track_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        game_time: str | None = None,
        board_type: str | None = None,
        timestamps: dict | None = None,
        raw_data: dict | None = None,
        first_seen_at: datetime | None = None,
    ) -> bool:
        """Track when a game first appears.

        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            game_time: Scheduled game time
            board_type: Board type (e.g., "college_basketball")
            timestamps: Extracted timestamp fields from game data
            raw_data: Raw game JSON data
            first_seen_at: When game was first seen (defaults to now)

        Returns:
            True if this is a new game, False if already tracked
        """
        if first_seen_at is None:
            first_seen_at = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO game_tracking
                (game_id, first_seen_at, game_time, home_team, away_team,
                 board_type, json_timestamps, raw_game_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                first_seen_at.isoformat(),
                game_time,
                home_team,
                away_team,
                board_type,
                json.dumps(timestamps) if timestamps else None,
                json.dumps(raw_data)[:50000] if raw_data else None,
            ))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Error tracking game {game_id}: {e}")
            return False
        finally:
            conn.close()

    def record_timestamp_field(
        self,
        field_path: str,
        sample_value: Any,
    ) -> None:
        """Record a discovered timestamp field.

        Args:
            field_path: JSON path to field (e.g., "game.created_at")
            sample_value: Sample value from field
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO timestamp_fields
                (field_path, sample_value, first_seen_at)
                VALUES (?, ?, ?)
                ON CONFLICT(field_path) DO UPDATE SET
                    occurrence_count = occurrence_count + 1
            """, (
                field_path,
                str(sample_value)[:500],
                datetime.now().isoformat(),
            ))
            conn.commit()
        except Exception as e:
            logger.debug(f"Error recording timestamp field: {e}")
        finally:
            conn.close()

    def get_game_count(self) -> int:
        """Get total number of tracked games."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM game_tracking")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_snapshot_count(self) -> int:
        """Get total number of API snapshots."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM api_snapshots")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_timestamp_fields(self) -> list[dict]:
        """Get all discovered timestamp fields.

        Returns:
            List of dicts with field_path, sample_value, occurrence_count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT field_path, sample_value, occurrence_count, first_seen_at
            FROM timestamp_fields
            ORDER BY occurrence_count DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'field_path': r[0],
                'sample_value': r[1],
                'occurrence_count': r[2],
                'first_seen_at': r[3],
            }
            for r in rows
        ]

    def get_games_with_timing(self) -> list[dict]:
        """Get games with first_seen and game_time for lead time analysis.

        Returns:
            List of game dicts with timing information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT game_id, first_seen_at, game_time,
                   home_team, away_team, json_timestamps
            FROM game_tracking
            WHERE game_time IS NOT NULL AND game_time != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'game_id': r[0],
                'first_seen_at': r[1],
                'game_time': r[2],
                'home_team': r[3],
                'away_team': r[4],
                'json_timestamps': r[5],
            }
            for r in rows
        ]

    def get_api_endpoints(self) -> list[dict]:
        """Get summary of discovered API endpoints.

        Returns:
            List of endpoint summaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT endpoint, COUNT(*) as captures,
                   AVG(response_size) as avg_size,
                   AVG(game_count) as avg_games
            FROM api_snapshots
            GROUP BY endpoint
            ORDER BY captures DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'endpoint': r[0],
                'captures': r[1],
                'avg_size': int(r[2]) if r[2] else 0,
                'avg_games': round(r[3], 1) if r[3] else 0,
            }
            for r in rows
        ]


def parse_datetime(dt_str: str) -> datetime | None:
    """Parse various datetime string formats.

    Args:
        dt_str: Datetime string to parse

    Returns:
        Parsed datetime or None if parsing fails
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(dt_str.replace('Z', ''), fmt)
        except ValueError:
            continue

    # Try Unix timestamp
    try:
        ts = float(dt_str)
        if ts > 1e12:  # Milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts)
    except (ValueError, TypeError, OSError):
        pass

    return None


def find_timestamp_fields(
    data: Any,
    path: str = "",
    db: TimingDatabase | None = None,
) -> dict[str, Any]:
    """Recursively find timestamp-like fields in JSON data.

    Args:
        data: JSON data to search
        path: Current path (for recursion)
        db: Optional database to record discoveries

    Returns:
        Dict mapping field paths to values
    """
    timestamps: dict[str, Any] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            key_lower = key.lower()

            # Check if key suggests a timestamp
            if any(p in key_lower for p in TIMESTAMP_PATTERNS):
                timestamps[current_path] = value
                if db:
                    db.record_timestamp_field(current_path, value)

            # Recurse into nested structures
            nested = find_timestamp_fields(value, current_path, db)
            timestamps.update(nested)

    elif isinstance(data, list) and data:
        # Check first item of lists
        nested = find_timestamp_fields(data[0], f"{path}[0]", db)
        timestamps.update(nested)

    return timestamps


def generate_game_id(game_data: dict) -> str | None:
    """Generate a unique game ID from game data.

    Args:
        game_data: Game data dictionary

    Returns:
        Game ID string or None if cannot be generated
    """
    # Try common ID fields
    for id_field in ['id', 'gameId', 'game_id', 'eventId', 'event_id']:
        if id_field in game_data:
            return str(game_data[id_field])

    # Try to construct from teams
    home = game_data.get('home', game_data.get('homeTeam', {}))
    away = game_data.get('away', game_data.get('awayTeam', {}))

    if isinstance(home, dict):
        home = home.get('name', home.get('team', ''))
    if isinstance(away, dict):
        away = away.get('name', away.get('team', ''))

    if home and away:
        return f"{away}@{home}"

    return None


class TimingAnalyzer:
    """Analyzes collected timing data to find odds release patterns.

    Example:
        ```python
        analyzer = TimingAnalyzer()

        # Get summary statistics
        stats = analyzer.get_lead_time_stats()
        print(f"Average lead time: {stats['average_hours']:.1f} hours")

        # Find best timestamp field for opening line detection
        best_field = analyzer.get_best_opening_field()
        print(f"Best opening field: {best_field}")

        # Generate full report
        report = analyzer.generate_report()
        print(report)
        ```
    """

    def __init__(self, db: TimingDatabase | None = None):
        """Initialize the analyzer.

        Args:
            db: TimingDatabase instance. Creates default if not provided.
        """
        self.db = db or TimingDatabase()

    def get_timestamp_field_analysis(self) -> dict:
        """Analyze discovered timestamp fields.

        Returns:
            Dict with categorized timestamp fields
        """
        fields = self.db.get_timestamp_fields()

        result = {
            'all_fields': fields,
            'opening_fields': [],
            'update_fields': [],
            'gametime_fields': [],
        }

        for field in fields:
            path_lower = field['field_path'].lower()

            if any(kw in path_lower for kw in OPENING_KEYWORDS):
                result['opening_fields'].append(field)
            elif any(kw in path_lower for kw in UPDATE_KEYWORDS):
                result['update_fields'].append(field)
            elif any(kw in path_lower for kw in GAMETIME_KEYWORDS):
                result['gametime_fields'].append(field)

        return result

    def get_best_opening_field(self) -> str | None:
        """Get the most likely field indicating opening line timestamp.

        Returns:
            Field path or None if no candidates found
        """
        analysis = self.get_timestamp_field_analysis()
        opening_fields = analysis['opening_fields']

        if opening_fields:
            # Return highest occurrence count
            return opening_fields[0]['field_path']
        return None

    def get_lead_time_stats(self) -> dict:
        """Calculate lead time statistics (time from odds post to game).

        Returns:
            Dict with average, median, min, max lead times in hours
        """
        games = self.db.get_games_with_timing()
        lead_times = []

        for game in games:
            first_seen = parse_datetime(game['first_seen_at'])
            game_time = parse_datetime(str(game['game_time']))

            if first_seen and game_time:
                delta = game_time - first_seen
                hours = delta.total_seconds() / 3600

                # Filter reasonable range (-48h to 7 days)
                if -48 < hours < 168:
                    lead_times.append(hours)

        if not lead_times:
            return {
                'average_hours': None,
                'median_hours': None,
                'min_hours': None,
                'max_hours': None,
                'game_count': 0,
            }

        sorted_times = sorted(lead_times)
        return {
            'average_hours': sum(lead_times) / len(lead_times),
            'median_hours': sorted_times[len(sorted_times) // 2],
            'min_hours': min(lead_times),
            'max_hours': max(lead_times),
            'game_count': len(lead_times),
        }

    def get_hourly_distribution(self) -> dict[int, int]:
        """Get distribution of when games first appear by hour.

        Returns:
            Dict mapping hour (0-23) to count of games discovered
        """
        games = self.db.get_games_with_timing()
        distribution: dict[int, int] = defaultdict(int)

        for game in games:
            first_seen = parse_datetime(game['first_seen_at'])
            if first_seen:
                distribution[first_seen.hour] += 1

        return dict(sorted(distribution.items()))

    def get_summary(self) -> dict:
        """Get a quick summary of collected data.

        Returns:
            Summary dict with counts and key findings
        """
        ts_analysis = self.get_timestamp_field_analysis()
        lead_stats = self.get_lead_time_stats()

        return {
            'games_tracked': self.db.get_game_count(),
            'snapshots_captured': self.db.get_snapshot_count(),
            'timestamp_fields_found': len(ts_analysis['all_fields']),
            'best_opening_field': self.get_best_opening_field(),
            'lead_time_stats': lead_stats,
        }

    def generate_report(self) -> str:
        """Generate a comprehensive timing analysis report.

        Returns:
            Markdown-formatted report string
        """
        lines = []
        lines.append("# Overtime.ag Odds Release Timing Analysis")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n" + "=" * 60 + "\n")

        # Summary
        summary = self.get_summary()
        lines.append("## Summary\n")
        lines.append(f"- Games tracked: {summary['games_tracked']}")
        lines.append(f"- API snapshots: {summary['snapshots_captured']}")
        lines.append(f"- Timestamp fields: {summary['timestamp_fields_found']}")

        if summary['best_opening_field']:
            lines.append(f"- Best opening field: `{summary['best_opening_field']}`")

        # Timestamp Fields
        ts_analysis = self.get_timestamp_field_analysis()
        lines.append("\n## Discovered Timestamp Fields\n")

        if ts_analysis['opening_fields']:
            lines.append("### Opening Line Indicators (KEY FIELDS)\n")
            for f in ts_analysis['opening_fields'][:5]:
                lines.append(f"- `{f['field_path']}`")
                lines.append(f"  - Sample: `{f['sample_value'][:50]}`")
                lines.append(f"  - Occurrences: {f['occurrence_count']}\n")

        if ts_analysis['gametime_fields']:
            lines.append("### Game Time Fields\n")
            for f in ts_analysis['gametime_fields'][:5]:
                lines.append(
                    f"- `{f['field_path']}`: {f['sample_value'][:40]}"
                )

        # Lead Time Analysis
        lead_stats = self.get_lead_time_stats()
        lines.append("\n## Lead Time Analysis\n")

        if lead_stats['game_count'] > 0:
            lines.append("Time between odds posting and game start:\n")
            lines.append(f"- Average: {lead_stats['average_hours']:.1f} hours")
            lines.append(f"- Median: {lead_stats['median_hours']:.1f} hours")
            lines.append(
                f"- Range: {lead_stats['min_hours']:.1f} to "
                f"{lead_stats['max_hours']:.1f} hours"
            )
            lines.append(f"- Games analyzed: {lead_stats['game_count']}")
        else:
            lines.append("*Not enough data for lead time analysis.*")

        # Hourly Distribution
        hourly = self.get_hourly_distribution()
        if hourly:
            lines.append("\n## Hourly Distribution\n")
            lines.append("When odds first appear (by hour):\n")
            lines.append("```")
            for hour in range(24):
                count = hourly.get(hour, 0)
                bar = "█" * min(count, 30)
                lines.append(f"{hour:02d}:00  {bar} ({count})")
            lines.append("```")

        # API Endpoints
        endpoints = self.db.get_api_endpoints()
        if endpoints:
            lines.append("\n## Discovered API Endpoints\n")
            for ep in endpoints[:10]:
                url_short = ep['endpoint'][:60]
                lines.append(f"- `{url_short}...`")
                lines.append(
                    f"  Captures: {ep['captures']}, "
                    f"Avg size: {ep['avg_size']} bytes"
                )

        # Recommendations
        lines.append("\n## Recommendations\n")

        if lead_stats['game_count'] > 0 and hourly:
            peak_hour = max(hourly.keys(), key=lambda h: hourly.get(h, 0))
            lines.append(f"1. **Primary capture window:** {peak_hour:02d}:00")
            lines.append(
                f"2. **Expected lead time:** "
                f"~{lead_stats['median_hours']:.0f} hours before tip-off"
            )
            lines.append("\n### Suggested Cron Schedule\n")
            lines.append("```bash")
            lines.append(f"# Capture odds at peak hour")
            lines.append(
                f"0 {peak_hour} * * * "
                f"python -m kenp0m_sp0rts_analyzer.overtime_scraper --scrape"
            )
            lines.append("```")
        else:
            lines.append("*Collect more data to generate recommendations.*")
            lines.append("\nRun monitoring for 24-48 hours to build dataset.")

        return "\n".join(lines)

    def export_report(self, output_path: Path | str) -> str:
        """Export report to a file.

        Args:
            output_path: Path to save report

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        report = self.generate_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Report exported to: {output_path}")
        return str(output_path)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze overtime.ag odds timing patterns"
    )
    parser.add_argument(
        "--db", "-d",
        type=Path,
        help="Database file path",
    )
    parser.add_argument(
        "--export", "-e",
        type=Path,
        help="Export report to file",
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show brief summary only",
    )

    args = parser.parse_args()

    # Initialize
    db = TimingDatabase(args.db) if args.db else TimingDatabase()
    analyzer = TimingAnalyzer(db)

    if args.summary:
        summary = analyzer.get_summary()
        print("\n=== OVERTIME.AG TIMING SUMMARY ===\n")
        print(f"Games tracked: {summary['games_tracked']}")
        print(f"Snapshots captured: {summary['snapshots_captured']}")
        print(f"Timestamp fields: {summary['timestamp_fields_found']}")

        if summary['best_opening_field']:
            print(f"Best opening field: {summary['best_opening_field']}")

        stats = summary['lead_time_stats']
        if stats['average_hours']:
            print(f"Avg lead time: {stats['average_hours']:.1f} hours")
    else:
        report = analyzer.generate_report()

        if args.export:
            analyzer.export_report(args.export)
            print(f"Report exported to: {args.export}")
        else:
            print(report)
