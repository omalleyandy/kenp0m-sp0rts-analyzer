#!/usr/bin/env python3
"""Populate point distribution table from KenPom API.

This script fetches point distribution data from the KenPom API and populates
the point_distribution table in the SQLite database for XGBoost feature engineering.

Point distribution tracks the percentage of points from FTs, 2-pointers, and 3-pointers
for both offense and defense, which helps identify:
- Teams with high 3-point reliance (higher variance)
- Shooting style matchups
- Offensive/defensive tendencies
"""

import argparse
import logging
from datetime import date, datetime

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

from .repository import KenPomRepository

logger = logging.getLogger(__name__)


def populate_point_distribution(
    year: int | None = None,
    snapshot_date: date | None = None,
    db_path: str = "data/kenpom.db",
) -> int:
    """Populate point distribution table from KenPom API.

    Args:
        year: Season year (e.g., 2025 for 2024-25 season). Defaults to current season.
        snapshot_date: Date to record snapshot. Defaults to today.
        db_path: Path to SQLite database.

    Returns:
        Number of records populated.
    """
    # Determine season if not specified
    if year is None:
        today = date.today()
        year = today.year if today.month >= 11 else today.year

    snapshot_date = snapshot_date or date.today()

    logger.info(f"Fetching point distribution data for {year} season...")

    # Initialize API client and repository
    api = KenPomAPI()
    repository = KenPomRepository(db_path)

    # Fetch point distribution data
    start_time = datetime.now()
    response = api.get_point_distribution(year=year)
    data = list(response.data)

    logger.info(f"Fetched {len(data)} teams in {(datetime.now() - start_time).total_seconds():.2f}s")

    # Save to database
    logger.info("Saving to database...")
    count = repository.save_point_distribution(snapshot_date, data)

    logger.info(f"Successfully populated {count} point distribution records")

    return count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate point distribution table from KenPom API"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Snapshot date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/kenpom.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse snapshot date if provided
    snapshot_date = None
    if args.date:
        try:
            snapshot_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1

    try:
        count = populate_point_distribution(
            year=args.year,
            snapshot_date=snapshot_date,
            db_path=args.db,
        )

        logger.info(f"[OK] Populated {count} records")
        return 0

    except Exception as e:
        logger.error(f"[ERROR] Failed to populate point distribution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
