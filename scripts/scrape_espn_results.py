#!/usr/bin/env python
"""Scrape ESPN historical game results for training data.

This script scrapes game results from ESPN's Men's College Basketball
scoreboard and saves them in a format compatible with the training pipeline.

Usage:
    # Scrape single date
    python scripts/scrape_espn_results.py --date 2024-12-15

    # Scrape date range
    python scripts/scrape_espn_results.py --start-date 2024-11-04 --end-date 2024-12-15

    # Scrape with visible browser (for debugging)
    python scripts/scrape_espn_results.py --date 2024-12-15 --visible

    # Scrape entire season (November to March)
    python scripts/scrape_espn_results.py --season 2024

Output:
    CSV file compatible with HistoricalDataLoader.load_from_csv()
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.espn_results_scraper import ESPNResultsScraper
from kenp0m_sp0rts_analyzer.utils.logging import setup_logging
from kenp0m_sp0rts_analyzer.config import config

logger = setup_logging(config.logs_dir, level="INFO", app_name="espn_scraper")


async def scrape_season(
    year: int,
    scraper: ESPNResultsScraper,
) -> list:
    """Scrape an entire college basketball season.

    The college basketball season typically runs from:
    - Early November (season start)
    - Through early April (Final Four)

    Args:
        year: The year the season STARTS (e.g., 2024 for 2024-25 season).
        scraper: ESPNResultsScraper instance.

    Returns:
        List of GameResult objects.
    """
    # Season date range
    start = date(year, 11, 4)  # First games typically early November
    end = date(year + 1, 4, 8)  # Final Four first week of April

    # Don't scrape future dates
    today = date.today()
    if end > today:
        end = today - timedelta(days=1)

    logger.info(f"Scraping {year}-{year + 1} season: {start} to {end}")

    results = await scraper.scrape_date_range(
        start_date=start,
        end_date=end,
        delay_seconds=2.5,  # Be respectful to ESPN
    )

    return results


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape ESPN college basketball results"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Single date to scrape (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Scrape entire season (year season starts, e.g., 2024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window (for debugging)",
    )

    args = parser.parse_args()

    # Initialize scraper
    scraper = ESPNResultsScraper(headless=not args.visible)

    # Determine what to scrape
    if args.season:
        results = await scrape_season(args.season, scraper)
        output_name = args.output or f"espn_results_{args.season}_{args.season + 1}.csv"

    elif args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
        results = await scraper.scrape_date(target)
        output_name = args.output or f"espn_results_{args.date}.csv"

    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        results = await scraper.scrape_date_range(start, end)
        output_name = (
            args.output or f"espn_results_{args.start_date}_to_{args.end_date}.csv"
        )

    else:
        # Default: yesterday
        yesterday = date.today() - timedelta(days=1)
        results = await scraper.scrape_date(yesterday)
        output_name = args.output or f"espn_results_{yesterday}.csv"

    # Save results
    if results:
        csv_path = scraper.save_to_csv(results, output_name)
        json_path = scraper.save_to_json(
            results,
            output_name.replace(".csv", ".json"),
        )

        print(f"\n{'=' * 60}")
        print(f"Scraping Complete!")
        print(f"{'=' * 60}")
        print(f"Total games: {len(results)}")
        print(f"CSV saved:   {csv_path}")
        print(f"JSON saved:  {json_path}")
        print(f"\nTo train with this data:")
        print(f"  python scripts/train_model.py --csv {csv_path}")
    else:
        print("No games found for the specified date(s)")


if __name__ == "__main__":
    asyncio.run(main())
