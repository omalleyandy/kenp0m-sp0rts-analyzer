#!/usr/bin/env python
"""Monitor college basketball odds from overtime.ag.

This script:
1. Scrapes today's CBB odds from overtime.ag (sole source)
2. Stores odds in the unified kenpom.db vegas_odds table
3. Displays games with spread, total, and moneyline
4. Optionally runs in continuous monitoring mode

Usage:
    # Single scrape and store
    uv run python scripts/monitor_cbb_odds.py

    # Continuous monitoring (every 30 minutes for 12 hours)
    uv run python scripts/monitor_cbb_odds.py --monitor -i 30 -d 12

    # Show stored odds for today
    uv run python scripts/monitor_cbb_odds.py --show

    # Headless mode (no browser window)
    uv run python scripts/monitor_cbb_odds.py --headless
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.kenpom import KenPomService
from kenp0m_sp0rts_analyzer.overtime_scraper import OvertimeScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def scrape_and_store_odds(
    headless: bool = False,
    snapshot_type: str = "current",
) -> int:
    """Scrape CBB odds from overtime.ag and store in database.

    Args:
        headless: Run browser in headless mode.
        snapshot_type: Type of snapshot ('open', 'current', 'close').

    Returns:
        Number of games stored.
    """
    print("=" * 70)
    print(f"OVERTIME.AG CBB ODDS MONITOR - {date.today()}")
    print("=" * 70)
    print(f"Snapshot type: {snapshot_type}")
    print(f"Headless: {headless}")
    print()

    # Scrape from overtime.ag
    print("Initializing browser...")
    async with OvertimeScraper(headless=headless) as scraper:
        print("Navigating to overtime.ag...")
        await scraper.login()

        print("Fetching college basketball lines...")
        games = await scraper.get_college_basketball_lines()

        if not games:
            print("[!] No games found on overtime.ag")
            return 0

        print(f"\n[OK] Found {len(games)} games")
        print("-" * 70)

        # Display games
        for game in games:
            spread_str = f"{game.spread:+.1f}" if game.spread else "N/A"
            total_str = f"{game.total}" if game.total else "N/A"
            away_ml_str = f"{game.away_ml:+d}" if game.away_ml else "N/A"
            home_ml_str = f"{game.home_ml:+d}" if game.home_ml else "N/A"

            print(
                f"{game.time:>8} | {game.away_team:25} @ {game.home_team:25}"
            )
            print(
                f"         | Spread: {spread_str:>7} | "
                f"Total: {total_str:>6} | "
                f"ML: {away_ml_str}/{home_ml_str}"
            )

        # Convert to dictionaries for storage
        game_dicts = [g.to_dict() for g in games]

    # Store in database
    print("\nStoring in kenpom.db...")
    service = KenPomService()
    count = service.repository.save_vegas_odds(
        game_date=date.today(),
        games=game_dicts,
        snapshot_type=snapshot_type,
    )

    print(f"[OK] Stored {count} games in vegas_odds table")

    # Also save to JSON for backup
    output_dir = Path(__file__).parent.parent / "data" / "overtime_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "vegas_lines.json"
    output_data = {
        "source": "overtime.ag",
        "date": str(date.today()),
        "scraped_at": datetime.now().isoformat(),
        "snapshot_type": snapshot_type,
        "game_count": len(game_dicts),
        "games": game_dicts,
    }

    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Backup saved to {output_file}")

    return count


async def run_continuous_monitor(
    interval_minutes: int = 30,
    duration_hours: int = 12,
    headless: bool = True,
) -> None:
    """Run continuous monitoring at specified intervals.

    Args:
        interval_minutes: Minutes between scrapes.
        duration_hours: Total hours to run.
        headless: Run in headless mode.
    """
    print("=" * 70)
    print("CONTINUOUS MONITORING MODE")
    print("=" * 70)
    print(f"Interval: {interval_minutes} minutes")
    print(f"Duration: {duration_hours} hours")
    print(f"Headless: {headless}")
    print()

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    iteration = 0

    while datetime.now() < end_time:
        iteration += 1
        print(f"\n{'=' * 70}")
        print(f"[Iteration {iteration}] {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 70)

        try:
            count = await scrape_and_store_odds(headless=headless)
            print(f"[OK] Iteration {iteration} complete: {count} games")
        except Exception as e:
            logger.error(f"Scrape failed: {e}")
            print(f"[X] Iteration {iteration} failed: {e}")

        # Calculate next run time
        next_run = datetime.now() + timedelta(minutes=interval_minutes)

        if next_run < end_time:
            sleep_seconds = interval_minutes * 60
            print(
                f"\nSleeping {interval_minutes} minutes until {next_run:%H:%M}"
            )
            await asyncio.sleep(sleep_seconds)

    print(f"\n[OK] Monitoring complete after {iteration} iterations")


def show_stored_odds(game_date: date | None = None) -> None:
    """Display stored odds from database.

    Args:
        game_date: Date to show (defaults to today).
    """
    game_date = game_date or date.today()

    print("=" * 70)
    print(f"STORED VEGAS ODDS - {game_date}")
    print("=" * 70)

    service = KenPomService()
    games = service.repository.get_vegas_odds_for_date(game_date)

    if not games:
        print(f"[!] No odds stored for {game_date}")
        print("    Run: uv run python scripts/monitor_cbb_odds.py")
        return

    print(f"Found {len(games)} games:\n")

    for game in games:
        spread = game.get("spread")
        total = game.get("total")
        away_ml = game.get("away_ml")
        home_ml = game.get("home_ml")

        spread_str = f"{spread:+.1f}" if spread else "N/A"
        total_str = f"{total}" if total else "N/A"
        away_ml_str = f"{away_ml:+d}" if away_ml else "N/A"
        home_ml_str = f"{home_ml:+d}" if home_ml else "N/A"

        game_time = game.get("game_time", "")
        away_team = game.get("away_team", "Unknown")
        home_team = game.get("home_team", "Unknown")

        print(f"{game_time:>8} | {away_team:25} @ {home_team:25}")
        print(
            f"         | Spread: {spread_str:>7} | "
            f"Total: {total_str:>6} | "
            f"ML: {away_ml_str}/{home_ml_str}"
        )

    snapshot_at = games[0].get("snapshot_at") if games else None
    if snapshot_at:
        print(f"\nLast updated: {snapshot_at}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor college basketball odds from overtime.ag"
    )
    parser.add_argument(
        "--monitor",
        "-m",
        action="store_true",
        help="Run continuous monitoring mode",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Monitoring interval in minutes (default: 30)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=12,
        help="Monitoring duration in hours (default: 12)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show stored odds for today (no scraping)",
    )
    parser.add_argument(
        "--snapshot-type",
        choices=["open", "current", "close"],
        default="current",
        help="Snapshot type (default: current)",
    )

    args = parser.parse_args()

    if args.show:
        show_stored_odds()
        return 0

    if args.monitor:
        asyncio.run(
            run_continuous_monitor(
                interval_minutes=args.interval,
                duration_hours=args.duration,
                headless=args.headless,
            )
        )
    else:
        count = asyncio.run(
            scrape_and_store_odds(
                headless=args.headless,
                snapshot_type=args.snapshot_type,
            )
        )
        if count == 0:
            print("\n[!] No games scraped - check if games are available")
            return 1

    print("\n[OK] Done!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[!] Cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
