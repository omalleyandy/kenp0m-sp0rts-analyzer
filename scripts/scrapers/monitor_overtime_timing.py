#!/usr/bin/env python
"""Monitor overtime.ag to discover odds release timing patterns.

This script runs continuous monitoring to empirically determine when
college basketball odds are posted on overtime.ag.

Usage:
    # Run monitoring for 24 hours (default), capturing every 30 minutes
    uv run python scripts/scrapers/monitor_overtime_timing.py

    # Custom interval and duration
    uv run python scripts/scrapers/monitor_overtime_timing.py -i 15 -d 48

    # Show browser for debugging
    uv run python scripts/scrapers/monitor_overtime_timing.py --show-browser

    # Analyze collected data
    uv run python scripts/scrapers/monitor_overtime_timing.py --analyze

    # Quick summary
    uv run python scripts/scrapers/monitor_overtime_timing.py --summary

    # Export report
    uv run python scripts/scrapers/monitor_overtime_timing.py --analyze -o report.md

Example workflow:
    1. Run monitoring for 24-48 hours to collect data
    2. Use --analyze to see timing patterns
    3. Schedule daily odds capture based on discovered patterns
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from kenp0m_sp0rts_analyzer.overtime_scraper import (
    OvertimeScraper,
    run_timing_monitor,
)
from kenp0m_sp0rts_analyzer.overtime_timing import (
    TimingAnalyzer,
    TimingDatabase,
    get_monitoring_dir,
)


def run_analysis(
    export_path: Path | None = None,
    summary_only: bool = False,
) -> None:
    """Run timing analysis on collected data."""
    db = TimingDatabase()
    analyzer = TimingAnalyzer(db)

    if summary_only:
        summary = analyzer.get_summary()
        print("\n" + "=" * 50)
        print("OVERTIME.AG TIMING SUMMARY")
        print("=" * 50)
        print(f"\nGames tracked: {summary['games_tracked']}")
        print(f"Snapshots captured: {summary['snapshots_captured']}")
        print(f"Timestamp fields: {summary['timestamp_fields_found']}")

        if summary['best_opening_field']:
            print(f"\nBest opening field: {summary['best_opening_field']}")

        stats = summary['lead_time_stats']
        if stats.get('average_hours'):
            print(f"\nLead Time Statistics:")
            print(f"  Average: {stats['average_hours']:.1f} hours")
            print(f"  Median: {stats['median_hours']:.1f} hours")
            print(f"  Range: {stats['min_hours']:.1f} - "
                  f"{stats['max_hours']:.1f} hours")

        if summary['games_tracked'] == 0:
            print("\n⚠️  No data collected yet.")
            print("Run monitoring first:")
            print("  uv run python scripts/scrapers/"
                  "monitor_overtime_timing.py")
    else:
        report = analyzer.generate_report()

        if export_path:
            analyzer.export_report(export_path)
            print(f"✅ Report exported to: {export_path}")
        else:
            print(report)


async def run_single_capture(headless: bool = True) -> None:
    """Run a single timing capture."""
    db = TimingDatabase()
    print(f"Database: {db.db_path}")

    async with OvertimeScraper(
        headless=headless,
        capture_network=True,
        timing_db=db,
    ) as scraper:
        snapshot = await scraper.capture_timing_snapshot()

    print(f"\n✅ Captured {snapshot['responses_captured']} responses")
    print(f"   Found {snapshot['games_found']} games")

    if snapshot['new_games']:
        print(f"\nNew games:")
        for game_id in snapshot['new_games'][:5]:
            print(f"  - {game_id}")
        if len(snapshot['new_games']) > 5:
            print(f"  ... and {len(snapshot['new_games']) - 5} more")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor overtime.ag odds release timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze collected data instead of monitoring",
    )
    mode_group.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show brief summary of collected data",
    )
    mode_group.add_argument(
        "--single",
        action="store_true",
        help="Run a single capture (for testing)",
    )

    # Monitoring options
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Capture interval in minutes (default: 30)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=24,
        help="Total duration in hours (default: 24)",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (default: headless)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Export analysis report to file",
    )

    args = parser.parse_args()

    # Handle analysis mode
    if args.analyze or args.summary:
        run_analysis(
            export_path=args.output,
            summary_only=args.summary,
        )
        return 0

    # Handle single capture
    if args.single:
        asyncio.run(run_single_capture(headless=not args.show_browser))
        return 0

    # Run continuous monitoring
    print("=" * 60)
    print("OVERTIME.AG TIMING MONITOR")
    print("=" * 60)
    print(f"\nInterval: {args.interval} minutes")
    print(f"Duration: {args.duration} hours")
    print(f"Headless: {not args.show_browser}")
    print(f"\nData directory: {get_monitoring_dir()}")
    print("\nPress Ctrl+C to stop early.\n")

    try:
        asyncio.run(run_timing_monitor(
            interval=args.interval,
            duration=args.duration,
            headless=not args.show_browser,
        ))
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Run with --analyze to see results.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
