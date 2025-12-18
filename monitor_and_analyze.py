"""Comprehensive monitoring system for college basketball odds and analysis.

This script:
1. Continuously checks overtime.ag for college basketball availability
2. When games appear, captures opening lines and tracks movement
3. Runs KenPom analysis to identify statistical edges
4. Generates reports comparing predictions to Vegas lines

Usage:
    # Start monitoring (runs indefinitely)
    uv run python monitor_and_analyze.py

    # Check every 15 minutes (faster)
    uv run python monitor_and_analyze.py --interval 15

    # Run KenPom analysis only (no monitoring)
    uv run python monitor_and_analyze.py --analyze-only
"""
import asyncio
import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kenp0m_sp0rts_analyzer.overtime_scraper import OvertimeScraper
from kenp0m_sp0rts_analyzer.overtime_timing import TimingDatabase
from kenp0m_sp0rts_analyzer.historical_odds_db import get_db


class ComprehensiveMonitor:
    """Monitors overtime.ag and analyzes KenPom data for edges."""

    def __init__(self, check_interval: int = 30, data_dir: Path | None = None):
        """Initialize monitor.

        Args:
            check_interval: Minutes between checks
            data_dir: Directory for data storage
        """
        self.check_interval = check_interval
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.timing_db = TimingDatabase()
        self.odds_db = get_db()

        self.games_found = False
        self.opening_lines_captured = False
        self.last_capture_time = None

        # State tracking
        self.monitoring_log = self.data_dir / "monitoring_log.json"
        self.load_state()

    def load_state(self):
        """Load monitoring state from disk."""
        if self.monitoring_log.exists():
            try:
                with open(self.monitoring_log) as f:
                    state = json.load(f)
                    self.games_found = state.get('games_found', False)
                    self.opening_lines_captured = state.get('opening_lines_captured', False)
                    self.last_capture_time = state.get('last_capture_time')
            except Exception:
                pass

    def save_state(self):
        """Save monitoring state to disk."""
        state = {
            'games_found': self.games_found,
            'opening_lines_captured': self.opening_lines_captured,
            'last_capture_time': self.last_capture_time,
            'updated_at': datetime.now().isoformat(),
        }
        with open(self.monitoring_log, 'w') as f:
            json.dump(state, f, indent=2)

    async def check_availability(self) -> tuple[bool, list]:
        """Check if college basketball is available on overtime.ag.

        Returns:
            Tuple of (available: bool, games: list)
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking overtime.ag...")

        try:
            async with OvertimeScraper(
                headless=True,
                capture_network=True,
                timing_db=self.timing_db,
            ) as scraper:
                # Capture timing data
                snapshot = await scraper.capture_timing_snapshot()

                # Try to get games
                await scraper.login()
                games = await scraper.get_college_basketball_lines()

                if games:
                    print(f"  [OK] Found {len(games)} games!")
                    return True, games
                else:
                    print("  [WAITING] No college basketball games yet...")
                    return False, []

        except Exception as e:
            print(f"  [ERROR] Check failed: {e}")
            return False, []

    def analyze_kenpom_edges(self, target_date: date | None = None) -> dict:
        """Analyze KenPom data to find statistical edges.

        Args:
            target_date: Date to analyze (default: today)

        Returns:
            Dict with game analysis and predictions
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.strftime('%Y-%m-%d')
        print(f"\n[KENPOM] Analyzing games for {date_str}...")

        # Save analysis to file for edge detection
        analysis_file = self.data_dir / f"kenpom_analysis_{date_str}.json"

        # Run pre-game analyzer
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "uv", "run", "python",
                "scripts/analysis/kenpom_pregame_analyzer.py",
                "--date", date_str,
                "-o", str(analysis_file)
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"  [OK] Analysis saved to {analysis_file}")

            # Load and return
            if analysis_file.exists():
                with open(analysis_file) as f:
                    return json.load(f)
        else:
            print(f"  [ERROR] Analysis failed: {result.stderr}")

        return {
            'date': date_str,
            'analyzed_at': datetime.now().isoformat(),
            'games': [],
            'edges_found': 0,
        }

    async def capture_and_store_lines(self, games: list, snapshot_type: str = "current"):
        """Capture and store betting lines.

        Args:
            games: List of game objects
            snapshot_type: 'open', 'current', or 'close'
        """
        print(f"\n[CAPTURE] Storing {snapshot_type} lines for {len(games)} games...")

        try:
            target_date = date.today().strftime("%Y-%m-%d")

            # Save to JSON
            output_path = self.data_dir / f"vegas_lines_{snapshot_type}.json"
            data = {
                "source": "overtime.ag",
                "date": target_date,
                "snapshot_type": snapshot_type,
                "scraped_at": datetime.now().isoformat(),
                "game_count": len(games),
                "games": [g.to_dict() for g in games],
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"  [SAVED] {output_path}")

            # Store in historical database
            game_dicts = [g.to_dict() for g in games]
            stored_count = self.odds_db.store_odds_snapshot(
                sport="Basketball",
                league="College Basketball",
                game_date=target_date,
                games=game_dicts,
                snapshot_type=snapshot_type,
                source="overtime.ag"
            )

            print(f"  [DATABASE] Stored {stored_count} games as {snapshot_type} lines")

            # Update state
            self.last_capture_time = datetime.now().isoformat()
            if snapshot_type == "open":
                self.opening_lines_captured = True
            self.save_state()

        except Exception as e:
            print(f"  [ERROR] Storage failed: {e}")

    def generate_edge_report(self) -> str:
        """Generate report comparing KenPom predictions to Vegas lines.

        Returns:
            Markdown-formatted report
        """
        print("\n[REPORT] Generating edge analysis...")

        # Check if Vegas lines exist
        lines_file = self.data_dir / "vegas_lines_current.json"
        if not lines_file.exists():
            print("  [WAITING] No Vegas lines available yet")
            return "No Vegas lines available yet."

        # Run edge detector
        report_path = self.data_dir / "edge_report.md"

        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "uv", "run", "python",
                "scripts/analysis/edge_detector.py",
                "--min-edge", "2.5",
                "-o", str(report_path)
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"  [OK] Edge report saved to {report_path}")

            # Read and return report
            if report_path.exists():
                with open(report_path) as f:
                    return f.read()
        else:
            print(f"  [ERROR] Edge detection failed: {result.stderr}")

        return "Edge detection failed."

    async def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        # Check availability
        available, games = await self.check_availability()

        if available and games:
            self.games_found = True

            # First time finding games - capture opening lines
            if not self.opening_lines_captured:
                print("\n[OPENING LINES DETECTED]")
                await self.capture_and_store_lines(games, snapshot_type="open")
                self.opening_lines_captured = True
            else:
                # Capture current lines for movement tracking
                await self.capture_and_store_lines(games, snapshot_type="current")

            # Run KenPom analysis
            self.analyze_kenpom_edges()

            # Generate edge report
            self.generate_edge_report()

        self.save_state()

    async def start_monitoring(self):
        """Start continuous monitoring loop."""
        print("=" * 70)
        print("COMPREHENSIVE MONITORING SYSTEM")
        print("=" * 70)
        print(f"Check interval: {self.check_interval} minutes")
        print(f"Data directory: {self.data_dir}")
        print(f"Database: {self.timing_db.db_path}")
        print("\nMonitoring for:")
        print("  1. College basketball availability on overtime.ag")
        print("  2. Line movement tracking")
        print("  3. KenPom statistical edges")
        print("\nPress Ctrl+C to stop.\n")
        print("=" * 70)

        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n{'='*70}")
                print(f"CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                await self.run_monitoring_cycle()

                # Wait for next check
                wait_seconds = self.check_interval * 60
                print(f"\n[WAITING] Next check in {self.check_interval} minutes...")
                print(f"           ({datetime.now() + timedelta(seconds=wait_seconds):%H:%M:%S})")

                await asyncio.sleep(wait_seconds)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            print(f"Total cycles completed: {cycle}")
            print(f"Games found: {self.games_found}")
            print(f"Opening lines captured: {self.opening_lines_captured}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive college basketball monitoring and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Check interval in minutes (default: 30)",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run KenPom analysis only (no monitoring)",
    )

    parser.add_argument(
        "--check-once",
        action="store_true",
        help="Run one check and exit",
    )

    args = parser.parse_args()

    monitor = ComprehensiveMonitor(check_interval=args.interval)

    if args.analyze_only:
        # Just run analysis
        monitor.analyze_kenpom_edges()
        monitor.generate_edge_report()
    elif args.check_once:
        # Single check
        await monitor.run_monitoring_cycle()
    else:
        # Continuous monitoring
        await monitor.start_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
