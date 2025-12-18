"""Quick script to capture today's odds without emoji encoding issues."""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kenp0m_sp0rts_analyzer.overtime_scraper import OvertimeScraper
from kenp0m_sp0rts_analyzer.overtime_timing import TimingDatabase


async def main():
    """Capture today's odds and timing data."""
    print("=" * 70)
    print("CAPTURING OVERTIME.AG ODDS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Initialize timing database
    db = TimingDatabase()
    print(f"\nDatabase: {db.db_path}")

    # Capture odds with timing
    print("\n[1/3] Connecting to overtime.ag...")
    async with OvertimeScraper(
        headless=True,
        capture_network=True,
        timing_db=db,
    ) as scraper:
        print("[2/3] Capturing timing snapshot...")
        snapshot = await scraper.capture_timing_snapshot()

        print(f"\n  -> Captured {snapshot['responses_captured']} API responses")
        print(f"  -> Found {snapshot['games_found']} games")

        if snapshot['new_games']:
            print(f"\n  New games detected: {len(snapshot['new_games'])}")
            for game_id in snapshot['new_games'][:5]:
                print(f"    - {game_id}")
            if len(snapshot['new_games']) > 5:
                print(f"    ... and {len(snapshot['new_games']) - 5} more")

        print("\n[3/3] Fetching college basketball lines...")
        try:
            await scraper.login()
            games = await scraper.get_college_basketball_lines()

            if games:
                print(f"\n[OK] Found {len(games)} games")

                # Save to file
                output_path = Path(__file__).parent / "data" / "vegas_lines.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                data = {
                    "source": "overtime.ag",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "scraped_at": datetime.now().isoformat(),
                    "game_count": len(games),
                    "games": [g.to_dict() for g in games],
                }

                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"\n[SAVED] {output_path}")

                # Print sample
                print(f"\n{'MATCHUP':<42} {'SPREAD':>10} {'TOTAL':>8}")
                print("-" * 65)
                for g in games[:10]:
                    away = g.away_team[:18]
                    home = g.home_team[:18]
                    matchup = f"{away} @ {home}"
                    spread = f"{g.spread:+.1f}" if g.spread else "N/A"
                    total = f"{g.total}" if g.total else "N/A"
                    print(f"{matchup:<42} {spread:>10} {total:>8}")
                if len(games) > 10:
                    print(f"... and {len(games) - 10} more")

            else:
                print("\n[WARNING] No games found")
                print("  - Lines may not be posted yet")
                print("  - Check overtime.ag directly")

        except Exception as e:
            print(f"\n[ERROR] Failed to fetch games: {e}")
            return 1

    print("\n" + "=" * 70)
    print("CAPTURE COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
