"""Check what games are available on overtime.ag."""
import asyncio
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kenp0m_sp0rts_analyzer.overtime_scraper import OvertimeScraper


async def check_games():
    """Check for available games."""
    today = date.today()
    tomorrow = today + timedelta(days=1)

    print("=" * 70)
    print("CHECKING OVERTIME.AG FOR AVAILABLE GAMES")
    print("=" * 70)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Today: {today.strftime('%A, %B %d, %Y')}")
    print(f"Tomorrow: {tomorrow.strftime('%A, %B %d, %Y')}")
    print("=" * 70)

    print("\n[CONNECTING] Launching browser (non-headless to see what's happening)...")
    print("             This will show the browser window so you can see the page.")
    print("             Close this script if you want to stop.\n")

    async with OvertimeScraper(headless=False) as scraper:
        print("[LOGIN] Attempting to log in to overtime.ag...")
        try:
            await scraper.login()
            print("[OK] Login successful\n")
        except Exception as e:
            print(f"[WARNING] Login failed (may work without login): {e}\n")

        print("[NAVIGATING] Attempting to navigate to College Basketball page...")
        print("             Watch the browser window to see what happens.\n")

        try:
            games = await scraper.get_college_basketball_lines()

            if games:
                print(f"\n[SUCCESS] Found {len(games)} games!\n")

                print(f"{'MATCHUP':<42} {'SPREAD':>10} {'TOTAL':>8}")
                print("-" * 65)

                for g in games[:15]:
                    away = g.away_team[:18]
                    home = g.home_team[:18]
                    matchup = f"{away} @ {home}"
                    spread = f"{g.spread:+.1f}" if g.spread else "N/A"
                    total = f"{g.total}" if g.total else "N/A"
                    print(f"{matchup:<42} {spread:>10} {total:>8}")

                if len(games) > 15:
                    print(f"... and {len(games) - 15} more games")

                print("\n[INFO] Lines are available!")
                print("       You can now run monitoring to track line movement.")

            else:
                print("\n[WARNING] No games found")
                print("\n  Possible reasons:")
                print("  1. Today's games have finished (it's past midnight)")
                print("  2. Tomorrow's lines not posted yet")
                print("  3. No college basketball games scheduled")
                print("  4. Site navigation changed")
                print("\n  What to try:")
                print("  - Run this script during the afternoon/evening")
                print("  - Lines typically post 12-24 hours before games")
                print("  - Check overtime.ag manually to see if CBB is available")

        except Exception as e:
            print(f"\n[ERROR] Failed to fetch games: {e}")
            print("\n  The browser window is still open.")
            print("  Check what page it's on to diagnose the issue.")
            print("  Press Ctrl+C to exit.")

            # Keep browser open for inspection
            await asyncio.sleep(300)  # 5 minutes

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(check_games())
