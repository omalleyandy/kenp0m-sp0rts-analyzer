"""Scrape Vegas lines from overtime.ag for today's games.

This script fetches college basketball betting lines from overtime.ag
and saves them to:
1. data/vegas_lines.json - Current day's lines for analysis
2. data/historical_odds.db - Historical database for long-term tracking

Uses Playwright browser automation by default for reliable scraping.

Usage:
    # Fetch today's lines (browser automation)
    uv run python scrape_overtime_lines.py
    
    # Fetch tomorrow's lines (usually available evening before)
    uv run python scrape_overtime_lines.py --tomorrow
    
    # Fetch lines for a specific date
    uv run python scrape_overtime_lines.py --date 2025-12-18
    
    # Run in headless mode (no browser window)
    uv run python scrape_overtime_lines.py --headless
    
    # Mark as opening lines (run in morning)
    uv run python scrape_overtime_lines.py --open
    
    # Mark as closing lines (run right before games start)
    uv run python scrape_overtime_lines.py --close
    
    # List available sports
    uv run python scrape_overtime_lines.py --sports

Environment Variables (add to .env):
    OVERTIME_USER - Your overtime.ag username (optional)
    OVERTIME_PASSWORD - Your overtime.ag password (optional)
"""

import asyncio
import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()


async def fetch_via_api(target_date: date, include_extra: bool = True) -> list[dict]:
    """Fetch lines using direct API (fast)."""
    from kenp0m_sp0rts_analyzer.overtime_api import OvertimeAPIClient
    
    async with OvertimeAPIClient() as client:
        games = await client.get_college_basketball_lines(include_extra=include_extra)
        
        # Filter by date
        date_str = target_date.strftime("%Y-%m-%d")
        games = [g for g in games if g.game_date == date_str]
        
        return [g.to_dict() for g in games]


async def fetch_via_browser(target_date: date, headless: bool = False, include_extra: bool = True) -> list[dict]:
    """Fetch lines using browser scraping (slower, more robust)."""
    from kenp0m_sp0rts_analyzer.overtime_scraper import OvertimeScraper
    
    async with OvertimeScraper(headless=headless) as scraper:
        await scraper.login()
        games = await scraper.get_college_basketball_lines(include_extra=include_extra)
        return [g.to_dict() for g in games]


async def list_sports():
    """List available sports from the API."""
    from kenp0m_sp0rts_analyzer.overtime_api import OvertimeAPIClient
    
    async with OvertimeAPIClient() as client:
        sports = await client.get_sports()
        
        print(f"\n{'SPORT':<15} {'LEAGUE':<35} {'ID':>6} {'STATUS':>8}")
        print("-" * 70)
        
        for s in sports:
            status = "Active" if s.active else "Inactive"
            print(f"{s.sport_type:<15} {s.sport_subtype:<35} {s.sport_subtype_id:>6} {status:>8}")
            
        # Highlight college basketball
        cbb = [s for s in sports if "College" in s.sport_subtype and s.sport_type == "Basketball"]
        if cbb:
            print(f"\n🏀 College Basketball Leagues: {len(cbb)}")
            for s in cbb:
                print(f"   - {s.sport_subtype} (ID: {s.sport_subtype_id})")


async def run_discovery(headless: bool = False):
    """Run API discovery using browser."""
    from kenp0m_sp0rts_analyzer.overtime_scraper import discover_overtime_api
    await discover_overtime_api(headless=headless)


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape Vegas lines from overtime.ag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scrape_overtime_lines.py              # Fetch today's lines
  uv run python scrape_overtime_lines.py --tomorrow   # Fetch tomorrow's lines  
  uv run python scrape_overtime_lines.py --date 2025-12-18
  uv run python scrape_overtime_lines.py --headless   # No browser window
  uv run python scrape_overtime_lines.py --open       # Mark as opening lines
  uv run python scrape_overtime_lines.py --close      # Mark as closing lines
        """
    )
    
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--tomorrow",
        action="store_true",
        help="Scrape lines for tomorrow's games"
    )
    parser.add_argument(
        "--sports",
        action="store_true",
        help="List available sports from API"
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run API discovery mode (uses browser)"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use direct API instead of browser (experimental)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Mark as opening lines (run in morning)"
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Mark as closing lines (run right before games start)"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip storing to historical database"
    )
    parser.add_argument(
        "--no-extra",
        action="store_true",
        help="Exclude 'College Extra' games (smaller schools)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: data/vegas_lines.json)"
    )
    
    args = parser.parse_args()
    
    # Determine target date
    if args.tomorrow:
        target_date = date.today() + timedelta(days=1)
    elif args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("   Use YYYY-MM-DD format (e.g., 2025-12-18)")
            return 1
    else:
        target_date = date.today()
    
    print("=" * 70)
    print("OVERTIME.AG VEGAS LINES")
    print("=" * 70)
    
    # Handle special modes
    if args.sports:
        print("Mode: List Available Sports")
        print("=" * 70)
        await list_sports()
        return 0
        
    if args.discover:
        print("Mode: API Discovery")
        print("=" * 70)
        await run_discovery(headless=args.headless)
        return 0
        
    # Standard fetch mode
    method = "API" if args.api else "Browser"
    print(f"Date: {target_date.strftime('%A, %B %d, %Y')}")
    print(f"Method: {method}")
    print("=" * 70)
    
    print(f"\n🏀 Fetching college basketball lines...")
    
    # Determine output path early so we can use it in error cases
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "data" / "vegas_lines.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.api:
            games = await fetch_via_api(target_date, include_extra=not args.no_extra)
        else:
            games = await fetch_via_browser(target_date, headless=args.headless, include_extra=not args.no_extra)
    except Exception as e:
        print(f"\n❌ Error fetching lines: {e}")
        if args.api:
            print("   API may require browser session. Try without --api flag.")
        return 1
        
    if not games:
        print(f"\n⚠️  No college basketball games found")
        print("\n   This could mean:")
        print("   • Navigation to College Basketball section failed")
        print("   • All games for today have completed or are in progress") 
        print("   • Tomorrow's lines haven't been posted yet (usually late evening)")
        print("   • Site structure may have changed")
        print("\n   💡 Tips:")
        print("   • Try running without --headless to see what's happening")
        print("   • Lines for next day typically appear late evening/overnight")
        print(f"   • Check overtime.ag directly to verify availability")
        
        # Save empty file to indicate we checked
        data = {
            "source": "overtime.ag",
            "date": target_date.strftime("%Y-%m-%d"),
            "scraped_at": datetime.now().isoformat(),
            "game_count": 0,
            "games": [],
            "note": "No games found - navigation may have failed or no CBB games available"
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n📁 Empty result saved to: {output_path}")
        
        return 0  # Not necessarily an error
        
    print(f"\n✅ Found {len(games)} games\n")
    
    # Print games table
    print(f"{'MATCHUP':<42} {'SPREAD':>10} {'TOTAL':>8} {'ML':>14}")
    print("-" * 78)
    
    for g in games:
        away = g.get('away_team', 'Unknown')[:18]
        home = g.get('home_team', 'Unknown')[:18]
        matchup = f"{away} @ {home}"
        
        spread = g.get('spread')
        spread_str = f"{spread:+.1f}" if spread is not None else "N/A"
        
        total = g.get('total')
        total_str = f"{total}" if total is not None else "N/A"
        
        away_ml = g.get('away_ml')
        home_ml = g.get('home_ml')
        if away_ml is not None and home_ml is not None:
            ml_str = f"{away_ml:+d}/{home_ml:+d}"
        else:
            ml_str = "N/A"
            
        print(f"{matchup:<42} {spread_str:>10} {total_str:>8} {ml_str:>14}")
        
    # Save to file
    data = {
        "source": "overtime.ag",
        "date": target_date.strftime("%Y-%m-%d"),
        "scraped_at": datetime.now().isoformat(),
        "game_count": len(games),
        "games": games,
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"\n📁 Saved to: {output_path}")
    
    # Store in historical database
    if not args.no_db:
        try:
            from kenp0m_sp0rts_analyzer.historical_odds_db import get_db
            
            # Determine snapshot type
            if args.open:
                snapshot_type = "open"
            elif args.close:
                snapshot_type = "close"
            else:
                snapshot_type = "current"
            
            db = get_db()
            stored_count = db.store_odds_snapshot(
                sport="Basketball",
                league="College Basketball",
                game_date=target_date.strftime("%Y-%m-%d"),
                games=games,
                snapshot_type=snapshot_type,
                source="overtime.ag"
            )
            db.close()
            
            print(f"📊 Stored {stored_count} games in historical database ({snapshot_type} lines)")
            
        except Exception as e:
            print(f"⚠️  Database storage failed: {e}")
            print("   Lines still saved to JSON file.")
    
    # Show next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"Run analysis with Vegas line comparison:")
    print(f"  uv run python analyze_todays_games.py --date {target_date}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
