"""Scrape today's college basketball odds from overtime.ag."""

import asyncio
import json
import sys
from datetime import datetime, date
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kenp0m_sp0rts_analyzer.overtime_scraper import OvertimeScraper


async def scrape_cbb():
    """Scrape college basketball lines from overtime.ag."""
    print("=" * 70)
    print(f"Overtime.ag College Basketball Scraper - {date.today()}")
    print("=" * 70)
    
    print("\nInitializing browser (non-headless for visibility)...")
    
    async with OvertimeScraper(headless=False) as scraper:
        print("Logging in to overtime.ag...")
        try:
            await scraper.login()
            print("Login successful!")
        except Exception as e:
            print(f"Login failed: {e}")
            return []
        
        print("\nNavigating to college basketball section...")
        try:
            games = await scraper.get_college_basketball_lines()
        except Exception as e:
            print(f"Error fetching lines: {e}")
            return []
        
        if not games:
            print("No college basketball games found!")
            return []
        
        print(f"\nFound {len(games)} games:")
        print("-" * 70)
        
        game_data = []
        for game in games:
            spread_str = f"{game.spread:+.1f}" if game.spread else "N/A"
            total_str = f"{game.total}" if game.total else "N/A"
            print(f"{game.time:>8} | {game.away_team:25} @ {game.home_team:25}")
            print(f"         | Spread: {spread_str:>7} | Total: {total_str}")
            game_data.append(game.to_dict())
        
        # Save to file
        output = {
            "source": "overtime.ag",
            "date": str(date.today()),
            "scraped_at": datetime.now().isoformat(),
            "game_count": len(games),
            "games": game_data,
        }
        
        output_path = Path(__file__).parent / "data" / "vegas_lines.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {output_path}")
        
        return games


if __name__ == "__main__":
    try:
        games = asyncio.run(scrape_cbb())
        print(f"\nTotal games scraped: {len(games)}")
    except KeyboardInterrupt:
        print("\nScraping cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
