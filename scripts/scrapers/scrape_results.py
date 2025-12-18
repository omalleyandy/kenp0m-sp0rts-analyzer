"""Scrape game results and update historical database.

This script fetches final scores for college basketball games
and updates the historical odds database for tracking and analysis.

Uses ESPN's public scoreboard data.

Usage:
    # Update results for today's completed games
    uv run python scrape_results.py
    
    # Update results for a specific date
    uv run python scrape_results.py --date 2025-12-17
    
    # Show pending games (not yet final)
    uv run python scrape_results.py --pending
    
    # Calculate prediction results after updating scores
    uv run python scrape_results.py --calculate
"""

import asyncio
import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()


# ESPN API endpoints
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


async def fetch_espn_scores(target_date: date) -> list[dict]:
    """Fetch scores from ESPN API.
    
    Args:
        target_date: Date to fetch scores for
        
    Returns:
        List of game dictionaries with scores
    """
    games = []
    
    # Format date for ESPN API (YYYYMMDD)
    date_str = target_date.strftime("%Y%m%d")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                ESPN_SCOREBOARD_URL,
                params={
                    "dates": date_str,
                    "limit": 200,
                }
            )
            response.raise_for_status()
            data = response.json()
            
        except Exception as e:
            print(f"❌ Error fetching ESPN data: {e}")
            return games
    
    # Parse games from ESPN response
    for event in data.get("events", []):
        try:
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) != 2:
                continue
                
            # ESPN lists home team first (or has homeAway field)
            home_team = None
            away_team = None
            home_score = None
            away_score = None
            
            for comp in competitors:
                team_name = comp.get("team", {}).get("displayName", "")
                score = comp.get("score", "")
                home_away = comp.get("homeAway", "")
                
                if home_away == "home":
                    home_team = team_name
                    home_score = int(score) if score.isdigit() else None
                else:
                    away_team = team_name
                    away_score = int(score) if score.isdigit() else None
            
            if not home_team or not away_team:
                continue
                
            # Get game status
            status = event.get("status", {}).get("type", {})
            status_name = status.get("name", "")
            is_final = status_name in ["STATUS_FINAL", "STATUS_FINAL_OT"]
            
            games.append({
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "status": "final" if is_final else status_name.lower(),
                "game_date": target_date.strftime("%Y-%m-%d"),
            })
            
        except Exception as e:
            print(f"  Warning: Could not parse game: {e}")
            continue
    
    return games


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    normalized = name.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def match_games(db_games: list[dict], espn_games: list[dict]) -> list[tuple]:
    """Match database games with ESPN results.
    
    Returns:
        List of (db_game, espn_game) tuples
    """
    matches = []
    
    for db_game in db_games:
        db_home = normalize_team_name(db_game.get("home_team", ""))
        db_away = normalize_team_name(db_game.get("away_team", ""))
        
        for espn_game in espn_games:
            espn_home = normalize_team_name(espn_game.get("home_team", ""))
            espn_away = normalize_team_name(espn_game.get("away_team", ""))
            
            # Check for match (fuzzy matching)
            home_match = (
                db_home in espn_home or 
                espn_home in db_home or
                db_home.split()[0] == espn_home.split()[0]  # First word match
            )
            away_match = (
                db_away in espn_away or 
                espn_away in db_away or
                db_away.split()[0] == espn_away.split()[0]
            )
            
            if home_match and away_match:
                matches.append((db_game, espn_game))
                break
                
    return matches


async def update_results(target_date: date, calculate_predictions: bool = False) -> dict:
    """Update database with final scores.
    
    Args:
        target_date: Date to update results for
        calculate_predictions: Also calculate prediction results
        
    Returns:
        Summary of updates
    """
    from kenp0m_sp0rts_analyzer.historical_odds_db import get_db
    
    print(f"\n🏀 Fetching ESPN scores for {target_date}...")
    espn_games = await fetch_espn_scores(target_date)
    
    if not espn_games:
        return {"error": "No games found on ESPN"}
    
    final_games = [g for g in espn_games if g["status"] == "final"]
    print(f"   Found {len(espn_games)} games, {len(final_games)} final")
    
    # Get games from our database
    db = get_db()
    db_games = db.get_games_by_date(
        target_date.strftime("%Y-%m-%d"),
        league="College Basketball"
    )
    
    if not db_games:
        db.close()
        return {"error": "No games in database for this date"}
    
    print(f"   {len(db_games)} games in database")
    
    # Match games
    matches = match_games(db_games, final_games)
    print(f"   Matched {len(matches)} games")
    
    # Update results
    updated = 0
    for db_game, espn_game in matches:
        if espn_game["away_score"] is not None and espn_game["home_score"] is not None:
            success = db.update_game_result(
                game_id=db_game["game_id"],
                away_score=espn_game["away_score"],
                home_score=espn_game["home_score"]
            )
            if success:
                updated += 1
                print(f"   ✓ {espn_game['away_team']} {espn_game['away_score']} @ "
                      f"{espn_game['home_team']} {espn_game['home_score']}")
    
    # Calculate prediction results if requested
    prediction_results = []
    if calculate_predictions:
        print(f"\n📊 Calculating prediction results...")
        for db_game, _ in matches:
            if db_game.get("status") == "final":
                result = db.calculate_prediction_results(db_game["game_id"])
                if "results" in result:
                    prediction_results.append(result)
    
    db.close()
    
    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "espn_games": len(espn_games),
        "final_games": len(final_games),
        "db_games": len(db_games),
        "matched": len(matches),
        "updated": updated,
        "prediction_results": len(prediction_results),
    }


async def show_pending_games(target_date: date):
    """Show games that don't have final scores yet."""
    from kenp0m_sp0rts_analyzer.historical_odds_db import get_db
    
    db = get_db()
    games = db.get_games_by_date(
        target_date.strftime("%Y-%m-%d"),
        league="College Basketball"
    )
    db.close()
    
    pending = [g for g in games if g.get("status") != "final"]
    
    if not pending:
        print(f"\n✅ All games for {target_date} have final scores!")
        return
        
    print(f"\n⏳ Pending games for {target_date}:")
    print(f"{'MATCHUP':<50} {'STATUS':>15}")
    print("-" * 65)
    
    for g in pending:
        matchup = f"{g['away_team'][:22]} @ {g['home_team'][:22]}"
        status = g.get("status", "scheduled")
        print(f"{matchup:<50} {status:>15}")


async def show_results_summary(target_date: date):
    """Show results summary with ATS outcomes."""
    from kenp0m_sp0rts_analyzer.historical_odds_db import get_db
    
    db = get_db()
    games = db.get_games_by_date(
        target_date.strftime("%Y-%m-%d"),
        league="College Basketball"
    )
    db.close()
    
    final_games = [g for g in games if g.get("status") == "final"]
    
    if not final_games:
        print(f"\n⚠️  No final games for {target_date}")
        return
        
    print(f"\n📊 Results for {target_date}:")
    print(f"{'MATCHUP':<40} {'SCORE':>12} {'SPREAD':>8} {'RESULT':>8}")
    print("-" * 75)
    
    for g in final_games:
        away = g['away_team'][:18]
        home = g['home_team'][:18]
        matchup = f"{away} @ {home}"
        
        score = f"{g.get('away_score', '?')}-{g.get('home_score', '?')}"
        
        closing_spread = g.get("closing_spread")
        spread_str = f"{closing_spread:+.1f}" if closing_spread else "N/A"
        
        # Calculate cover result
        result = ""
        if closing_spread is not None and g.get("actual_spread") is not None:
            actual = g["actual_spread"]
            if actual < closing_spread:
                result = "HOME ✓"
            elif actual > closing_spread:
                result = "AWAY ✓"
            else:
                result = "PUSH"
        
        print(f"{matchup:<40} {score:>12} {spread_str:>8} {result:>8}")


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape game results and update historical database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Show games pending final scores"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show results summary with ATS outcomes"
    )
    parser.add_argument(
        "--calculate",
        action="store_true",
        help="Calculate prediction results after updating"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to update (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            return 1
    else:
        target_date = date.today()
    
    print("=" * 70)
    print("GAME RESULTS UPDATER")
    print("=" * 70)
    
    if args.pending:
        await show_pending_games(target_date)
        return 0
        
    if args.summary:
        await show_results_summary(target_date)
        return 0
    
    # Update results for specified days
    for day_offset in range(args.days):
        current_date = target_date - timedelta(days=day_offset)
        result = await update_results(current_date, calculate_predictions=args.calculate)
        
        if "error" in result:
            print(f"\n⚠️  {result['error']}")
        else:
            print(f"\n✅ Updated {result['updated']} games for {result['date']}")
            
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("View results summary:")
    print(f"  uv run python scrape_results.py --summary --date {target_date}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
