"""Dynamic analysis of NCAA Basketball games with Vegas line comparison.

This script automatically fetches today's games from the KenPom FanMatch API
and compares predictions against Vegas lines from overtime.ag.

Usage:
    uv run python analyze_todays_games.py
    uv run python analyze_todays_games.py --date 2025-12-20
    uv run python analyze_todays_games.py --min-edge 3
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.analysis import analyze_matchup
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer
from kenp0m_sp0rts_analyzer.experience_chemistry_analysis import (
    ExperienceChemistryAnalyzer,
)
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup
from kenp0m_sp0rts_analyzer.luck_regression import LuckRegressionAnalyzer
from kenp0m_sp0rts_analyzer.point_distribution_analysis import (
    PointDistributionAnalyzer,
)
from kenp0m_sp0rts_analyzer.size_athleticism_analysis import (
    SizeAthleticismAnalyzer,
)
from kenp0m_sp0rts_analyzer.utils import normalize_team_name

# Load environment variables
load_dotenv()

# Vegas lines source
VEGAS_SOURCE = "overtime.ag"


def get_current_season() -> int:
    """Get the current basketball season year.

    The college basketball season spans two calendar years (e.g., 2024-25).
    KenPom uses the ending year (2025 for 2024-25 season).
    Season typically starts in November.
    """
    now = datetime.now()
    if now.month >= 8:
        return now.year + 1
    return now.year


def load_vegas_lines(date_str: str) -> dict:
    """Load Vegas lines from JSON file.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        Dictionary mapping (away_team, home_team) to line data
    """
    # Try to load from data directory
    data_dir = Path(__file__).parent / "data"
    vegas_file = data_dir / "vegas_lines.json"

    if not vegas_file.exists():
        return {}

    with vegas_file.open() as f:
        data = json.load(f)

    # Check if the file is for the requested date
    if data.get("date") != date_str:
        file_date = data.get("date")
        print(f"[WARN] Vegas lines file is for {file_date}, not {date_str}")
        print("       Update data/vegas_lines.json with current lines")
        return {}

    # Create lookup dictionary with normalized team names
    lines = {}
    for game in data.get("games", []):
        away = normalize_team_name(game["away_team"])
        home = normalize_team_name(game["home_team"])
        lines[(away, home)] = {
            "spread": game["spread"],  # Negative = home favored
            "total": game["total"],
            "away_ml": game.get("away_ml"),
            "home_ml": game.get("home_ml"),
            "time": game.get("time", ""),
        }

    return lines


def load_injuries() -> dict[str, list[dict]]:
    """Load injury data from JSON file.

    Returns:
        Dictionary mapping normalized team names to list of injuries
    """
    data_dir = Path(__file__).parent / "data"
    injury_file = data_dir / "injuries_covers.json"

    if not injury_file.exists():
        return {}

    with injury_file.open() as f:
        data = json.load(f)

    # Build lookup by normalized team name
    injuries_by_team: dict[str, list[dict]] = {}

    for injury in data.get("injuries", []):
        # Covers format: "Alabama CRIMSON TIDE" -> normalize to "Alabama"
        raw_team = injury.get("team", "")
        # Extract first part before all-caps mascot
        parts = raw_team.split()
        team_parts = []
        for part in parts:
            if part.isupper() and len(part) > 2:
                break
            team_parts.append(part)
        team_name = " ".join(team_parts) if team_parts else raw_team

        # Normalize for KenPom matching
        normalized = normalize_team_name(team_name)

        if normalized not in injuries_by_team:
            injuries_by_team[normalized] = []

        injuries_by_team[normalized].append(
            {
                "player": injury.get("player", "Unknown"),
                "position": injury.get("position", "?"),
                "status": injury.get("status", "Unknown"),
                "injury_type": injury.get("injury_type", ""),
            }
        )

    return injuries_by_team


def get_team_luck(api: KenPomAPI, team_name: str, season: int) -> float | None:
    """Get luck factor for a team from KenPom ratings.

    Args:
        api: KenPomAPI client
        team_name: Team name to look up
        season: Season year

    Returns:
        Luck factor or None if not found
    """
    try:
        ratings = api.get_ratings(year=season)
        for team in ratings.data:
            if team.get("TeamName") == team_name:
                return team.get("Luck", 0.0)
        # Try normalized name lookup
        normalized = normalize_team_name(team_name)
        for team in ratings.data:
            if normalize_team_name(team.get("TeamName", "")) == normalized:
                return team.get("Luck", 0.0)
    except Exception:
        pass
    return None


def print_separator():
    """Print a visual separator line."""
    print("\n" + "=" * 80)


def analyze_game(
    api: KenPomAPI,
    game: dict,
    season: int,
    vegas_lines: dict,
    injuries: dict[str, list[dict]],
    luck_analyzer: LuckRegressionAnalyzer,
    ratings_cache: dict,
) -> dict | None:
    """Analyze a single game and return key insights.

    Args:
        api: KenPomAPI client instance
        game: Game data from fanmatch endpoint
        season: Season year (e.g., 2025)
        vegas_lines: Dictionary of Vegas lines from overtime.ag
        injuries: Dictionary of injuries by normalized team name
        luck_analyzer: LuckRegressionAnalyzer instance
        ratings_cache: Cache of team ratings to avoid repeated API calls

    Returns:
        Dictionary with analysis results or None if analysis fails
    """
    team1 = game["Visitor"]  # Away team
    team2 = game["Home"]  # Home team

    # KenPom predictions from fanmatch
    kenpom_visitor_score = game["VisitorPred"]
    kenpom_home_score = game["HomePred"]
    # Negative spread = home favored
    kenpom_spread = kenpom_home_score - kenpom_visitor_score
    home_win_prob = game["HomeWP"]
    thrill_score = game.get("ThrillScore", 0)
    predicted_tempo = game.get("PredTempo", 0)
    kenpom_total = kenpom_visitor_score + kenpom_home_score

    # Look up Vegas lines
    normalized_away = normalize_team_name(team1)
    normalized_home = normalize_team_name(team2)
    vegas = vegas_lines.get((normalized_away, normalized_home), {})
    vegas_spread = vegas.get("spread")
    vegas_total = vegas.get("total")

    print(f"\nAnalyzing: {team1} @ {team2}")
    print(f"  KenPom: {team2} {-kenpom_spread:+.1f}")
    if vegas_spread is not None:
        print(f"  Vegas:  {team2} {vegas_spread:+.1f} (via {VEGAS_SOURCE})")
    else:
        print("  Vegas:  No line available")
    print(f"  Home Win Prob: {home_win_prob}% | Thrill: {thrill_score}")
    print("-" * 80)

    try:
        # Basic efficiency analysis
        basic = analyze_matchup(
            team1, team2, season, neutral_site=False, home_team=team2
        )

        # Four Factors analysis
        ff = FourFactorsMatchup(api)
        ff_analysis = ff.analyze_matchup(team1, team2, season)

        # Point Distribution analysis
        pd_analyzer = PointDistributionAnalyzer(api)
        pd_analysis = pd_analyzer.analyze_matchup(team1, team2, season)

        # Defensive analysis
        defense = DefensiveAnalyzer(api)
        def_analysis = defense.analyze_matchup(team1, team2, season)

        # Size & Athleticism analysis
        size = SizeAthleticismAnalyzer(api)
        size_analysis = size.analyze_matchup(team1, team2, season)

        # Experience & Chemistry analysis
        exp = ExperienceChemistryAnalyzer(api)
        exp_analysis = exp.analyze_matchup(team1, team2, season)

        # Count dimensional wins
        team1_wins = 0
        team2_wins = 0

        # Basic efficiency
        if basic.em_difference > 0:
            team1_wins += 1
        else:
            team2_wins += 1

        # Four Factors
        if ff_analysis.overall_advantage == team1.lower():
            team1_wins += 1
        elif ff_analysis.overall_advantage == team2.lower():
            team2_wins += 1

        # Defense
        if def_analysis.better_defense == team1:
            team1_wins += 1
        elif def_analysis.better_defense == team2:
            team2_wins += 1

        # Size
        if size_analysis.better_size_team == team1:
            team1_wins += 1
        elif size_analysis.better_size_team == team2:
            team2_wins += 1

        # Experience
        if exp_analysis.better_intangibles == team1:
            team1_wins += 1
        elif exp_analysis.better_intangibles == team2:
            team2_wins += 1

        # Print results
        winner = basic.predicted_winner
        margin = basic.predicted_margin
        print("\nRESULTS:")
        print(f"  Prediction: {winner} by {margin:.1f}")
        print(f"  KenPom Total: {kenpom_total:.1f}")
        print(f"  Tempo: {predicted_tempo:.1f} possessions")
        print(f"  Dimensions: {team2} {team2_wins}-{team1_wins} {team1}")

        # Four Factors
        ff_adv = ff_analysis.overall_advantage.upper()
        ff_score = ff_analysis.advantage_score
        print(f"\n  Four Factors: {ff_adv} ({ff_score:+.2f})")

        # Key advantages
        three_pt_adv = pd_analysis.three_point_advantage
        if three_pt_adv > 2.0:
            print(f"  -> {team1} 3PT advantage ({three_pt_adv:+.1f}%)")
        elif three_pt_adv < -2.0:
            print(f"  -> {team2} 3PT advantage ({three_pt_adv:+.1f}%)")

        height_adv = size_analysis.overall_height_advantage
        if height_adv > 1.5:
            print(f'  -> {team1} size advantage ({height_adv:+.1f}")')
        elif height_adv < -1.5:
            print(f'  -> {team2} size advantage ({height_adv:+.1f}")')

        # Luck Regression Analysis (evidence-backed)
        luck_edge = 0.0
        team1_luck = None
        team2_luck = None
        luck_recommendation = None

        # Get luck values from ratings cache
        for team_data in ratings_cache.values():
            if team_data.get("TeamName") == team1:
                team1_luck = team_data.get("Luck", 0.0)
            if team_data.get("TeamName") == team2:
                team2_luck = team_data.get("Luck", 0.0)

        # Also try normalized names
        if team1_luck is None or team2_luck is None:
            norm1 = normalize_team_name(team1)
            norm2 = normalize_team_name(team2)
            for team_data in ratings_cache.values():
                raw_name = team_data.get("TeamName", "")
                normalized_name = normalize_team_name(raw_name)
                if normalized_name == norm1 and team1_luck is None:
                    team1_luck = team_data.get("Luck", 0.0)
                if normalized_name == norm2 and team2_luck is None:
                    team2_luck = team_data.get("Luck", 0.0)

        if team1_luck is not None and team2_luck is not None:
            # Get AdjEM for luck analysis
            team1_adjEM = basic.team1_adj_em
            team2_adjEM = basic.team2_adj_em

            luck_result = luck_analyzer.analyze_matchup_luck(
                team1_name=team1,
                team1_adjEM=team1_adjEM,
                team1_luck=team1_luck,
                team2_name=team2,
                team2_adjEM=team2_adjEM,
                team2_luck=team2_luck,
                games_remaining=15,  # Mid-season estimate
                neutral_site=False,
                home_court_advantage=3.5,
            )
            luck_edge = luck_result.luck_edge
            luck_recommendation = luck_result.betting_recommendation

            print("\n  [LUCK REGRESSION] (Evidence-Backed: 10,000+ games)")
            print(f"     {team1} Luck: {team1_luck:+.3f}", end="")
            if team1_luck > 0.08:
                print(" (LUCKY - fade)")
            elif team1_luck < -0.08:
                print(" (UNLUCKY - back)")
            else:
                print(" (neutral)")

            print(f"     {team2} Luck: {team2_luck:+.3f}", end="")
            if team2_luck > 0.08:
                print(" (LUCKY - fade)")
            elif team2_luck < -0.08:
                print(" (UNLUCKY - back)")
            else:
                print(" (neutral)")

            print(f"     Luck Edge: {luck_edge:+.1f} pts")
            if abs(luck_edge) >= 1.5:
                print(f"     -> {luck_recommendation}")

        # Injury Report (Display Only - No Adjustment)
        team1_injuries = injuries.get(normalize_team_name(team1), [])
        team2_injuries = injuries.get(normalize_team_name(team2), [])

        if team1_injuries or team2_injuries:
            print("\n  [INJURY REPORT] (Display Only - No Adjustment)")
            if team1_injuries:
                print(f"     {team1}:")
                for inj in team1_injuries[:3]:  # Show max 3
                    icon = "[OUT]" if inj["status"] == "Out" else "[Q]"
                    pos = inj["position"]
                    inj_type = inj["injury_type"]
                    print(
                        f"       {icon} {inj['player']} ({pos}) - {inj_type}"
                    )
            if team2_injuries:
                print(f"     {team2}:")
                for inj in team2_injuries[:3]:  # Show max 3
                    icon = "[OUT]" if inj["status"] == "Out" else "[Q]"
                    pos = inj["position"]
                    inj_type = inj["injury_type"]
                    print(
                        f"       {icon} {inj['player']} ({pos}) - {inj_type}"
                    )

        # Value Analysis vs Vegas
        spread_edge = None
        total_edge = None
        spread_pick = None
        total_pick = None
        composite_edge = None

        if vegas_spread is not None:
            # KenPom spread (negative = home favored)
            kp_spread = -kenpom_spread
            spread_edge = kp_spread - vegas_spread

            # Composite edge = KenPom edge + luck regression edge
            composite_edge = spread_edge + luck_edge

            print("\n  [SPREAD ANALYSIS]")
            print(f"     KenPom: {team2} {kp_spread:+.1f}")
            print(f"     Vegas:  {team2} {vegas_spread:+.1f}")
            print(f"     KenPom Edge:    {spread_edge:+.1f} pts")
            print(f"     Luck Edge:      {luck_edge:+.1f} pts")
            print(f"     COMPOSITE EDGE: {composite_edge:+.1f} pts")

            # Use COMPOSITE edge for value determination (KenPom + Luck)
            if abs(composite_edge) >= 3:
                if composite_edge < 0:
                    # Composite analysis favors HOME team
                    # VALUE: Bet HOME team (they cover with extra points)
                    if vegas_spread > 0:
                        spread_pick = f"{team2} +{vegas_spread}"
                    else:
                        spread_pick = f"{team2} {vegas_spread}"
                    print(f"     [VALUE] {spread_pick}")
                    print("        (Home favored by composite analysis)")
                else:
                    # Composite analysis favors AWAY team
                    # VALUE: Bet AWAY team (home is overvalued)
                    if vegas_spread > 0:
                        # Home is underdog, away is favorite
                        spread_pick = f"{team1} -{vegas_spread}"
                    else:
                        # Home is favorite, away is underdog
                        spread_pick = f"{team1} +{abs(vegas_spread)}"
                    print(f"     [VALUE] {spread_pick}")
                    print("        (Away favored by composite analysis)")
            elif abs(composite_edge) < 1:
                print("     -- NO EDGE - Lines agree")
            else:
                print("     [WARN] Small edge - proceed with caution")

        if vegas_total is not None:
            total_edge = kenpom_total - vegas_total

            print("\n  [TOTAL ANALYSIS]")
            print(f"     KenPom: {kenpom_total:.1f}")
            print(f"     Vegas:  {vegas_total}")
            print(f"     Edge:   {total_edge:+.1f} points")

            if abs(total_edge) >= 5:
                if total_edge > 0:
                    total_pick = f"OVER {vegas_total}"
                    print(f"     [VALUE] {total_pick}")
                else:
                    total_pick = f"UNDER {vegas_total}"
                    print(f"     [VALUE] {total_pick}")
            elif abs(total_edge) < 2:
                print("     -- NO EDGE - Lines agree")
            else:
                print("     [WARN] Small edge - proceed with caution")

        return {
            "team1": team1,
            "team2": team2,
            "kenpom_spread": -kenpom_spread,
            "vegas_spread": vegas_spread,
            "spread_edge": spread_edge,
            "composite_edge": composite_edge,
            "luck_edge": luck_edge,
            "team1_luck": team1_luck,
            "team2_luck": team2_luck,
            "spread_pick": spread_pick,
            "kenpom_total": kenpom_total,
            "vegas_total": vegas_total,
            "total_edge": total_edge,
            "total_pick": total_pick,
            "home_win_prob": home_win_prob,
            "thrill_score": thrill_score,
            "system_winner": basic.predicted_winner,
            "system_margin": basic.predicted_margin,
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "ff_advantage": ff_analysis.overall_advantage,
            "is_close_game": 35 <= home_win_prob <= 65,
        }

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    """Main entry point for the analysis script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze NCAA Basketball games vs Vegas lines"
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default=None,
        help="Date to analyze (YYYY-MM-DD format). Defaults to today.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=3.0,
        help="Minimum spread edge to flag as value (default: 3.0 points)",
    )
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only show close games (35-65%% win probability)",
    )
    args = parser.parse_args()

    # Determine the date to analyze
    if args.date:
        try:
            analysis_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}")
            print("        Use YYYY-MM-DD format (e.g., 2025-12-20)")
            return
    else:
        analysis_date = date.today()

    season = get_current_season()
    date_display = analysis_date.strftime("%A, %B %d, %Y")
    date_api = analysis_date.strftime("%Y-%m-%d")

    print("=" * 80)
    print(f"NCAA BASKETBALL ANALYSIS - {date_display.upper()}")
    season_str = f"{season - 1}-{str(season)[2:]}"
    print(f"Season: {season_str} | Vegas Source: {VEGAS_SOURCE}")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("\n[ERROR] KENPOM_API_KEY not found")
        return

    print("\n[OK] API key loaded")

    # Initialize API client
    try:
        api = KenPomAPI()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize API: {e}")
        return

    # Load Vegas lines
    vegas_lines = load_vegas_lines(date_api)
    if vegas_lines:
        line_count = len(vegas_lines)
        print(f"[OK] Loaded {line_count} Vegas lines from {VEGAS_SOURCE}")
    else:
        print("[WARN] No Vegas lines loaded - update data/vegas_lines.json")

    # Load injury data
    injuries = load_injuries()
    if injuries:
        team_count = len(injuries)
        print(f"[OK] Loaded injuries for {team_count} teams")
    else:
        print("[INFO] No injury data available")

    # Initialize luck regression analyzer
    luck_analyzer = LuckRegressionAnalyzer()
    print("[OK] Luck regression analyzer initialized")

    # Load ratings cache for luck values
    ratings_cache: dict[str, dict] = {}
    try:
        ratings_response = api.get_ratings(year=season)
        for team in ratings_response.data:
            team_name = team.get("TeamName", "")
            if team_name:
                ratings_cache[team_name] = team
        print(f"[OK] Cached ratings for {len(ratings_cache)} teams")
    except Exception as e:
        print(f"[WARN] Failed to load ratings cache: {e}")

    # Fetch games from KenPom fanmatch
    print(f"[OK] Fetching games for {date_api}...")

    try:
        response = api.get_fanmatch(date_api)
        games = response.data
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch games: {e}")
        return

    if not games:
        print(f"\n[INFO] No games found for {date_display}")
        return

    # Filter games
    filtered_games = games
    if args.close_only:
        filtered_games = [
            g for g in filtered_games if 35 <= g.get("HomeWP", 50) <= 65
        ]
        print("[OK] Filtered to close games only")

    game_count = len(filtered_games)
    print(f"[OK] Found {game_count} games to analyze\n")

    if not filtered_games:
        print("[INFO] No games match your filter criteria")
        return

    # Analyze each game
    results = []

    for i, game in enumerate(filtered_games, 1):
        print(f"\n[{i}/{len(filtered_games)}]", end="")
        result = analyze_game(
            api,
            game,
            season,
            vegas_lines,
            injuries,
            luck_analyzer,
            ratings_cache,
        )
        if result:
            results.append(result)
        print_separator()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL GAMES")
    print("=" * 80)

    if not results:
        print("\nNo games were successfully analyzed.")
        return

    # All games table (shows COMPOSITE edge = KenPom + Luck)
    header = f"{'MATCHUP':<33} {'KP':>6} {'VEG':>6} {'COMP':>6} {'TOT':>4}"
    print(f"\n{header}")
    print("-" * 58)

    for r in results:
        matchup = f"{r['team1'][:15]} @ {r['team2'][:15]}"
        kp_spread = f"{r['kenpom_spread']:+.1f}"
        if r["vegas_spread"] is not None:
            vegas_spread = f"{r['vegas_spread']:+.1f}"
        else:
            vegas_spread = "N/A"
        # Show COMPOSITE edge (KenPom + Luck)
        if r["composite_edge"] is not None:
            edge = f"{r['composite_edge']:+.1f}"
        elif r["spread_edge"] is not None:
            edge = f"{r['spread_edge']:+.1f}"
        else:
            edge = "N/A"
        total = f"{r['kenpom_total']:.0f}"
        fmt = f"{matchup:<33} {kp_spread:>6} {vegas_spread:>6}"
        fmt += f" {edge:>6} {total:>4}"
        print(fmt)

    # Value picks (spread) - using COMPOSITE edge
    spread_value = [r for r in results if r.get("spread_pick")]
    if spread_value:
        print(
            f"\n[PICK] SPREAD VALUE PICKS (Composite >= {args.min_edge} pts):"
        )
        print("-" * 65)
        for r in spread_value:
            print(f"   {r['team1']} @ {r['team2']}")
            print(f"      PICK: {r['spread_pick']}")
            kp_edge = r.get("spread_edge") or 0
            luck_e = r.get("luck_edge") or 0
            comp_e = r.get("composite_edge") or 0
            wp = r["home_win_prob"]
            print(f"      KenPom: {kp_edge:+.1f} | Luck: {luck_e:+.1f}")
            print(f"      COMPOSITE: {comp_e:+.1f} pts | WP: {wp}%")
            print()

    # Value picks (totals)
    total_value = [r for r in results if r.get("total_pick")]
    if total_value:
        print("\n[PICK] TOTAL VALUE PICKS (Edge >= 5 pts):")
        print("-" * 60)
        for r in total_value:
            print(f"   {r['team1']} @ {r['team2']}")
            print(f"      PICK: {r['total_pick']}")
            kp_tot = r["kenpom_total"]
            v_tot = r["vegas_total"]
            t_edge = r["total_edge"]
            print(f"      KP: {kp_tot:.1f} | V: {v_tot} | E: {t_edge:+.1f}")
            print()

    # Close games
    close_games = [r for r in results if r.get("is_close_game")]
    if close_games:
        print(f"\n[WARN] CLOSE GAMES ({len(close_games)} games, 35-65% WP):")
        for r in close_games:
            wp = r["home_win_prob"]
            print(f"   {r['team1']} @ {r['team2']} ({wp}% home)")

    # Stats
    games_with_lines = [r for r in results if r["vegas_spread"] is not None]
    print("\n[OK] ANALYSIS COMPLETE")
    print(f"   Date: {date_display}")
    print(f"   Total Games: {len(results)}")
    print(f"   Games with Vegas Lines: {len(games_with_lines)}")
    print(f"   Spread Value Picks: {len(spread_value)}")
    print(f"   Total Value Picks: {len(total_value)}")
    print(f"   Close Games: {len(close_games)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
