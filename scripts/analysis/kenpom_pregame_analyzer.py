"""KenPom Pre-Game Statistical Analyzer.

Analyzes games BEFORE Vegas lines post to identify statistical edges.
This is the Billy Walters approach - know your number before the market sets theirs.

Usage:
    # Analyze today's games
    uv run python scripts/analysis/kenpom_pregame_analyzer.py

    # Analyze specific date
    uv run python scripts/analysis/kenpom_pregame_analyzer.py --date 2025-12-19

    # Export to file
    uv run python scripts/analysis/kenpom_pregame_analyzer.py -o pregame_analysis.json
"""
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import ComprehensiveMatchupAnalyzer
from kenp0m_sp0rts_analyzer.prediction import GamePredictor
from kenp0m_sp0rts_analyzer.luck_regression import LuckRegressionAnalyzer


class PreGameAnalyzer:
    """Analyzes games before Vegas lines are available."""

    def __init__(self, api_key: str | None = None):
        """Initialize analyzer.

        Args:
            api_key: KenPom API key (reads from env if not provided)
        """
        self.api = KenPomAPI(api_key=api_key)
        self.analyzer = ComprehensiveMatchupAnalyzer(api_key=api_key)
        self.predictor = None  # Initialize when needed
        self.luck_analyzer = LuckRegressionAnalyzer()  # NEW: Luck regression

    def get_todays_games(self, target_date: date | None = None) -> list[dict]:
        """Get list of games for target date.

        Args:
            target_date: Date to get games for (default: today)

        Returns:
            List of game dicts with team names and times
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.strftime("%Y-%m-%d")

        print(f"[KENPOM] Fetching games for {date_str}...")

        try:
            # Get FanMatch predictions (includes game schedule)
            fanmatch_response = self.api.get_fanmatch(date_str)

            if not fanmatch_response.data:
                print(f"  [WARNING] No games found for {date_str}")
                return []

            games = []
            for game in fanmatch_response.data:
                games.append({
                    'home_team': game.get('Home'),
                    'away_team': game.get('Visitor'),
                    'game_time': game.get('Time'),
                    'venue': game.get('Venue'),
                    'kenpom_prediction': {
                        'home_win_prob': game.get('HomeWP'),
                        'predicted_score': game.get('PredScore'),
                    }
                })

            print(f"  [OK] Found {len(games)} games")
            return games

        except Exception as e:
            print(f"  [ERROR] Failed to fetch games: {e}")
            return []

    def analyze_game(self, home_team: str, away_team: str, neutral_site: bool = False) -> dict:
        """Perform comprehensive analysis of a single game.

        Args:
            home_team: Home team name
            away_team: Away team name
            neutral_site: Whether game is at neutral site

        Returns:
            Dict with complete analysis
        """
        print(f"\n[ANALYZING] {away_team} @ {home_team}")

        try:
            # Run comprehensive matchup analysis
            report = self.analyzer.analyze_matchup(
                team1=home_team,
                team2=away_team,
                neutral_site=neutral_site,
            )

            # NEW: Get team ratings for luck analysis
            home_ratings = self.api.get_team_by_name(home_team, year=datetime.now().year)
            away_ratings = self.api.get_team_by_name(away_team, year=datetime.now().year)

            # NEW: Analyze luck regression
            luck_edge = None
            if home_ratings and away_ratings:
                luck_edge = self.luck_analyzer.analyze_matchup_luck(
                    team1_name=home_team,
                    team1_adjEM=home_ratings.get('AdjEM', report.prediction.predicted_margin),
                    team1_luck=home_ratings.get('Luck', 0.0),
                    team2_name=away_team,
                    team2_adjEM=away_ratings.get('AdjEM', 0.0),
                    team2_luck=away_ratings.get('Luck', 0.0),
                    neutral_site=neutral_site,
                )

            # Extract key metrics
            analysis = {
                'matchup': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'neutral_site': neutral_site,
                'analyzed_at': datetime.now().isoformat(),

                # Prediction
                'kenpom_prediction': {
                    'home_win_prob': report.prediction.home_win_prob,
                    'away_win_prob': report.prediction.away_win_prob,
                    'predicted_margin': report.prediction.predicted_margin,
                    'predicted_total': report.prediction.predicted_total,
                    'confidence': report.prediction.confidence_level,
                },

                # NEW: Luck regression analysis
                'luck_regression': luck_edge.to_dict() if luck_edge else None,

                # Four Factors edges
                'four_factors': {
                    'efg_advantage': report.four_factors.summary.get('eFG% Edge'),
                    'to_advantage': report.four_factors.summary.get('TO% Edge'),
                    'or_advantage': report.four_factors.summary.get('OR% Edge'),
                    'ft_advantage': report.four_factors.summary.get('FT Rate Edge'),
                },

                # Tempo/Pace
                'tempo': {
                    'expected_pace': report.tempo_analysis.expected_pace,
                    'pace_advantage': report.tempo_analysis.pace_advantage,
                },

                # Size/Athleticism (TIER 2)
                'size_athleticism': {
                    'height_advantage': getattr(report, 'size_analysis', None) and report.size_analysis.height_advantage,
                    'experience_edge': getattr(report, 'experience_analysis', None) and report.experience_analysis.experience_edge,
                },

                # Key strengths/weaknesses
                'edges': self._identify_edges(report),
            }

            print(f"  [OK] Analysis complete")
            print(f"       Prediction: {home_team} by {analysis['kenpom_prediction']['predicted_margin']:.1f}")
            print(f"       Win Prob: {analysis['kenpom_prediction']['home_win_prob']:.1%}")
            print(f"       Confidence: {analysis['kenpom_prediction']['confidence']}")

            # NEW: Display luck regression analysis
            if analysis['luck_regression']:
                luck_data = analysis['luck_regression']
                print(f"\n  [LUCK REGRESSION]")
                print(f"       {home_team} Luck: {luck_data['team1_luck']:+.3f}")
                print(f"       {away_team} Luck: {luck_data['team2_luck']:+.3f}")
                print(f"       Luck Edge: {luck_data['luck_edge']:+.1f} points")
                print(f"       {luck_data['betting_recommendation']}")
                if luck_data['expected_clv'] > 0:
                    print(f"       Expected CLV: +{luck_data['expected_clv']:.1f} points")

            return analysis

        except Exception as e:
            print(f"  [ERROR] Analysis failed: {e}")
            return {
                'matchup': f"{away_team} @ {home_team}",
                'error': str(e),
                'analyzed_at': datetime.now().isoformat(),
            }

    def _identify_edges(self, report) -> list[dict]:
        """Identify significant statistical edges.

        Args:
            report: ComprehensiveMatchupReport

        Returns:
            List of edge dicts
        """
        edges = []

        # Check Four Factors edges
        ff = report.four_factors.summary
        for factor, value in ff.items():
            if 'Edge' in factor and abs(value) > 0.05:  # >5% edge
                edges.append({
                    'type': 'four_factors',
                    'factor': factor,
                    'magnitude': value,
                    'significance': 'high' if abs(value) > 0.10 else 'medium',
                })

        # Check win probability edge
        home_wp = report.prediction.home_win_prob
        if home_wp > 0.65 or home_wp < 0.35:  # Strong favorite/underdog
            edges.append({
                'type': 'win_probability',
                'factor': 'confidence',
                'magnitude': max(home_wp, 1 - home_wp),
                'significance': 'high' if max(home_wp, 1 - home_wp) > 0.75 else 'medium',
            })

        return edges

    def analyze_all_games(self, target_date: date | None = None) -> dict:
        """Analyze all games for a given date.

        Args:
            target_date: Date to analyze (default: today)

        Returns:
            Dict with all game analyses
        """
        if target_date is None:
            target_date = date.today()

        games = self.get_todays_games(target_date)

        if not games:
            return {
                'date': target_date.strftime("%Y-%m-%d"),
                'game_count': 0,
                'games': [],
                'analyzed_at': datetime.now().isoformat(),
            }

        print(f"\n{'='*70}")
        print(f"ANALYZING {len(games)} GAMES")
        print(f"{'='*70}")

        analyses = []
        for i, game in enumerate(games, 1):
            print(f"\n[{i}/{len(games)}]")
            analysis = self.analyze_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                neutral_site=False,  # TODO: Detect from venue
            )
            analyses.append(analysis)

        # Summarize
        high_confidence_picks = [
            a for a in analyses
            if 'kenpom_prediction' in a and
            a['kenpom_prediction'].get('confidence') == 'high'
        ]

        result = {
            'date': target_date.strftime("%Y-%m-%d"),
            'game_count': len(games),
            'analyzed_at': datetime.now().isoformat(),
            'games': analyses,
            'summary': {
                'high_confidence_picks': len(high_confidence_picks),
                'total_edges': sum(len(a.get('edges', [])) for a in analyses),
            }
        }

        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Games analyzed: {len(games)}")
        print(f"High confidence picks: {len(high_confidence_picks)}")
        print(f"Total edges identified: {result['summary']['total_edges']}")

        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KenPom pre-game statistical analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Target date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (JSON format)",
    )

    parser.add_argument(
        "--game", "-g",
        type=str,
        nargs=2,
        metavar=("AWAY", "HOME"),
        help="Analyze single game: --game 'Duke' 'North Carolina'",
    )

    args = parser.parse_args()

    # Parse date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}")
            print("        Use YYYY-MM-DD format")
            return 1
    else:
        target_date = date.today()

    # Initialize analyzer
    print("=" * 70)
    print("KENPOM PRE-GAME STATISTICAL ANALYZER")
    print("=" * 70)
    print(f"Date: {target_date.strftime('%A, %B %d, %Y')}")
    print("=" * 70)

    analyzer = PreGameAnalyzer()

    # Single game or all games
    if args.game:
        away_team, home_team = args.game
        result = analyzer.analyze_game(home_team, away_team)
    else:
        result = analyzer.analyze_all_games(target_date)

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[SAVED] {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
