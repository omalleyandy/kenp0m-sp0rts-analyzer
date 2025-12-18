"""Edge Detection System - Compare KenPom vs Vegas Lines.

Identifies betting edges by comparing KenPom predictions to Vegas lines.
Calculates expected value and recommends bets with positive CLV potential.

Usage:
    # Compare KenPom to current Vegas lines
    uv run python scripts/analysis/edge_detector.py

    # Specify date
    uv run python scripts/analysis/edge_detector.py --date 2025-12-19

    # Minimum edge threshold (only show games with 3+ point edge)
    uv run python scripts/analysis/edge_detector.py --min-edge 3.0

    # Export report
    uv run python scripts/analysis/edge_detector.py -o edge_report.md
"""
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kenp0m_sp0rts_analyzer.luck_regression import calculate_luck_edge


class EdgeDetector:
    """Detects betting edges by comparing KenPom to Vegas."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize detector.

        Args:
            data_dir: Directory containing analysis and Vegas data
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"

    def load_kenpom_analysis(self, target_date: date) -> dict | None:
        """Load KenPom pre-game analysis.

        Args:
            target_date: Date of analysis

        Returns:
            Analysis dict or None if not found
        """
        date_str = target_date.strftime("%Y-%m-%d")
        analysis_file = self.data_dir / f"kenpom_analysis_{date_str}.json"

        if not analysis_file.exists():
            print(f"[WARNING] No KenPom analysis found for {date_str}")
            print(f"          Expected: {analysis_file}")
            print(f"          Run: uv run python scripts/analysis/kenpom_pregame_analyzer.py --date {date_str} -o {analysis_file}")
            return None

        with open(analysis_file) as f:
            return json.load(f)

    def load_vegas_lines(self, snapshot_type: str = "current") -> dict | None:
        """Load Vegas lines.

        Args:
            snapshot_type: 'open', 'current', or 'close'

        Returns:
            Lines dict or None if not found
        """
        lines_file = self.data_dir / f"vegas_lines_{snapshot_type}.json"

        if not lines_file.exists():
            print(f"[WARNING] No Vegas {snapshot_type} lines found")
            print(f"          Expected: {lines_file}")
            return None

        with open(lines_file) as f:
            return json.load(f)

    def match_teams(self, kenpom_name: str, vegas_name: str) -> bool:
        """Fuzzy match team names between KenPom and Vegas.

        Args:
            kenpom_name: Team name from KenPom
            vegas_name: Team name from Vegas

        Returns:
            True if names likely refer to same team
        """
        # Normalize names
        k_norm = kenpom_name.lower().strip()
        v_norm = vegas_name.lower().strip()

        # Exact match
        if k_norm == v_norm:
            return True

        # Remove common suffixes
        for suffix in [' wildcats', ' tigers', ' bulldogs', ' eagles', ' huskies']:
            k_norm = k_norm.replace(suffix, '')
            v_norm = v_norm.replace(suffix, '')

        # Check if one contains the other
        if k_norm in v_norm or v_norm in k_norm:
            return True

        # Check main word match
        k_words = set(k_norm.split())
        v_words = set(v_norm.split())
        if k_words & v_words:  # Intersection
            return True

        return False

    def find_matching_game(self, kenpom_game: dict, vegas_games: list[dict]) -> dict | None:
        """Find matching Vegas game for KenPom analysis.

        Args:
            kenpom_game: KenPom game analysis
            vegas_games: List of Vegas games

        Returns:
            Matching Vegas game or None
        """
        k_home = kenpom_game.get('home_team')
        k_away = kenpom_game.get('away_team')

        for v_game in vegas_games:
            v_home = v_game.get('home_team')
            v_away = v_game.get('away_team')

            if self.match_teams(k_home, v_home) and self.match_teams(k_away, v_away):
                return v_game

        return None

    def calculate_edge(self, kenpom_game: dict, vegas_game: dict) -> dict:
        """Calculate betting edge for a game.

        Args:
            kenpom_game: KenPom analysis
            vegas_game: Vegas lines

        Returns:
            Edge analysis dict
        """
        kp_pred = kenpom_game.get('kenpom_prediction', {})
        kp_margin = kp_pred.get('predicted_margin', 0)
        kp_total = kp_pred.get('predicted_total', 0)
        kp_home_wp = kp_pred.get('home_win_prob', 0.5)

        vegas_spread = vegas_game.get('spread', 0)
        vegas_total = vegas_game.get('total', 0)

        # NEW: Get luck regression data
        luck_regression = kenpom_game.get('luck_regression', {})
        luck_adjusted_margin = luck_regression.get('luck_adjusted_margin', kp_margin)
        luck_edge = luck_regression.get('luck_edge', 0.0)
        expected_clv = luck_regression.get('expected_clv', 0.0)

        # Calculate edges (use luck-adjusted margin)
        spread_edge = abs(luck_adjusted_margin) - abs(vegas_spread) if vegas_spread else None
        total_edge = kp_total - vegas_total if vegas_total else None

        # Determine recommendations
        recommendation = []

        if spread_edge and abs(spread_edge) >= 2.5:  # 2.5+ point edge
            if spread_edge > 0:  # KenPom predicts bigger margin
                if kp_margin > 0:  # Home team favored
                    recommendation.append({
                        'bet': f"{vegas_game['home_team']} {vegas_spread}",
                        'edge': spread_edge,
                        'confidence': kp_pred.get('confidence', 'medium'),
                        'reason': f"KenPom predicts {kp_margin:.1f}, Vegas has {vegas_spread}",
                    })
                else:  # Away team favored
                    recommendation.append({
                        'bet': f"{vegas_game['away_team']} +{abs(vegas_spread)}",
                        'edge': spread_edge,
                        'confidence': kp_pred.get('confidence', 'medium'),
                        'reason': f"KenPom predicts {kp_margin:.1f}, Vegas has {vegas_spread}",
                    })

        if total_edge and abs(total_edge) >= 5.0:  # 5+ point edge on total
            bet_type = "OVER" if total_edge > 0 else "UNDER"
            recommendation.append({
                'bet': f"{bet_type} {vegas_total}",
                'edge': total_edge,
                'confidence': 'medium',
                'reason': f"KenPom predicts {kp_total:.1f}, Vegas has {vegas_total}",
            })

        return {
            'matchup': kenpom_game.get('matchup'),
            'kenpom': {
                'predicted_margin': kp_margin,
                'luck_adjusted_margin': luck_adjusted_margin,  # NEW
                'predicted_total': kp_total,
                'home_win_prob': kp_home_wp,
                'confidence': kp_pred.get('confidence'),
            },
            'vegas': {
                'spread': vegas_spread,
                'total': vegas_total,
            },
            'edges': {
                'spread_edge': spread_edge,
                'total_edge': total_edge,
                'luck_edge': luck_edge,  # NEW
            },
            'luck_regression': luck_regression,  # NEW: Full luck data
            'recommendations': recommendation,
            'has_edge': len(recommendation) > 0,
        }

    def detect_all_edges(self, target_date: date, min_edge: float = 0.0) -> dict:
        """Detect edges for all games on a date.

        Args:
            target_date: Date to analyze
            min_edge: Minimum edge threshold in points

        Returns:
            Dict with all edge analyses
        """
        print(f"\n{'='*70}")
        print(f"EDGE DETECTION - {target_date.strftime('%A, %B %d, %Y')}")
        print(f"{'='*70}")

        # Load data
        kenpom_data = self.load_kenpom_analysis(target_date)
        vegas_data = self.load_vegas_lines("current")

        if not kenpom_data:
            return {'error': 'No KenPom analysis available'}

        if not vegas_data:
            return {'error': 'No Vegas lines available'}

        print(f"\nKenPom games: {len(kenpom_data.get('games', []))}")
        print(f"Vegas games: {len(vegas_data.get('games', []))}")

        # Match and analyze
        edges = []
        unmatched_kenpom = []

        for kp_game in kenpom_data.get('games', []):
            vegas_game = self.find_matching_game(kp_game, vegas_data.get('games', []))

            if vegas_game:
                edge_analysis = self.calculate_edge(kp_game, vegas_game)

                # Apply minimum edge filter
                if edge_analysis['edges']['spread_edge']:
                    if abs(edge_analysis['edges']['spread_edge']) >= min_edge:
                        edges.append(edge_analysis)
                elif edge_analysis['edges']['total_edge']:
                    if abs(edge_analysis['edges']['total_edge']) >= min_edge:
                        edges.append(edge_analysis)
                elif min_edge == 0:
                    edges.append(edge_analysis)
            else:
                unmatched_kenpom.append(kp_game.get('matchup'))

        # Sort by edge magnitude
        edges_with_bets = [e for e in edges if e['has_edge']]
        edges_with_bets.sort(key=lambda x: max(
            [rec['edge'] for rec in x['recommendations']],
            default=0
        ), reverse=True)

        result = {
            'date': target_date.strftime("%Y-%m-%d"),
            'analyzed_at': datetime.now().isoformat(),
            'games_analyzed': len(edges),
            'games_with_edges': len(edges_with_bets),
            'min_edge_threshold': min_edge,
            'edges': edges,
            'unmatched_games': unmatched_kenpom,
        }

        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Games analyzed: {len(edges)}")
        print(f"Games with betting edges: {len(edges_with_bets)}")
        print(f"Unmatched games: {len(unmatched_kenpom)}")

        return result

    def generate_report(self, edge_data: dict, output_path: Path | None = None) -> str:
        """Generate markdown report.

        Args:
            edge_data: Edge detection results
            output_path: Optional output file path

        Returns:
            Markdown report text
        """
        lines = []
        lines.append("# College Basketball Edge Detection Report")
        lines.append(f"\nDate: {edge_data.get('date')}")
        lines.append(f"Generated: {edge_data.get('analyzed_at', '')[:19]}")
        lines.append("\n" + "=" * 70)

        lines.append("\n## Summary")
        lines.append(f"\n- Games analyzed: {edge_data.get('games_analyzed')}")
        lines.append(f"- Games with betting edges: {edge_data.get('games_with_edges')}")
        lines.append(f"- Minimum edge threshold: {edge_data.get('min_edge_threshold')} points")

        # Recommended bets
        edges_with_bets = [e for e in edge_data.get('edges', []) if e['has_edge']]

        if edges_with_bets:
            lines.append("\n## Recommended Bets")

            for i, edge in enumerate(edges_with_bets, 1):
                lines.append(f"\n### {i}. {edge['matchup']}")

                for rec in edge['recommendations']:
                    lines.append(f"\n**BET: {rec['bet']}**")
                    lines.append(f"- Edge: {rec['edge']:.1f} points")
                    lines.append(f"- Confidence: {rec['confidence'].upper()}")
                    lines.append(f"- Reason: {rec['reason']}")

                # Show comparison
                lines.append(f"\n**Analysis:**")
                lines.append(f"- KenPom Margin: {edge['kenpom']['predicted_margin']:+.1f}")

                # NEW: Show luck adjustment if significant
                if abs(edge['edges'].get('luck_edge', 0)) > 0.5:
                    lines.append(f"- Luck-Adjusted Margin: {edge['kenpom']['luck_adjusted_margin']:+.1f}")
                    lines.append(f"- Luck Edge: {edge['edges']['luck_edge']:+.1f} points")

                lines.append(f"- Vegas Spread: {edge['vegas']['spread']:+.1f}")
                lines.append(f"- Spread Edge: {edge['edges']['spread_edge']:.1f}")
                lines.append(f"- KenPom Total: {edge['kenpom']['predicted_total']:.1f}")
                lines.append(f"- Vegas Total: {edge['vegas']['total']:.1f}")
                if edge['edges']['total_edge']:
                    lines.append(f"- Total Edge: {edge['edges']['total_edge']:.1f}")

                # NEW: Show luck regression details if present
                luck_reg = edge.get('luck_regression')
                if luck_reg and luck_reg.get('betting_recommendation') != "NO SIGNIFICANT LUCK EDGE":
                    lines.append(f"\n**Luck Regression:**")
                    lines.append(f"- {luck_reg.get('betting_recommendation')}")
                    if luck_reg.get('expected_clv', 0) > 0:
                        lines.append(f"- Expected CLV: +{luck_reg['expected_clv']:.1f} points")
        else:
            lines.append("\n## No Significant Edges Found")
            lines.append("\nNo games met the minimum edge threshold.")

        # Unmatched games
        unmatched = edge_data.get('unmatched_games', [])
        if unmatched:
            lines.append("\n## Unmatched Games")
            lines.append("\nKenPom games without matching Vegas lines:")
            for game in unmatched:
                lines.append(f"- {game}")

        report_text = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n[SAVED] {output_path}")

        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Edge detection: KenPom vs Vegas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Target date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--min-edge",
        type=float,
        default=2.5,
        help="Minimum edge threshold in points (default: 2.5)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output report path (markdown format)",
    )

    args = parser.parse_args()

    # Parse date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}")
            return 1
    else:
        target_date = date.today()

    # Run detection
    detector = EdgeDetector()
    edge_data = detector.detect_all_edges(target_date, min_edge=args.min_edge)

    if 'error' in edge_data:
        print(f"\n[ERROR] {edge_data['error']}")
        return 1

    # Generate report
    output_path = Path(args.output) if args.output else None
    report = detector.generate_report(edge_data, output_path)

    if not args.output:
        print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
