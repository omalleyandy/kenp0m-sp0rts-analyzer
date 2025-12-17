#!/usr/bin/env python3
"""
Edge Validation Script

Validates betting edges by comparing:
1. KenPom model prediction
2. Historical game totals (actual results)
3. Market betting lines

Usage:
    python scripts/validate_edge.py "Montana St." "Cal Poly" --market-spread -3 --market-total 160 --home team2
    python scripts/validate_edge.py Duke Houston --market-spread 0 --market-total 180 --neutral
"""

import argparse
from pathlib import Path

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


class EdgeValidator:
    """Validates betting edges using KenPom data and actual game results."""

    HOME_COURT_ADVANTAGE = 3.5  # Points
    EDGE_THRESHOLD = 3.0  # Minimum edge in points to recommend a bet

    def __init__(self, season: int = 2025):
        self.season = season
        self.api = KenPomAPI()
        self._data_cache = None

    def get_team_data(self, team_name: str) -> dict | None:
        """Get team data from KenPom API."""
        if self._data_cache is None:
            print(f"Loading {self.season} season data...")
            ratings = self.api.get_ratings(year=self.season)
            self._data_cache = ratings.to_dataframe()

        team_name_lower = team_name.lower().strip()

        # Try exact match first
        match = self._data_cache[
            self._data_cache["TeamName"].str.lower() == team_name_lower
        ]

        # Try partial match if no exact match
        if len(match) == 0:
            match = self._data_cache[
                self._data_cache["TeamName"].str.lower().str.contains(team_name_lower)
            ]

        if len(match) == 0:
            return None

        if len(match) > 1:
            print(f"Warning: Multiple matches found for '{team_name}':")
            for idx, row in match.iterrows():
                print(f"  - {row['TeamName']}")
            print(f"Using first match: {match.iloc[0]['TeamName']}")

        return match.iloc[0].to_dict()

    def predict_game(
        self,
        team1_name: str,
        team2_name: str,
        home_team: str | None = None,
    ) -> dict:
        """Predict game using KenPom efficiency metrics."""
        team1 = self.get_team_data(team1_name)
        team2 = self.get_team_data(team2_name)

        if team1 is None:
            raise ValueError(f"Team not found: {team1_name}")
        if team2 is None:
            raise ValueError(f"Team not found: {team2_name}")

        # Calculate base margin
        margin = team1["AdjEM"] - team2["AdjEM"]

        # Apply home court advantage
        if home_team == "team1":
            margin += self.HOME_COURT_ADVANTAGE
            location = f"{team1['TeamName']} (Home)"
        elif home_team == "team2":
            margin -= self.HOME_COURT_ADVANTAGE
            location = f"{team2['TeamName']} (Home)"
        else:
            location = "Neutral Site"

        # Calculate expected tempo and scores
        expected_tempo = (team1["AdjTempo"] + team2["AdjTempo"]) / 2
        team1_score = (team1["AdjOE"] / 100) * expected_tempo * (100 / team2["AdjDE"])
        team2_score = (team2["AdjOE"] / 100) * expected_tempo * (100 / team1["AdjDE"])

        # Apply home court to scores
        if home_team == "team1":
            team1_score += self.HOME_COURT_ADVANTAGE / 2
            team2_score -= self.HOME_COURT_ADVANTAGE / 2
        elif home_team == "team2":
            team2_score += self.HOME_COURT_ADVANTAGE / 2
            team1_score -= self.HOME_COURT_ADVANTAGE / 2

        return {
            "team1": team1,
            "team2": team2,
            "predicted_margin": abs(margin),
            "predicted_winner": team1["TeamName"] if margin > 0 else team2["TeamName"],
            "team1_score": team1_score,
            "team2_score": team2_score,
            "predicted_total": team1_score + team2_score,
            "expected_tempo": expected_tempo,
            "location": location,
            "raw_margin": margin,
        }

    def get_historical_averages(self, team_name: str) -> dict:
        """
        Calculate historical game averages from actual games played.

        Note: This requires schedule data. For now, we'll use season averages
        from the team's overall performance metrics.

        TODO: Implement actual game-by-game scraping from KenPom team pages.
        """
        team = self.get_team_data(team_name)
        if team is None:
            raise ValueError(f"Team not found: {team_name}")

        # Use season averages as proxy (not ideal, but better than nothing)
        # These are adjusted metrics, so multiply by average tempo to get raw scores
        avg_tempo = team["AdjTempo"]

        # Estimate points scored/allowed per game
        points_scored = (team["AdjOE"] / 100) * avg_tempo
        points_allowed = (team["AdjDE"] / 100) * avg_tempo
        avg_total = points_scored + points_allowed

        # Use record to validate
        games_played = team["Wins"] + team["Losses"]

        return {
            "team_name": team["TeamName"],
            "games_played": games_played,
            "estimated_ppg": points_scored,
            "estimated_opponent_ppg": points_allowed,
            "estimated_total_ppg": avg_total,
            "record": f"{team['Wins']:.0f}-{team['Losses']:.0f}",
            "warning": "Using season averages - not actual game-by-game data",
        }

    def validate_edge(
        self,
        team1_name: str,
        team2_name: str,
        market_spread: float,
        market_total: float,
        home_team: str | None = None,
    ) -> dict:
        """
        Validate betting edge by comparing model vs historical vs market.

        Returns:
            Dictionary with validation results and recommendations
        """
        # Get KenPom prediction
        prediction = self.predict_game(team1_name, team2_name, home_team)

        # Get historical averages
        team1_hist = self.get_historical_averages(team1_name)
        team2_hist = self.get_historical_averages(team2_name)

        # Calculate historical total expectation
        historical_total = (
            team1_hist["estimated_ppg"] + team2_hist["estimated_opponent_ppg"] +
            team2_hist["estimated_ppg"] + team1_hist["estimated_opponent_ppg"]
        ) / 2

        # Compare predictions
        spread_edge = abs(prediction["raw_margin"] - market_spread)
        total_edge = abs(prediction["predicted_total"] - market_total)
        historical_vs_market = abs(historical_total - market_total)
        model_vs_historical = abs(prediction["predicted_total"] - historical_total)

        # Determine recommendations
        spread_recommendation = "PASS"
        total_recommendation = "PASS"

        if spread_edge >= self.EDGE_THRESHOLD:
            spread_recommendation = "BET" if prediction["raw_margin"] != market_spread else "PASS"

        # Check if model and historical agree (within 10 points)
        if model_vs_historical > 10:
            total_recommendation = "CONFLICT - Model and historical disagree (>10 points)"
        elif total_edge >= self.EDGE_THRESHOLD:
            # Only recommend if model and historical agree on direction
            if (prediction["predicted_total"] < market_total and historical_total < market_total):
                total_recommendation = "BET UNDER"
            elif (prediction["predicted_total"] > market_total and historical_total > market_total):
                total_recommendation = "BET OVER"
            else:
                total_recommendation = "CONFLICT - Direction mismatch"

        return {
            "prediction": prediction,
            "team1_historical": team1_hist,
            "team2_historical": team2_hist,
            "market": {
                "spread": market_spread,
                "total": market_total,
            },
            "edges": {
                "spread_edge": spread_edge,
                "total_edge": total_edge,
                "historical_vs_market": historical_vs_market,
                "model_vs_historical": model_vs_historical,
            },
            "analysis": {
                "kenpom_total": prediction["predicted_total"],
                "historical_total": historical_total,
                "market_total": market_total,
                "spread_recommendation": spread_recommendation,
                "total_recommendation": total_recommendation,
            },
            "warnings": [
                team1_hist.get("warning"),
                team2_hist.get("warning"),
                "Manual validation required - check actual game results" if model_vs_historical > 10 else None,
            ],
        }

    def print_validation_report(self, validation: dict):
        """Print formatted validation report."""
        pred = validation["prediction"]
        team1 = pred["team1"]
        team2 = pred["team2"]
        market = validation["market"]
        edges = validation["edges"]
        analysis = validation["analysis"]

        print(f"\n{'='*80}")
        print(f"EDGE VALIDATION REPORT - {self.season} Season")
        print(f"{'='*80}\n")

        print(f"{team1['TeamName']} vs {team2['TeamName']}")
        print(f"Location: {pred['location']}\n")

        print(f"{'MODEL PREDICTION':-^80}")
        print(f"\nSpread: {pred['predicted_winner']} by {pred['predicted_margin']:.1f}")
        print(f"Total: {pred['predicted_total']:.1f} points")
        print(f"Expected Score: {pred['team1_score']:.1f} - {pred['team2_score']:.1f}")
        print(f"Expected Tempo: {pred['expected_tempo']:.1f} possessions\n")

        print(f"{'HISTORICAL AVERAGES':-^80}\n")
        team1_hist = validation["team1_historical"]
        team2_hist = validation["team2_historical"]
        print(f"{team1['TeamName']} ({team1_hist['record']}):")
        print(f"  Estimated PPG: {team1_hist['estimated_ppg']:.1f}")
        print(f"  Opponent PPG: {team1_hist['estimated_opponent_ppg']:.1f}")
        print(f"  Game Total: {team1_hist['estimated_total_ppg']:.1f}\n")
        print(f"{team2['TeamName']} ({team2_hist['record']}):")
        print(f"  Estimated PPG: {team2_hist['estimated_ppg']:.1f}")
        print(f"  Opponent PPG: {team2_hist['estimated_opponent_ppg']:.1f}")
        print(f"  Game Total: {team2_hist['estimated_total_ppg']:.1f}\n")
        print(f"Combined Historical Total: {analysis['historical_total']:.1f}\n")

        print(f"{'MARKET COMPARISON':-^80}\n")
        print(f"Market Spread: {market['spread']:+.1f}")
        print(f"Market Total: {market['total']:.1f}\n")

        print(f"{'EDGE ANALYSIS':-^80}\n")
        print(f"Spread Edge: {edges['spread_edge']:.1f} points")
        print(f"Total Edge (Model vs Market): {edges['total_edge']:.1f} points")
        print(f"Total Edge (Historical vs Market): {edges['historical_vs_market']:.1f} points")
        print(f"Model vs Historical: {edges['model_vs_historical']:.1f} points\n")

        print(f"{'RECOMMENDATIONS':-^80}\n")
        print(f"Spread: {analysis['spread_recommendation']}")
        print(f"Total: {analysis['total_recommendation']}\n")

        # Print warnings
        warnings = [w for w in validation["warnings"] if w]
        if warnings:
            print(f"{'WARNINGS':-^80}\n")
            for warning in warnings:
                print(f"WARNING: {warning}")
            print()

        print(f"{'='*80}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate betting edges using KenPom data"
    )
    parser.add_argument("team1", help="First team name")
    parser.add_argument("team2", help="Second team name")
    parser.add_argument(
        "--market-spread",
        type=float,
        required=True,
        help="Market spread (positive = team1 favored, negative = team2 favored)",
    )
    parser.add_argument(
        "--market-total",
        type=float,
        required=True,
        help="Market total points line",
    )
    parser.add_argument("--season", type=int, default=2025, help="Season year")
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Game at neutral site",
    )
    parser.add_argument(
        "--home",
        choices=["team1", "team2"],
        help="Which team is home",
    )

    args = parser.parse_args()

    # Create validator
    validator = EdgeValidator(season=args.season)

    try:
        # Validate edge
        validation = validator.validate_edge(
            args.team1,
            args.team2,
            args.market_spread,
            args.market_total,
            home_team=args.home,
        )

        # Print report
        validator.print_validation_report(validation)

    except ValueError as e:
        print(f"\nError: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
