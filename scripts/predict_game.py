#!/usr/bin/env python3
"""
Simple Game Prediction Script

Predict college basketball games using KenPom efficiency metrics.

Usage:
    python scripts/predict_game.py Duke "North Carolina"
    python scripts/predict_game.py Duke "North Carolina" --season 2025 --neutral
    python scripts/predict_game.py --team1 "Duke" --team2 "Virginia" --home team1
"""

import argparse
from pathlib import Path

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


class GamePredictor:
    """Simple game predictor using KenPom efficiency metrics."""

    HOME_COURT_ADVANTAGE = 3.5  # Points

    def __init__(self, season: int = 2025):
        self.season = season
        self.api = KenPomAPI()
        self._data_cache = None

    def get_team_data(self, team_name: str) -> dict | None:
        """
        Get team data from KenPom API.

        Args:
            team_name: Team name (e.g., "Duke", "North Carolina")

        Returns:
            Team data dictionary or None if not found
        """
        # Check cache first
        if self._data_cache is None:
            print(f"Fetching {self.season} season data...")
            ratings = self.api.get_ratings(year=self.season)
            self._data_cache = ratings.to_dataframe()

        # Normalize team name
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

    def predict_simple(
        self,
        team1_name: str,
        team2_name: str,
        neutral_site: bool = True,
        home_team: str | None = None,
    ) -> dict:
        """
        Predict game outcome using simple AdjEM difference.

        Args:
            team1_name: First team name
            team2_name: Second team name
            neutral_site: If True, no home court advantage
            home_team: "team1" or "team2" if not neutral (overrides neutral_site)

        Returns:
            Prediction dictionary with margin, winner, confidence
        """
        # Get team data
        team1 = self.get_team_data(team1_name)
        team2 = self.get_team_data(team2_name)

        if team1 is None:
            raise ValueError(f"Team not found: {team1_name}")
        if team2 is None:
            raise ValueError(f"Team not found: {team2_name}")

        # Calculate base margin (efficiency margin difference)
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

        # Determine winner
        if margin > 0:
            winner = team1["TeamName"]
            predicted_margin = margin
        else:
            winner = team2["TeamName"]
            predicted_margin = abs(margin)

        # Confidence levels based on margin
        if abs(margin) > 15:
            confidence = "Very High"
            confidence_pct = 0.95
        elif abs(margin) > 10:
            confidence = "High"
            confidence_pct = 0.85
        elif abs(margin) > 5:
            confidence = "Medium"
            confidence_pct = 0.70
        elif abs(margin) > 2:
            confidence = "Low"
            confidence_pct = 0.60
        else:
            confidence = "Very Low (Toss-up)"
            confidence_pct = 0.55

        return {
            "team1": team1,
            "team2": team2,
            "predicted_winner": winner,
            "predicted_margin": predicted_margin,
            "raw_margin": margin,
            "confidence": confidence,
            "confidence_pct": confidence_pct,
            "location": location,
        }

    def predict_with_tempo(
        self,
        team1_name: str,
        team2_name: str,
        neutral_site: bool = True,
        home_team: str | None = None,
    ) -> dict:
        """
        Predict game outcome with tempo and scoring estimates.

        Args:
            team1_name: First team name
            team2_name: Second team name
            neutral_site: If True, no home court advantage
            home_team: "team1" or "team2" if not neutral

        Returns:
            Prediction dictionary with scores, total, tempo
        """
        # Get base prediction
        prediction = self.predict_simple(team1_name, team2_name, neutral_site, home_team)

        team1 = prediction["team1"]
        team2 = prediction["team2"]

        # Calculate expected tempo (average of both teams)
        expected_tempo = (team1["AdjTempo"] + team2["AdjTempo"]) / 2

        # Estimate scores based on efficiency and tempo
        # Score = Efficiency × (Tempo / 100) × 100 possessions
        # Simplified: Score ≈ Efficiency × (Tempo / 100)
        team1_score = (team1["AdjOE"] / 100) * expected_tempo
        team2_score = (team2["AdjOE"] / 100) * expected_tempo

        # Adjust for defensive matchup
        # Team 1 faces Team 2's defense, Team 2 faces Team 1's defense
        team1_score_adjusted = team1_score * (100 / team2["AdjDE"])
        team2_score_adjusted = team2_score * (100 / team1["AdjDE"])

        # Apply home court (roughly +1.75 points to score, -1.75 from opponent)
        if home_team == "team1":
            team1_score_adjusted += self.HOME_COURT_ADVANTAGE / 2
            team2_score_adjusted -= self.HOME_COURT_ADVANTAGE / 2
        elif home_team == "team2":
            team2_score_adjusted += self.HOME_COURT_ADVANTAGE / 2
            team1_score_adjusted -= self.HOME_COURT_ADVANTAGE / 2

        predicted_total = team1_score_adjusted + team2_score_adjusted

        # Add to prediction
        prediction.update(
            {
                "team1_score": team1_score_adjusted,
                "team2_score": team2_score_adjusted,
                "predicted_total": predicted_total,
                "expected_tempo": expected_tempo,
            }
        )

        return prediction

    def print_prediction(self, prediction: dict, detailed: bool = False):
        """Print formatted prediction."""
        team1 = prediction["team1"]
        team2 = prediction["team2"]

        print(f"\n{'='*80}")
        print(f"GAME PREDICTION - {self.season} Season")
        print(f"{'='*80}\n")

        print(f"{team1['TeamName']} vs {team2['TeamName']}")
        print(f"Location: {prediction['location']}\n")

        print(f"{'TEAM STATS':-^80}")
        print(
            f"\n{team1['TeamName']:40s} | {team2['TeamName']:37s}"
        )
        print(f"{'-'*80}")
        print(
            f"AdjEM: {team1['AdjEM']:+6.2f} (#{team1['RankAdjEM']:<3.0f})           "
            f"| AdjEM: {team2['AdjEM']:+6.2f} (#{team2['RankAdjEM']:<3.0f})"
        )
        print(
            f"AdjO:  {team1['AdjOE']:6.2f} (#{team1['RankAdjOE']:<3.0f})           "
            f"| AdjO:  {team2['AdjOE']:6.2f} (#{team2['RankAdjOE']:<3.0f})"
        )
        print(
            f"AdjD:  {team1['AdjDE']:6.2f} (#{team1['RankAdjDE']:<3.0f})           "
            f"| AdjD:  {team2['AdjDE']:6.2f} (#{team2['RankAdjDE']:<3.0f})"
        )
        print(
            f"Tempo: {team1['AdjTempo']:6.2f} (#{team1['RankAdjTempo']:<3.0f})           "
            f"| Tempo: {team2['AdjTempo']:6.2f} (#{team2['RankAdjTempo']:<3.0f})"
        )
        print(
            f"Record: {team1['Wins']:.0f}-{team1['Losses']:.0f}                               "
            f"| Record: {team2['Wins']:.0f}-{team2['Losses']:.0f}"
        )

        print(f"\n{'PREDICTION':-^80}\n")
        print(f"Winner: {prediction['predicted_winner']}")
        print(f"Margin: {prediction['predicted_margin']:.1f} points")
        print(f"Confidence: {prediction['confidence']} ({prediction['confidence_pct']:.1%})")

        if "predicted_total" in prediction:
            print(f"\nExpected Score: {prediction['team1_score']:.1f} - {prediction['team2_score']:.1f}")
            print(f"Predicted Total: {prediction['predicted_total']:.1f} points")
            print(f"Expected Tempo: {prediction['expected_tempo']:.1f} possessions/game")

        if detailed:
            print(f"\n{'DETAILS':-^80}\n")
            print(f"Efficiency Margin Difference: {prediction['raw_margin']:+.2f}")
            print(
                f"Offensive Advantage: {team1['TeamName'] if team1['AdjOE'] > team2['AdjOE'] else team2['TeamName']} "
                f"({abs(team1['AdjOE'] - team2['AdjOE']):.2f} pts/100 poss)"
            )
            print(
                f"Defensive Advantage: {team1['TeamName'] if team1['AdjDE'] < team2['AdjDE'] else team2['TeamName']} "
                f"({abs(team1['AdjDE'] - team2['AdjDE']):.2f} pts allowed/100 poss)"
            )
            print(
                f"Pace Control: {team1['TeamName'] if team1['AdjTempo'] > team2['AdjTempo'] else team2['TeamName']} "
                f"({abs(team1['AdjTempo'] - team2['AdjTempo']):.2f} poss/game)"
            )

        print(f"\n{'='*80}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Predict college basketball games using KenPom data"
    )
    parser.add_argument("team1", nargs="?", help="First team name")
    parser.add_argument("team2", nargs="?", help="Second team name")
    parser.add_argument("--season", type=int, default=2025, help="Season year (default: 2025)")
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Game at neutral site (default: neutral)",
    )
    parser.add_argument(
        "--home",
        choices=["team1", "team2"],
        help="Which team is home (overrides --neutral)",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed breakdown"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple prediction (no tempo/scoring)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.team1 or not args.team2:
        parser.error("Both team1 and team2 are required")

    # Create predictor
    predictor = GamePredictor(season=args.season)

    try:
        # Generate prediction
        if args.simple:
            prediction = predictor.predict_simple(
                args.team1, args.team2, neutral_site=args.neutral, home_team=args.home
            )
        else:
            prediction = predictor.predict_with_tempo(
                args.team1, args.team2, neutral_site=args.neutral, home_team=args.home
            )

        # Print result
        predictor.print_prediction(prediction, detailed=args.detailed)

    except ValueError as e:
        print(f"\nError: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
