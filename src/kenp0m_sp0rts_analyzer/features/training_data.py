"""Training data generation for XGBoost model.

This module provides tools to create training datasets from historical
KenPom ratings and optionally game results.

Training data sources:
1. Historical rating snapshots (for feature generation)
2. Game results (for targets) - when available
3. Simulated outcomes (for initial model development)
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import random

import numpy as np
import pandas as pd

from ..kenpom import KenPomService
from ..utils.logging import logger


@dataclass
class TrainingExample:
    """A single training example with features and target."""

    game_id: str
    game_date: str
    home_team: str
    away_team: str
    features: dict[str, float]
    actual_margin: float  # Home - Away (positive = home won)
    actual_total: int
    vegas_spread: float | None = None
    vegas_total: float | None = None


class HistoricalDataLoader:
    """Load and prepare historical training data.

    Usage:
        loader = HistoricalDataLoader(KenPomService())

        # Get training data from ratings (simulated outcomes)
        X, y_margin, y_total = loader.generate_training_data(
            start_date=date(2024, 11, 1),
            end_date=date(2025, 3, 1),
            n_samples=5000,
        )

        # If you have game results CSV
        X, y_margin, y_total = loader.load_from_csv(
            "historical_games.csv",
            start_date=date(2024, 11, 1),
        )
    """

    def __init__(self, kenpom_service: KenPomService):
        """Initialize with KenPom service.

        Args:
            kenpom_service: Initialized KenPomService.
        """
        self.service = kenpom_service
        self.repository = kenpom_service.repository
        # Home court advantage (points)
        self.hca = 3.75

    def get_team_ids(self) -> dict[str, int]:
        """Get mapping of team names to IDs."""
        teams = {}
        with self.repository.db.connection() as conn:
            rows = conn.execute(
                "SELECT team_id, team_name FROM teams"
            ).fetchall()
            for row in rows:
                teams[row["team_name"]] = row["team_id"]
        return teams

    def get_rating_snapshot(
        self,
        team_id: int,
        snapshot_date: date,
    ) -> dict[str, Any] | None:
        """Get team rating for a specific date.

        Args:
            team_id: Team ID.
            snapshot_date: Date to get ratings for.

        Returns:
            Dict of ratings or None if not found.
        """
        with self.repository.db.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM ratings
                WHERE team_id = ? AND snapshot_date <= ?
                ORDER BY snapshot_date DESC
                LIMIT 1
                """,
                (team_id, str(snapshot_date)),
            ).fetchone()

            if row:
                return {
                    "adj_em": row["adj_em"],
                    "adj_oe": row["adj_oe"],
                    "adj_de": row["adj_de"],
                    "adj_tempo": row["adj_tempo"],
                    "pythag": row["pythag"],
                    "luck": row["luck"],
                    "sos": row["sos"],
                }
        return None

    def calculate_expected_margin(
        self,
        home_rating: dict[str, Any],
        away_rating: dict[str, Any],
        neutral_site: bool = False,
    ) -> tuple[float, int]:
        """Calculate expected margin and total using KenPom formula.

        Args:
            home_rating: Home team ratings.
            away_rating: Away team ratings.
            neutral_site: Whether game is at neutral site.

        Returns:
            Tuple of (expected_margin, expected_total).
        """
        # Efficiency margin difference
        em_diff = home_rating["adj_em"] - away_rating["adj_em"]

        # Tempo adjustment
        avg_tempo = (home_rating["adj_tempo"] + away_rating["adj_tempo"]) / 2
        tempo_factor = avg_tempo / 67.5

        # Home court advantage
        hca = 0 if neutral_site else self.hca

        # Expected margin
        expected_margin = em_diff * tempo_factor + hca

        # Expected total
        avg_oe = (home_rating["adj_oe"] + away_rating["adj_oe"]) / 2
        avg_de = (home_rating["adj_de"] + away_rating["adj_de"]) / 2
        expected_ppp = (avg_oe + (200 - avg_de)) / 200
        expected_total = int(avg_tempo * 2 * expected_ppp)

        return expected_margin, expected_total

    def simulate_outcome(
        self,
        expected_margin: float,
        expected_total: int,
        noise_std: float = 11.0,
    ) -> tuple[float, int]:
        """Simulate a game outcome with realistic variance.

        Args:
            expected_margin: KenPom expected margin.
            expected_total: KenPom expected total.
            noise_std: Standard deviation of margin noise (default 11).

        Returns:
            Tuple of (actual_margin, actual_total).
        """
        # Add gaussian noise to margin (typical game variance ~11 pts)
        actual_margin = expected_margin + np.random.normal(0, noise_std)

        # Total has less variance
        total_noise = np.random.normal(0, 8)
        actual_total = max(80, int(expected_total + total_noise))

        return actual_margin, actual_total

    def generate_training_data(
        self,
        start_date: date,
        end_date: date,
        n_samples: int = 5000,
        include_neutrals: bool = True,
        seed: int | None = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Generate training data from rating snapshots.

        Creates matchups between randomly selected teams and simulates
        game outcomes based on rating differentials.

        Args:
            start_date: Start date for sampling.
            end_date: End date for sampling.
            n_samples: Number of training examples to generate.
            include_neutrals: Include neutral site games.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (X, y_margin, y_total, feature_names).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        logger.info(f"Generating {n_samples} training examples...")

        # Get all teams
        teams = self.get_team_ids()
        team_list = list(teams.items())

        if len(team_list) < 2:
            raise ValueError("Need at least 2 teams to generate matchups")

        # Get available dates
        with self.repository.db.connection() as conn:
            dates = conn.execute(
                """
                SELECT DISTINCT snapshot_date FROM ratings
                WHERE snapshot_date BETWEEN ? AND ?
                ORDER BY snapshot_date
                """,
                (str(start_date), str(end_date)),
            ).fetchall()
            available_dates = []
            for row in dates:
                d = row[0]
                if isinstance(d, str):
                    d = datetime.strptime(d, "%Y-%m-%d").date()
                available_dates.append(d)

        if not available_dates:
            raise ValueError(
                f"No ratings found between {start_date} and {end_date}"
            )

        logger.info(
            f"Found {len(available_dates)} dates with ratings, "
            f"{len(team_list)} teams"
        )

        # Import feature engineer here to avoid circular import
        from .feature_engineer import DatabaseFeatureEngineer

        engineer = DatabaseFeatureEngineer(self.service)

        examples = []
        attempts = 0
        max_attempts = n_samples * 3

        while len(examples) < n_samples and attempts < max_attempts:
            attempts += 1

            try:
                # Random date
                game_date = random.choice(available_dates)

                # Random teams (ensure different teams)
                team1_name, team1_id = random.choice(team_list)
                team2_name, team2_id = random.choice(team_list)
                if team1_id == team2_id:
                    continue

                # Random home/away or neutral
                neutral = include_neutrals and random.random() < 0.15
                if random.random() < 0.5:
                    home_name, home_id = team1_name, team1_id
                    away_name, away_id = team2_name, team2_id
                else:
                    home_name, home_id = team2_name, team2_id
                    away_name, away_id = team1_name, team1_id

                # Get ratings
                home_rating = self.get_rating_snapshot(home_id, game_date)
                away_rating = self.get_rating_snapshot(away_id, game_date)

                if not home_rating or not away_rating:
                    continue

                # Calculate expected outcome
                exp_margin, exp_total = self.calculate_expected_margin(
                    home_rating, away_rating, neutral
                )

                # Simulate actual outcome
                actual_margin, actual_total = self.simulate_outcome(
                    exp_margin, exp_total
                )

                # Generate mock Vegas line (close to expected)
                spread_noise = random.uniform(-1, 1)
                vegas_spread = round(exp_margin * 2) / 2 + spread_noise
                vegas_total = exp_total + random.uniform(-3, 3)

                # Create features
                features = engineer.create_features(
                    home_team=home_name,
                    away_team=away_name,
                    vegas_spread=-vegas_spread,  # Negative = home favored
                    vegas_total=vegas_total,
                    game_date=game_date,
                )

                examples.append(TrainingExample(
                    game_id=f"{game_date}_{away_name}@{home_name}",
                    game_date=str(game_date),
                    home_team=home_name,
                    away_team=away_name,
                    features=features.features,
                    actual_margin=actual_margin,
                    actual_total=actual_total,
                    vegas_spread=-vegas_spread,
                    vegas_total=vegas_total,
                ))

            except Exception as e:
                logger.debug(f"Skipped example: {e}")
                continue

        logger.info(f"Generated {len(examples)} training examples")

        if len(examples) < 100:
            raise ValueError(
                f"Only generated {len(examples)} examples, need at least 100"
            )

        # Convert to arrays
        feature_names = list(examples[0].features.keys())
        X = np.array([
            [ex.features[f] for f in feature_names]
            for ex in examples
        ])
        y_margin = np.array([ex.actual_margin for ex in examples])
        y_total = np.array([ex.actual_total for ex in examples])

        return X, y_margin, y_total, feature_names

    def load_from_csv(
        self,
        csv_path: str,
        date_col: str = "game_date",
        home_col: str = "home_team",
        away_col: str = "away_team",
        margin_col: str = "margin",
        total_col: str = "total",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Load training data from a CSV with game results.

        Args:
            csv_path: Path to CSV file.
            date_col: Column name for game date.
            home_col: Column name for home team.
            away_col: Column name for away team.
            margin_col: Column name for margin (home - away).
            total_col: Column name for total points.

        Returns:
            Tuple of (X, y_margin, y_total, feature_names).
        """
        logger.info(f"Loading training data from {csv_path}")

        df = pd.read_csv(csv_path)
        df[date_col] = pd.to_datetime(df[date_col]).dt.date

        # Import feature engineer
        from .feature_engineer import DatabaseFeatureEngineer

        engineer = DatabaseFeatureEngineer(self.service)

        examples = []
        for _, row in df.iterrows():
            try:
                game_date = row[date_col]

                # Default Vegas lines if not in CSV
                vegas_spread = row.get("vegas_spread", -3.0)
                vegas_total = row.get("vegas_total", 145.0)

                features = engineer.create_features(
                    home_team=row[home_col],
                    away_team=row[away_col],
                    vegas_spread=vegas_spread,
                    vegas_total=vegas_total,
                    game_date=game_date,
                )

                examples.append(TrainingExample(
                    game_id=f"{game_date}_{row[away_col]}@{row[home_col]}",
                    game_date=str(game_date),
                    home_team=row[home_col],
                    away_team=row[away_col],
                    features=features.features,
                    actual_margin=float(row[margin_col]),
                    actual_total=int(row[total_col]),
                    vegas_spread=vegas_spread,
                    vegas_total=vegas_total,
                ))
            except Exception as e:
                logger.warning(f"Skipped row: {e}")
                continue

        logger.info(f"Loaded {len(examples)} examples from CSV")

        if not examples:
            raise ValueError("No valid examples loaded from CSV")

        # Convert to arrays
        feature_names = list(examples[0].features.keys())
        X = np.array([
            [ex.features[f] for f in feature_names]
            for ex in examples
        ])
        y_margin = np.array([ex.actual_margin for ex in examples])
        y_total = np.array([ex.actual_total for ex in examples])

        return X, y_margin, y_total, feature_names
