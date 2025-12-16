"""Validate the 2-3% improvement claim for tempo/pace features.

This script compares prediction performance with and without APL (Average Possession
Length) tempo features to validate the claimed 2-3% accuracy improvement.

Usage:
    # With synthetic data (for testing):
    python scripts/validate_tempo_features.py --synthetic

    # With real KenPom data (requires API key):
    python scripts/validate_tempo_features.py --year 2024 --games-csv games.csv

    # Custom train/test split:
    python scripts/validate_tempo_features.py --train-split 0.8

Requirements:
    - Historical game data with KenPom stats (AdjEM, AdjO, AdjD, AdjT, APL_Off, APL_Def)
    - Actual game results (margins and totals)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.prediction import (
    BacktestMetrics,
    BacktestingFramework,
    FeatureEngineer,
    GamePredictor,
)


def generate_synthetic_data(n_games: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic game data for testing the validation framework.

    Args:
        n_games: Number of games to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic game features and outcomes
    """
    np.random.seed(seed)

    # Generate team statistics (realistic ranges)
    data = []
    for _ in range(n_games):
        # Team 1 stats
        team1_em = np.random.normal(15.0, 10.0)  # Efficiency margin
        team1_tempo = np.random.normal(68.0, 4.0)  # Tempo (possessions)
        team1_oe = np.random.normal(110.0, 8.0)  # Offensive efficiency
        team1_de = 100.0 + (team1_em - (team1_oe - 110.0))  # Defensive efficiency
        team1_apl_off = np.random.normal(17.5, 1.5)  # APL offense
        team1_apl_def = np.random.normal(17.5, 1.5)  # APL defense

        # Team 2 stats
        team2_em = np.random.normal(10.0, 10.0)
        team2_tempo = np.random.normal(68.0, 4.0)
        team2_oe = np.random.normal(110.0, 8.0)
        team2_de = 100.0 + (team2_em - (team2_oe - 110.0))
        team2_apl_off = np.random.normal(17.5, 1.5)
        team2_apl_def = np.random.normal(17.5, 1.5)

        # Neutral site (50% of games)
        neutral_site = np.random.rand() < 0.5

        # Create feature vector
        team1_stats = {
            "AdjEM": team1_em,
            "AdjT": team1_tempo,
            "AdjO": team1_oe,
            "AdjD": team1_de,
            "AdjTempo": team1_tempo,
            "AdjOE": team1_oe,
            "AdjDE": team1_de,
            "Pythag": 0.5 + (team1_em / 40.0),
            "SOS": np.random.normal(5.0, 3.0),
            "APL_Off": team1_apl_off,
            "APL_Def": team1_apl_def,
        }

        team2_stats = {
            "AdjEM": team2_em,
            "AdjT": team2_tempo,
            "AdjO": team2_oe,
            "AdjD": team2_de,
            "AdjTempo": team2_tempo,
            "AdjOE": team2_oe,
            "AdjDE": team2_de,
            "Pythag": 0.5 + (team2_em / 40.0),
            "SOS": np.random.normal(5.0, 3.0),
            "APL_Off": team2_apl_off,
            "APL_Def": team2_apl_def,
        }

        features = FeatureEngineer.create_features(
            team1_stats, team2_stats, neutral_site
        )

        # Generate actual outcome (margin and total)
        # Base prediction on efficiency margin
        base_margin = (team1_em - team2_em) * 0.8
        avg_tempo = (team1_tempo + team2_tempo) / 2

        # Add tempo effect (simplified)
        tempo_effect = (avg_tempo - 68.0) * (team1_em - team2_em) / 100.0

        # Add noise
        noise = np.random.normal(0, 8.0)

        actual_margin = base_margin + tempo_effect + noise
        actual_total = avg_tempo * 1.5 + np.random.normal(0, 6.0)

        # Add features and outcomes
        game = features.copy()
        game["actual_margin"] = actual_margin
        game["actual_total"] = actual_total

        data.append(game)

    return pd.DataFrame(data)


def remove_apl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a copy of dataset without APL features (for baseline comparison).

    Args:
        df: DataFrame with all features

    Returns:
        DataFrame with APL features set to defaults (neutral impact)
    """
    df_no_apl = df.copy()

    # Set APL features to neutral values (no impact)
    df_no_apl["apl_off_diff"] = 0.0
    df_no_apl["apl_def_diff"] = 0.0
    df_no_apl["apl_off_mismatch_team1"] = 0.0
    df_no_apl["apl_off_mismatch_team2"] = 0.0
    df_no_apl["tempo_control_factor"] = 0.0

    return df_no_apl


def compare_models(
    df_with_apl: pd.DataFrame,
    df_without_apl: pd.DataFrame,
    train_split: float = 0.8,
) -> tuple[BacktestMetrics, BacktestMetrics]:
    """Train and compare models with and without APL features.

    Args:
        df_with_apl: Full feature dataset (including APL)
        df_without_apl: Baseline dataset (without APL)
        train_split: Fraction for training (default 0.8)

    Returns:
        Tuple of (metrics_with_apl, metrics_without_apl)
    """
    framework = BacktestingFramework()

    print("\n" + "=" * 70)
    print("Training model WITH APL tempo features...")
    print("=" * 70)
    metrics_with = framework.run_backtest(df_with_apl, train_split=train_split)

    print("\n" + "=" * 70)
    print("Training model WITHOUT APL tempo features (baseline)...")
    print("=" * 70)
    metrics_without = framework.run_backtest(df_without_apl, train_split=train_split)

    return metrics_with, metrics_without


def print_comparison(
    metrics_with: BacktestMetrics, metrics_without: BacktestMetrics
) -> None:
    """Print detailed comparison of model performance.

    Args:
        metrics_with: Metrics with APL features
        metrics_without: Metrics without APL features (baseline)
    """
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS: APL Tempo Features Impact")
    print("=" * 70)

    # Margin prediction comparison
    print("\n[MARGIN PREDICTION]")
    print(f"  Without APL (Baseline):  MAE = {metrics_without.mae_margin:.2f} points")
    print(f"  With APL (Enhanced):     MAE = {metrics_with.mae_margin:.2f} points")

    mae_improvement = (
        (metrics_without.mae_margin - metrics_with.mae_margin)
        / metrics_without.mae_margin
        * 100
    )
    print(f"  -> Improvement:           {mae_improvement:+.2f}%")

    print(f"\n  Without APL (Baseline):  RMSE = {metrics_without.rmse_margin:.2f}")
    print(f"  With APL (Enhanced):     RMSE = {metrics_with.rmse_margin:.2f}")

    rmse_improvement = (
        (metrics_without.rmse_margin - metrics_with.rmse_margin)
        / metrics_without.rmse_margin
        * 100
    )
    print(f"  -> Improvement:           {rmse_improvement:+.2f}%")

    print(f"\n  Without APL (Baseline):  R^2 = {metrics_without.r2_margin:.3f}")
    print(f"  With APL (Enhanced):     R^2 = {metrics_with.r2_margin:.3f}")

    # Winner prediction comparison
    print("\n[WINNER PREDICTION]")
    print(
        f"  Without APL (Baseline):  Accuracy = {metrics_without.accuracy * 100:.1f}%"
    )
    print(f"  With APL (Enhanced):     Accuracy = {metrics_with.accuracy * 100:.1f}%")

    acc_improvement = (metrics_with.accuracy - metrics_without.accuracy) * 100
    print(f"  -> Improvement:           {acc_improvement:+.1f} percentage points")

    # Probability calibration comparison
    print("\n[PROBABILITY CALIBRATION]")
    print(f"  Without APL (Baseline):  Brier = {metrics_without.brier_score:.3f}")
    print(f"  With APL (Enhanced):     Brier = {metrics_with.brier_score:.3f}")

    brier_improvement = (
        (metrics_without.brier_score - metrics_with.brier_score)
        / metrics_without.brier_score
        * 100
    )
    print(
        f"  -> Improvement:           {brier_improvement:+.2f}% (lower Brier is better)"
    )

    # Total prediction comparison
    print("\n[TOTAL SCORE PREDICTION]")
    print(f"  Without APL (Baseline):  MAE = {metrics_without.mae_total:.2f}")
    print(f"  With APL (Enhanced):     MAE = {metrics_with.mae_total:.2f}")

    total_improvement = (
        (metrics_without.mae_total - metrics_with.mae_total)
        / metrics_without.mae_total
        * 100
    )
    print(f"  -> Improvement:           {total_improvement:+.2f}%")

    # Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)

    improvements = [mae_improvement, rmse_improvement, acc_improvement * 10]
    avg_improvement = np.mean(improvements)

    if avg_improvement >= 2.0:
        verdict = "[VALIDATED]"
        msg = f"APL tempo features improve prediction accuracy by {avg_improvement:.1f}%"
    elif avg_improvement >= 1.0:
        verdict = "[WARNING]"
        msg = (
            f"APL tempo features show {avg_improvement:.1f}% improvement "
            f"(below claimed 2-3%)"
        )
    else:
        verdict = "[NOT VALIDATED]"
        msg = f"APL tempo features show only {avg_improvement:.1f}% improvement"

    print(f"{verdict} {msg}")
    print("=" * 70)


def main() -> None:
    """Run tempo feature validation."""
    parser = argparse.ArgumentParser(
        description="Validate APL tempo features improve predictions by 2-3%"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (for testing framework)",
    )
    parser.add_argument(
        "--games-csv", type=str, help="Path to CSV file with historical game data"
    )
    parser.add_argument(
        "--year", type=int, help="Season year to fetch from KenPom API"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training split fraction (default: 0.8)",
    )
    parser.add_argument(
        "--n-games", type=int, default=500, help="Number of synthetic games to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for synthetic data"
    )

    args = parser.parse_args()

    # Load or generate data
    if args.synthetic:
        print(f"Generating {args.n_games} synthetic games (seed={args.seed})...")
        df = generate_synthetic_data(n_games=args.n_games, seed=args.seed)
        print(f"[OK] Generated {len(df)} games with APL features")
    elif args.games_csv:
        print(f"Loading games from {args.games_csv}...")
        df = pd.read_csv(args.games_csv)
        print(f"[OK] Loaded {len(df)} games")
    elif args.year:
        print(f"Fetching {args.year} season data from KenPom API...")
        print("[ERROR] Real data fetching not yet implemented")
        print("   Use --synthetic for testing or --games-csv to load existing data")
        sys.exit(1)
    else:
        print("[ERROR] Must specify --synthetic, --games-csv, or --year")
        parser.print_help()
        sys.exit(1)

    # Validate data has required columns
    required_cols = set(FeatureEngineer.FEATURE_NAMES) | {
        "actual_margin",
        "actual_total",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
        sys.exit(1)

    # Create baseline dataset without APL features
    print("\nPreparing datasets:")
    print(f"  - Full dataset: {len(df)} games with {len(df.columns)} features")
    df_no_apl = remove_apl_features(df)
    print(f"  - Baseline dataset: APL features neutralized")

    # Compare models
    metrics_with, metrics_without = compare_models(
        df_with_apl=df, df_without_apl=df_no_apl, train_split=args.train_split
    )

    # Print results
    print_comparison(metrics_with, metrics_without)


if __name__ == "__main__":
    main()
