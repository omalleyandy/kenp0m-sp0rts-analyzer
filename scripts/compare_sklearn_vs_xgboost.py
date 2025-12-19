"""Compare sklearn GradientBoosting vs XGBoost performance.

This script runs backtests on both prediction models to measure:
- Accuracy improvement
- MAE/RMSE reduction
- Training time differences
- Feature importance insights (XGBoost only)

Usage:
    python scripts/compare_sklearn_vs_xgboost.py
    python scripts/compare_sklearn_vs_xgboost.py --season 2024 --verbose
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from kenp0m_sp0rts_analyzer.prediction import (
    BacktestingFramework,
    FeatureEngineer,
    GamePredictor,
    XGBoostGamePredictor,
)


def load_historical_games(season: int = 2024) -> pd.DataFrame:
    """Load historical game data for backtesting.

    Args:
        season: Season year to load (default 2024)

    Returns:
        DataFrame with features and actual outcomes

    Note:
        This function expects a parquet file with historical game data.
        File should contain:
        - All feature columns from FeatureEngineer.FEATURE_NAMES
        - actual_margin: Actual point margin (team1 - team2)
        - actual_total: Actual total points (team1 + team2)

        To create this file, use the KenPom API to fetch historical games
        and calculate features using FeatureEngineer.
    """
    data_path = Path(f"data/historical_games_{season}.parquet")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Historical data not found: {data_path}\n\n"
            "To create historical game data:\n"
            "1. Use KenPom API to fetch game results for the season\n"
            "2. Calculate features using FeatureEngineer.create_features()\n"
            "3. Add actual_margin and actual_total columns\n"
            "4. Save as parquet: df.to_parquet(data_path)\n"
        )

    df = pd.read_parquet(data_path)

    # Validate columns
    required_cols = set(FeatureEngineer.FEATURE_NAMES) | {
        "actual_margin",
        "actual_total",
    }
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Historical data missing required columns: {missing_cols}\n"
            f"Expected columns: {required_cols}"
        )

    return df


def run_sklearn_backtest(games_df: pd.DataFrame) -> dict:
    """Run backtest using sklearn GradientBoosting.

    Args:
        games_df: Historical games with features and outcomes

    Returns:
        Dictionary with metrics and timing
    """
    print("\n" + "=" * 60)
    print("SKLEARN GRADIENTBOOSTING BACKTEST")
    print("=" * 60)

    framework = BacktestingFramework()

    # Time the backtest
    start_time = time.time()
    metrics = framework.run_backtest(games_df, train_split=0.8)
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\nTraining Time: {elapsed_time:.2f} seconds")
    print(f"\nPerformance Metrics:")
    print(f"  MAE Margin:  {metrics.mae_margin:.2f} points")
    print(f"  RMSE Margin: {metrics.rmse_margin:.2f} points")
    print(f"  MAE Total:   {metrics.mae_total:.2f} points")
    print(f"  RMSE Total:  {metrics.rmse_total:.2f} points")
    print(f"  Accuracy:    {metrics.accuracy:.1%}")
    print(f"  Brier Score: {metrics.brier_score:.3f}")
    print(f"  R² Margin:   {metrics.r2_margin:.3f}")

    if metrics.ats_record != (0, 0):
        wins, losses = metrics.ats_record
        print(f"  ATS Record:  {wins}-{losses} ({metrics.ats_percentage:.1%})")

    return {
        "metrics": metrics,
        "elapsed_time": elapsed_time,
    }


def run_xgboost_backtest(games_df: pd.DataFrame, verbose: bool = False) -> dict:
    """Run backtest using XGBoost.

    Args:
        games_df: Historical games with features and outcomes
        verbose: Print training progress

    Returns:
        Dictionary with metrics, timing, and feature importance
    """
    print("\n" + "=" * 60)
    print("XGBOOST BACKTEST")
    print("=" * 60)

    # Split data
    split_idx = int(len(games_df) * 0.8)
    train_df = games_df.iloc[:split_idx].copy()
    test_df = games_df.iloc[split_idx:].copy()

    # Create predictor
    predictor = XGBoostGamePredictor()

    # Time the training
    start_time = time.time()

    # Train predictor (Phase 1: no early stopping yet)
    predictor.fit(
        games_df=train_df[FeatureEngineer.FEATURE_NAMES],
        margins=train_df["actual_margin"],
        totals=train_df["actual_total"],
    )

    training_time = time.time() - start_time

    # Make predictions
    pred_margins = predictor.margin_model.predict(
        test_df[FeatureEngineer.FEATURE_NAMES]
    )
    pred_totals = predictor.total_model.predict(
        test_df[FeatureEngineer.FEATURE_NAMES]
    )
    pred_upper = predictor.margin_upper.predict(
        test_df[FeatureEngineer.FEATURE_NAMES]
    )
    pred_lower = predictor.margin_lower.predict(
        test_df[FeatureEngineer.FEATURE_NAMES]
    )

    # Calculate win probabilities
    import numpy as np

    margin_stds = np.maximum((pred_upper - pred_lower) / 1.35, 1.0)
    z_scores = pred_margins / margin_stds
    win_probs = 0.5 * (1.0 + np.tanh(z_scores / 2))

    # Calculate metrics
    framework = BacktestingFramework()
    metrics = framework._calculate_metrics(
        pred_margins=pred_margins,
        pred_totals=pred_totals,
        win_probs=win_probs,
        test_df=test_df,
    )

    # Get feature importance
    importance_gain = predictor.get_feature_importance(importance_type="gain", top_n=15)
    importance_weight = predictor.get_feature_importance(
        importance_type="weight", top_n=15
    )

    # Print results
    print(f"\nTraining Time: {training_time:.2f} seconds")
    print(f"\nPerformance Metrics:")
    print(f"  MAE Margin:  {metrics.mae_margin:.2f} points")
    print(f"  RMSE Margin: {metrics.rmse_margin:.2f} points")
    print(f"  MAE Total:   {metrics.mae_total:.2f} points")
    print(f"  RMSE Total:  {metrics.rmse_total:.2f} points")
    print(f"  Accuracy:    {metrics.accuracy:.1%}")
    print(f"  Brier Score: {metrics.brier_score:.3f}")
    print(f"  R² Margin:   {metrics.r2_margin:.3f}")

    if metrics.ats_record != (0, 0):
        wins, losses = metrics.ats_record
        print(f"  ATS Record:  {wins}-{losses} ({metrics.ats_percentage:.1%})")

    # Print feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (GAIN)")
    print("=" * 60)
    print("\nTop 10 Features by Gain (loss reduction):")
    for idx, row in importance_gain.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:8.1f}")

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (WEIGHT)")
    print("=" * 60)
    print("\nTop 10 Features by Weight (split frequency):")
    for idx, row in importance_weight.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:8.0f}")

    return {
        "metrics": metrics,
        "training_time": training_time,
        "importance_gain": importance_gain,
        "importance_weight": importance_weight,
    }


def compare_results(sklearn_results: dict, xgboost_results: dict) -> None:
    """Print comparison summary.

    Args:
        sklearn_results: Results from sklearn backtest
        xgboost_results: Results from XGBoost backtest
    """
    sklearn_metrics = sklearn_results["metrics"]
    xgboost_metrics = xgboost_results["metrics"]

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: SKLEARN VS XGBOOST")
    print("=" * 60)

    # Accuracy improvement
    accuracy_improvement = (
        xgboost_metrics.accuracy - sklearn_metrics.accuracy
    ) * 100
    print(f"\nAccuracy Improvement: {accuracy_improvement:+.1f}%")
    print(
        f"  sklearn:  {sklearn_metrics.accuracy:.1%} "
        f"-> XGBoost: {xgboost_metrics.accuracy:.1%}"
    )

    # MAE improvement
    mae_improvement = sklearn_metrics.mae_margin - xgboost_metrics.mae_margin
    print(f"\nMAE Improvement: {mae_improvement:+.2f} points")
    print(
        f"  sklearn:  {sklearn_metrics.mae_margin:.2f} pts "
        f"-> XGBoost: {xgboost_metrics.mae_margin:.2f} pts"
    )

    # RMSE improvement
    rmse_improvement = sklearn_metrics.rmse_margin - xgboost_metrics.rmse_margin
    print(f"\nRMSE Improvement: {rmse_improvement:+.2f} points")
    print(
        f"  sklearn:  {sklearn_metrics.rmse_margin:.2f} pts "
        f"-> XGBoost: {xgboost_metrics.rmse_margin:.2f} pts"
    )

    # Training time
    time_ratio = sklearn_results["elapsed_time"] / xgboost_results["training_time"]
    print(f"\nTraining Speed:")
    print(f"  sklearn:  {sklearn_results['elapsed_time']:.2f}s")
    print(f"  XGBoost:  {xgboost_results['training_time']:.2f}s")
    print(f"  Speedup:  {time_ratio:.1f}x {'faster' if time_ratio > 1 else 'slower'}")

    # Overall verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    improvements = []
    if accuracy_improvement > 0:
        improvements.append(f"+{accuracy_improvement:.1f}% accuracy")
    if mae_improvement > 0:
        improvements.append(f"-{mae_improvement:.2f} MAE")
    if rmse_improvement > 0:
        improvements.append(f"-{rmse_improvement:.2f} RMSE")
    if time_ratio > 1:
        improvements.append(f"{time_ratio:.1f}x faster training")

    if improvements:
        print("\nXGBoost Advantages:")
        for imp in improvements:
            print(f"  [OK] {imp}")

    print("\nAdditional XGBoost Benefits:")
    print("  [OK] Feature importance (gain, weight, cover)")
    print("  [OK] Better regularization (L1/L2)")
    print("  [OK] Early stopping support")
    print("  [OK] Native handling of missing values")

    # Betting edge opportunities
    print("\n" + "=" * 60)
    print("BETTING EDGE OPPORTUNITIES")
    print("=" * 60)
    print("\nUse XGBoost feature importance to identify high-value bets:")

    importance = xgboost_results["importance_gain"]

    # Example edge detection
    top_features = importance.head(5)
    print(f"\nTop 5 Most Important Features:")
    for idx, row in top_features.iterrows():
        feature = row["feature"]
        print(f"\n  {feature}:")

        # Provide betting insights based on feature name
        if "luck" in feature.lower():
            print("    -> EDGE: Fade lucky teams, back unlucky teams")
        elif "momentum" in feature.lower():
            print("    -> EDGE: Follow hot teams, fade cold teams")
        elif "pace" in feature.lower() or "tempo" in feature.lower():
            print("    -> EDGE: Target pace mismatch games")
        elif "em_diff" in feature.lower():
            print("    -> Core predictor: Efficiency margin differential")
        elif "pythag" in feature.lower():
            print("    -> EDGE: Target teams deviating from Pythagorean expectation")
        else:
            print(f"    -> Analyze {feature} for betting opportunities")


def main():
    """Run sklearn vs XGBoost comparison."""
    parser = argparse.ArgumentParser(description="Compare sklearn vs XGBoost")
    parser.add_argument(
        "--season", type=int, default=2024, help="Season year (default: 2024)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print training progress"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SKLEARN VS XGBOOST COMPARISON")
    print("=" * 60)
    print(f"\nSeason: {args.season}")

    try:
        # Load data
        print("\nLoading historical game data...")
        games_df = load_historical_games(season=args.season)
        print(f"Loaded {len(games_df)} games")
        print(
            f"Train set: {int(len(games_df) * 0.8)} games | "
            f"Test set: {len(games_df) - int(len(games_df) * 0.8)} games"
        )

        # Run backtests
        sklearn_results = run_sklearn_backtest(games_df)
        xgboost_results = run_xgboost_backtest(games_df, verbose=args.verbose)

        # Compare results
        compare_results(sklearn_results, xgboost_results)

        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo prepare historical data, run:")
        print("  python scripts/prepare_historical_data.py --season 2024")

    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
