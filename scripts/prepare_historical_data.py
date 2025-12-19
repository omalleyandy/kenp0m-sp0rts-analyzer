"""Prepare historical game data for backtesting.

This script fetches historical NCAA basketball games from KenPom and
creates a feature-engineered dataset for model training and backtesting.

Usage:
    python scripts/prepare_historical_data.py --season 2024
    python scripts/prepare_historical_data.py --season 2024 --output data/games_2024.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.prediction import FeatureEngineer


def fetch_historical_games(api: KenPomAPI, season: int) -> pd.DataFrame:
    """Fetch historical games for a season.

    Args:
        api: KenPom API client
        season: Season year (e.g., 2024)

    Returns:
        DataFrame with game results and team stats

    Note:
        This is a placeholder implementation. You'll need to:
        1. Use api.get_fanmatch() to get scheduled games
        2. Match with actual game results
        3. Or parse KenPom game data directly
    """
    print(f"\nFetching games for {season} season...")

    # TODO: Implement actual data fetching
    # Option 1: Use FanMatch endpoint for predictions + actual results
    # Option 2: Scrape KenPom game results page
    # Option 3: Use external data source (e.g., ESPN API)

    # Placeholder - return empty DataFrame with correct structure
    raise NotImplementedError(
        "Historical game fetching not yet implemented.\n\n"
        "To implement:\n"
        "1. Fetch game results from KenPom or external source\n"
        "2. For each game, get team ratings at time of game\n"
        "3. Calculate actual_margin and actual_total\n"
        "4. Return DataFrame with columns:\n"
        "   - game_date, team1, team2, neutral_site, home_team1\n"
        "   - team1_stats (dict), team2_stats (dict)\n"
        "   - actual_margin, actual_total\n"
    )


def engineer_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to games DataFrame.

    Args:
        games_df: DataFrame with team stats and outcomes

    Returns:
        DataFrame with all features for ML training
    """
    print("\nEngineering features...")

    engineer = FeatureEngineer()
    features_list = []

    for idx, game in games_df.iterrows():
        # Create features for this game
        features = engineer.create_features(
            team1_stats=game["team1_stats"],
            team2_stats=game["team2_stats"],
            neutral_site=game.get("neutral_site", True),
            home_team1=game.get("home_team1", False),
        )

        # Add outcomes
        features["actual_margin"] = game["actual_margin"]
        features["actual_total"] = game["actual_total"]

        # Add metadata
        features["game_date"] = game.get("game_date")
        features["team1"] = game.get("team1")
        features["team2"] = game.get("team2")

        features_list.append(features)

    result_df = pd.DataFrame(features_list)

    print(f"Created {len(FeatureEngineer.FEATURE_NAMES)} features for {len(result_df)} games")

    return result_df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset has required columns and data quality.

    Args:
        df: Dataset to validate

    Raises:
        ValueError: If dataset is invalid
    """
    print("\nValidating dataset...")

    # Check required columns
    required_cols = set(FeatureEngineer.FEATURE_NAMES) | {
        "actual_margin",
        "actual_total",
    }
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for missing values
    missing_values = df[list(required_cols)].isnull().sum()
    if missing_values.any():
        print("\nWarning: Missing values detected:")
        print(missing_values[missing_values > 0])

    # Check data ranges
    if df["actual_margin"].abs().max() > 100:
        print("\nWarning: Extreme margin values detected (>100 points)")

    if df["actual_total"].min() < 50 or df["actual_total"].max() > 250:
        print("\nWarning: Unusual total scores detected")

    print(f"\nDataset validated: {len(df)} games")
    print(f"  Margins: {df['actual_margin'].min():.1f} to {df['actual_margin'].max():.1f}")
    print(f"  Totals:  {df['actual_total'].min():.1f} to {df['actual_total'].max():.1f}")


def create_sample_dataset(season: int) -> pd.DataFrame:
    """Create sample dataset for testing (placeholder).

    Args:
        season: Season year

    Returns:
        Sample DataFrame with synthetic data

    Note:
        This is a temporary helper for testing the backtest script.
        Replace with actual historical data when available.
    """
    print("\n[WARNING] Creating SYNTHETIC sample data for testing")
    print("This is NOT real data. Replace with actual KenPom data for production use.\n")

    import numpy as np

    np.random.seed(42)
    n_games = 1000

    # Generate synthetic features
    data = {
        "em_diff": np.random.normal(0, 10, n_games),
        "tempo_avg": np.random.normal(70, 5, n_games),
        "tempo_diff": np.random.normal(0, 3, n_games),
        "oe_diff": np.random.normal(0, 8, n_games),
        "de_diff": np.random.normal(0, 8, n_games),
        "pythag_diff": np.random.normal(0, 0.2, n_games),
        "sos_diff": np.random.normal(0, 3, n_games),
        "home_advantage": np.random.choice([0, 1, -1], n_games),
        "em_tempo_interaction": np.random.normal(0, 100, n_games),
        "apl_off_diff": np.random.normal(0, 2, n_games),
        "apl_def_diff": np.random.normal(0, 2, n_games),
        "apl_off_mismatch_team1": np.random.normal(0, 1.5, n_games),
        "apl_off_mismatch_team2": np.random.normal(0, 1.5, n_games),
        "tempo_control_factor": np.random.normal(0, 0.5, n_games),
    }

    df = pd.DataFrame(data)

    # Generate synthetic outcomes (correlated with em_diff)
    df["actual_margin"] = (
        df["em_diff"] * 0.8
        + df["home_advantage"] * 3
        + np.random.normal(0, 8, n_games)
    )

    df["actual_total"] = (
        df["tempo_avg"] * 2
        + 140
        + np.random.normal(0, 10, n_games)
    )

    return df


def main():
    """Prepare historical data for backtesting."""
    parser = argparse.ArgumentParser(description="Prepare historical game data")
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: data/historical_games_{season}.parquet)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create synthetic sample data for testing",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/historical_games_{args.season}.parquet")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPARE HISTORICAL DATA")
    print("=" * 60)
    print(f"\nSeason: {args.season}")
    print(f"Output: {output_path}")

    try:
        if args.sample:
            # Create synthetic sample data
            df = create_sample_dataset(args.season)

        else:
            # Fetch real data
            api = KenPomAPI()
            games_df = fetch_historical_games(api, args.season)
            df = engineer_features(games_df)

        # Validate
        validate_dataset(df)

        # Save
        print(f"\nSaving to {output_path}...")
        df.to_parquet(output_path, index=False)

        print("\n" + "=" * 60)
        print("DATASET CREATED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nFile: {output_path}")
        print(f"Games: {len(df)}")
        print(f"Features: {len(FeatureEngineer.FEATURE_NAMES)}")

        print("\nNext steps:")
        print("  1. Run comparison: python scripts/compare_sklearn_vs_xgboost.py")
        print(f"  2. View results for {args.season} season backtest")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
