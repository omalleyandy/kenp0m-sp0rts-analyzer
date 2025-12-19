"""Fetch historical data and train XGBoost model for game predictions.

This script:
1. Fetches historical ratings from KenPom archive
2. Creates synthetic game outcomes for training (placeholder for real results)
3. Trains XGBoost model with enhanced features
4. Saves model for use in live predictions

Usage:
    python scripts/fetch_and_train_xgboost.py --season 2025
    python scripts/fetch_and_train_xgboost.py --season 2025 --enhanced
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.prediction import (
    XGBoostGamePredictor,
    XGBoostFeatureEngineer,
    FeatureEngineer,
    BacktestingFramework,
)

# Load environment variables
load_dotenv()


def fetch_ratings_over_season(
    api: KenPomAPI, season: int
) -> dict[str, pd.DataFrame]:
    """Fetch team ratings at multiple points throughout the season.

    Args:
        api: KenPom API client
        season: Season year (e.g., 2025)

    Returns:
        Dictionary mapping date strings to DataFrames of ratings
    """
    print(f"\nFetching ratings throughout {season} season...")

    # Define key dates throughout the season
    # College basketball: November - April
    start_date = datetime(season - 1, 11, 15)  # Mid-November
    end_date = datetime(season, 3, 15)  # Mid-March (before tournament)

    ratings_by_date = {}

    # Sample ratings every 2 weeks
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")

        try:
            print(f"  Fetching {date_str}...", end=" ")
            response = api.get_archive(archive_date=date_str)
            df = pd.DataFrame(response.data)
            ratings_by_date[date_str] = df
            print(f"OK ({len(df)} teams)")

        except Exception as e:
            print(f"SKIP ({e})")

        # Move to next date (2 weeks)
        current_date += timedelta(days=14)

    print(f"\nFetched ratings from {len(ratings_by_date)} dates")
    return ratings_by_date


def generate_synthetic_games(
    ratings_df: pd.DataFrame, n_games: int = 100, season: int = 2025
) -> pd.DataFrame:
    """Generate synthetic game matchups with realistic outcomes.

    Args:
        ratings_df: DataFrame of team ratings
        n_games: Number of games to generate
        season: Season year

    Returns:
        DataFrame with game features and outcomes
    """
    print(f"\nGenerating {n_games} synthetic game matchups...")

    np.random.seed(42)
    games = []

    # Get teams with complete data
    teams_raw = ratings_df[
        ratings_df["AdjEM"].notna()
        & ratings_df["AdjOE"].notna()
        & ratings_df["AdjDE"].notna()
        & ratings_df["AdjTempo"].notna()
    ].to_dict("records")

    # Map API field names to feature engineer expected names
    teams = []
    for team in teams_raw:
        tempo_value = team.get("AdjTempo")
        mapped_team = {
            "TeamName": team.get("TeamName"),
            "TeamID": team.get("TeamID"),
            "AdjEM": team.get("AdjEM"),
            "AdjO": team.get("AdjOE"),  # API uses AdjOE
            "AdjD": team.get("AdjDE"),  # API uses AdjDE
            "AdjT": tempo_value,  # FeatureEngineer expects AdjT
            "AdjTempo": tempo_value,  # tempo_analysis expects AdjTempo
            "Luck": team.get("Luck", 0.0),
            "Pythag": team.get("Pythag", 0.5),
            "SOS": team.get("SOS", 5.0),
            "SOSO": team.get("SOSO", 5.0),
            "SOSD": team.get("SOSD", 5.0),
        }
        # Add APL fields if available
        if "APL_Off" in team:
            mapped_team["APL_Off"] = team["APL_Off"]
            mapped_team["APL_Def"] = team["APL_Def"]
        else:
            # Estimate from tempo (18 seconds per possession average)
            mapped_team["APL_Off"] = 18.0
            mapped_team["APL_Def"] = 18.0

        teams.append(mapped_team)

    if len(teams) < 10:
        raise ValueError(f"Not enough teams with complete data: {len(teams)}")

    engineer = FeatureEngineer()

    for _ in range(n_games):
        # Randomly select two teams
        team1, team2 = np.random.choice(teams, size=2, replace=False)

        # Random home court advantage
        neutral_site = np.random.choice([True, False], p=[0.3, 0.7])
        home_team1 = False if neutral_site else np.random.choice([True, False])

        # Create features
        features = engineer.create_features(
            team1_stats=team1,
            team2_stats=team2,
            neutral_site=neutral_site,
            home_team1=home_team1,
        )

        # Generate realistic outcome based on efficiency margin
        em_diff = features["em_diff"]
        home_adv = (
            features["home_advantage"] * 3.5
        )  # ~3.5 point home advantage
        tempo_factor = (
            features["tempo_avg"] / 70
        )  # Normalize around 70 possessions

        # Predicted margin (with noise)
        base_margin = em_diff + home_adv
        actual_margin = base_margin + np.random.normal(0, 10 * tempo_factor)

        # Predicted total (with noise)
        base_total = features["tempo_avg"] * 2 + 140
        actual_total = base_total + np.random.normal(0, 12)

        # Add outcomes to features
        features["actual_margin"] = actual_margin
        features["actual_total"] = actual_total

        # Add metadata
        features["team1"] = team1["TeamName"]
        features["team2"] = team2["TeamName"]
        features["season"] = season

        games.append(features)

    df = pd.DataFrame(games)
    print(f"Generated {len(df)} games with realistic outcomes")

    return df


def generate_enhanced_synthetic_games(
    ratings_df: pd.DataFrame, n_games: int = 100, season: int = 2025
) -> pd.DataFrame:
    """Generate synthetic games with enhanced Phase 2 features.

    Args:
        ratings_df: DataFrame of team ratings
        n_games: Number of games to generate
        season: Season year

    Returns:
        DataFrame with enhanced features and outcomes
    """
    print(f"\nGenerating {n_games} synthetic games with enhanced features...")

    np.random.seed(42)
    games = []

    # Get teams with complete data
    teams_raw = ratings_df[
        ratings_df["AdjEM"].notna()
        & ratings_df["AdjOE"].notna()
        & ratings_df["AdjDE"].notna()
        & ratings_df["AdjTempo"].notna()
    ].to_dict("records")

    # Map API field names to feature engineer expected names
    teams = []
    for team in teams_raw:
        tempo_value = team.get("AdjTempo")
        mapped_team = {
            "TeamName": team.get("TeamName"),
            "TeamID": team.get("TeamID"),
            "AdjEM": team.get("AdjEM"),
            "AdjO": team.get("AdjOE"),  # API uses AdjOE
            "AdjD": team.get("AdjDE"),  # API uses AdjDE
            "AdjT": tempo_value,  # FeatureEngineer expects AdjT
            "AdjTempo": tempo_value,  # tempo_analysis expects AdjTempo
            "Luck": team.get("Luck", 0.0),
            "Pythag": team.get("Pythag", 0.5),
            "SOS": team.get("SOS", 5.0),
            "SOSO": team.get("SOSO", 5.0),
            "SOSD": team.get("SOSD", 5.0),
        }
        # Add APL fields if available
        if "APL_Off" in team:
            mapped_team["APL_Off"] = team["APL_Off"]
            mapped_team["APL_Def"] = team["APL_Def"]
        else:
            # Estimate from tempo (18 seconds per possession average)
            mapped_team["APL_Off"] = 18.0
            mapped_team["APL_Def"] = 18.0

        teams.append(mapped_team)

    if len(teams) < 10:
        raise ValueError(f"Not enough teams with complete data: {len(teams)}")

    engineer = XGBoostFeatureEngineer()

    for _ in range(n_games):
        # Randomly select two teams
        team1, team2 = np.random.choice(teams, size=2, replace=False)

        # Random home court advantage
        neutral_site = np.random.choice([True, False], p=[0.3, 0.7])
        home_team1 = False if neutral_site else np.random.choice([True, False])

        # Create enhanced features
        features = engineer.create_enhanced_features(
            team1_stats=team1,
            team2_stats=team2,
            neutral_site=neutral_site,
            home_team1=home_team1,
        )

        # Generate realistic outcome incorporating luck regression
        em_diff = features["em_diff"]
        home_adv = features["home_advantage"] * 3.5
        luck_regression = features.get("luck_regression_expected", 0)
        tempo_factor = features["tempo_avg"] / 70

        # Predicted margin (with luck regression and noise)
        base_margin = em_diff + home_adv + luck_regression
        actual_margin = base_margin + np.random.normal(0, 10 * tempo_factor)

        # Predicted total (with noise)
        base_total = features["tempo_avg"] * 2 + 140
        actual_total = base_total + np.random.normal(0, 12)

        # Add outcomes
        features["actual_margin"] = actual_margin
        features["actual_total"] = actual_total

        # Add metadata
        features["team1"] = team1["TeamName"]
        features["team2"] = team2["TeamName"]
        features["season"] = season

        games.append(features)

    df = pd.DataFrame(games)
    print(f"Generated {len(df)} games with {len(features)} enhanced features")

    return df


def train_xgboost_model(
    games_df: pd.DataFrame, enhanced: bool = False
) -> XGBoostGamePredictor:
    """Train XGBoost model on historical games.

    Args:
        games_df: DataFrame with game features and outcomes
        enhanced: Whether to use enhanced features

    Returns:
        Trained XGBoostGamePredictor
    """
    print(f"\nTraining XGBoost model ({len(games_df)} games)...")

    # Select feature columns
    if enhanced:
        feature_cols = XGBoostFeatureEngineer.ENHANCED_FEATURE_NAMES
    else:
        feature_cols = FeatureEngineer.FEATURE_NAMES

    # Prepare data
    X = games_df[feature_cols]
    y_margin = games_df["actual_margin"]
    y_total = games_df["actual_total"]

    # Split into train/validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_margin_train, y_margin_val = (
        y_margin.iloc[:split_idx],
        y_margin.iloc[split_idx:],
    )
    y_total_train, y_total_val = (
        y_total.iloc[:split_idx],
        y_total.iloc[split_idx:],
    )

    # Train model - pass enhanced flag to use correct feature set
    predictor = XGBoostGamePredictor(use_enhanced_features=enhanced)

    print(f"  Training on {len(X_train)} games...")
    print(f"  Validating on {len(X_val)} games...")

    predictor.fit(
        games_df=X_train,
        margins=y_margin_train,
        totals=y_total_train,
    )

    # Evaluate on validation set
    print("\n  Validation Performance:")
    predictions = []
    actuals_margin = []
    actuals_total = []

    for idx in X_val.index:
        # Get team stats from original features
        tempo_avg = games_df.loc[idx, "tempo_avg"]
        team1_stats = {
            "AdjEM": games_df.loc[idx, "em_diff"] / 2,  # Approximate
            "AdjO": games_df.loc[idx, "oe_diff"] / 2 + 110,
            "AdjD": games_df.loc[idx, "de_diff"] / 2 + 100,
            "AdjOE": games_df.loc[idx, "oe_diff"] / 2 + 110,
            "AdjDE": games_df.loc[idx, "de_diff"] / 2 + 100,
            "AdjT": tempo_avg,  # FeatureEngineer expects AdjT
            "AdjTempo": tempo_avg,
            "Pythag": games_df.loc[idx, "pythag_diff"] / 2 + 0.5,
            "SOS": 5.0,
            "APL_Off": 18.0,
            "APL_Def": 18.0,
        }
        team2_stats = {
            "AdjEM": -games_df.loc[idx, "em_diff"] / 2,
            "AdjO": -games_df.loc[idx, "oe_diff"] / 2 + 110,
            "AdjD": -games_df.loc[idx, "de_diff"] / 2 + 100,
            "AdjOE": -games_df.loc[idx, "oe_diff"] / 2 + 110,
            "AdjDE": -games_df.loc[idx, "de_diff"] / 2 + 100,
            "AdjT": tempo_avg,  # FeatureEngineer expects AdjT
            "AdjTempo": tempo_avg,
            "Pythag": -games_df.loc[idx, "pythag_diff"] / 2 + 0.5,
            "SOS": 5.0,
            "APL_Off": 18.0,
            "APL_Def": 18.0,
        }

        result = predictor.predict_with_confidence(
            team1_stats, team2_stats, neutral_site=True
        )

        predictions.append(result.predicted_margin)
        actuals_margin.append(y_margin_val.loc[idx])
        actuals_total.append(y_total_val.loc[idx])

    # Calculate metrics
    mae_margin = np.mean(
        np.abs(np.array(predictions) - np.array(actuals_margin))
    )
    rmse_margin = np.sqrt(
        np.mean((np.array(predictions) - np.array(actuals_margin)) ** 2)
    )

    print(f"    MAE (Margin):  {mae_margin:.2f} points")
    print(f"    RMSE (Margin): {rmse_margin:.2f} points")

    return predictor


def save_model(
    predictor: XGBoostGamePredictor, season: int, enhanced: bool = False
):
    """Save trained model to disk.

    Args:
        predictor: Trained predictor
        season: Season year
        enhanced: Whether model uses enhanced features
    """
    # Create models directory
    models_dir = Path("data/xgboost_models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save margin model
    model_type = "enhanced" if enhanced else "base"
    margin_path = models_dir / f"margin_model_{season}_{model_type}.json"
    predictor.margin_model.save_model(margin_path)

    # Save total model
    total_path = models_dir / f"total_model_{season}_{model_type}.json"
    predictor.total_model.save_model(total_path)

    # Save metadata
    metadata = {
        "season": season,
        "enhanced_features": enhanced,
        "created_at": datetime.now().isoformat(),
        "feature_count": (
            len(XGBoostFeatureEngineer.ENHANCED_FEATURE_NAMES)
            if enhanced
            else len(FeatureEngineer.FEATURE_NAMES)
        ),
    }

    metadata_path = models_dir / f"metadata_{season}_{model_type}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved to {models_dir}/")
    print(f"    - {margin_path.name}")
    print(f"    - {total_path.name}")
    print(f"    - {metadata_path.name}")


def main():
    """Fetch data and train XGBoost model."""
    parser = argparse.ArgumentParser(
        description="Fetch data and train XGBoost"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2025)",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced Phase 2 features (luck, point dist, momentum)",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=1000,
        help="Number of synthetic games to generate (default: 1000)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("XGBOOST MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nSeason: {args.season}")
    print(f"Enhanced Features: {'Yes' if args.enhanced else 'No'}")
    print(f"Synthetic Games: {args.n_games}")

    try:
        # Initialize API
        api = KenPomAPI()

        # Fetch current ratings (use as baseline for synthetic games)
        print("\nFetching current season ratings...")
        ratings_response = api.get_ratings(year=args.season)
        ratings_df = pd.DataFrame(ratings_response.data)
        print(f"Loaded ratings for {len(ratings_df)} teams")

        # Generate synthetic games
        # TODO: Replace with actual game results when available
        if args.enhanced:
            games_df = generate_enhanced_synthetic_games(
                ratings_df, n_games=args.n_games, season=args.season
            )
        else:
            games_df = generate_synthetic_games(
                ratings_df, n_games=args.n_games, season=args.season
            )

        # Train model
        predictor = train_xgboost_model(games_df, enhanced=args.enhanced)

        # Save model
        save_model(predictor, args.season, enhanced=args.enhanced)

        # Feature importance analysis
        if args.enhanced:
            print("\n" + "=" * 70)
            print("FEATURE IMPORTANCE ANALYSIS (Top 15)")
            print("=" * 70)

            importance = predictor.get_feature_importance(
                importance_type="gain", top_n=15
            )

            print(f"\n{'Feature':<30} {'Gain Score':>12}")
            print("-" * 45)
            for idx, row in importance.iterrows():
                print(f"{row['feature']:<30} {row['importance']:>12.1f}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nModel Type: {'Enhanced' if args.enhanced else 'Base'}")
        print(f"Season: {args.season}")
        print(f"Training Games: {len(games_df)}")

        print("\nNext Steps:")
        print("  1. Update analyze_todays_games.py to use trained model")
        print("  2. Run live predictions on upcoming games")
        print(
            "  3. Replace synthetic data with real game results for better accuracy"
        )

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
