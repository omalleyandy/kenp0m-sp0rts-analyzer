"""Train XGBoost model on real NCAA basketball game results.

This script replaces synthetic training data with actual game outcomes:
1. Loads real game results from ESPN scraped data
2. Fetches/loads historical ratings for each game date
3. Generates features from pre-game team ratings
4. Trains XGBoost model on actual margins and totals

Usage:
    python scripts/train_on_real_data.py --season 2025
    python scripts/train_on_real_data.py --season 2025 --backfill-ratings
    python scripts/train_on_real_data.py --season 2025 --enhanced
"""

import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.helpers import normalize_team_name
from kenp0m_sp0rts_analyzer.prediction import (
    FeatureEngineer,
    XGBoostFeatureEngineer,
    XGBoostGamePredictor,
)

load_dotenv()

# Constants
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "xgboost_models"
DB_PATH = DATA_DIR / "kenpom.db"
ESPN_RESULTS_DIR = DATA_DIR / "espn_results"


def load_espn_results(season: int) -> pd.DataFrame:
    """Load ESPN game results for a season.

    Args:
        season: Season year (e.g., 2025 for 2024-25 season)

    Returns:
        DataFrame with game results
    """
    print(f"\nLoading ESPN game results for {season - 1}-{season} season...")

    all_results = []

    # Look for all ESPN result files
    if ESPN_RESULTS_DIR.exists():
        for csv_file in ESPN_RESULTS_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                # Filter out non-D1 games (approximation: both teams should be known)
                all_results.append(df)
                print(f"  Loaded {len(df)} games from {csv_file.name}")
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")

    if not all_results:
        raise FileNotFoundError(
            f"No ESPN results found in {ESPN_RESULTS_DIR}. "
            "Run the ESPN scraper first to collect game results."
        )

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Parse dates and filter to season
    combined["game_date"] = pd.to_datetime(combined["game_date"])

    # Season runs Nov (year-1) to April (year)
    season_start = datetime(season - 1, 11, 1)
    season_end = datetime(season, 4, 30)

    mask = (combined["game_date"] >= season_start) & (
        combined["game_date"] <= season_end
    )
    season_games = combined[mask].copy()

    # Remove duplicates (same game might appear in multiple files)
    season_games = season_games.drop_duplicates(
        subset=["game_date", "home_team", "away_team"]
    )

    print(f"\nTotal unique games for {season} season: {len(season_games)}")

    return season_games


def get_available_archive_dates(db_path: Path) -> set[date]:
    """Get all dates that have archive data in the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Set of dates with archive data
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT archive_date FROM archive")
    dates = {
        date.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
        for row in cursor.fetchall()
    }

    conn.close()
    return dates


def get_ratings_for_date(
    db_path: Path, archive_date: date
) -> dict[str, dict[str, Any]]:
    """Load team ratings from archive for a specific date.

    Args:
        db_path: Path to SQLite database
        archive_date: Date to get ratings for

    Returns:
        Dictionary mapping team name -> rating dict
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT team_name, adj_em, adj_oe, adj_de, adj_tempo,
               rank_adj_em, rank_adj_oe, rank_adj_de
        FROM archive
        WHERE archive_date = ?
    """,
        (archive_date.isoformat(),),
    )

    ratings = {}
    for row in cursor.fetchall():
        team_name = row[0]
        ratings[team_name] = {
            "TeamName": team_name,
            "AdjEM": row[1],
            "AdjO": row[2],  # Map to FeatureEngineer expected names
            "AdjOE": row[2],
            "AdjD": row[3],
            "AdjDE": row[3],
            "AdjT": row[4],
            "AdjTempo": row[4],
            "RankEM": row[5],
            "RankOE": row[6],
            "RankDE": row[7],
            # Default values for fields that may not be in archive
            "Luck": 0.0,
            "Pythag": 0.5,
            "SOS": 5.0,
            "SOSO": 5.0,
            "SOSD": 5.0,
            "APL_Off": 18.0,
            "APL_Def": 18.0,
        }

    conn.close()
    return ratings


def get_current_ratings(db_path: Path) -> dict[str, dict[str, Any]]:
    """Load current team ratings from the ratings table.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary mapping team name -> rating dict
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT team_name, adj_em, adj_oe, adj_de, adj_tempo,
               rank_adj_em, luck, pythag, sos
        FROM ratings
    """
    )

    ratings = {}
    for row in cursor.fetchall():
        team_name = row[0]
        ratings[team_name] = {
            "TeamName": team_name,
            "AdjEM": row[1],
            "AdjO": row[2],
            "AdjOE": row[2],
            "AdjD": row[3],
            "AdjDE": row[3],
            "AdjT": row[4],
            "AdjTempo": row[4],
            "RankEM": row[5],
            "Luck": row[6] or 0.0,
            "Pythag": row[7] or 0.5,
            "SOS": row[8] or 5.0,
            "SOSO": 5.0,
            "SOSD": 5.0,
            "APL_Off": 18.0,
            "APL_Def": 18.0,
        }

    conn.close()
    return ratings


def find_team_in_ratings(
    team_name: str, ratings: dict[str, dict]
) -> dict[str, Any] | None:
    """Find a team in the ratings dictionary, trying normalization.

    Args:
        team_name: Raw team name
        ratings: Dictionary of team ratings

    Returns:
        Team rating dict or None if not found
    """
    # Try exact match first
    if team_name in ratings:
        return ratings[team_name]

    # Try normalized name
    normalized = normalize_team_name(team_name)
    if normalized in ratings:
        return ratings[normalized]

    # Try case-insensitive match
    team_lower = team_name.lower()
    for known_team, data in ratings.items():
        if known_team.lower() == team_lower:
            return data

    return None


def backfill_archive_ratings(
    db_path: Path, dates_needed: list[date], progress: bool = True
) -> int:
    """Backfill archive ratings for missing dates.

    Args:
        db_path: Path to SQLite database
        dates_needed: List of dates to backfill
        progress: Show progress

    Returns:
        Number of dates successfully backfilled
    """
    from kenp0m_sp0rts_analyzer.kenpom.archive_loader import ArchiveLoader

    print(f"\nBackfilling {len(dates_needed)} archive dates...")

    loader = ArchiveLoader(db_path=str(db_path))
    filled = 0

    for i, archive_date in enumerate(dates_needed):
        if progress and i % 10 == 0:
            print(f"  Progress: {i}/{len(dates_needed)} dates...")

        try:
            loader.load_date(archive_date, store=True)
            filled += 1
        except Exception as e:
            print(f"  Skipping {archive_date}: {e}")

    print(f"  Backfilled {filled} dates")
    return filled


def create_training_data(
    games_df: pd.DataFrame,
    db_path: Path,
    enhanced: bool = False,
    use_current_ratings: bool = False,
) -> pd.DataFrame:
    """Create training dataset from real game results.

    For each game, looks up pre-game team ratings and creates features.

    Args:
        games_df: DataFrame with game results
        db_path: Path to SQLite database
        enhanced: Whether to use enhanced features
        use_current_ratings: Use current ratings instead of historical

    Returns:
        DataFrame with features and actual outcomes
    """
    print(f"\nCreating training data from {len(games_df)} games...")

    # Get available archive dates
    available_dates = get_available_archive_dates(db_path)
    print(f"  Archive has {len(available_dates)} snapshot dates")

    # Load current ratings as fallback
    current_ratings = get_current_ratings(db_path)
    print(f"  Current ratings for {len(current_ratings)} teams")

    # Initialize feature engineer
    if enhanced:
        engineer = XGBoostFeatureEngineer()
        create_features = engineer.create_enhanced_features
    else:
        engineer = FeatureEngineer()
        create_features = engineer.create_features

    training_data = []
    skipped_no_team1 = 0
    skipped_no_team2 = 0
    used_current = 0
    used_historical = 0

    for _, game in games_df.iterrows():
        game_date = game["game_date"].date()
        home_team = game["home_team"]
        away_team = game["away_team"]

        # Try to get historical ratings for this date
        ratings = None

        if not use_current_ratings:
            # Find the closest available archive date on or before game date
            valid_dates = [d for d in available_dates if d <= game_date]
            if valid_dates:
                closest_date = max(valid_dates)
                ratings = get_ratings_for_date(db_path, closest_date)
                used_historical += 1

        # Fall back to current ratings if no historical available
        if ratings is None or len(ratings) == 0:
            ratings = current_ratings
            used_current += 1

        # Look up teams
        team1_stats = find_team_in_ratings(away_team, ratings)
        team2_stats = find_team_in_ratings(home_team, ratings)

        if team1_stats is None:
            skipped_no_team1 += 1
            continue

        if team2_stats is None:
            skipped_no_team2 += 1
            continue

        # Create features (away team is team1, home team is team2)
        try:
            features = create_features(
                team1_stats=team1_stats,
                team2_stats=team2_stats,
                neutral_site=False,  # Assume home game
                home_team1=False,  # Away team (team1) is not home
            )
        except Exception as e:
            print(
                f"  Error creating features for {away_team} @ {home_team}: {e}"
            )
            continue

        # Add actual outcomes (margin from away team perspective)
        # ESPN: margin = home_score - away_score (positive = home win)
        # Our model: margin = team1 - team2 (positive = away win)
        actual_margin = -float(
            game["margin"]
        )  # Flip sign for away team perspective
        actual_total = float(game["total"])

        features["actual_margin"] = actual_margin
        features["actual_total"] = actual_total

        # Add metadata (not used for training)
        features["team1"] = away_team
        features["team2"] = home_team
        features["game_date"] = game_date.isoformat()
        features["overtime"] = game.get("overtime", False)

        training_data.append(features)

    print(f"\n  Training games created: {len(training_data)}")
    print(f"  Skipped (team1 not found): {skipped_no_team1}")
    print(f"  Skipped (team2 not found): {skipped_no_team2}")
    print(f"  Used historical ratings: {used_historical}")
    print(f"  Used current ratings: {used_current}")

    return pd.DataFrame(training_data)


def train_xgboost_model(
    games_df: pd.DataFrame, enhanced: bool = False
) -> XGBoostGamePredictor:
    """Train XGBoost model on real game data.

    Args:
        games_df: DataFrame with game features and outcomes
        enhanced: Whether to use enhanced features

    Returns:
        Trained XGBoostGamePredictor
    """
    print(f"\nTraining XGBoost model on {len(games_df)} real games...")

    # Exclude overtime games (skewed totals and margins)
    non_overtime = games_df[~games_df["overtime"].astype(bool)].copy()
    print(f"  Non-overtime games for training: {len(non_overtime)}")

    # Select feature columns
    if enhanced:
        feature_cols = XGBoostFeatureEngineer.ENHANCED_FEATURE_NAMES
    else:
        feature_cols = FeatureEngineer.FEATURE_NAMES

    # Verify all feature columns exist
    missing_cols = [c for c in feature_cols if c not in games_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing features: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in games_df.columns]

    # Use non-overtime games for both margin and total training
    X = non_overtime[feature_cols]
    y_margin = non_overtime["actual_margin"]
    y_total = non_overtime["actual_total"]

    # Split into train/validation (80/20)
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_margin_train = y_margin.iloc[:split_idx]
    y_margin_val = y_margin.iloc[split_idx:]
    y_total_train = y_total.iloc[:split_idx]
    y_total_val = y_total.iloc[split_idx:]

    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # Train model
    predictor = XGBoostGamePredictor(use_enhanced_features=enhanced)

    predictor.fit(
        games_df=X_train,
        margins=y_margin_train,
        totals=y_total_train,
    )

    # Evaluate on validation set
    print("\n  Validation Performance:")

    # Margin predictions
    margin_preds = predictor.margin_model.predict(X_val)
    mae_margin = np.mean(np.abs(margin_preds - y_margin_val))
    rmse_margin = np.sqrt(np.mean((margin_preds - y_margin_val) ** 2))

    # Total predictions
    total_preds = predictor.total_model.predict(X_val)
    mae_total = np.mean(np.abs(total_preds - y_total_val))
    rmse_total = np.sqrt(np.mean((total_preds - y_total_val) ** 2))

    print(f"    Margin MAE:  {mae_margin:.2f} points")
    print(f"    Margin RMSE: {rmse_margin:.2f} points")
    print(f"    Total MAE:   {mae_total:.2f} points")
    print(f"    Total RMSE:  {rmse_total:.2f} points")

    # ATS accuracy (if we have enough games)
    if len(X_val) > 20:
        # Simulated ATS: did we predict the right side?
        correct_side = np.sum(np.sign(margin_preds) == np.sign(y_margin_val))
        ats_accuracy = correct_side / len(y_margin_val) * 100
        print(f"    Side Accuracy: {ats_accuracy:.1f}%")

    return predictor


def save_model(
    predictor: XGBoostGamePredictor,
    season: int,
    enhanced: bool = False,
    training_games: int = 0,
):
    """Save trained model to disk.

    Args:
        predictor: Trained predictor
        season: Season year
        enhanced: Whether model uses enhanced features
        training_games: Number of games used for training
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_type = "real_enhanced" if enhanced else "real_base"

    # Save margin models
    margin_path = MODELS_DIR / f"margin_model_{season}_{model_type}.json"
    predictor.margin_model.get_booster().save_model(str(margin_path))

    margin_upper_path = MODELS_DIR / f"margin_upper_{season}_{model_type}.json"
    predictor.margin_upper.get_booster().save_model(str(margin_upper_path))

    margin_lower_path = MODELS_DIR / f"margin_lower_{season}_{model_type}.json"
    predictor.margin_lower.get_booster().save_model(str(margin_lower_path))

    # Save total model
    total_path = MODELS_DIR / f"total_model_{season}_{model_type}.json"
    predictor.total_model.get_booster().save_model(str(total_path))

    # Save metadata
    metadata = {
        "season": season,
        "enhanced_features": enhanced,
        "trained_on_real_data": True,
        "training_games": training_games,
        "created_at": datetime.now().isoformat(),
        "feature_count": (
            len(XGBoostFeatureEngineer.ENHANCED_FEATURE_NAMES)
            if enhanced
            else len(FeatureEngineer.FEATURE_NAMES)
        ),
        "quantile_models": {
            "upper_quantile": 0.75,
            "lower_quantile": 0.25,
        },
    }

    metadata_path = MODELS_DIR / f"metadata_{season}_{model_type}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved to {MODELS_DIR}/")
    print(f"    - {margin_path.name}")
    print(f"    - {margin_upper_path.name}")
    print(f"    - {margin_lower_path.name}")
    print(f"    - {total_path.name}")
    print(f"    - {metadata_path.name}")


def main():
    """Train XGBoost model on real game data."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost on real NCAA game results"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2025 for 2024-25 season)",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced Phase 2 features",
    )
    parser.add_argument(
        "--backfill-ratings",
        action="store_true",
        help="Backfill missing historical ratings from KenPom archive",
    )
    parser.add_argument(
        "--use-current-ratings",
        action="store_true",
        help="Use current ratings only (faster, less accurate)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("REAL DATA XGBOOST TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nSeason: {args.season}")
    print(f"Enhanced Features: {'Yes' if args.enhanced else 'No'}")
    print(
        f"Use Current Ratings: {'Yes' if args.use_current_ratings else 'No'}"
    )

    try:
        # Load ESPN game results
        games_df = load_espn_results(args.season)

        if len(games_df) == 0:
            print("\nNo games found. Please run the ESPN scraper first.")
            return

        # Check if we need to backfill ratings
        if args.backfill_ratings and not args.use_current_ratings:
            # Get unique game dates
            game_dates = games_df["game_date"].dt.date.unique()

            # Check which dates are missing
            available_dates = get_available_archive_dates(DB_PATH)
            missing_dates = [d for d in game_dates if d not in available_dates]

            if missing_dates:
                print(
                    f"\nMissing archive data for {len(missing_dates)} game dates"
                )
                backfill_archive_ratings(DB_PATH, sorted(missing_dates))
            else:
                print("\nAll game dates have archive data")

        # Create training data
        training_df = create_training_data(
            games_df,
            DB_PATH,
            enhanced=args.enhanced,
            use_current_ratings=args.use_current_ratings,
        )

        if len(training_df) < 50:
            print(f"\nNot enough training data ({len(training_df)} games).")
            print("Need at least 50 games for meaningful training.")
            return

        # Train model
        predictor = train_xgboost_model(training_df, enhanced=args.enhanced)

        # Save model
        save_model(
            predictor,
            args.season,
            enhanced=args.enhanced,
            training_games=len(training_df),
        )

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
            for _, row in importance.iterrows():
                print(f"{row['feature']:<30} {row['importance']:>12.1f}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(
            f"\nModel Type: Real Data {'Enhanced' if args.enhanced else 'Base'}"
        )
        print(f"Season: {args.season}")
        print(f"Training Games: {len(training_df)}")
        print(f"Model Path: {MODELS_DIR}")

        print("\nNext Steps:")
        print("  1. Compare performance vs synthetic model")
        print("  2. Update IntegratedPredictor to use real data model")
        print("  3. Run predictions on today's games")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
