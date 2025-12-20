"""
Train XGBoost models for margin and total predictions
"""

import sys
from pathlib import Path
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.config import config
from kenp0m_sp0rts_analyzer.features.feature_engineer import (
    AdvancedFeatureEngineer,
)
from kenp0m_sp0rts_analyzer.models.xgboost_model import XGBoostModelWrapper
from kenp0m_sp0rts_analyzer.utils.logging import setup_logging
from kenp0m_sp0rts_analyzer.utils.exceptions import NCAABException


logger = setup_logging(config.logs_dir, level="INFO", app_name="training")


def load_training_data():
    """Load historical game data for training"""
    logger.info("Loading historical game data...")

    # This would load your actual historical data
    # For now, returning placeholder structure
    kenpom_data = {}  # Load KenPom stats
    historical_games = []  # Load game results

    # TODO: Implement actual data loading from database or CSV
    # Example structure:
    # historical_games = [
    #     {
    #         "game_id": "2025-01-01_Duke_UNC",
    #         "date": "2025-01-01",
    #         "home_team": "Duke",
    #         "away_team": "UNC",
    #         "opening_spread": -3.5,
    #         "opening_total": 152.0,
    #         "actual_margin": 5,  # Home won by 5
    #         "actual_total": 145,
    #     },
    #     ...
    # ]

    return kenpom_data, historical_games


def prepare_training_data(kenpom_data, historical_games):
    """Prepare features and targets for training"""
    logger.info("Preparing training data...")

    feature_engineer = AdvancedFeatureEngineer(kenpom_data, {})

    all_features = []
    all_margins = []
    all_totals = []
    skipped_games = []

    for game in historical_games:
        try:
            game_features = feature_engineer.create_features(
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_date=game["date"],
                vegas_spread=game.get("opening_spread", -2.5),
                vegas_total=game.get("opening_total", 152.0),
            )

            all_features.append(game_features.to_array())
            all_margins.append(game["actual_margin"])
            all_totals.append(game["actual_total"])

        except Exception as e:
            logger.warning(
                f"Skipping game {game.get('game_id', 'unknown')}: {str(e)}"
            )
            skipped_games.append(game.get("game_id", "unknown"))
            continue

    X = np.array(all_features)
    y_margin = np.array(all_margins)
    y_total = np.array(all_totals)

    logger.info(
        f"Prepared {len(all_features)} games for training (skipped {len(skipped_games)})"
    )

    return X, y_margin, y_total, feature_engineer.FEATURE_NAMES


def split_training_data(X, y_margin, y_total):
    """Split data into train/val/test sets"""
    logger.info("Splitting data into train/val/test sets...")

    # First split: 70% train, 30% temp
    X_train, X_temp, y_m_train, y_m_temp, y_t_train, y_t_temp = (
        train_test_split(
            X,
            y_margin,
            y_total,
            test_size=0.3,
            random_state=config.data_config.random_state,
        )
    )

    # Second split: 50/50 of temp into val/test
    X_val, X_test, y_m_val, y_m_test, y_t_val, y_t_test = train_test_split(
        X_temp,
        y_m_temp,
        y_t_temp,
        test_size=0.5,
        random_state=config.data_config.random_state,
    )

    logger.info(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_m_train,
        y_m_val,
        y_m_test,
        y_t_train,
        y_t_val,
        y_t_test,
    )


def train_margin_model(X_train, y_train, X_val, y_val, feature_names):
    """Train margin prediction model"""
    logger.info("Training margin prediction model...")

    model = XGBoostModelWrapper(model_name="ncaab", model_type="margin")

    try:
        metrics = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
        )

        logger.info(f"Margin Model Metrics:")
        logger.info(f"  Train MAE: {metrics['train_mae']:.4f}")
        logger.info(f"  Val MAE: {metrics['val_mae']:.4f}")
        logger.info(f"  Val RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"  Best Iteration: {metrics['best_iteration']}")

        return model, metrics

    except Exception as e:
        logger.error(f"Failed to train margin model: {str(e)}")
        raise


def train_total_model(X_train, y_train, X_val, y_val, feature_names):
    """Train total prediction model"""
    logger.info("Training total prediction model...")

    model = XGBoostModelWrapper(model_name="ncaab", model_type="total")

    try:
        metrics = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
        )

        logger.info(f"Total Model Metrics:")
        logger.info(f"  Train MAE: {metrics['train_mae']:.4f}")
        logger.info(f"  Val MAE: {metrics['val_mae']:.4f}")
        logger.info(f"  Val RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"  Best Iteration: {metrics['best_iteration']}")

        return model, metrics

    except Exception as e:
        logger.error(f"Failed to train total model: {str(e)}")
        raise


def save_training_summary(
    margin_model, total_model, margin_metrics, total_metrics
):
    """Save training summary to file"""
    logger.info("Saving training summary...")

    summary = {
        "training_timestamp": str(Path.ctime(Path("."))),
        "margin_model": {
            "train_mae": margin_metrics["train_mae"],
            "val_mae": margin_metrics["val_mae"],
            "val_rmse": margin_metrics["val_rmse"],
            "best_iteration": margin_metrics["best_iteration"],
            "feature_importance": margin_model.get_feature_importance(
                top_n=20
            ),
        },
        "total_model": {
            "train_mae": total_metrics["train_mae"],
            "val_mae": total_metrics["val_mae"],
            "val_rmse": total_metrics["val_rmse"],
            "best_iteration": total_metrics["best_iteration"],
            "feature_importance": total_model.get_feature_importance(top_n=20),
        },
    }

    summary_path = config.models_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved to {summary_path}")
    return summary


def main():
    """Main training routine"""
    try:
        logger.info("=" * 60)
        logger.info("Starting XGBoost Model Training Pipeline")
        logger.info("=" * 60)

        # 1. Load data
        kenpom_data, historical_games = load_training_data()

        if not historical_games:
            logger.error("No historical game data loaded!")
            return False

        # 2. Prepare data
        X, y_margin, y_total, feature_names = prepare_training_data(
            kenpom_data, historical_games
        )

        if len(X) < 100:
            logger.error(
                f"Insufficient training data: {len(X)} games (need at least 100)"
            )
            return False

        # 3. Split data
        (
            X_train,
            X_val,
            X_test,
            y_m_train,
            y_m_val,
            y_m_test,
            y_t_train,
            y_t_val,
            y_t_test,
        ) = split_training_data(X, y_margin, y_total)

        # 4. Train models
        logger.info("\\n" + "=" * 60)
        logger.info("Training Models")
        logger.info("=" * 60)

        margin_model, margin_metrics = train_margin_model(
            X_train, y_m_train, X_val, y_m_val, feature_names
        )

        total_model, total_metrics = train_total_model(
            X_train, y_t_train, X_val, y_t_val, feature_names
        )

        # 5. Evaluate on test set
        logger.info("\\n" + "=" * 60)
        logger.info("Evaluating on Test Set")
        logger.info("=" * 60)

        y_m_pred = margin_model.predict(X_test)
        y_t_pred = total_model.predict(X_test)

        test_margin_mae = np.mean(np.abs(y_m_test - y_m_pred))
        test_total_mae = np.mean(np.abs(y_t_test - y_t_pred))

        logger.info(f"Test Set - Margin MAE: {test_margin_mae:.4f}")
        logger.info(f"Test Set - Total MAE: {test_total_mae:.4f}")

        # 6. Save models
        logger.info("\\n" + "=" * 60)
        logger.info("Saving Models")
        logger.info("=" * 60)

        margin_model.save(config.models_dir)
        total_model.save(config.models_dir)

        # 7. Save summary
        summary = save_training_summary(
            margin_model, total_model, margin_metrics, total_metrics
        )

        logger.info("\\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Models saved to: {config.models_dir}")
        logger.info(f"Summary: {summary}")

        return True

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
