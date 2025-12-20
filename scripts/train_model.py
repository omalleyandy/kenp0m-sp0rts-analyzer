"""
Train XGBoost models for margin and total predictions.

Usage:
    python scripts/train_model.py [--samples N] [--start-date YYYY-MM-DD]

This script generates training data from historical ratings and trains
XGBoost models for margin and total predictions.
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.config import config
from kenp0m_sp0rts_analyzer.features import HistoricalDataLoader
from kenp0m_sp0rts_analyzer.kenpom import KenPomService
from kenp0m_sp0rts_analyzer.models.xgboost_model import XGBoostModelWrapper
from kenp0m_sp0rts_analyzer.utils.logging import setup_logging

logger = setup_logging(config.logs_dir, level="INFO", app_name="training")


def load_training_data(
    n_samples: int = 5000,
    start_date: date | None = None,
    end_date: date | None = None,
):
    """Load or generate historical game data for training.

    Args:
        n_samples: Number of training examples to generate.
        start_date: Start date for data range.
        end_date: End date for data range.

    Returns:
        Tuple of (X, y_margin, y_total, feature_names).
    """
    logger.info("Loading historical game data...")

    # Default date range
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(end_date.year - 1, 11, 1)

    # Initialize loader
    service = KenPomService()
    loader = HistoricalDataLoader(service)

    # Generate training data
    X, y_margin, y_total, feature_names = loader.generate_training_data(
        start_date=start_date,
        end_date=end_date,
        n_samples=n_samples,
        seed=42,
    )

    logger.info(
        f"Generated {len(X)} training examples with {len(feature_names)} "
        f"features"
    )

    return X, y_margin, y_total, feature_names


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

        logger.info("Margin Model Metrics:")
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

        logger.info("Total Model Metrics:")
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
        "training_timestamp": datetime.now().isoformat(),
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for basketball predictions"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with historical game results",
    )
    return parser.parse_args()


def main():
    """Main training routine"""
    args = parse_args()

    try:
        logger.info("=" * 60)
        logger.info("Starting XGBoost Model Training Pipeline")
        logger.info("=" * 60)

        # 1. Load training data
        if args.csv:
            # Load from CSV file with real game results
            logger.info(f"Loading training data from CSV: {args.csv}")
            service = KenPomService()
            loader = HistoricalDataLoader(service)
            X, y_margin, y_total, feature_names = loader.load_from_csv(
                args.csv
            )
        else:
            # Generate simulated training data
            start_date = None
            end_date = None
            if args.start_date:
                start_date = datetime.strptime(
                    args.start_date, "%Y-%m-%d"
                ).date()
            if args.end_date:
                end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

            X, y_margin, y_total, feature_names = load_training_data(
                n_samples=args.samples,
                start_date=start_date,
                end_date=end_date,
            )

        if len(X) < 100:
            logger.error(
                f"Insufficient training data: {len(X)} games "
                "(need at least 100)"
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
