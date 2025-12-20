#!/usr/bin/env python
"""Daily prediction and training workflow.

This script manages the daily cycle of:
1. Making predictions for today's games
2. Collecting yesterday's results
3. Tracking prediction accuracy
4. Accumulating training data
5. Periodic model retraining

Usage:
    # Morning: Make predictions for today
    python scripts/daily_workflow.py predict

    # Evening: Collect results and track accuracy
    python scripts/daily_workflow.py collect

    # Weekly: Retrain model with accumulated data
    python scripts/daily_workflow.py retrain

    # Full daily cycle
    python scripts/daily_workflow.py --all
"""

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.config import config
from kenp0m_sp0rts_analyzer.espn_results_scraper import ESPNResultsScraper
from kenp0m_sp0rts_analyzer.features import HistoricalDataLoader
from kenp0m_sp0rts_analyzer.integrated_predictor import IntegratedPredictor
from kenp0m_sp0rts_analyzer.kenpom import KenPomService
from kenp0m_sp0rts_analyzer.models.xgboost_model import XGBoostModelWrapper
from kenp0m_sp0rts_analyzer.utils.logging import setup_logging

logger = setup_logging(
    config.logs_dir, level="INFO", app_name="daily_workflow"
)


# Paths for accumulated data
ACCUMULATED_RESULTS_PATH = config.data_dir / "accumulated_results.csv"
PREDICTIONS_LOG_PATH = config.data_dir / "predictions_log.json"
ACCURACY_LOG_PATH = config.data_dir / "accuracy_log.json"


def load_predictions_log() -> dict:
    """Load the predictions log from disk."""
    if PREDICTIONS_LOG_PATH.exists():
        with PREDICTIONS_LOG_PATH.open() as f:
            return json.load(f)
    return {"predictions": []}


def save_predictions_log(log: dict) -> None:
    """Save the predictions log to disk."""
    with PREDICTIONS_LOG_PATH.open("w") as f:
        json.dump(log, f, indent=2, default=str)


def load_accuracy_log() -> dict:
    """Load accuracy tracking log."""
    if ACCURACY_LOG_PATH.exists():
        with ACCURACY_LOG_PATH.open() as f:
            return json.load(f)
    return {
        "daily_accuracy": [],
        "cumulative": {
            "total_games": 0,
            "margin_mae": 0.0,
            "total_mae": 0.0,
            "spread_ats_wins": 0,
            "spread_ats_losses": 0,
            "spread_pushes": 0,
        },
    }


def save_accuracy_log(log: dict) -> None:
    """Save accuracy tracking log."""
    with ACCURACY_LOG_PATH.open("w") as f:
        json.dump(log, f, indent=2, default=str)


async def make_predictions(game_date: date | None = None) -> list[dict]:
    """Make predictions for games on a given date.

    Args:
        game_date: Date to predict (default: today).

    Returns:
        List of prediction dictionaries.
    """
    if game_date is None:
        game_date = date.today()

    logger.info(f"Making predictions for {game_date}")

    # Initialize predictor with trained model
    model_path = (
        config.data_dir / "xgboost_models" / "margin_model_2025_enhanced.json"
    )
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training first.")
        return []

    _predictor = IntegratedPredictor(
        db_path=str(config.data_dir / "kenpom.db"),
        model_path=str(model_path),
        use_xgb_wrapper=True,
    )  # Placeholder for schedule integration

    # Get today's games from KenPom schedule (if available)
    # For now, we'll need to get matchups from another source
    # This is a placeholder - in production, you'd fetch from schedule API
    _ = KenPomService()  # Will be used for schedule integration

    predictions: list[dict] = []
    _ = load_predictions_log()  # Will be used to log predictions

    # Example: Load games from a schedule file or API
    # For demonstration, we'll show how to predict a specific matchup
    logger.info(
        "To predict games, provide matchups via schedule API or manual input"
    )
    logger.info(f"Predictor ready with model: {model_path.name}")

    return predictions


async def collect_results(game_date: date | None = None) -> list[dict]:
    """Collect game results from ESPN.

    Args:
        game_date: Date to collect (default: yesterday).

    Returns:
        List of game result dictionaries.
    """
    if game_date is None:
        game_date = date.today() - timedelta(days=1)

    logger.info(f"Collecting results for {game_date}")

    scraper = ESPNResultsScraper()
    results = await scraper.scrape_date(game_date)

    if results:
        # Save to daily file
        daily_filename = f"espn_results_{game_date}.csv"
        csv_path = scraper.save_to_csv(results, daily_filename)
        logger.info(f"Saved {len(results)} results to {csv_path}")

        # Append to accumulated results
        await append_to_accumulated_results(results)

    return [r.to_dict() for r in results]


async def append_to_accumulated_results(results: list) -> None:
    """Append new results to the accumulated training data file."""
    import csv

    fieldnames = [
        "game_date",
        "home_team",
        "away_team",
        "margin",
        "total",
        "home_score",
        "away_score",
        "overtime",
    ]

    # Check if file exists to determine if we need headers
    file_exists = ACCUMULATED_RESULTS_PATH.exists()

    with ACCUMULATED_RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "game_date": str(result.game_date),
                    "home_team": result.home_team_normalized,
                    "away_team": result.away_team_normalized,
                    "margin": result.margin,
                    "total": result.total,
                    "home_score": result.home_score,
                    "away_score": result.away_score,
                    "overtime": result.overtime,
                }
            )

    logger.info(
        f"Appended {len(results)} results to {ACCUMULATED_RESULTS_PATH}"
    )


def track_accuracy(predictions: list[dict], results: list[dict]) -> dict:
    """Compare predictions to actual results and track accuracy.

    Args:
        predictions: List of prediction dicts with predicted_margin,
            predicted_total.
        results: List of result dicts with actual margin, total.

    Returns:
        Accuracy metrics for the day.
    """
    if not predictions or not results:
        return {}

    # Match predictions to results by teams
    matched = []
    for pred in predictions:
        for res in results:
            if (
                pred["home_team"] == res["home_team"]
                and pred["away_team"] == res["away_team"]
            ):
                matched.append(
                    {
                        "home_team": pred["home_team"],
                        "away_team": pred["away_team"],
                        "predicted_margin": pred["predicted_margin"],
                        "actual_margin": res["margin"],
                        "predicted_total": pred.get("predicted_total", 0),
                        "actual_total": res["total"],
                        "vegas_spread": pred.get("vegas_spread"),
                    }
                )
                break

    if not matched:
        logger.warning("No predictions matched to results")
        return {}

    # Calculate metrics
    margin_errors = [
        abs(m["predicted_margin"] - m["actual_margin"]) for m in matched
    ]
    total_errors = [
        abs(m["predicted_total"] - m["actual_total"]) for m in matched
    ]

    margin_mae = np.mean(margin_errors)
    total_mae = np.mean(total_errors)

    # ATS performance (if we have Vegas lines)
    ats_wins = 0
    ats_losses = 0
    pushes = 0

    for m in matched:
        if m["vegas_spread"] is not None:
            # Vegas spread is from home team perspective
            predicted_cover = m["predicted_margin"] > -m["vegas_spread"]
            actual_cover = m["actual_margin"] > -m["vegas_spread"]

            if abs(m["actual_margin"] - (-m["vegas_spread"])) < 0.5:
                pushes += 1
            elif predicted_cover == actual_cover:
                ats_wins += 1
            else:
                ats_losses += 1

    metrics = {
        "date": str(date.today()),
        "games_matched": len(matched),
        "margin_mae": round(margin_mae, 2),
        "total_mae": round(total_mae, 2),
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": pushes,
        "ats_pct": (
            round(ats_wins / (ats_wins + ats_losses) * 100, 1)
            if (ats_wins + ats_losses) > 0
            else 0
        ),
    }

    # Update cumulative tracking
    accuracy_log = load_accuracy_log()
    accuracy_log["daily_accuracy"].append(metrics)

    cumulative = accuracy_log["cumulative"]
    cumulative["total_games"] += len(matched)
    # Running average for MAE
    n = cumulative["total_games"]
    old_n = n - len(matched)
    if old_n > 0:
        cumulative["margin_mae"] = (
            cumulative["margin_mae"] * old_n + margin_mae * len(matched)
        ) / n
        cumulative["total_mae"] = (
            cumulative["total_mae"] * old_n + total_mae * len(matched)
        ) / n
    else:
        cumulative["margin_mae"] = margin_mae
        cumulative["total_mae"] = total_mae

    cumulative["spread_ats_wins"] += ats_wins
    cumulative["spread_ats_losses"] += ats_losses
    cumulative["spread_pushes"] += pushes

    save_accuracy_log(accuracy_log)

    logger.info(
        f"Daily accuracy: Margin MAE={margin_mae:.2f}, "
        f"ATS={ats_wins}-{ats_losses}"
    )

    return metrics


def retrain_model(min_games: int = 200) -> bool:
    """Retrain the model with accumulated results.

    Args:
        min_games: Minimum games required for retraining.

    Returns:
        True if retraining succeeded.
    """
    logger.info("Checking for model retraining...")

    if not ACCUMULATED_RESULTS_PATH.exists():
        logger.warning(
            "No accumulated results found. Collect more data first."
        )
        return False

    # Count games in accumulated file
    with ACCUMULATED_RESULTS_PATH.open() as f:
        game_count = sum(1 for _ in f) - 1  # Subtract header

    if game_count < min_games:
        logger.info(
            f"Only {game_count} games accumulated. "
            f"Need {min_games} for retraining."
        )
        return False

    logger.info(f"Retraining with {game_count} accumulated games...")

    # Load data and retrain
    from sklearn.model_selection import train_test_split

    service = KenPomService()
    loader = HistoricalDataLoader(service)

    try:
        X, y_margin, y_total, feature_names = loader.load_from_csv(
            str(ACCUMULATED_RESULTS_PATH)
        )

        if len(X) < 100:
            logger.error(f"Insufficient valid games: {len(X)}")
            return False

        # Split data
        X_train, X_temp, y_m_train, y_m_temp, y_t_train, y_t_temp = (
            train_test_split(
                X, y_margin, y_total, test_size=0.3, random_state=42
            )
        )
        X_val, X_test, y_m_val, y_m_test, y_t_val, y_t_test = train_test_split(
            X_temp, y_m_temp, y_t_temp, test_size=0.5, random_state=42
        )

        # Train margin model
        margin_model = XGBoostModelWrapper(
            model_name="ncaab", model_type="margin"
        )
        margin_metrics = margin_model.train(
            X_train=X_train,
            y_train=y_m_train,
            X_val=X_val,
            y_val=y_m_val,
            feature_names=feature_names,
        )

        # Train total model
        total_model = XGBoostModelWrapper(
            model_name="ncaab", model_type="total"
        )
        total_metrics = total_model.train(
            X_train=X_train,
            y_train=y_t_train,
            X_val=X_val,
            y_val=y_t_val,
            feature_names=feature_names,
        )

        # Save models
        margin_model.save(config.models_dir)
        total_model.save(config.models_dir)

        logger.info(
            f"Retraining complete! "
            f"Margin MAE: {margin_metrics['val_mae']:.2f}, "
            f"Total MAE: {total_metrics['val_mae']:.2f}"
        )

        # Save training summary
        summary = {
            "retrained_at": datetime.now().isoformat(),
            "games_used": len(X),
            "margin_val_mae": margin_metrics["val_mae"],
            "total_val_mae": total_metrics["val_mae"],
        }

        summary_path = config.models_dir / "retrain_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        return True

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False


def show_status() -> None:
    """Display current status of the prediction system."""
    print("\n" + "=" * 60)
    print("KenPom Sports Analyzer - System Status")
    print("=" * 60)

    # Model status
    model_path = (
        config.data_dir / "xgboost_models" / "margin_model_2025_enhanced.json"
    )
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"\n[Model] Last trained: {mtime.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("\n[Model] Not trained yet!")

    # Accumulated data
    if ACCUMULATED_RESULTS_PATH.exists():
        with ACCUMULATED_RESULTS_PATH.open() as f:
            game_count = sum(1 for _ in f) - 1
        print(f"[Data] Accumulated games: {game_count}")
    else:
        print("[Data] No accumulated results yet")

    # Accuracy tracking
    if ACCURACY_LOG_PATH.exists():
        log = load_accuracy_log()
        cum = log["cumulative"]
        if cum["total_games"] > 0:
            print(f"\n[Accuracy] Total games tracked: {cum['total_games']}")
            print(f"  - Margin MAE: {cum['margin_mae']:.2f}")
            print(f"  - Total MAE: {cum['total_mae']:.2f}")
            total_ats = cum["spread_ats_wins"] + cum["spread_ats_losses"]
            if total_ats > 0:
                ats_pct = cum["spread_ats_wins"] / total_ats * 100
                print(
                    f"  - ATS Record: {cum['spread_ats_wins']}-"
                    f"{cum['spread_ats_losses']} ({ats_pct:.1f}%)"
                )

    print("\n" + "=" * 60 + "\n")


async def run_collect_workflow() -> None:
    """Run the evening collection workflow."""
    yesterday = date.today() - timedelta(days=1)

    logger.info(f"Starting collection workflow for {yesterday}")

    # 1. Collect yesterday's results
    results = await collect_results(yesterday)
    logger.info(f"Collected {len(results)} game results")

    # 2. Load yesterday's predictions (if any)
    predictions_log = load_predictions_log()
    yesterday_preds = [
        p
        for p in predictions_log.get("predictions", [])
        if p.get("game_date") == str(yesterday)
    ]

    # 3. Track accuracy
    if yesterday_preds and results:
        metrics = track_accuracy(yesterday_preds, results)
        if metrics:
            logger.info(f"Accuracy tracked: {metrics}")

    logger.info("Collection workflow complete")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Daily prediction and training workflow"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["predict", "collect", "retrain", "status"],
        default="status",
        help="Command to run (default: status)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date for predictions/collection (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full daily cycle",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.all:
        # Full cycle: collect yesterday, make today's predictions
        await run_collect_workflow()
        await make_predictions()
        show_status()

    elif args.command == "predict":
        await make_predictions(target_date)

    elif args.command == "collect":
        await run_collect_workflow()

    elif args.command == "retrain":
        retrain_model()

    elif args.command == "status":
        show_status()


if __name__ == "__main__":
    asyncio.run(main())
