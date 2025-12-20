"""
XGBoost model wrapper with training, prediction, and management
"""

import xgboost as xgb
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

from ..config import config
from ..utils.exceptions import ModelError
from ..utils.logging import logger


class XGBoostModelWrapper:
    """
    Wrapper around XGBoost model with full lifecycle management
    """

    def __init__(
        self, model_name: str = "ncaab_xgb", model_type: str = "margin"
    ):
        """
        Initialize XGBoost model wrapper

        Args:
            model_name: Name of the model
            model_type: "margin" or "total"
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model: Optional[xgb.Booster] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list] = None
        self.feature_importance: Optional[Dict] = None
        self.metadata: Dict = {
            "created_at": None,
            "trained_at": None,
            "model_type": model_type,
            "num_games": 0,
            "mae": None,
            "rmse": None,
        }
        self.logger = logger

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
        **kwargs,
    ) -> Dict:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Names of features in order
            **kwargs: Additional XGBoost parameters

        Returns:
            Training metrics dictionary
        """
        try:
            self.logger.info(
                f"Starting training for {self.model_name} ({self.model_type})"
            )

            self.feature_names = feature_names

            # Fit scaler on training data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Prepare DMatrix
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)

            # Model parameters
            params = {
                "objective": "reg:squarederror",
                "max_depth": config.model_config.max_depth,
                "learning_rate": config.model_config.learning_rate,
                "subsample": config.model_config.subsample,
                "colsample_bytree": config.model_config.colsample_bytree,
                "gamma": config.model_config.gamma,
                "reg_alpha": config.model_config.reg_alpha,
                "reg_lambda": config.model_config.reg_lambda,
                "tree_method": "hist",
                "device": "cuda" if self._has_gpu() else "cpu",
            }
            params.update(kwargs)

            # Train with early stopping
            evals = [(dtrain, "train"), (dval, "eval")]
            evals_result = {}

            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=config.model_config.num_boost_rounds,
                evals=evals,
                evals_result=evals_result,
                early_stopping_rounds=config.model_config.early_stopping_rounds,
                verbose_eval=False,
            )

            # Calculate metrics
            y_train_pred = self.model.predict(dtrain)
            y_val_pred = self.model.predict(dval)

            train_mae = np.mean(np.abs(y_train - y_train_pred))
            val_mae = np.mean(np.abs(y_val - y_val_pred))
            val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))

            # Feature importance
            self.feature_importance = self.model.get_score(
                importance_type="weight"
            )

            # Update metadata
            self.metadata.update(
                {
                    "created_at": datetime.now().isoformat(),
                    "trained_at": datetime.now().isoformat(),
                    "num_games": len(X_train) + len(X_val),
                    "train_games": len(X_train),
                    "val_games": len(X_val),
                    "mae": float(val_mae),
                    "rmse": float(val_rmse),
                    "train_mae": float(train_mae),
                }
            )

            metrics = {
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
                "best_iteration": self.model.best_iteration,
                "feature_importance": self.feature_importance,
            }

            self.logger.info(
                f"Training complete. Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}")

    def predict(
        self,
        X: np.ndarray,
        scale: bool = True,
        feature_names: list | None = None,
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features
            scale: Whether to scale features
            feature_names: Feature names (required if model was trained with them)

        Returns:
            Predictions
        """
        if self.model is None:
            raise ModelError("Model not trained or loaded")

        try:
            if scale and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Use feature names if provided or stored
            names = feature_names or self.feature_names
            dmatrix = xgb.DMatrix(X_scaled, feature_names=names)
            predictions = self.model.predict(dmatrix)

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")

    def save(self, model_dir: Path) -> Path:
        """
        Save model and scaler to disk

        Args:
            model_dir: Directory to save model

        Returns:
            Path to saved model
        """
        try:
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save XGBoost model
            model_path = (
                model_dir / f"{self.model_name}_{self.model_type}.json"
            )
            self.model.save_model(str(model_path))

            # Save scaler
            scaler_path = model_dir / f"{self.model_name}_scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            # Save metadata
            metadata_path = model_dir / f"{self.model_name}_metadata.json"
            metadata_copy = self.metadata.copy()
            metadata_copy["feature_importance"] = str(self.feature_importance)
            with open(metadata_path, "w") as f:
                json.dump(metadata_copy, f, indent=2)

            # Save feature names
            features_path = model_dir / f"{self.model_name}_features.json"
            with open(features_path, "w") as f:
                json.dump({"feature_names": self.feature_names}, f)

            self.logger.info(f"Model saved to {model_dir}")
            return model_path

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")

    def load(self, model_dir: Path) -> bool:
        """
        Load model and optional scaler from disk.

        Args:
            model_dir: Directory containing model files

        Returns:
            True if successful
        """
        try:
            # Load model
            model_path = (
                model_dir / f"{self.model_name}_{self.model_type}.json"
            )
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))

            # Load scaler (optional - some models trained without scaling)
            scaler_path = model_dir / f"{self.model_name}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = None
                self.logger.info("No scaler found, using unscaled features")

            # Load metadata (optional)
            metadata_path = model_dir / f"{self.model_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}

            # Load feature names (optional)
            features_path = model_dir / f"{self.model_name}_features.json"
            if features_path.exists():
                with open(features_path) as f:
                    feature_data = json.load(f)
                    self.feature_names = feature_data["feature_names"]

            self.logger.info(f"Model loaded from {model_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """
        Get top N important features

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary of feature importance
        """
        if self.feature_importance is None:
            return {}

        # Sort by importance and return top N
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        return dict(sorted_features[:top_n])

    @staticmethod
    def _has_gpu() -> bool:
        """Check if GPU is available"""
        try:
            xgb.get_config()
            return True
        except:
            return False
