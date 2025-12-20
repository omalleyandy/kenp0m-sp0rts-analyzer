"""
Configuration management for the XGBoost prediction system.
Supports multiple environments: dev, test, prod
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """XGBoost model configuration"""

    model_type: str = "xgboost"
    max_depth: int = 7
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 1.0
    reg_alpha: float = 0.5
    reg_lambda: float = 1.0
    num_boost_rounds: int = 500
    early_stopping_rounds: int = 20


@dataclass
class DataConfig:
    """Data loading and processing configuration"""

    test_size: float = 0.2
    random_state: int = 42
    val_size: float = 0.1  # Validation set during training
    min_games: int = 100  # Minimum games for training


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""

    normalize_features: bool = True
    handle_missing: str = "mean"  # "mean", "median", "drop"
    outlier_threshold: float = 3.0  # Standard deviations
    interaction_terms: bool = True


@dataclass
class PredictionConfig:
    """Prediction configuration"""

    min_edge: float = 1.5  # Minimum spread edge
    min_roi: float = 0.05  # Minimum ROI percentage
    confidence_threshold: float = 0.55  # Minimum confidence
    use_moving_average: bool = True
    ma_window: int = 5


@dataclass
class DatabaseConfig:
    """Database configuration"""

    db_type: str = "sqlite"
    db_path: str = "data/predictions.db"
    echo_sql: bool = False  # SQL logging
    pool_size: int = 5
    max_overflow: int = 10


class Config:
    """Main configuration class"""

    def __init__(self, env: str = "development"):
        self.env = env
        self.root_dir = Path(__file__).parent.parent.parent
        self.src_dir = self.root_dir / "src"
        self.data_dir = self.root_dir / "data"
        self.models_dir = self.root_dir / "models"
        self.logs_dir = self.root_dir / "logs"
        self.config_dir = self.root_dir / "config"
        self.tests_dir = self.root_dir / "tests"

        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load environment-specific config
        self._load_environment_config()

    def _load_environment_config(self):
        """Load environment-specific YAML configuration"""
        config_file = self.config_dir / f"{self.env}.yaml"

        if config_file.exists():
            with open(config_file) as f:
                env_config = yaml.safe_load(f)
                self._apply_config(env_config)

    def _apply_config(self, config_dict: dict):
        """Apply configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # Model configuration
    model_config = ModelConfig()

    # Data configuration
    data_config = DataConfig()

    # Feature configuration
    feature_config = FeatureConfig()

    # Prediction configuration
    prediction_config = PredictionConfig()

    # Database configuration
    database_config = DatabaseConfig()

    @property
    def is_development(self) -> bool:
        return self.env == "development"

    @property
    def is_testing(self) -> bool:
        return self.env == "testing"

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# Global config instance
config = Config(env=os.getenv("APP_ENV", "development"))
