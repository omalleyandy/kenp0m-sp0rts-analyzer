"""
End-to-end integration tests for complete prediction pipeline
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kenp0m_sp0rts_analyzer.features.feature_engineer import (
    AdvancedFeatureEngineer,
)
from kenp0m_sp0rts_analyzer.models.xgboost_model import XGBoostModelWrapper
from sklearn.model_selection import train_test_split


@pytest.fixture
def kenpom_data():
    """Sample KenPom data"""
    teams = {
        "Duke": {
            "adj_em": 28.5,
            "adj_oe": 125.3,
            "adj_de": 96.8,
            "pace": 70.5,
            "fg_pct": 0.495,
            "three_pct": 0.385,
            "ft_rate": 0.285,
            "to_rate": 0.165,
            "pythag": 0.85,
            "luck": 0.02,
            "sos": 25.5,
            "elevation": 150,
        },
        "North Carolina": {
            "adj_em": 24.2,
            "adj_oe": 122.1,
            "adj_de": 97.9,
            "pace": 68.3,
            "fg_pct": 0.485,
            "three_pct": 0.375,
            "ft_rate": 0.280,
            "to_rate": 0.175,
            "pythag": 0.78,
            "luck": -0.01,
            "sos": 24.2,
            "elevation": 300,
        },
        "Kansas": {
            "adj_em": 26.8,
            "adj_oe": 123.9,
            "adj_de": 97.1,
            "pace": 69.2,
            "fg_pct": 0.490,
            "three_pct": 0.380,
            "ft_rate": 0.282,
            "to_rate": 0.170,
            "pythag": 0.82,
            "luck": 0.01,
            "sos": 24.8,
            "elevation": 800,
        },
        "Virginia": {
            "adj_em": 25.1,
            "adj_oe": 120.5,
            "adj_de": 95.4,
            "pace": 67.1,
            "fg_pct": 0.492,
            "three_pct": 0.378,
            "ft_rate": 0.275,
            "to_rate": 0.168,
            "pythag": 0.80,
            "luck": 0.00,
            "sos": 26.1,
            "elevation": 250,
        },
    }
    return teams


@pytest.fixture
def game_history():
    """Sample game history"""
    return {}


@pytest.fixture
def temp_model_dir():
    """Temporary model directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEndToEndPipeline:
    """End-to-end integration tests"""

    def test_full_prediction_pipeline(
        self, kenpom_data, game_history, temp_model_dir
    ):
        """Test complete pipeline from features to predictions"""

        # 1. Feature engineering
        feature_engineer = AdvancedFeatureEngineer(kenpom_data, game_history)

        games = [
            ("Duke", "North Carolina", "2025-12-20", -3.5, 152.5),
            ("Kansas", "Virginia", "2025-12-20", -5.0, 155.0),
            ("Duke", "Kansas", "2025-12-25", -2.0, 153.0),
        ]

        all_features = []
        all_targets = []

        # Create synthetic data
        np.random.seed(42)
        for home, away, date, spread, total in games:
            # Generate features
            game_features = feature_engineer.create_features(
                home_team=home,
                away_team=away,
                game_date=date,
                vegas_spread=spread,
                vegas_total=total,
            )

            all_features.append(game_features.to_array())
            # Synthetic target based on spread
            all_targets.append(-spread + np.random.randn() * 2)

        # Generate more synthetic data for training
        np.random.seed(42)
        for i in range(50):
            team_pairs = [
                ("Duke", "North Carolina"),
                ("Kansas", "Virginia"),
                ("Duke", "Kansas"),
                ("North Carolina", "Virginia"),
            ]
            home, away = team_pairs[i % len(team_pairs)]

            game_features = feature_engineer.create_features(
                home_team=home,
                away_team=away,
                game_date=f"2025-12-{15 + (i % 5)}",
                vegas_spread=-3.0 + np.random.randn() * 2,
                vegas_total=152.0 + np.random.randn() * 5,
            )

            all_features.append(game_features.to_array())
            all_targets.append(np.random.randn() * 5)

        X = np.array(all_features)
        y = np.array(all_targets)

        # 2. Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # 3. Train model
        model = XGBoostModelWrapper(
            model_name="ncaab_margin", model_type="margin"
        )
        metrics = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_engineer.FEATURE_NAMES,
        )

        assert metrics["val_mae"] < 10  # Reasonable accuracy

        # 4. Make predictions
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert np.all(np.isfinite(predictions))

        # 5. Save model
        model_path = model.save(temp_model_dir)
        assert model_path.exists()

        # 6. Load and verify
        loaded_model = XGBoostModelWrapper(
            model_name="ncaab_margin", model_type="margin"
        )
        assert loaded_model.load(temp_model_dir)

        predictions_loaded = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(
            predictions, predictions_loaded, decimal=5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])