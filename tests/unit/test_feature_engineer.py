"""
Unit tests for feature engineering module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kenp0m_sp0rts_analyzer.features.feature_engineer import (
    AdvancedFeatureEngineer,
    GameFeatures,
)
from kenp0m_sp0rts_analyzer.utils.exceptions import FeatureEngineeringError


@pytest.fixture
def sample_kenpom_data():
    """Sample KenPom data for testing"""
    return {
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
    }


@pytest.fixture
def sample_game_history():
    """Sample game history for testing"""
    return {
        "Duke": [
            {
                "date": "2025-12-15",
                "opponent": "NC State",
                "pace": 70.2,
                "oe": 125.1,
                "de": 96.9,
            },
            {
                "date": "2025-12-12",
                "opponent": "Purdue",
                "pace": 70.8,
                "oe": 125.5,
                "de": 96.5,
            },
        ],
        "North Carolina": [
            {
                "date": "2025-12-14",
                "opponent": "Wake Forest",
                "pace": 68.1,
                "oe": 122.0,
                "de": 98.0,
            },
            {
                "date": "2025-12-11",
                "opponent": "Virginia",
                "pace": 68.5,
                "oe": 122.2,
                "de": 97.8,
            },
        ],
    }


@pytest.fixture
def feature_engineer(sample_kenpom_data, sample_game_history):
    """Initialize feature engineer"""
    return AdvancedFeatureEngineer(sample_kenpom_data, sample_game_history)


class TestFeatureEngineer:
    """Test suite for AdvancedFeatureEngineer"""

    def test_feature_engineer_initialization(self, feature_engineer):
        """Test that feature engineer initializes correctly"""
        assert feature_engineer is not None
        assert len(feature_engineer.FEATURE_NAMES) == 40
        assert "adj_em_diff" in feature_engineer.FEATURE_NAMES

    def test_create_features_basic(self, feature_engineer):
        """Test basic feature creation"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        assert isinstance(game_features, GameFeatures)
        assert game_features.game_id == "2025-12-20_North Carolina_Duke"
        assert len(game_features.features) == 40
        assert len(game_features.feature_names) == 40

    def test_feature_values_efficiency_metrics(self, feature_engineer):
        """Test efficiency metric calculations"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        # Duke should have higher AdjEM than UNC
        assert game_features.features["adj_em_diff"] > 0
        assert game_features.features["adj_oe_diff"] > 0
        assert (
            game_features.features["adj_de_diff"] < 0
        )  # Duke has worse defense

    def test_feature_values_four_factors(self, feature_engineer):
        """Test four factors calculations"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        # FG% should favor Duke
        assert game_features.features["fg_pct_diff"] > 0
        # TO rate should favor Duke (lower is better)
        assert game_features.features["to_rate_diff"] > 0

    def test_feature_values_tempo(self, feature_engineer):
        """Test tempo feature calculations"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        # Tempo average should be between both teams
        expected_avg = (70.5 + 68.3) / 2
        assert game_features.features["tempo_avg"] == pytest.approx(
            expected_avg
        )

        # Duke plays faster
        assert game_features.features["tempo_diff"] > 0

    def test_feature_values_vegas_market(self, feature_engineer):
        """Test Vegas market signal features"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
            opening_spread=-3.0,
            opening_total=151.0,
        )

        # Vegas features should be set correctly
        assert game_features.features["vegas_spread"] == -3.5
        assert game_features.features["vegas_total"] == 152.5
        assert (
            game_features.features["spread_direction"] < 0
        )  # Negative spread

    def test_feature_values_interactions(self, feature_engineer):
        """Test interaction term calculations"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        # Interaction terms should be product of components
        expected_em_pace = (28.5 - 24.2) * ((70.5 + 68.3) / 2)
        assert game_features.features["em_pace_interaction"] == pytest.approx(
            expected_em_pace
        )

    def test_feature_array_conversion(self, feature_engineer):
        """Test conversion to numpy array"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        feature_array = game_features.to_array()
        assert isinstance(feature_array, np.ndarray)
        assert len(feature_array) == 40
        assert not np.any(np.isnan(feature_array))

    def test_invalid_team_home(self, feature_engineer):
        """Test error handling for invalid home team"""
        with pytest.raises(FeatureEngineeringError):
            feature_engineer.create_features(
                home_team="Unknown Team",
                away_team="North Carolina",
                game_date="2025-12-20",
                vegas_spread=-3.5,
                vegas_total=152.5,
            )

    def test_invalid_team_away(self, feature_engineer):
        """Test error handling for invalid away team"""
        with pytest.raises(FeatureEngineeringError):
            feature_engineer.create_features(
                home_team="Duke",
                away_team="Unknown Team",
                game_date="2025-12-20",
                vegas_spread=-3.5,
                vegas_total=152.5,
            )

    def test_feature_count(self, feature_engineer):
        """Test that all 40 features are created"""
        game_features = feature_engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            game_date="2025-12-20",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )

        assert len(game_features.features) == 40
        assert len(game_features.feature_names) == 40

        # Ensure no NaN values
        for feature_name, value in game_features.features.items():
            assert not np.isnan(value), f"NaN found in {feature_name}"


class TestGameFeatures:
    """Test suite for GameFeatures dataclass"""

    def test_game_features_creation(self):
        """Test GameFeatures object creation"""
        features = {"adj_em_diff": 4.3, "adj_oe_diff": 3.2}
        game_features = GameFeatures(
            game_id="test_game",
            features=features,
            feature_names=["adj_em_diff", "adj_oe_diff"],
            timestamp=str(pd.Timestamp.now()),
        )

        assert game_features.game_id == "test_game"
        assert game_features.features["adj_em_diff"] == 4.3

    def test_game_features_to_array(self):
        """Test conversion to array preserves feature order"""
        features = {
            "feature_a": 1.0,
            "feature_b": 2.0,
            "feature_c": 3.0,
        }
        game_features = GameFeatures(
            game_id="test",
            features=features,
            feature_names=["feature_a", "feature_b", "feature_c"],
            timestamp=str(pd.Timestamp.now()),
        )

        array = game_features.to_array()
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(array, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
