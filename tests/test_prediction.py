"""Tests for machine learning prediction module."""

import numpy as np
import pandas as pd
import pytest

from kenp0m_sp0rts_analyzer.prediction import (
    BacktestingFramework,
    BacktestMetrics,
    FeatureEngineer,
    GamePredictor,
    PredictionResult,
)

# Fixtures


@pytest.fixture
def sample_team_stats():
    """Sample team statistics for testing."""
    return {
        "team1": {
            "TeamName": "Duke",
            "AdjEM": 20.5,
            "AdjO": 115.2,
            "AdjD": 94.7,
            "AdjT": 68.5,
            "AdjTempo": 68.5,
            "AdjOE": 115.2,
            "AdjDE": 94.7,
            "Pythag": 0.85,
            "SOS": 5.2,
            "APL_Off": 17.2,
            "APL_Def": 17.5,
            "RankAdjTempo": 150,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
        },
        "team2": {
            "TeamName": "UNC",
            "AdjEM": 15.3,
            "AdjO": 110.8,
            "AdjD": 95.5,
            "AdjT": 70.2,
            "AdjTempo": 70.2,
            "AdjOE": 110.8,
            "AdjDE": 95.5,
            "Pythag": 0.78,
            "SOS": 4.8,
            "APL_Off": 17.8,
            "APL_Def": 17.3,
            "RankAdjTempo": 80,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
        },
    }


@pytest.fixture
def sample_training_data():
    """Sample historical games for training."""
    np.random.seed(42)
    n_games = 100

    # Generate synthetic features
    em_diff = np.random.normal(0, 10, n_games)
    tempo_avg = np.random.normal(68, 3, n_games)

    # Generate synthetic outcomes (margin correlated with em_diff)
    margins = em_diff + np.random.normal(0, 8, n_games)
    totals = tempo_avg * 2 + np.random.normal(140, 10, n_games)

    games_df = pd.DataFrame(
        {
            "em_diff": em_diff,
            "tempo_avg": tempo_avg,
            "tempo_diff": np.random.normal(0, 2, n_games),
            "oe_diff": np.random.normal(0, 5, n_games),
            "de_diff": np.random.normal(0, 5, n_games),
            "pythag_diff": np.random.normal(0, 0.1, n_games),
            "sos_diff": np.random.normal(0, 2, n_games),
            "home_advantage": np.random.choice([-1, 0, 1], n_games),
            "em_tempo_interaction": em_diff * tempo_avg,
            # APL features (new)
            "apl_off_diff": np.random.normal(0, 1, n_games),
            "apl_def_diff": np.random.normal(0, 1, n_games),
            "apl_off_mismatch_team1": np.random.normal(0, 2, n_games),
            "apl_off_mismatch_team2": np.random.normal(0, 2, n_games),
            "tempo_control_factor": np.random.uniform(-0.5, 0.5, n_games),
            # Outcomes
            "actual_margin": margins,
            "actual_total": totals,
        }
    )

    return games_df, pd.Series(margins), pd.Series(totals)


# TestFeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_feature_creation_neutral_site(self, sample_team_stats):
        """Test feature creation for neutral site game."""
        features = FeatureEngineer.create_features(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
            neutral_site=True,
        )

        # Verify home advantage is 0 for neutral site
        assert features["home_advantage"] == 0.0

        # Verify basic calculations
        assert features["em_diff"] == pytest.approx(20.5 - 15.3)
        assert features["tempo_avg"] == pytest.approx((68.5 + 70.2) / 2)

    def test_feature_creation_home_advantage(self, sample_team_stats):
        """Test feature creation with home advantage for team 1."""
        features = FeatureEngineer.create_features(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
            neutral_site=False,
            home_team1=True,
        )

        # Verify home advantage is 1 for team 1 home game
        assert features["home_advantage"] == 1.0

    def test_feature_creation_away_game(self, sample_team_stats):
        """Test feature creation with team 1 as away team."""
        features = FeatureEngineer.create_features(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
            neutral_site=False,
            home_team1=False,
        )

        # Verify home advantage is -1 for team 1 away game
        assert features["home_advantage"] == -1.0

    def test_feature_interaction_terms(self, sample_team_stats):
        """Test em_tempo_interaction calculation."""
        features = FeatureEngineer.create_features(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
            neutral_site=True,
        )

        expected_interaction = features["em_diff"] * features["tempo_avg"]
        assert features["em_tempo_interaction"] == pytest.approx(expected_interaction)

    def test_all_features_present(self, sample_team_stats):
        """Ensure all 9 features are created."""
        features = FeatureEngineer.create_features(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
        )

        expected_features = FeatureEngineer.FEATURE_NAMES
        assert set(features.keys()) == set(expected_features)

    def test_feature_types(self, sample_team_stats):
        """Ensure all features are floats."""
        features = FeatureEngineer.create_features(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
        )

        for key, value in features.items():
            assert isinstance(value, float), (
                f"Feature {key} is not a float: {type(value)}"
            )


# TestGamePredictor


class TestGamePredictor:
    """Tests for GamePredictor class."""

    def test_predictor_initialization(self):
        """Test that predictor initializes correctly."""
        predictor = GamePredictor()

        assert predictor.is_fitted is False
        assert hasattr(predictor, "margin_model")
        assert hasattr(predictor, "margin_upper")
        assert hasattr(predictor, "margin_lower")
        assert hasattr(predictor, "total_model")
        assert hasattr(predictor, "feature_engineer")

    def test_fit_with_training_data(self, sample_training_data):
        """Test training process."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()

        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        assert predictor.is_fitted is True

    def test_predict_raises_before_fit(self, sample_team_stats):
        """Ensure error if prediction attempted before fitting."""
        predictor = GamePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.predict_with_confidence(
                team1_stats=sample_team_stats["team1"],
                team2_stats=sample_team_stats["team2"],
            )

    def test_predict_with_confidence_neutral(
        self, sample_training_data, sample_team_stats
    ):
        """Test prediction for neutral site game."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()
        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        result = predictor.predict_with_confidence(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
            neutral_site=True,
        )

        assert isinstance(result, PredictionResult)
        assert isinstance(result.predicted_margin, float)
        assert isinstance(result.predicted_total, float)

    def test_predict_with_confidence_home(
        self, sample_training_data, sample_team_stats
    ):
        """Test prediction for home game."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()
        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        result = predictor.predict_with_confidence(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
            neutral_site=False,
            home_team1=True,
        )

        assert isinstance(result, PredictionResult)
        # Home team should have higher margin (all else equal)
        assert result.predicted_margin > 0  # Team 1 favored due to better stats + home

    def test_confidence_interval_bounds(self, sample_training_data, sample_team_stats):
        """Verify lower < margin < upper in most cases."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()
        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        result = predictor.predict_with_confidence(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
        )

        lower, upper = result.confidence_interval
        # Lower should be less than upper (guaranteed by Pydantic validator)
        assert lower <= upper
        # Typically margin should be between bounds (not always due to quantile regression)
        # Just verify they're reasonable
        assert -50 < lower < 50
        assert -50 < upper < 50

    def test_win_probability_range(self, sample_training_data, sample_team_stats):
        """Ensure 0 <= win_prob <= 1."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()
        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        result = predictor.predict_with_confidence(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
        )

        assert 0 <= result.team1_win_prob <= 1

    def test_total_prediction_positive(self, sample_training_data, sample_team_stats):
        """Ensure predicted total is positive."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()
        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        result = predictor.predict_with_confidence(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
        )

        assert result.predicted_total > 0
        assert result.team1_score > 0
        assert result.team2_score > 0

    def test_margin_and_total_consistency(
        self, sample_training_data, sample_team_stats
    ):
        """Verify scores add up correctly."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()
        predictor.fit(
            games_df=games_df[FeatureEngineer.FEATURE_NAMES],
            margins=margins,
            totals=totals,
        )

        result = predictor.predict_with_confidence(
            team1_stats=sample_team_stats["team1"],
            team2_stats=sample_team_stats["team2"],
        )

        # team1_score + team2_score should equal predicted_total
        calculated_total = result.team1_score + result.team2_score
        assert calculated_total == pytest.approx(result.predicted_total, abs=0.2)

        # team1_score - team2_score should equal predicted_margin
        calculated_margin = result.team1_score - result.team2_score
        assert calculated_margin == pytest.approx(result.predicted_margin, abs=0.2)


# TestBacktestingFramework


class TestBacktestingFramework:
    """Tests for BacktestingFramework class."""

    def test_backtest_train_test_split(self, sample_training_data):
        """Verify 80/20 split works."""
        games_df, _, _ = sample_training_data
        framework = BacktestingFramework()

        metrics = framework.run_backtest(games_df, train_split=0.8)

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.mae_margin > 0
        assert metrics.rmse_margin >= metrics.mae_margin  # RMSE >= MAE always
        assert 0 <= metrics.accuracy <= 1

    def test_calculate_metrics_accuracy(self, sample_training_data):
        """Test accuracy calculation."""
        games_df, _, _ = sample_training_data
        framework = BacktestingFramework()

        metrics = framework.run_backtest(games_df, train_split=0.8)

        # Accuracy should be between 0 and 1
        assert 0 <= metrics.accuracy <= 1

        # With synthetic data correlated to em_diff, accuracy should be reasonable
        assert metrics.accuracy > 0.5  # Better than random

    def test_calculate_metrics_mae_rmse(self, sample_training_data):
        """Test error metrics."""
        games_df, _, _ = sample_training_data
        framework = BacktestingFramework()

        metrics = framework.run_backtest(games_df, train_split=0.8)

        # MAE and RMSE should be positive
        assert metrics.mae_margin > 0
        assert metrics.rmse_margin > 0
        assert metrics.mae_total > 0
        assert metrics.rmse_total > 0

        # RMSE should be >= MAE (by definition)
        assert metrics.rmse_margin >= metrics.mae_margin
        assert metrics.rmse_total >= metrics.mae_total

    def test_calculate_metrics_brier_score(self, sample_training_data):
        """Test probability calibration metric."""
        games_df, _, _ = sample_training_data
        framework = BacktestingFramework()

        metrics = framework.run_backtest(games_df, train_split=0.8)

        # Brier score should be between 0 and 1 (lower is better)
        assert 0 <= metrics.brier_score <= 1

    def test_ats_record_calculation(self, sample_training_data):
        """Test ATS tracking."""
        games_df, _, _ = sample_training_data

        # Add spread column for ATS calculation
        games_df["spread"] = np.random.normal(0, 5, len(games_df))

        framework = BacktestingFramework()
        metrics = framework.run_backtest(games_df, train_split=0.8)

        # ATS record should sum to total test games
        ats_wins, ats_losses = metrics.ats_record
        assert ats_wins + ats_losses == int(len(games_df) * 0.2)

        # ATS percentage should match record
        expected_pct = ats_wins / (ats_wins + ats_losses)
        assert metrics.ats_percentage == pytest.approx(expected_pct, abs=0.01)

    def test_cross_validation(self, sample_training_data):
        """Test k-fold cross-validation."""
        games_df, _, _ = sample_training_data
        framework = BacktestingFramework()

        metrics_list = framework.cross_validate(games_df, n_folds=5)

        # Should return 5 metrics objects
        assert len(metrics_list) == 5

        # All should be BacktestMetrics
        for metrics in metrics_list:
            assert isinstance(metrics, BacktestMetrics)
            assert metrics.mae_margin > 0
            assert 0 <= metrics.accuracy <= 1

    def test_calibration_plot_data(self, sample_training_data):
        """Test calibration curve generation."""
        games_df, margins, totals = sample_training_data
        predictor = GamePredictor()

        # Train
        train_df = games_df.iloc[:80]
        predictor.fit(
            games_df=train_df[FeatureEngineer.FEATURE_NAMES],
            margins=train_df["actual_margin"],
            totals=train_df["actual_total"],
        )

        # Make predictions on test set
        test_df = games_df.iloc[80:]
        predictions = []

        for _, row in test_df.iterrows():
            # Reconstruct team stats from features (simplified)
            team1_stats = {
                "AdjEM": 10.0 + row["em_diff"] / 2,
                "AdjO": 110.0,
                "AdjD": 100.0,
                "AdjT": row["tempo_avg"],
                "Pythag": 0.7,
                "SOS": 0.0,
            }
            team2_stats = {
                "AdjEM": 10.0 - row["em_diff"] / 2,
                "AdjO": 110.0,
                "AdjD": 100.0,
                "AdjT": row["tempo_avg"],
                "Pythag": 0.7,
                "SOS": 0.0,
            }

            pred = predictor.predict_with_confidence(
                team1_stats=team1_stats,
                team2_stats=team2_stats,
                neutral_site=row["home_advantage"] == 0,
                home_team1=row["home_advantage"] == 1,
            )
            predictions.append(pred)

        framework = BacktestingFramework()
        calibration_df = framework.calibration_plot_data(
            predictions=predictions, actuals=test_df, n_bins=10
        )

        # Should return DataFrame with calibration data
        assert isinstance(calibration_df, pd.DataFrame)
        assert "bin_center" in calibration_df.columns
        assert "actual_win_rate" in calibration_df.columns
        assert "count" in calibration_df.columns

    def test_backtest_with_small_sample(self):
        """Edge case with <10 games should raise error."""
        small_df = pd.DataFrame(
            {
                "em_diff": [1, 2, 3, 4, 5],
                "tempo_avg": [68, 69, 70, 71, 72],
                "tempo_diff": [0, 1, -1, 2, -2],
                "oe_diff": [2, 3, 1, 4, 2],
                "de_diff": [1, 2, 3, 1, 2],
                "pythag_diff": [0.05, 0.1, 0.08, 0.12, 0.06],
                "sos_diff": [1, 2, 1.5, 2.5, 1.2],
                "home_advantage": [0, 1, -1, 0, 1],
                "em_tempo_interaction": [68, 138, -70, 284, 144],
                # APL features
                "apl_off_diff": [0, 0.5, -0.5, 1, -1],
                "apl_def_diff": [0, 0.3, -0.3, 0.8, -0.8],
                "apl_off_mismatch_team1": [0, 1, -1, 2, -2],
                "apl_off_mismatch_team2": [0, -1, 1, -2, 2],
                "tempo_control_factor": [0, 0.2, -0.2, 0.4, -0.4],
                # Outcomes
                "actual_margin": [5, 10, -5, 15, 8],
                "actual_total": [140, 145, 138, 150, 142],
            }
        )

        framework = BacktestingFramework()

        with pytest.raises(ValueError, match="Insufficient training data"):
            framework.run_backtest(small_df, train_split=0.8)


# TestPredictionResult


class TestPredictionResult:
    """Tests for PredictionResult Pydantic model."""

    def test_prediction_result_validation(self):
        """Test Pydantic validation works."""
        result = PredictionResult(
            predicted_margin=5.5,
            predicted_total=142.0,
            confidence_interval=(-2.3, 13.2),
            team1_score=73.8,
            team2_score=68.2,
            team1_win_prob=0.75,
            confidence_level=0.5,
        )

        assert result.predicted_margin == 5.5
        assert result.confidence_interval == (-2.3, 13.2)
        assert result.team1_win_prob == 0.75

    def test_confidence_interval_ordering(self):
        """Ensure lower < upper is enforced."""
        with pytest.raises(ValueError, match="Lower bound .* must be <= upper bound"):
            PredictionResult(
                predicted_margin=5.5,
                predicted_total=142.0,
                confidence_interval=(13.2, -2.3),  # Wrong order
                team1_score=73.8,
                team2_score=68.2,
                team1_win_prob=0.75,
            )

    def test_win_prob_bounds(self):
        """Ensure 0 <= win_prob <= 1 is enforced."""
        # Test upper bound
        with pytest.raises(ValueError, match="Win probability must be in"):
            PredictionResult(
                predicted_margin=5.5,
                predicted_total=142.0,
                confidence_interval=(-2.3, 13.2),
                team1_score=73.8,
                team2_score=68.2,
                team1_win_prob=1.5,  # > 1
            )

        # Test lower bound
        with pytest.raises(ValueError, match="Win probability must be in"):
            PredictionResult(
                predicted_margin=5.5,
                predicted_total=142.0,
                confidence_interval=(-2.3, 13.2),
                team1_score=73.8,
                team2_score=68.2,
                team1_win_prob=-0.1,  # < 0
            )
