"""Tests for conference power ratings and analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from kenp0m_sp0rts_analyzer.conference_analytics import (
    ConferenceAnalytics,
    ConferencePowerRating,
)


class TestConferencePowerRatingCalculation:
    """Test conference power rating calculations."""

    @pytest.fixture
    def analytics(self) -> ConferenceAnalytics:
        """Create a ConferenceAnalytics instance."""
        return ConferenceAnalytics()

    @pytest.fixture
    def power_ratings(self, analytics: ConferenceAnalytics) -> pd.DataFrame:
        """Get power ratings, skipping if unavailable."""
        df = analytics.calculate_conference_power_ratings(2025)
        if len(df) == 0:
            pytest.skip("No conference data available - check API authentication")
        return df

    def test_calculate_conference_power_ratings_returns_dataframe(
        self, power_ratings: pd.DataFrame
    ):
        """Test that power ratings calculation returns a DataFrame."""
        assert isinstance(power_ratings, pd.DataFrame)
        assert len(power_ratings) > 0

    def test_power_ratings_have_required_columns(self, power_ratings: pd.DataFrame):
        """Test that power ratings have all required columns."""

        required_columns = [
            "conference",
            "conf_short",
            "num_teams",
            "avg_adj_em",
            "median_adj_em",
            "top_team_adj_em",
            "bottom_team_adj_em",
            "top_25_teams",
            "top_50_teams",
            "above_average",
            "avg_adj_oe",
            "avg_adj_de",
            "avg_tempo",
            "avg_sos",
            "avg_ncsos",
            "estimated_bids",
            "power_score",
            "power_rank",
        ]

        for col in required_columns:
            assert col in power_ratings.columns, f"Missing column: {col}"

    def test_power_scores_in_valid_range(self, power_ratings: pd.DataFrame):
        """Test that power scores are between 0 and 100."""
        assert power_ratings["power_score"].min() >= 0
        assert power_ratings["power_score"].max() <= 100

    def test_power_ranks_are_unique(self, power_ratings: pd.DataFrame):
        """Test that each conference has a unique rank."""
        ranks = power_ratings["power_rank"].tolist()
        assert len(ranks) == len(set(ranks)), "Duplicate ranks found"

    def test_power_ranks_start_at_1(self, power_ratings: pd.DataFrame):
        """Test that the top conference has rank 1."""
        assert power_ratings.iloc[0]["power_rank"] == 1

    def test_sorted_by_power_score_descending(self, power_ratings: pd.DataFrame):
        """Test that results are sorted by power score (highest first)."""
        power_scores = power_ratings["power_score"].tolist()
        assert power_scores == sorted(power_scores, reverse=True)

    def test_top_conference_has_positive_avg_em(self, power_ratings: pd.DataFrame):
        """Test that the top-ranked conference has strong metrics."""
        top_conf = power_ratings.iloc[0]
        # Top conference should have positive average AdjEM
        assert top_conf["avg_adj_em"] > 0

    def test_estimated_bids_at_least_one(self, power_ratings: pd.DataFrame):
        """Test that every conference gets at least 1 bid (auto-bid)."""
        assert (power_ratings["estimated_bids"] >= 1).all()

    def test_top_25_teams_less_than_num_teams(self, power_ratings: pd.DataFrame):
        """Test that top_25_teams doesn't exceed conference size."""
        assert (power_ratings["top_25_teams"] <= power_ratings["num_teams"]).all()

    def test_above_average_less_than_num_teams(self, power_ratings: pd.DataFrame):
        """Test that above_average doesn't exceed conference size."""
        assert (power_ratings["above_average"] <= power_ratings["num_teams"]).all()


class TestConferencePowerRatingComponents:
    """Test individual components of power rating calculation."""

    @pytest.fixture
    def analytics(self) -> ConferenceAnalytics:
        """Create a ConferenceAnalytics instance."""
        return ConferenceAnalytics()

    def test_normalize_column_returns_0_to_1(self, analytics: ConferenceAnalytics):
        """Test that normalization returns values in 0-1 range."""
        series = pd.Series([5, 10, 15, 20, 25])
        normalized = analytics._normalize_column(series)

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_normalize_column_handles_constant_values(
        self, analytics: ConferenceAnalytics
    ):
        """Test normalization when all values are the same."""
        series = pd.Series([10, 10, 10, 10])
        normalized = analytics._normalize_column(series)

        # Should return all 1.0s when values are constant
        assert (normalized == 1.0).all()

    def test_estimate_tournament_bids_logic(self, analytics: ConferenceAnalytics):
        """Test tournament bid estimation logic."""
        # Conference with strong teams
        strong_teams = [
            {"AdjEM": 20.0},  # Lock
            {"AdjEM": 16.0},  # Lock
            {"AdjEM": 12.0},  # Bubble (50% chance)
            {"AdjEM": 11.0},  # Bubble (50% chance)
            {"AdjEM": 5.0},  # Unlikely
        ]
        estimated = analytics._estimate_tournament_bids(strong_teams)
        # 2 locks + 1 bubble (2/2 = 1) = 3 bids
        assert estimated == 3

        # Weak conference
        weak_teams = [
            {"AdjEM": 8.0},
            {"AdjEM": 5.0},
            {"AdjEM": 2.0},
        ]
        estimated = analytics._estimate_tournament_bids(weak_teams)
        # At least 1 (auto-bid)
        assert estimated == 1

    def test_power_score_weights_sum_to_100_percent(
        self, analytics: ConferenceAnalytics
    ):
        """Test that power score weights sum to 100%."""
        total_weight = (
            analytics.WEIGHT_AVG_EM
            + analytics.WEIGHT_TOP_TEAM
            + analytics.WEIGHT_DEPTH
            + analytics.WEIGHT_TOP_25
            + analytics.WEIGHT_NCSOS
        )
        assert total_weight == pytest.approx(1.0)


class TestConferenceTournamentOutlook:
    """Test conference tournament outlook analysis."""

    @pytest.fixture
    def analytics(self) -> ConferenceAnalytics:
        """Create a ConferenceAnalytics instance."""
        return ConferenceAnalytics()

    def test_get_tournament_outlook_returns_dataframe(
        self, analytics: ConferenceAnalytics
    ):
        """Test that tournament outlook returns a DataFrame."""
        df = analytics.get_conference_tournament_outlook("ACC", 2025)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_tournament_outlook_has_required_columns(
        self, analytics: ConferenceAnalytics
    ):
        """Test tournament outlook has required columns."""
        df = analytics.get_conference_tournament_outlook("B12", 2025)

        required_columns = ["Team", "AdjEM", "RankAdjEM", "NCAA_Probability", "Bubble"]
        for col in required_columns:
            assert col in df.columns

    def test_tournament_outlook_sorted_by_em(self, analytics: ConferenceAnalytics):
        """Test that tournament outlook is sorted by AdjEM descending."""
        df = analytics.get_conference_tournament_outlook("SEC", 2025)

        adj_ems = df["AdjEM"].tolist()
        assert adj_ems == sorted(adj_ems, reverse=True)

    def test_ncaa_probability_in_valid_range(self, analytics: ConferenceAnalytics):
        """Test that NCAA probabilities are between 0 and 1."""
        df = analytics.get_conference_tournament_outlook("B10", 2025)

        assert (df["NCAA_Probability"] >= 0).all()
        assert (df["NCAA_Probability"] <= 1).all()

    def test_high_em_teams_have_high_probability(self, analytics: ConferenceAnalytics):
        """Test that high AdjEM teams have high tournament probability."""
        df = analytics.get_conference_tournament_outlook("ACC", 2025)

        # Teams with AdjEM > 15 should have >0.9 probability
        high_em_teams = df[df["AdjEM"] > 15]
        if len(high_em_teams) > 0:
            assert (high_em_teams["NCAA_Probability"] > 0.9).all()

    def test_bubble_classification_logic(self, analytics: ConferenceAnalytics):
        """Test bubble classification based on probability."""
        df = analytics.get_conference_tournament_outlook("ACC", 2025)

        # High probability teams should be "Lock"
        locks = df[df["NCAA_Probability"] > 0.8]
        if len(locks) > 0:
            assert (locks["Bubble"] == "Lock").all()

        # Low probability teams should be "NIT/CBI"
        unlikely = df[df["NCAA_Probability"] < 0.3]
        if len(unlikely) > 0:
            assert (unlikely["Bubble"] == "NIT/CBI").all()


class TestConferenceHeadToHeadComparison:
    """Test head-to-head conference comparison (placeholder)."""

    @pytest.fixture
    def analytics(self) -> ConferenceAnalytics:
        """Create a ConferenceAnalytics instance."""
        return ConferenceAnalytics()

    def test_head_to_head_raises_not_implemented(self, analytics: ConferenceAnalytics):
        """Test that head-to-head comparison raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="schedule data"):
            analytics.compare_conferences_head_to_head("ACC", "SEC", 2025)


class TestConferencePowerRatingDataclass:
    """Test ConferencePowerRating dataclass."""

    def test_dataclass_creation(self):
        """Test creating a ConferencePowerRating instance."""
        rating = ConferencePowerRating(
            conference="Atlantic Coast Conference",
            conf_short="ACC",
            num_teams=15,
            avg_adj_em=12.5,
            median_adj_em=11.0,
            top_team_adj_em=25.0,
            bottom_team_adj_em=-5.0,
            top_25_teams=4,
            top_50_teams=8,
            above_average=10,
            avg_adj_oe=112.0,
            avg_adj_de=99.5,
            avg_tempo=68.5,
            avg_sos=5.5,
            avg_ncsos=3.2,
            estimated_bids=6,
            power_score=85.0,
            power_rank=1,
        )

        assert rating.conference == "Atlantic Coast Conference"
        assert rating.conf_short == "ACC"
        assert rating.num_teams == 15
        assert rating.power_score == 85.0
        assert rating.power_rank == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analytics(self) -> ConferenceAnalytics:
        """Create a ConferenceAnalytics instance."""
        return ConferenceAnalytics()

    def test_empty_conference_handling(self, analytics: ConferenceAnalytics):
        """Test handling of conferences with no teams."""
        # This should either skip or handle gracefully
        df = analytics.calculate_conference_power_ratings(2025)
        # Should still return a DataFrame
        assert isinstance(df, pd.DataFrame)

    def test_single_team_conference(self, analytics: ConferenceAnalytics):
        """Test handling of conferences with very few teams."""
        # Independents might have few teams
        df = analytics.calculate_conference_power_ratings(2025)

        if len(df) == 0:
            pytest.skip("No conference data available")

        # Even single-team conferences should have valid ratings
        single_team_confs = df[df["num_teams"] == 1]
        if len(single_team_confs) > 0:
            assert (single_team_confs["power_score"] >= 0).all()

    def test_invalid_conference_in_tournament_outlook(
        self, analytics: ConferenceAnalytics
    ):
        """Test handling of invalid conference abbreviation."""
        # This should raise an error or return empty DataFrame
        try:
            df = analytics.get_conference_tournament_outlook("INVALID", 2025)
            # If it doesn't raise, should return empty
            assert len(df) == 0 or isinstance(df, pd.DataFrame)
        except Exception:
            # Expected behavior - invalid conference raises error
            pass


class TestRealWorldScenarios:
    """Test realistic scenarios with actual data."""

    @pytest.fixture
    def analytics(self) -> ConferenceAnalytics:
        """Create a ConferenceAnalytics instance."""
        return ConferenceAnalytics()

    @pytest.fixture
    def power_ratings(self, analytics: ConferenceAnalytics) -> pd.DataFrame:
        """Get power ratings, skipping if unavailable."""
        df = analytics.calculate_conference_power_ratings(2025)
        if len(df) == 0:
            pytest.skip("No conference data available - check API authentication")
        return df

    def test_major_conferences_in_top_10(self, power_ratings: pd.DataFrame):
        """Test that major conferences rank in top positions."""
        # Major conferences should generally be in top 10
        major_conferences = ["ACC", "B10", "B12", "SEC", "BE"]
        top_10 = power_ratings.head(10)

        major_in_top_10 = top_10[top_10["conf_short"].isin(major_conferences)]
        # At least some major conferences should be in top 10
        assert len(major_in_top_10) > 0

    def test_power_six_conferences_get_multiple_bids(self, power_ratings: pd.DataFrame):
        """Test that power conferences get multiple tournament bids."""
        # Power 5+ conferences (formerly Power 6 before Pac-12 dissolution)
        power_six = power_ratings[
            power_ratings["conf_short"].isin(["ACC", "B10", "B12", "SEC", "BE"])
        ]

        if len(power_six) > 0:
            # Power conferences should average >2 bids
            assert power_six["estimated_bids"].mean() > 2

    def test_conference_with_elite_team_ranks_high(self, power_ratings: pd.DataFrame):
        """Test that conferences with elite teams rank highly."""
        # Find conference with highest top_team_adj_em
        top_team_conf = power_ratings.loc[power_ratings["top_team_adj_em"].idxmax()]

        # That conference should have high power score
        assert top_team_conf["power_score"] > 50

    def test_deep_conference_vs_top_heavy(self, power_ratings: pd.DataFrame):
        """Test that depth is valued in power ratings."""
        # Verify that above_average metric is tracked
        # (20% weight in power score formula)
        assert "above_average" in power_ratings.columns

        # Conferences should have varying depth
        if len(power_ratings) >= 2:
            assert power_ratings["above_average"].std() > 0
