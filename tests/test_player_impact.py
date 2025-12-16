"""Tests for player impact modeling and injury analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from kenp0m_sp0rts_analyzer.player_impact import (
    InjuryImpact,
    PlayerImpactModel,
    PlayerValue,
)


class TestPlayerValueCalculation:
    """Test player value calculation across different player types."""

    @pytest.fixture
    def model(self) -> PlayerImpactModel:
        """Create a PlayerImpactModel instance."""
        return PlayerImpactModel()

    @pytest.fixture
    def team_stats(self) -> dict:
        """Duke's 2025 team statistics."""
        return {
            "TeamName": "Duke",
            "AdjEM": 25.0,
            "AdjOE": 120.0,
            "AdjDE": 95.0,
            "AdjTempo": 70.0,
        }

    def test_star_player_value(self, model: PlayerImpactModel, team_stats: dict):
        """Test value calculation for a star player."""
        # Star player: high usage, elite efficiency
        star_stats = {
            "Player": "Star Player",
            "Team": "Duke",
            "Poss%": 25.0,  # High usage
            "ORtg": 130.0,  # Elite efficiency (10 pts above team)
            "eFG%": 62.5,
            "TS%": 65.0,
            "TRB%": 15.0,
            "STL%": 3.0,
            "BLK%": 5.0,
        }

        value = model.calculate_player_value(star_stats, team_stats)

        assert value.player_name == "Star Player"
        assert value.team == "Duke"
        assert value.possession_pct == 25.0
        assert value.offensive_rating == 130.0

        # Star player should have high estimated value
        assert value.estimated_value > 2.0
        assert value.offensive_contribution > 2.0  # (130 - 120) * 0.25 = 2.5
        assert value.defensive_contribution > 0.0
        assert value.value_over_replacement > 0.0

    def test_role_player_value(self, model: PlayerImpactModel, team_stats: dict):
        """Test value calculation for a role player."""
        # Role player: moderate usage, average efficiency
        role_stats = {
            "Player": "Role Player",
            "Team": "Duke",
            "Poss%": 15.0,
            "ORtg": 118.0,  # Slightly below team
            "eFG%": 55.0,
            "TS%": 58.0,
            "TRB%": 10.0,
            "STL%": 2.0,
            "BLK%": 1.0,
        }

        value = model.calculate_player_value(role_stats, team_stats)

        assert value.player_name == "Role Player"
        # Role player: lower value than star
        assert -1.0 < value.estimated_value < 2.0
        # Offensive contribution can be negative (below team average)
        assert value.offensive_contribution < 0.5

    def test_bench_player_value(self, model: PlayerImpactModel, team_stats: dict):
        """Test value calculation for a bench player."""
        bench_stats = {
            "Player": "Bench Player",
            "Team": "Duke",
            "Poss%": 8.0,  # Low usage
            "ORtg": 110.0,  # Below team
            "eFG%": 48.0,
            "TS%": 52.0,
            "TRB%": 6.0,
            "STL%": 1.0,
            "BLK%": 0.5,
        }

        value = model.calculate_player_value(bench_stats, team_stats)

        assert value.player_name == "Bench Player"
        # Bench player: minimal value
        assert value.estimated_value < 1.0
        assert value.possession_pct == 8.0

    def test_minutes_estimation(self, model: PlayerImpactModel, team_stats: dict):
        """Test minutes percentage estimation from possession usage."""
        high_usage_stats = {
            "Player": "High Usage",
            "Team": "Duke",
            "Poss%": 30.0,
            "ORtg": 120.0,
            "eFG%": 55.0,
            "TS%": 58.0,
            "TRB%": 10.0,
        }

        value = model.calculate_player_value(high_usage_stats, team_stats)

        # Minutes should be estimated as Poss% * 1.2, capped at 100
        expected_minutes = min(100.0, 30.0 * 1.2)
        assert value.minutes_pct == expected_minutes

    def test_replacement_level_calculation(
        self, model: PlayerImpactModel, team_stats: dict
    ):
        """Test replacement level and VOR calculation."""
        player_stats = {
            "Player": "Test Player",
            "Team": "Duke",
            "Poss%": 20.0,
            "ORtg": 125.0,
            "eFG%": 60.0,
            "TS%": 62.0,
            "TRB%": 12.0,
            "STL%": 2.5,
            "BLK%": 3.0,
        }

        value = model.calculate_player_value(player_stats, team_stats)

        # Replacement level should be 70% of estimated value
        assert value.replacement_level == pytest.approx(
            value.estimated_value * 0.70, rel=1e-5
        )
        # VOR = value - replacement
        assert value.value_over_replacement == pytest.approx(
            value.estimated_value - value.replacement_level, rel=1e-5
        )


class TestInjuryImpactEstimation:
    """Test injury impact estimation with different severities."""

    @pytest.fixture
    def model(self) -> PlayerImpactModel:
        """Create a PlayerImpactModel instance."""
        return PlayerImpactModel()

    @pytest.fixture
    def team_stats(self) -> dict:
        """Duke's team statistics."""
        return {
            "TeamName": "Duke",
            "AdjEM": 25.0,
            "AdjOE": 120.0,
            "AdjDE": 95.0,
        }

    @pytest.fixture
    def star_player(self, model: PlayerImpactModel, team_stats: dict) -> PlayerValue:
        """Create a star player for testing."""
        star_stats = {
            "Player": "Star Player",
            "Team": "Duke",
            "Poss%": 25.0,
            "ORtg": 130.0,
            "eFG%": 62.5,
            "TS%": 65.0,
            "TRB%": 15.0,
            "STL%": 3.0,
            "BLK%": 5.0,
        }
        return model.calculate_player_value(star_stats, team_stats)

    def test_injury_out_full_impact(
        self, model: PlayerImpactModel, star_player: PlayerValue, team_stats: dict
    ):
        """Test injury impact when player is fully out."""
        injury = model.estimate_injury_impact(star_player, team_stats, "out")

        assert injury.player == star_player
        assert injury.team_adj_em_baseline == 25.0

        # Full impact: 100% of VOR lost
        assert injury.estimated_adj_em_loss == pytest.approx(
            star_player.value_over_replacement, rel=1e-5
        )
        # Adjusted AdjEM = baseline - loss
        assert injury.adjusted_adj_em == pytest.approx(
            25.0 - star_player.value_over_replacement, rel=1e-5
        )
        # Offensive rating should decrease
        assert injury.adjusted_adj_oe < 120.0

        # Should be classified as significant impact
        assert injury.severity in ["Moderate", "Major", "Devastating"]

    def test_injury_doubtful_75_percent_impact(
        self, model: PlayerImpactModel, star_player: PlayerValue, team_stats: dict
    ):
        """Test injury impact when player is doubtful (75% impact)."""
        injury = model.estimate_injury_impact(star_player, team_stats, "doubtful")

        # 75% of VOR lost
        expected_loss = star_player.value_over_replacement * 0.75
        assert injury.estimated_adj_em_loss == pytest.approx(expected_loss, rel=1e-5)
        assert injury.adjusted_adj_em == pytest.approx(25.0 - expected_loss, rel=1e-5)

    def test_injury_questionable_50_percent_impact(
        self, model: PlayerImpactModel, star_player: PlayerValue, team_stats: dict
    ):
        """Test injury impact when player is questionable (50% impact)."""
        injury = model.estimate_injury_impact(star_player, team_stats, "questionable")

        # 50% of VOR lost
        expected_loss = star_player.value_over_replacement * 0.5
        assert injury.estimated_adj_em_loss == pytest.approx(expected_loss, rel=1e-5)
        assert injury.adjusted_adj_em == pytest.approx(25.0 - expected_loss, rel=1e-5)

    def test_confidence_intervals(
        self, model: PlayerImpactModel, star_player: PlayerValue, team_stats: dict
    ):
        """Test confidence interval calculation."""
        injury = model.estimate_injury_impact(star_player, team_stats, "out")

        ci_lower, ci_upper = injury.confidence_interval

        # CI should be centered around adjusted AdjEM
        ci_center = (ci_lower + ci_upper) / 2
        assert ci_center == pytest.approx(injury.adjusted_adj_em, rel=1e-5)

        # CI width should be ~35% of value lost
        expected_uncertainty = abs(injury.estimated_adj_em_loss) * 0.35
        actual_uncertainty = (ci_upper - ci_lower) / 2
        assert actual_uncertainty == pytest.approx(expected_uncertainty, rel=1e-5)

    def test_severity_classifications(
        self, model: PlayerImpactModel, team_stats: dict
    ):
        """Test severity classification thresholds."""
        # Minor: < 1.0 AdjEM loss
        minor_stats = {
            "Player": "Bench",
            "Team": "Duke",
            "Poss%": 5.0,
            "ORtg": 110.0,
            "eFG%": 50.0,
            "TS%": 52.0,
            "TRB%": 5.0,
        }
        minor_player = model.calculate_player_value(minor_stats, team_stats)
        minor_injury = model.estimate_injury_impact(minor_player, team_stats, "out")
        assert minor_injury.severity == "Minor"

        # Moderate: 1.0-3.0 AdjEM loss
        moderate_stats = {
            "Player": "Role",
            "Team": "Duke",
            "Poss%": 18.0,
            "ORtg": 122.0,
            "eFG%": 58.0,
            "TS%": 60.0,
            "TRB%": 10.0,
            "STL%": 2.0,
        }
        moderate_player = model.calculate_player_value(moderate_stats, team_stats)
        moderate_injury = model.estimate_injury_impact(
            moderate_player, team_stats, "out"
        )
        assert moderate_injury.severity in ["Minor", "Moderate"]

    def test_invalid_injury_severity_raises_error(
        self, model: PlayerImpactModel, star_player: PlayerValue, team_stats: dict
    ):
        """Test that invalid injury severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid injury_severity"):
            model.estimate_injury_impact(star_player, team_stats, "maybe")

    def test_defensive_impact_weighted(
        self, model: PlayerImpactModel, team_stats: dict
    ):
        """Test that defensive impact is weighted at 50%."""
        # Player with high defensive stats
        defensive_stats = {
            "Player": "Defender",
            "Team": "Duke",
            "Poss%": 20.0,
            "ORtg": 118.0,
            "eFG%": 54.0,
            "TS%": 56.0,
            "TRB%": 18.0,
            "STL%": 4.0,
            "BLK%": 6.0,
        }

        player = model.calculate_player_value(defensive_stats, team_stats)
        injury = model.estimate_injury_impact(player, team_stats, "out")

        # Defensive contribution should be weighted at 0.5
        expected_de_change = player.defensive_contribution * 0.5
        actual_de_change = injury.adjusted_adj_de - team_stats["AdjDE"]
        assert actual_de_change == pytest.approx(expected_de_change, rel=1e-5)


class TestDepthChart:
    """Test depth chart generation."""

    @pytest.fixture
    def model(self) -> PlayerImpactModel:
        """Create a PlayerImpactModel instance."""
        return PlayerImpactModel()

    @pytest.fixture
    def team_stats(self) -> dict:
        """Duke's team statistics."""
        return {
            "TeamName": "Duke",
            "AdjEM": 25.0,
            "AdjOE": 120.0,
            "AdjDE": 95.0,
        }

    @pytest.fixture
    def player_stats_df(self) -> pd.DataFrame:
        """Sample player statistics DataFrame."""
        return pd.DataFrame(
            [
                {
                    "Player": "Star A",
                    "Team": "Duke",
                    "yr": "Jr",
                    "ht": "6-8",
                    "Poss%": 28.0,
                    "ORtg": 132.0,
                    "eFG%": 64.0,
                    "TS%": 66.0,
                    "TRB%": 16.0,
                    "STL%": 3.5,
                    "BLK%": 4.0,
                },
                {
                    "Player": "Star B",
                    "Team": "Duke",
                    "yr": "So",
                    "ht": "6-6",
                    "Poss%": 24.0,
                    "ORtg": 128.0,
                    "eFG%": 60.0,
                    "TS%": 62.0,
                    "TRB%": 12.0,
                    "STL%": 2.5,
                    "BLK%": 2.0,
                },
                {
                    "Player": "Role Player",
                    "Team": "Duke",
                    "yr": "Sr",
                    "ht": "6-4",
                    "Poss%": 15.0,
                    "ORtg": 118.0,
                    "eFG%": 56.0,
                    "TS%": 58.0,
                    "TRB%": 8.0,
                    "STL%": 2.0,
                    "BLK%": 1.0,
                },
                {
                    "Player": "Bench",
                    "Team": "Duke",
                    "yr": "Fr",
                    "ht": "6-10",
                    "Poss%": 6.0,
                    "ORtg": 112.0,
                    "eFG%": 52.0,
                    "TS%": 54.0,
                    "TRB%": 10.0,
                    "STL%": 1.0,
                    "BLK%": 2.0,
                },
                # Different team (should be filtered out)
                {
                    "Player": "UNC Player",
                    "Team": "North Carolina",
                    "yr": "Jr",
                    "ht": "6-5",
                    "Poss%": 25.0,
                    "ORtg": 125.0,
                    "eFG%": 58.0,
                    "TS%": 60.0,
                    "TRB%": 12.0,
                },
            ]
        )

    def test_depth_chart_creation(
        self,
        model: PlayerImpactModel,
        team_stats: dict,
        player_stats_df: pd.DataFrame,
    ):
        """Test basic depth chart creation."""
        depth_chart = model.get_team_depth_chart("Duke", player_stats_df, team_stats)

        # Should have 4 Duke players
        assert len(depth_chart) == 4
        # Should not include UNC player
        assert "UNC Player" not in depth_chart["Player"].values

    def test_depth_chart_sorted_by_value(
        self,
        model: PlayerImpactModel,
        team_stats: dict,
        player_stats_df: pd.DataFrame,
    ):
        """Test that depth chart is sorted by estimated value."""
        depth_chart = model.get_team_depth_chart("Duke", player_stats_df, team_stats)

        # Values should be in descending order
        values = depth_chart["EstimatedValue"].tolist()
        assert values == sorted(values, reverse=True)

        # Star A should be first (highest usage and efficiency)
        assert depth_chart.iloc[0]["Player"] == "Star A"

    def test_depth_chart_columns(
        self,
        model: PlayerImpactModel,
        team_stats: dict,
        player_stats_df: pd.DataFrame,
    ):
        """Test depth chart has correct columns."""
        depth_chart = model.get_team_depth_chart("Duke", player_stats_df, team_stats)

        expected_columns = [
            "Player",
            "yr",
            "ht",
            "Poss%",
            "ORtg",
            "eFG%",
            "EstimatedValue",
            "VOR",
        ]
        assert list(depth_chart.columns) == expected_columns

    def test_empty_depth_chart_for_nonexistent_team(
        self,
        model: PlayerImpactModel,
        team_stats: dict,
        player_stats_df: pd.DataFrame,
    ):
        """Test that non-existent team returns empty DataFrame."""
        depth_chart = model.get_team_depth_chart(
            "Fake Team", player_stats_df, team_stats
        )

        assert depth_chart.empty

    def test_depth_chart_value_calculations(
        self,
        model: PlayerImpactModel,
        team_stats: dict,
        player_stats_df: pd.DataFrame,
    ):
        """Test that depth chart values match individual calculations."""
        depth_chart = model.get_team_depth_chart("Duke", player_stats_df, team_stats)

        # Get Star A's row
        star_a_row = depth_chart[depth_chart["Player"] == "Star A"].iloc[0]

        # Manually calculate value for Star A
        star_a_stats = player_stats_df[
            player_stats_df["Player"] == "Star A"
        ].iloc[0].to_dict()
        manual_value = model.calculate_player_value(star_a_stats, team_stats)

        # Values should match
        assert star_a_row["EstimatedValue"] == pytest.approx(
            manual_value.estimated_value, rel=1e-5
        )
        assert star_a_row["VOR"] == pytest.approx(
            manual_value.value_over_replacement, rel=1e-5
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def model(self) -> PlayerImpactModel:
        """Create a PlayerImpactModel instance."""
        return PlayerImpactModel()

    @pytest.fixture
    def team_stats(self) -> dict:
        """Basic team statistics."""
        return {
            "TeamName": "Test Team",
            "AdjEM": 10.0,
            "AdjOE": 110.0,
            "AdjDE": 100.0,
        }

    def test_missing_optional_stats(
        self, model: PlayerImpactModel, team_stats: dict
    ):
        """Test handling of missing optional statistics (STL%, BLK%)."""
        player_stats = {
            "Player": "Test",
            "Team": "Test Team",
            "Poss%": 20.0,
            "ORtg": 115.0,
            "eFG%": 55.0,
            "TS%": 58.0,
            "TRB%": 10.0,
            # STL% and BLK% missing
        }

        value = model.calculate_player_value(player_stats, team_stats)

        # Should default to 0 for missing stats
        assert value.player_name == "Test"
        assert value.defensive_contribution >= 0

    def test_zero_possession_percentage(
        self, model: PlayerImpactModel, team_stats: dict
    ):
        """Test handling of zero possession percentage."""
        player_stats = {
            "Player": "Benchwarmer",
            "Team": "Test Team",
            "Poss%": 0.0,
            "ORtg": 100.0,
            "eFG%": 50.0,
            "TS%": 52.0,
            "TRB%": 5.0,
        }

        value = model.calculate_player_value(player_stats, team_stats)

        # Should handle gracefully
        assert value.possession_pct == 0.0
        assert value.offensive_contribution == 0.0

    def test_negative_efficiency_player(
        self, model: PlayerImpactModel, team_stats: dict
    ):
        """Test player with efficiency below team average."""
        player_stats = {
            "Player": "Inefficient",
            "Team": "Test Team",
            "Poss%": 15.0,
            "ORtg": 95.0,  # Well below team (110)
            "eFG%": 40.0,
            "TS%": 45.0,
            "TRB%": 8.0,
        }

        value = model.calculate_player_value(player_stats, team_stats)

        # Offensive contribution should be negative
        assert value.offensive_contribution < 0
        # Total value can be negative
        assert value.estimated_value < 0
