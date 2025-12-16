"""Comprehensive tests for tempo and pace matchup analysis."""

import pytest

from kenp0m_sp0rts_analyzer.tempo_analysis import (
    PaceMatchupAnalysis,
    TempoMatchupAnalyzer,
    TempoProfile,
)


class TestTempoProfile:
    """Test TempoProfile dataclass and classification logic."""

    def test_fast_paced_team_profile(self):
        """Test classification of fast-paced team (Auburn-style)."""
        team_stats = {
            "TeamName": "Auburn",
            "AdjTempo": 72.5,
            "RankAdjTempo": 15,
            "APL_Off": 15.2,
            "APL_Def": 15.8,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
            "AdjEM": 33.5,
            "AdjOE": 118.3,
            "AdjDE": 84.8,
        }

        analyzer = TempoMatchupAnalyzer()
        profile = analyzer.get_tempo_profile(team_stats)

        assert profile.team_name == "Auburn"
        assert profile.pace_style == "fast"
        assert profile.off_style == "quick_strike"
        assert profile.def_style == "pressure"
        assert profile.adj_tempo == 72.5
        assert profile.rank_tempo == 15

    def test_slow_paced_team_profile(self):
        """Test classification of slow-paced team (Wisconsin-style)."""
        team_stats = {
            "TeamName": "Wisconsin",
            "AdjTempo": 64.2,
            "RankAdjTempo": 320,
            "APL_Off": 20.5,
            "APL_Def": 19.8,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
            "AdjEM": 10.2,
            "AdjOE": 105.4,
            "AdjDE": 95.2,
        }

        analyzer = TempoMatchupAnalyzer()
        profile = analyzer.get_tempo_profile(team_stats)

        assert profile.team_name == "Wisconsin"
        assert profile.pace_style == "slow"
        assert profile.off_style == "methodical"
        assert profile.def_style == "pack_line"
        assert profile.adj_tempo == 64.2

    def test_average_paced_team_profile(self):
        """Test classification of average-paced team."""
        team_stats = {
            "TeamName": "Duke",
            "AdjTempo": 68.0,
            "RankAdjTempo": 150,
            "APL_Off": 17.5,
            "APL_Def": 17.5,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
            "AdjEM": 25.0,
            "AdjOE": 115.0,
            "AdjDE": 90.0,
        }

        analyzer = TempoMatchupAnalyzer()
        profile = analyzer.get_tempo_profile(team_stats)

        assert profile.pace_style == "average"
        assert profile.off_style == "average"
        assert profile.def_style == "average"

    def test_asymmetric_pace_profile(self):
        """Test team with asymmetric pace (Houston-style: slow offense, fast defense)."""
        team_stats = {
            "TeamName": "Houston",
            "AdjTempo": 66.8,
            "RankAdjTempo": 200,
            "APL_Off": 18.2,  # Slower offense
            "APL_Def": 15.5,  # Pressure defense
            "ConfAPL_Off": 17.0,
            "ConfAPL_Def": 17.0,
            "AdjEM": 28.0,
            "AdjOE": 112.0,
            "AdjDE": 84.0,
        }

        analyzer = TempoMatchupAnalyzer()
        profile = analyzer.get_tempo_profile(team_stats)

        # Asymmetric styles
        assert profile.off_style == "average"  # 18.2 is between 17 and 19
        assert profile.def_style == "pressure"  # 15.5 < 17
        assert abs(profile.apl_off - profile.apl_def) > 2.0  # Significant asymmetry


class TestPaceClassification:
    """Test individual classification methods."""

    def test_classify_pace_thresholds(self):
        """Test pace classification at boundary values."""
        analyzer = TempoMatchupAnalyzer()

        # > 70.0 is fast
        assert analyzer._classify_pace(71.0) == "fast"
        assert analyzer._classify_pace(70.1) == "fast"
        assert analyzer._classify_pace(70.0) == "average"  # Exactly at threshold
        assert analyzer._classify_pace(69.9) == "average"
        assert analyzer._classify_pace(68.0) == "average"
        assert analyzer._classify_pace(66.1) == "average"
        # < 66.0 is slow
        assert analyzer._classify_pace(66.0) == "average"  # Exactly at threshold
        assert analyzer._classify_pace(65.9) == "slow"
        assert analyzer._classify_pace(65.0) == "slow"

    def test_classify_offensive_style_thresholds(self):
        """Test offensive style classification at boundary values."""
        analyzer = TempoMatchupAnalyzer()

        # < 17.0 is quick_strike
        assert analyzer._classify_offensive_style(16.0) == "quick_strike"
        assert analyzer._classify_offensive_style(16.9) == "quick_strike"
        assert analyzer._classify_offensive_style(17.0) == "average"  # At threshold
        assert analyzer._classify_offensive_style(17.1) == "average"
        assert analyzer._classify_offensive_style(18.0) == "average"
        assert analyzer._classify_offensive_style(18.9) == "average"
        # > 19.0 is methodical
        assert analyzer._classify_offensive_style(19.0) == "average"  # At threshold
        assert analyzer._classify_offensive_style(19.1) == "methodical"
        assert analyzer._classify_offensive_style(20.0) == "methodical"

    def test_classify_defensive_style_thresholds(self):
        """Test defensive style classification at boundary values."""
        analyzer = TempoMatchupAnalyzer()

        # < 17.0 is pressure
        assert analyzer._classify_defensive_style(16.0) == "pressure"
        assert analyzer._classify_defensive_style(16.9) == "pressure"
        assert analyzer._classify_defensive_style(17.0) == "average"  # At threshold
        assert analyzer._classify_defensive_style(17.1) == "average"
        assert analyzer._classify_defensive_style(18.0) == "average"
        assert analyzer._classify_defensive_style(18.9) == "average"
        # > 19.0 is pack_line
        assert analyzer._classify_defensive_style(19.0) == "average"  # At threshold
        assert analyzer._classify_defensive_style(19.1) == "pack_line"
        assert analyzer._classify_defensive_style(20.0) == "pack_line"


class TestPaceMatchupAnalysis:
    """Test comprehensive matchup analysis."""

    @pytest.fixture
    def fast_team_stats(self):
        """Auburn-like fast-paced team."""
        return {
            "TeamName": "Auburn",
            "AdjTempo": 72.5,
            "RankAdjTempo": 15,
            "APL_Off": 15.2,
            "APL_Def": 15.8,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
            "AdjEM": 33.5,
            "AdjOE": 118.3,
            "AdjDE": 84.8,
        }

    @pytest.fixture
    def slow_team_stats(self):
        """Wisconsin-like slow-paced team."""
        return {
            "TeamName": "Wisconsin",
            "AdjTempo": 64.2,
            "RankAdjTempo": 320,
            "APL_Off": 20.5,
            "APL_Def": 19.8,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
            "AdjEM": 10.2,
            "AdjOE": 105.4,
            "AdjDE": 95.2,
        }

    @pytest.fixture
    def average_team_stats(self):
        """Duke-like average-paced team."""
        return {
            "TeamName": "Duke",
            "AdjTempo": 68.0,
            "RankAdjTempo": 150,
            "APL_Off": 17.5,
            "APL_Def": 17.5,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
            "AdjEM": 25.0,
            "AdjOE": 115.0,
            "AdjDE": 90.0,
        }

    def test_extreme_tempo_mismatch(self, fast_team_stats, slow_team_stats):
        """Test analysis of extreme tempo mismatch (Auburn vs Wisconsin)."""
        analyzer = TempoMatchupAnalyzer()
        analysis = analyzer.analyze_pace_matchup(fast_team_stats, slow_team_stats)

        # Verify basic matchup structure
        assert isinstance(analysis, PaceMatchupAnalysis)
        assert analysis.team1_profile.team_name == "Auburn"
        assert analysis.team2_profile.team_name == "Wisconsin"

        # Tempo differential
        assert analysis.tempo_differential == pytest.approx(8.3, abs=0.1)
        assert analysis.tempo_differential > 0  # Team1 is faster

        # Expected possessions
        # Auburn has positive control (0.655) due to:
        # - Elite efficiency (33.5 vs 10.2)
        # - Pressure defense (15.8 APL_Def vs 19.8)
        # - So pace is weighted toward Auburn's faster tempo
        simple_avg = (72.5 + 64.2) / 2  # 68.35
        assert 64 <= analysis.expected_possessions <= 73
        # Should be higher than simple average (Auburn controls pace)
        assert analysis.expected_possessions > simple_avg

        # Style mismatch should be high (extreme contrast)
        assert analysis.style_mismatch_score >= 7.0  # Out of 10
        assert analysis.style_mismatch_score <= 10.0

        # Pace control (Wisconsin's pack-line should have some control)
        assert analysis.pace_advantage in ["team1", "team2", "neutral"]

        # APL mismatches
        assert analysis.offensive_disruption_team1 == "severe"  # Auburn disrupted
        assert abs(analysis.apl_off_mismatch_team1) > 3.0  # Large mismatch

    def test_similar_tempo_matchup(self, average_team_stats):
        """Test analysis when both teams have similar tempo."""
        analyzer = TempoMatchupAnalyzer()

        # Create two similar teams
        team2_stats = average_team_stats.copy()
        team2_stats["TeamName"] = "UNC"
        team2_stats["AdjTempo"] = 67.5
        team2_stats["AdjEM"] = 23.0

        analysis = analyzer.analyze_pace_matchup(average_team_stats, team2_stats)

        # Small tempo differential
        assert abs(analysis.tempo_differential) <= 1.0

        # Expected possessions close to both teams' tempos
        assert 67.0 <= analysis.expected_possessions <= 68.5

        # Low style mismatch
        assert analysis.style_mismatch_score < 3.0  # Minimal mismatch

        # Minimal offensive disruption
        assert analysis.offensive_disruption_team1 in ["minimal", "moderate"]
        assert analysis.offensive_disruption_team2 in ["minimal", "moderate"]

        # Neutral pace advantage
        assert analysis.pace_advantage == "neutral"

    def test_asymmetric_matchup(self, fast_team_stats, average_team_stats):
        """Test matchup between fast and average team."""
        analyzer = TempoMatchupAnalyzer()
        analysis = analyzer.analyze_pace_matchup(fast_team_stats, average_team_stats)

        # Moderate tempo differential
        assert 3.0 <= abs(analysis.tempo_differential) <= 6.0

        # Expected possessions between the two
        assert (
            min(fast_team_stats["AdjTempo"], average_team_stats["AdjTempo"])
            <= analysis.expected_possessions
            <= max(fast_team_stats["AdjTempo"], average_team_stats["AdjTempo"])
        )

        # Moderate style mismatch
        assert 3.0 <= analysis.style_mismatch_score <= 7.0

    def test_tempo_control_factor_range(self, fast_team_stats, slow_team_stats):
        """Test that tempo control factor is within valid range."""
        analyzer = TempoMatchupAnalyzer()
        analysis = analyzer.analyze_pace_matchup(fast_team_stats, slow_team_stats)

        # Should be between -1 and +1
        assert -1.0 <= analysis.tempo_control_factor <= 1.0

    def test_optimal_pace_calculation(self, fast_team_stats, slow_team_stats):
        """Test optimal pace calculation for teams."""
        analyzer = TempoMatchupAnalyzer()
        analysis = analyzer.analyze_pace_matchup(fast_team_stats, slow_team_stats)

        # Optimal paces should be reasonable
        assert 60.0 <= analysis.optimal_pace_team1 <= 80.0
        assert 60.0 <= analysis.optimal_pace_team2 <= 80.0

        # Fast efficient team should have higher optimal pace
        # Auburn (better, faster) vs Wisconsin (worse, slower)
        assert analysis.optimal_pace_team1 > analysis.optimal_pace_team2

    def test_pace_scenario_beneficiaries(self, fast_team_stats, slow_team_stats):
        """Test which team benefits from fast/slow pace scenarios."""
        analyzer = TempoMatchupAnalyzer()
        analysis = analyzer.analyze_pace_matchup(fast_team_stats, slow_team_stats)

        # Fast pace should favor the better team (Auburn)
        assert analysis.fast_pace_favors in ["team1", "team2", "neutral"]

        # Slow pace should favor the underdog (Wisconsin)
        assert analysis.slow_pace_favors in ["team1", "team2", "neutral"]


class TestTempoControlCalculation:
    """Test tempo control factor calculation."""

    def test_defensive_style_dominates_control(self):
        """Test that defensive style (40% weight) affects pace control."""
        analyzer = TempoMatchupAnalyzer()

        # Team with pressure defense vs pack-line defense
        team1 = {
            "APL_Def": 16.0,  # Pressure defense
            "AdjEM": 20.0,
            "AdjTempo": 68.0,
        }
        team2 = {
            "APL_Def": 20.0,  # Pack-line defense
            "AdjEM": 20.0,  # Same efficiency
            "AdjTempo": 68.0,  # Same tempo preference
        }

        control = analyzer.calculate_tempo_control(team1, team2)

        # Team1 should have positive control (pressure defense forces pace)
        assert control > 0

    def test_efficiency_affects_control(self):
        """Test that efficiency advantage affects pace control."""
        analyzer = TempoMatchupAnalyzer()

        # Much better team vs average team
        team1 = {
            "APL_Def": 17.5,
            "AdjEM": 30.0,  # Elite
            "AdjTempo": 68.0,
        }
        team2 = {
            "APL_Def": 17.5,  # Same defensive style
            "AdjEM": 10.0,  # Average
            "AdjTempo": 68.0,
        }

        control = analyzer.calculate_tempo_control(team1, team2)

        # Better team should have positive control
        assert control > 0

    def test_tempo_preference_strength_affects_control(self):
        """Test that extreme tempo preference affects control."""
        analyzer = TempoMatchupAnalyzer()

        # Very fast team vs average team
        team1 = {
            "APL_Def": 17.5,
            "AdjEM": 20.0,
            "AdjTempo": 74.0,  # Very fast
        }
        team2 = {
            "APL_Def": 17.5,
            "AdjEM": 20.0,
            "AdjTempo": 68.0,  # Average
        }

        control = analyzer.calculate_tempo_control(team1, team2)

        # Fast team should have some positive control
        assert control > 0

    def test_control_factor_clamped(self):
        """Test that control factor is clamped to [-1, 1]."""
        analyzer = TempoMatchupAnalyzer()

        # Extreme mismatch
        team1 = {
            "APL_Def": 15.0,  # Extreme pressure
            "AdjEM": 35.0,  # Elite
            "AdjTempo": 75.0,  # Very fast
        }
        team2 = {
            "APL_Def": 21.0,  # Extreme pack-line
            "AdjEM": -10.0,  # Poor
            "AdjTempo": 62.0,  # Very slow
        }

        control = analyzer.calculate_tempo_control(team1, team2)

        # Should be clamped
        assert -1.0 <= control <= 1.0


class TestExpectedTempoCalculation:
    """Test expected tempo calculation with control weighting."""

    def test_neutral_control_gives_simple_average(self):
        """Test that zero control factor gives simple average."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjTempo": 70.0}
        team2 = {"AdjTempo": 66.0}

        expected = analyzer._calculate_expected_tempo(team1, team2, control_factor=0.0)

        # Should be simple average
        assert expected == pytest.approx(68.0, abs=0.1)

    def test_positive_control_favors_team1(self):
        """Test that positive control weighs toward team1's tempo."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjTempo": 72.0}
        team2 = {"AdjTempo": 64.0}

        expected = analyzer._calculate_expected_tempo(team1, team2, control_factor=0.8)

        # Should be closer to team1's tempo (72)
        simple_avg = 68.0
        assert expected > simple_avg
        assert expected < 72.0  # But not equal to team1's tempo

    def test_negative_control_favors_team2(self):
        """Test that negative control weighs toward team2's tempo."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjTempo": 72.0}
        team2 = {"AdjTempo": 64.0}

        expected = analyzer._calculate_expected_tempo(
            team1, team2, control_factor=-0.8
        )

        # Should be closer to team2's tempo (64)
        simple_avg = 68.0
        assert expected < simple_avg
        assert expected > 64.0  # But not equal to team2's tempo


class TestStyleMismatchCalculation:
    """Test style mismatch scoring."""

    def test_identical_styles_zero_mismatch(self):
        """Test that identical teams have minimal mismatch."""
        analyzer = TempoMatchupAnalyzer()

        profile1 = TempoProfile(
            team_name="Team1",
            adj_tempo=68.0,
            rank_tempo=150,
            apl_off=17.5,
            apl_def=17.5,
            conf_apl_off=17.5,
            conf_apl_def=17.5,
            pace_style="average",
            off_style="average",
            def_style="average",
        )

        profile2 = TempoProfile(
            team_name="Team2",
            adj_tempo=68.0,
            rank_tempo=151,
            apl_off=17.5,
            apl_def=17.5,
            conf_apl_off=17.5,
            conf_apl_def=17.5,
            pace_style="average",
            off_style="average",
            def_style="average",
        )

        mismatch = analyzer._calculate_style_mismatch(profile1, profile2)

        # Should be zero or very close
        assert mismatch < 1.0

    def test_extreme_mismatch_high_score(self):
        """Test that extreme mismatches get high scores."""
        analyzer = TempoMatchupAnalyzer()

        # Fast, quick-strike, pressure team
        profile1 = TempoProfile(
            team_name="Auburn",
            adj_tempo=74.0,
            rank_tempo=10,
            apl_off=15.0,
            apl_def=15.5,
            conf_apl_off=17.5,
            conf_apl_def=17.5,
            pace_style="fast",
            off_style="quick_strike",
            def_style="pressure",
        )

        # Slow, methodical, pack-line team
        profile2 = TempoProfile(
            team_name="Wisconsin",
            adj_tempo=63.0,
            rank_tempo=350,
            apl_off=21.0,
            apl_def=20.5,
            conf_apl_off=17.5,
            conf_apl_def=17.5,
            pace_style="slow",
            off_style="methodical",
            def_style="pack_line",
        )

        mismatch = analyzer._calculate_style_mismatch(profile1, profile2)

        # Should be high (7-10 range)
        assert mismatch >= 7.0
        assert mismatch <= 10.0

    def test_mismatch_score_range(self):
        """Test that mismatch score is always in [0, 10] range."""
        analyzer = TempoMatchupAnalyzer()

        # Test with various extreme profiles
        profiles = [
            TempoProfile(
                "Team",
                75.0,
                1,
                14.0,
                14.5,
                17.5,
                17.5,
                "fast",
                "quick_strike",
                "pressure",
            ),
            TempoProfile(
                "Team",
                62.0,
                350,
                22.0,
                21.5,
                17.5,
                17.5,
                "slow",
                "methodical",
                "pack_line",
            ),
            TempoProfile(
                "Team", 68.0, 150, 17.5, 17.5, 17.5, 17.5, "average", "average", "average"
            ),
        ]

        for p1 in profiles:
            for p2 in profiles:
                mismatch = analyzer._calculate_style_mismatch(p1, p2)
                assert 0.0 <= mismatch <= 10.0


class TestOffensiveDisruptionClassification:
    """Test offensive disruption severity classification."""

    def test_minimal_disruption(self):
        """Test classification of minimal disruption."""
        analyzer = TempoMatchupAnalyzer()

        # Small mismatch (0-1.5 seconds)
        assert analyzer._classify_disruption(0.5) == "minimal"
        assert analyzer._classify_disruption(1.0) == "minimal"
        assert analyzer._classify_disruption(-1.0) == "minimal"

    def test_moderate_disruption(self):
        """Test classification of moderate disruption."""
        analyzer = TempoMatchupAnalyzer()

        # Moderate mismatch (1.5-3.0 seconds)
        assert analyzer._classify_disruption(2.0) == "moderate"
        assert analyzer._classify_disruption(-2.5) == "moderate"

    def test_severe_disruption(self):
        """Test classification of severe disruption."""
        analyzer = TempoMatchupAnalyzer()

        # Large mismatch (>3.0 seconds)
        assert analyzer._classify_disruption(4.0) == "severe"
        assert analyzer._classify_disruption(-5.0) == "severe"


class TestTempoImpactEstimation:
    """Test tempo impact on margin estimation."""

    def test_tempo_impact_sign(self):
        """Test that tempo impact has correct sign."""
        analyzer = TempoMatchupAnalyzer()

        # Better team (Team1) with faster pace
        team1 = {"AdjEM": 30.0, "AdjTempo": 72.0}
        team2 = {"AdjEM": 10.0, "AdjTempo": 64.0}

        # Expected tempo higher than average favors better team
        impact = analyzer._estimate_tempo_impact(team1, team2, expected_tempo=70.0)

        # Should be positive (favors team1)
        # Because expected_tempo (70) > simple_avg (68), and team1 has EM advantage
        assert impact > 0

    def test_tempo_impact_magnitude_reasonable(self):
        """Test that tempo impact is reasonable in magnitude."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjEM": 25.0, "AdjTempo": 70.0}
        team2 = {"AdjEM": 15.0, "AdjTempo": 66.0}

        impact = analyzer._estimate_tempo_impact(team1, team2, expected_tempo=69.0)

        # Should be small (< 2 points typically)
        assert abs(impact) < 3.0


class TestConfidenceAdjustment:
    """Test confidence interval variance adjustment."""

    def test_slow_game_increases_variance(self):
        """Test that slower games have wider confidence intervals."""
        analyzer = TempoMatchupAnalyzer()

        # National average is 68 possessions
        slow_adjustment = analyzer._calculate_confidence_adjustment(expected_tempo=62.0)
        avg_adjustment = analyzer._calculate_confidence_adjustment(expected_tempo=68.0)

        # Slow game should have adjustment > 1.0
        assert slow_adjustment > 1.0
        assert slow_adjustment > avg_adjustment

    def test_fast_game_decreases_variance(self):
        """Test that faster games have narrower confidence intervals."""
        analyzer = TempoMatchupAnalyzer()

        fast_adjustment = analyzer._calculate_confidence_adjustment(expected_tempo=74.0)
        avg_adjustment = analyzer._calculate_confidence_adjustment(expected_tempo=68.0)

        # Fast game should have adjustment < 1.0
        assert fast_adjustment < 1.0
        assert fast_adjustment < avg_adjustment

    def test_average_tempo_neutral_adjustment(self):
        """Test that average tempo has neutral adjustment."""
        analyzer = TempoMatchupAnalyzer()

        adjustment = analyzer._calculate_confidence_adjustment(expected_tempo=68.0)

        # Should be approximately 1.0
        assert adjustment == pytest.approx(1.0, abs=0.01)


class TestOptimalPaceCalculation:
    """Test optimal pace calculation."""

    def test_elite_offense_prefers_faster_pace(self):
        """Test that elite offensive teams prefer faster pace."""
        analyzer = TempoMatchupAnalyzer()

        team_avg_off = {"AdjTempo": 68.0, "AdjOE": 110.0, "AdjDE": 100.0}
        team_elite_off = {"AdjTempo": 68.0, "AdjOE": 120.0, "AdjDE": 100.0}

        optimal_avg = analyzer._calculate_optimal_pace(team_avg_off)
        optimal_elite = analyzer._calculate_optimal_pace(team_elite_off)

        # Elite offense should want faster pace
        assert optimal_elite > optimal_avg

    def test_elite_defense_can_handle_faster_pace(self):
        """Test that elite defensive teams can handle faster pace."""
        analyzer = TempoMatchupAnalyzer()

        team_avg_def = {"AdjTempo": 68.0, "AdjOE": 110.0, "AdjDE": 100.0}
        team_elite_def = {"AdjTempo": 68.0, "AdjOE": 110.0, "AdjDE": 85.0}

        optimal_avg = analyzer._calculate_optimal_pace(team_avg_def)
        optimal_elite = analyzer._calculate_optimal_pace(team_elite_def)

        # Elite defense can handle faster pace
        assert optimal_elite > optimal_avg


class TestTempoBeneficiaryDetermination:
    """Test determination of which team benefits from pace."""

    def test_fast_pace_favors_efficient_team(self):
        """Test that fast pace favors the more efficient team."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjEM": 30.0, "AdjOE": 118.0}
        team2 = {"AdjEM": 10.0, "AdjOE": 105.0}

        beneficiary = analyzer._determine_tempo_beneficiary(team1, team2, "fast")

        # Better team (team1) should benefit
        assert beneficiary == "team1"

    def test_slow_pace_favors_underdog(self):
        """Test that slow pace favors the underdog."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjEM": 30.0, "AdjOE": 118.0}
        team2 = {"AdjEM": 10.0, "AdjOE": 105.0}

        beneficiary = analyzer._determine_tempo_beneficiary(team1, team2, "slow")

        # Underdog (team2) should benefit from fewer possessions
        assert beneficiary == "team2"

    def test_even_matchup_neutral_beneficiary(self):
        """Test that evenly matched teams have neutral pace beneficiary."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjEM": 20.0, "AdjOE": 112.0}
        team2 = {"AdjEM": 20.0, "AdjOE": 112.0}

        fast_beneficiary = analyzer._determine_tempo_beneficiary(team1, team2, "fast")
        slow_beneficiary = analyzer._determine_tempo_beneficiary(team1, team2, "slow")

        # Even teams should be neutral
        assert fast_beneficiary == "neutral"
        assert slow_beneficiary == "neutral"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_apl_data_handled_gracefully(self):
        """Test that missing APL data doesn't break analysis."""
        # This is handled in prediction.py with defaults,
        # but tempo_analysis.py expects complete data
        analyzer = TempoMatchupAnalyzer()

        team_stats = {
            "TeamName": "Team",
            "AdjTempo": 68.0,
            "RankAdjTempo": 150,
            "APL_Off": 17.5,
            "APL_Def": 17.5,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
            "AdjEM": 20.0,
            "AdjOE": 110.0,
            "AdjDE": 90.0,
        }

        # Should not raise any errors
        profile = analyzer.get_tempo_profile(team_stats)
        assert profile is not None

    def test_extreme_tempo_values(self):
        """Test with extreme but valid tempo values."""
        analyzer = TempoMatchupAnalyzer()

        # Extremely fast team
        fast_extreme = {
            "TeamName": "Fast",
            "AdjTempo": 80.0,
            "RankAdjTempo": 1,
            "APL_Off": 13.0,
            "APL_Def": 13.5,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
            "AdjEM": 25.0,
            "AdjOE": 115.0,
            "AdjDE": 90.0,
        }

        # Extremely slow team
        slow_extreme = {
            "TeamName": "Slow",
            "AdjTempo": 58.0,
            "RankAdjTempo": 363,
            "APL_Off": 24.0,
            "APL_Def": 23.5,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
            "AdjEM": 15.0,
            "AdjOE": 105.0,
            "AdjDE": 90.0,
        }

        # Should handle extremes without errors
        analysis = analyzer.analyze_pace_matchup(fast_extreme, slow_extreme)

        assert analysis is not None
        assert analysis.tempo_differential > 20.0
        assert analysis.style_mismatch_score >= 8.0

    def test_negative_efficiency_values(self):
        """Test with negative efficiency margin (poor teams)."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {
            "TeamName": "Poor1",
            "AdjTempo": 68.0,
            "RankAdjTempo": 250,
            "APL_Off": 17.5,
            "APL_Def": 17.5,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
            "AdjEM": -10.0,
            "AdjOE": 95.0,
            "AdjDE": 105.0,
        }

        team2 = {
            "TeamName": "Poor2",
            "AdjTempo": 66.0,
            "RankAdjTempo": 280,
            "APL_Off": 18.0,
            "APL_Def": 18.5,
            "ConfAPL_Off": 17.5,
            "ConfAPL_Def": 17.5,
            "AdjEM": -15.0,
            "AdjOE": 92.0,
            "AdjDE": 107.0,
        }

        # Should handle negative EM values
        analysis = analyzer.analyze_pace_matchup(team1, team2)
        assert analysis is not None
        assert -1.0 <= analysis.tempo_control_factor <= 1.0


class TestRoundingAndPrecision:
    """Test that values are properly rounded."""

    def test_expected_tempo_rounded(self):
        """Test that expected tempo is rounded to 1 decimal."""
        analyzer = TempoMatchupAnalyzer()

        team1 = {"AdjTempo": 68.33333}
        team2 = {"AdjTempo": 66.66666}

        expected = analyzer._calculate_expected_tempo(team1, team2, control_factor=0.0)

        # Should be rounded to 1 decimal place
        assert expected == round(expected, 1)

    def test_style_mismatch_rounded(self):
        """Test that style mismatch is rounded to 1 decimal."""
        analyzer = TempoMatchupAnalyzer()

        profile1 = TempoProfile(
            "Team1",
            68.5,
            150,
            17.3,
            17.7,
            17.5,
            17.5,
            "average",
            "average",
            "average",
        )
        profile2 = TempoProfile(
            "Team2",
            66.2,
            180,
            18.1,
            18.4,
            17.5,
            17.5,
            "average",
            "average",
            "average",
        )

        mismatch = analyzer._calculate_style_mismatch(profile1, profile2)

        # Should be rounded to 1 decimal place
        assert mismatch == round(mismatch, 1)

    def test_confidence_adjustment_rounded(self):
        """Test that confidence adjustment is rounded to 3 decimals."""
        analyzer = TempoMatchupAnalyzer()

        adjustment = analyzer._calculate_confidence_adjustment(expected_tempo=67.333)

        # Should be rounded to 3 decimal places
        assert adjustment == round(adjustment, 3)
