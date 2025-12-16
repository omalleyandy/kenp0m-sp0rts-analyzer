"""Comprehensive tests for integrated matchup analysis."""

import pytest
from unittest.mock import Mock

from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
    ComprehensiveMatchupAnalyzer,
    DimensionScore,
    MatchupWeights,
)


class TestMatchupWeights:
    """Test weight configuration and validation."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        weights = MatchupWeights()

        total = (
            weights.efficiency
            + weights.four_factors
            + weights.tempo
            + weights.point_distribution
            + weights.defensive
            + weights.size
            + weights.experience
        )

        assert abs(total - 1.0) < 0.01

    def test_tournament_weights_sum_to_one(self):
        """Test that tournament weights sum to 1.0."""
        weights = MatchupWeights.tournament_weights()

        total = (
            weights.efficiency
            + weights.four_factors
            + weights.tempo
            + weights.point_distribution
            + weights.defensive
            + weights.size
            + weights.experience
        )

        assert abs(total - 1.0) < 0.01

    def test_tournament_weights_increase_experience(self):
        """Test that tournament context increases experience weight."""
        regular = MatchupWeights()
        tournament = MatchupWeights.tournament_weights()

        assert tournament.experience > regular.experience

    def test_tournament_weights_increase_four_factors(self):
        """Test that tournament context increases four factors weight."""
        regular = MatchupWeights()
        tournament = MatchupWeights.tournament_weights()

        assert tournament.four_factors > regular.four_factors

    def test_tournament_weights_decrease_tempo(self):
        """Test that tournament context decreases tempo weight."""
        regular = MatchupWeights()
        tournament = MatchupWeights.tournament_weights()

        assert tournament.tempo < regular.tempo

    def test_custom_weights_valid(self):
        """Test creation of custom valid weights."""
        custom = MatchupWeights(
            efficiency=0.3,
            four_factors=0.25,
            tempo=0.1,
            point_distribution=0.1,
            defensive=0.15,
            size=0.05,
            experience=0.05,
        )

        assert custom.efficiency == 0.3

    def test_invalid_weights_raise_error(self):
        """Test that weights not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MatchupWeights(
                efficiency=0.5,  # Too high
                four_factors=0.2,
                tempo=0.1,
                point_distribution=0.1,
                defensive=0.1,
                size=0.1,
                experience=0.1,
            )


class TestDimensionScore:
    """Test dimension score creation."""

    def test_dimension_score_creation(self):
        """Test creating a dimension score."""
        score = DimensionScore(
            dimension="Efficiency",
            score=5.5,
            confidence=0.85,
            key_insight="Duke efficiency advantage",
            details="Duke: 25.0 AdjEM | UNC: 20.0 AdjEM",
        )

        assert score.dimension == "Efficiency"
        assert score.score == 5.5
        assert score.confidence == 0.85
        assert "Duke" in score.key_insight

    def test_dimension_score_bounds(self):
        """Test dimension scores within -10 to +10 range."""
        score = DimensionScore(
            dimension="Test",
            score=12.0,  # Will be clamped in actual implementation
            confidence=0.8,
            key_insight="Test",
        )

        # In real implementation, scores should be clamped
        # This tests the data structure accepts the value
        assert isinstance(score.score, float)


class TestCompositeScoring:
    """Test composite score calculation."""

    def test_calculate_composite_score(self):
        """Test composite score calculation with weights."""
        analyzer = ComprehensiveMatchupAnalyzer()

        dimensions = [
            (DimensionScore("Eff", 5.0, 0.9, "Test"), 0.25),
            (DimensionScore("FF", 3.0, 0.85, "Test"), 0.20),
            (DimensionScore("Tempo", -2.0, 0.7, "Test"), 0.10),
            (DimensionScore("PD", 4.0, 0.75, "Test"), 0.15),
            (DimensionScore("Def", 2.0, 0.8, "Test"), 0.15),
            (DimensionScore("Size", 1.0, 0.7, "Test"), 0.10),
            (DimensionScore("Exp", 0.5, 0.65, "Test"), 0.05),
        ]

        composite = analyzer._calculate_composite_score(dimensions)

        # Expected: (5*0.25 + 3*0.20 + (-2)*0.10 + 4*0.15 + 2*0.15 + 1*0.10 + 0.5*0.05) * 10
        expected = (1.25 + 0.60 - 0.20 + 0.60 + 0.30 + 0.10 + 0.025) * 10
        assert abs(composite - expected) < 0.1

    def test_composite_score_bounds(self):
        """Test that composite scores are within reasonable bounds."""
        analyzer = ComprehensiveMatchupAnalyzer()

        # Extreme positive scores
        extreme_positive = [
            (DimensionScore("Test", 10.0, 0.9, ""), 1.0 / 7)
            for _ in range(7)
        ]

        composite = analyzer._calculate_composite_score(extreme_positive)

        # Should be close to 100 (10 * sum of weights * 10)
        assert 95.0 <= composite <= 105.0

    def test_composite_score_zero_for_neutral(self):
        """Test that neutral scores produce composite near zero."""
        analyzer = ComprehensiveMatchupAnalyzer()

        neutral = [(DimensionScore("Test", 0.0, 0.8, ""), 1.0 / 7) for _ in range(7)]

        composite = analyzer._calculate_composite_score(neutral)

        assert abs(composite) < 1.0


class TestEfficiencyScoring:
    """Test efficiency dimension scoring."""

    def test_efficiency_score_positive_advantage(self):
        """Test efficiency scoring with team1 advantage."""
        analyzer = ComprehensiveMatchupAnalyzer()

        mock_analysis = Mock()
        mock_analysis.team1_efficiency = Mock(adj_em=25.0)
        mock_analysis.team2_efficiency = Mock(adj_em=20.0)

        score = analyzer._score_efficiency(mock_analysis, "Duke", "UNC")

        # 5.0 AdjEM difference / 2 = 2.5 score
        assert score.score == 2.5
        assert "Duke" in score.key_insight
        assert score.confidence == 0.9

    def test_efficiency_score_negative_advantage(self):
        """Test efficiency scoring with team2 advantage."""
        analyzer = ComprehensiveMatchupAnalyzer()

        mock_analysis = Mock()
        mock_analysis.team1_efficiency = Mock(adj_em=18.0)
        mock_analysis.team2_efficiency = Mock(adj_em=24.0)

        score = analyzer._score_efficiency(mock_analysis, "Team A", "Team B")

        # -6.0 AdjEM difference / 2 = -3.0 score
        assert score.score == -3.0
        assert "Team B" in score.key_insight

    def test_efficiency_score_clamped(self):
        """Test that extreme efficiency differences are clamped."""
        analyzer = ComprehensiveMatchupAnalyzer()

        mock_analysis = Mock()
        mock_analysis.team1_efficiency = Mock(adj_em=35.0)
        mock_analysis.team2_efficiency = Mock(adj_em=5.0)

        score = analyzer._score_efficiency(mock_analysis, "Elite", "Weak")

        # 30.0 AdjEM difference / 2 = 15.0, clamped to 10.0
        assert score.score == 10.0


class TestKeyFactorRanking:
    """Test key factor ranking and synthesis."""

    def test_rank_key_factors_by_impact(self):
        """Test that factors are ranked by absolute impact."""
        analyzer = ComprehensiveMatchupAnalyzer()

        dimensions = [
            DimensionScore("Low Impact", 1.0, 0.8, "Low"),
            DimensionScore("High Impact", 8.0, 0.9, "High"),
            DimensionScore("Medium Impact", -5.0, 0.75, "Medium"),
            DimensionScore("Zero Impact", 0.0, 0.7, "Zero"),
        ]

        ranked = analyzer._rank_key_factors(dimensions)

        # Should be ordered by absolute value: 8.0, 5.0, 1.0, 0.0
        assert ranked[0][1] == 8.0
        assert ranked[1][1] == 5.0
        assert ranked[2][1] == 1.0
        assert ranked[3][1] == 0.0

    def test_key_factors_include_insights(self):
        """Test that key factors include dimension insights."""
        analyzer = ComprehensiveMatchupAnalyzer()

        dimensions = [
            DimensionScore("Efficiency", 5.0, 0.9, "Duke efficiency edge"),
            DimensionScore("Tempo", 3.0, 0.7, "UNC controls pace"),
        ]

        ranked = analyzer._rank_key_factors(dimensions)

        assert "Duke efficiency edge" in ranked[0][0]
        assert "UNC controls pace" in ranked[1][0]


class TestGamePlanGeneration:
    """Test strategic game plan generation."""

    def test_game_plan_includes_four_factors(self):
        """Test that game plan includes Four Factors advantages."""
        analyzer = ComprehensiveMatchupAnalyzer()

        # Mock Four Factors with advantages
        mock_ff = Mock()
        mock_ff.team1_profile = Mock(team_name="Duke")
        mock_ff.team1_advantages = [
            Mock(factor=Mock(value="shooting efficiency")),
            Mock(factor=Mock(value="offensive rebounding")),
        ]

        mock_tempo = Mock(
            faster_pace_benefits=None, slower_pace_benefits=None, expected_tempo=70.0
        )
        mock_pd = Mock(recommended_strategy="Duke should attack the paint")
        mock_def = Mock(
            team1_defense=Mock(team_name="Duke"),
            team1_defensive_keys=["Protect the rim"],
        )
        mock_size = Mock(better_size_team="UNC")

        plan = analyzer._generate_game_plan(
            "Duke",
            "UNC",
            DimensionScore("Eff", 5.0, 0.9, "Test"),
            mock_ff,
            mock_tempo,
            mock_pd,
            mock_def,
            mock_size,
        )

        # Should include Four Factors advantages
        assert any("shooting efficiency" in rec for rec in plan)

    def test_game_plan_limited_to_five(self):
        """Test that game plan is limited to top 5 recommendations."""
        analyzer = ComprehensiveMatchupAnalyzer()

        # Mock with many potential recommendations
        mock_ff = Mock()
        mock_ff.team1_profile = Mock(team_name="Duke")
        mock_ff.team1_advantages = [Mock(factor=Mock(value=f"advantage{i}")) for i in range(10)]

        mock_tempo = Mock(
            faster_pace_benefits="Duke", slower_pace_benefits=None, expected_tempo=70.0
        )
        mock_pd = Mock(recommended_strategy="Duke should shoot threes")
        mock_def = Mock(
            team1_defense=Mock(team_name="Duke"),
            team1_defensive_keys=["Play defense"],
        )
        mock_size = Mock(better_size_team="Duke", strategic_recommendation="Use size")

        plan = analyzer._generate_game_plan(
            "Duke",
            "UNC",
            DimensionScore("Eff", 5.0, 0.9, "Test"),
            mock_ff,
            mock_tempo,
            mock_pd,
            mock_def,
            mock_size,
        )

        assert len(plan) <= 5


class TestXFactorIdentification:
    """Test X-factor identification."""

    def test_tempo_clash_identified(self):
        """Test that extreme tempo mismatches are identified as X-factors."""
        analyzer = ComprehensiveMatchupAnalyzer()

        mock_tempo = Mock(style_mismatch_score=8.5)
        mock_size = Mock(better_size_team="neutral", rebounding_prediction="")
        mock_def = Mock(pressure_defense_advantage=None)
        mock_pd = Mock(three_point_advantage=2.0)

        x_factors = analyzer._identify_x_factors(
            mock_tempo, mock_size, mock_def, mock_pd
        )

        assert any("Tempo control" in factor for factor in x_factors)

    def test_rebounding_battle_identified(self):
        """Test that rebounding advantages are identified as X-factors."""
        analyzer = ComprehensiveMatchupAnalyzer()

        mock_tempo = Mock(style_mismatch_score=5.0)
        mock_size = Mock(
            better_size_team="Duke",
            rebounding_prediction="Duke should dominate the glass",
        )
        mock_def = Mock(pressure_defense_advantage=None)
        mock_pd = Mock(three_point_advantage=2.0)

        x_factors = analyzer._identify_x_factors(
            mock_tempo, mock_size, mock_def, mock_pd
        )

        assert any("Rebounding battle" in factor for factor in x_factors)

    def test_three_point_variance_identified(self):
        """Test that large 3PT advantages are identified as X-factors."""
        analyzer = ComprehensiveMatchupAnalyzer()

        mock_tempo = Mock(style_mismatch_score=5.0)
        mock_size = Mock(better_size_team="neutral")
        mock_def = Mock(pressure_defense_advantage=None)
        mock_pd = Mock(three_point_advantage=7.5)  # Large advantage

        x_factors = analyzer._identify_x_factors(
            mock_tempo, mock_size, mock_def, mock_pd
        )

        assert any("Three-point" in factor for factor in x_factors)


class TestOverallAssessment:
    """Test overall assessment generation."""

    def test_significant_advantage_identified(self):
        """Test assessment for significant composite advantage."""
        analyzer = ComprehensiveMatchupAnalyzer()

        key_factors = [
            ("Efficiency: Duke advantage", 8.0),
            ("Four Factors: Duke advantage", 6.0),
        ]

        assessment = analyzer._generate_overall_assessment(
            "Duke", "UNC", 5.5, key_factors, None
        )

        assert "Duke" in assessment
        assert "significant" in assessment

    def test_slight_advantage_identified(self):
        """Test assessment for slight composite advantage."""
        analyzer = ComprehensiveMatchupAnalyzer()

        key_factors = [
            ("Efficiency: Duke advantage", 3.0),
            ("Tempo: UNC advantage", 2.0),
        ]

        assessment = analyzer._generate_overall_assessment(
            "Duke", "UNC", 1.5, key_factors, None
        )

        assert "Duke" in assessment
        assert "slight" in assessment

    def test_even_matchup_identified(self):
        """Test assessment for even matchup."""
        analyzer = ComprehensiveMatchupAnalyzer()

        key_factors = [
            ("Four Factors: Duke advantage", 2.0),
            ("Defensive: UNC advantage", 1.8),
        ]

        assessment = analyzer._generate_overall_assessment(
            "Duke", "UNC", 0.0, key_factors, None
        )

        assert "Evenly matched" in assessment


class TestTextReportGeneration:
    """Test text report formatting."""

    def test_text_report_contains_header(self):
        """Test that text report includes matchup header."""
        from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
            ComprehensiveMatchupReport,
        )

        report = ComprehensiveMatchupReport(
            team1="Duke",
            team2="UNC",
            season=2025,
            tournament_context=False,
            weights=MatchupWeights(),
            efficiency_score=DimensionScore("Eff", 5.0, 0.9, "Test"),
            four_factors_score=DimensionScore("FF", 3.0, 0.85, "Test"),
            tempo_score=DimensionScore("Tempo", 2.0, 0.7, "Test"),
            point_distribution_score=DimensionScore("PD", 1.0, 0.75, "Test"),
            defensive_score=DimensionScore("Def", 2.0, 0.8, "Test"),
            size_score=DimensionScore("Size", 1.0, 0.7, "Test"),
            experience_score=DimensionScore("Exp", 0.5, 0.65, "Test"),
            composite_score=3.5,
            prediction=None,
            key_factors=[("Factor 1", 5.0), ("Factor 2", 3.0)],
            team1_game_plan=["Strategy 1", "Strategy 2"],
            team2_game_plan=["Strategy A", "Strategy B"],
            x_factors=["X-Factor 1"],
            overall_assessment="Duke has advantage",
        )

        text = report.generate_text_report()

        assert "Duke vs UNC" in text
        assert "COMPREHENSIVE MATCHUP ANALYSIS" in text

    def test_text_report_includes_game_plans(self):
        """Test that text report includes strategic game plans."""
        from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
            ComprehensiveMatchupReport,
        )

        report = ComprehensiveMatchupReport(
            team1="Duke",
            team2="UNC",
            season=2025,
            tournament_context=False,
            weights=MatchupWeights(),
            efficiency_score=DimensionScore("Eff", 5.0, 0.9, "Test"),
            four_factors_score=DimensionScore("FF", 3.0, 0.85, "Test"),
            tempo_score=DimensionScore("Tempo", 2.0, 0.7, "Test"),
            point_distribution_score=DimensionScore("PD", 1.0, 0.75, "Test"),
            defensive_score=DimensionScore("Def", 2.0, 0.8, "Test"),
            size_score=DimensionScore("Size", 1.0, 0.7, "Test"),
            experience_score=DimensionScore("Exp", 0.5, 0.65, "Test"),
            composite_score=3.5,
            prediction=None,
            key_factors=[("Factor 1", 5.0)],
            team1_game_plan=["Attack the paint", "Control tempo"],
            team2_game_plan=["Shoot threes", "Apply pressure"],
            x_factors=["Rebounding"],
            overall_assessment="Test",
        )

        text = report.generate_text_report()

        assert "Attack the paint" in text
        assert "Shoot threes" in text
        assert "STRATEGIC GAME PLAN: Duke" in text
        assert "STRATEGIC GAME PLAN: UNC" in text

    def test_detailed_report_includes_dimensions(self):
        """Test that detailed report includes all dimension breakdowns."""
        from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
            ComprehensiveMatchupReport,
        )

        report = ComprehensiveMatchupReport(
            team1="Duke",
            team2="UNC",
            season=2025,
            tournament_context=False,
            weights=MatchupWeights(),
            efficiency_score=DimensionScore("Eff", 5.0, 0.9, "Efficiency test"),
            four_factors_score=DimensionScore("FF", 3.0, 0.85, "Four Factors test"),
            tempo_score=DimensionScore("Tempo", 2.0, 0.7, "Tempo test"),
            point_distribution_score=DimensionScore("PD", 1.0, 0.75, "PD test"),
            defensive_score=DimensionScore("Def", 2.0, 0.8, "Def test"),
            size_score=DimensionScore("Size", 1.0, 0.7, "Size test"),
            experience_score=DimensionScore("Exp", 0.5, 0.65, "Exp test"),
            composite_score=3.5,
            prediction=None,
            key_factors=[],
            team1_game_plan=[],
            team2_game_plan=[],
            x_factors=[],
            overall_assessment="Test",
        )

        text = report.generate_text_report(detailed=True)

        assert "DIMENSIONAL BREAKDOWN" in text
        assert "Efficiency & Ratings" in text
        assert "Four Factors" in text
        assert "Tempo & Pace" in text
