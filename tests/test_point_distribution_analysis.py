"""Comprehensive tests for point distribution and scoring style analysis."""

import pytest
from unittest.mock import Mock

from kenp0m_sp0rts_analyzer.point_distribution_analysis import (
    PointDistributionAnalyzer,
    ScoringStyleMatchup,
    ScoringStyleProfile,
)


class TestScoringStyleProfile:
    """Test scoring style classification and profile generation."""

    def test_perimeter_team_classification(self):
        """Test classification of perimeter-heavy team (>40% from 3pt)."""
        # Mock API response for 3pt-heavy team
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Villanova",
                "OffFt": 18.5,
                "OffFg2": 38.2,
                "OffFg3": 43.3,  # >40% threshold
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.5,
                "RankOffFt": 100,
                "RankOffFg2": 200,
                "RankOffFg3": 15,
            }
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        profile = analyzer.get_scoring_profile("Villanova", 2025)

        assert profile.team_name == "Villanova"
        assert profile.style == "perimeter"
        assert profile.fg3_pct == 43.3
        assert "Three-point heavy" in profile.primary_strength
        assert profile.season == 2025

    def test_interior_team_classification(self):
        """Test classification of interior-oriented team (>55% from 2pt)."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Gonzaga",
                "OffFt": 22.0,
                "OffFg2": 56.5,  # >55% threshold
                "OffFg3": 21.5,
                "DefFt": 21.0,
                "DefFg2": 48.0,
                "DefFg3": 31.0,
                "RankOffFt": 50,
                "RankOffFg2": 10,
                "RankOffFg3": 250,
            }
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        profile = analyzer.get_scoring_profile("Gonzaga", 2025)

        assert profile.style == "interior"
        assert profile.fg2_pct == 56.5
        assert "Interior oriented" in profile.primary_strength

    def test_balanced_team_classification(self):
        """Test classification of balanced scoring team."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Duke",
                "OffFt": 20.0,
                "OffFg2": 50.0,  # Neither >55% nor <40%
                "OffFg3": 30.0,  # Neither >40%
                "DefFt": 22.0,
                "DefFg2": 46.0,
                "DefFg3": 33.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            }
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        profile = analyzer.get_scoring_profile("Duke", 2025)

        assert profile.style == "balanced"
        assert profile.primary_strength == "Balanced scoring attack"

    def test_defensive_weakness_identification(self):
        """Test identification of defensive weaknesses."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team A",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 18.0,
                "DefFg2": 44.0,
                "DefFg3": 45.0,  # Highest = biggest weakness
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            }
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        profile = analyzer.get_scoring_profile("Team A", 2025)

        assert profile.defensive_weakness == "three-point defense"

    def test_team_not_found_raises_error(self):
        """Test that ValueError is raised when team is not found."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Duke",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 22.0,
                "DefFg2": 46.0,
                "DefFg3": 33.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            }
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)

        with pytest.raises(ValueError, match="not found"):
            analyzer.get_scoring_profile("NonexistentTeam", 2025)


class TestScoringStyleMatchup:
    """Test scoring style matchup analysis."""

    def test_three_point_advantage_calculation(self):
        """Test calculation of 3pt advantage (perimeter offense vs weak perimeter defense)."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team 3PT",
                "OffFt": 18.0,
                "OffFg2": 40.0,
                "OffFg3": 42.0,  # Strong 3pt offense
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 20,
            },
            {
                "TeamName": "Team Weak 3PT Def",
                "OffFt": 20.0,
                "OffFg2": 52.0,
                "OffFg3": 28.0,
                "DefFt": 20.0,
                "DefFg2": 44.0,
                "DefFg3": 36.0,  # Weak 3pt defense
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 200,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team 3PT", "Team Weak 3PT Def", 2025)

        # Team 3PT has 42% offense vs Team Weak 3PT Def's 36% defense allowed
        assert matchup.three_point_advantage > 0
        assert matchup.three_point_advantage == pytest.approx(6.0, abs=0.1)

    def test_interior_advantage_calculation(self):
        """Test calculation of 2pt interior advantage."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Interior",
                "OffFt": 22.0,
                "OffFg2": 58.0,  # Strong interior offense
                "OffFg3": 20.0,
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.0,
                "RankOffFt": 50,
                "RankOffFg2": 5,
                "RankOffFg3": 300,
            },
            {
                "TeamName": "Team Weak Paint",
                "OffFt": 20.0,
                "OffFg2": 48.0,
                "OffFg3": 32.0,
                "DefFt": 20.0,
                "DefFg2": 52.0,  # Weak interior defense
                "DefFg3": 31.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Interior", "Team Weak Paint", 2025)

        # Team Interior has 58% offense vs Team Weak Paint's 52% defense allowed
        assert matchup.two_point_advantage > 0
        assert matchup.two_point_advantage == pytest.approx(6.0, abs=0.1)

    def test_style_mismatch_score(self):
        """Test style mismatch score calculation."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team A",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
            {
                "TeamName": "Team B",
                "OffFt": 22.0,
                "OffFg2": 48.0,
                "OffFg3": 30.0,
                "DefFt": 22.0,
                "DefFg2": 46.0,
                "DefFg3": 32.0,
                "RankOffFt": 50,
                "RankOffFg2": 150,
                "RankOffFg3": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team A", "Team B", 2025)

        # Style mismatch should be 0-10 scale
        assert 0.0 <= matchup.style_mismatch_score <= 10.0

    def test_exploitable_areas_identification(self):
        """Test identification of exploitable mismatches."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Strong 3PT",
                "OffFt": 20.0,
                "OffFg2": 45.0,
                "OffFg3": 38.0,  # >35% threshold
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 50,
            },
            {
                "TeamName": "Weak 3PT Def",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 35.0,  # >33% threshold (weak)
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Strong 3PT", "Weak 3PT Def", 2025)

        # Should identify 3pt shooting as exploitable
        assert len(matchup.team1_exploitable_areas) > 0
        assert any("Three-point" in area for area in matchup.team1_exploitable_areas)

    def test_key_matchup_factor_identification(self):
        """Test identification of key matchup factor."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Elite 3PT",
                "OffFt": 18.0,
                "OffFg2": 40.0,
                "OffFg3": 42.0,  # Massive 3pt advantage
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 10,
            },
            {
                "TeamName": "Team Weak Perimeter",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 36.0,  # Poor 3pt defense
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Elite 3PT", "Team Weak Perimeter", 2025)

        # Key factor should be 3pt shooting
        assert "three-point" in matchup.key_matchup_factor.lower()

    def test_strategy_recommendation(self):
        """Test strategic recommendation generation."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team A",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 42.0,  # Strong 3pt
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 20,
            },
            {
                "TeamName": "Team B",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 28.0,
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 35.0,  # Weak 3pt defense
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 200,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team A", "Team B", 2025)

        # Strategy should mention 3pt shooting
        assert "three-point" in matchup.recommended_strategy.lower()
        assert "Team A" in matchup.recommended_strategy

    def test_negative_advantage_favors_team2(self):
        """Test that negative advantages correctly favor team 2."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Weak 3PT",
                "OffFt": 20.0,
                "OffFg2": 52.0,
                "OffFg3": 28.0,  # Weak 3pt offense
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 36.0,  # Weak 3pt defense
                "RankOffFt": 100,
                "RankOffFg2": 50,
                "RankOffFg3": 250,
            },
            {
                "TeamName": "Team Strong 3PT",
                "OffFt": 18.0,
                "OffFg2": 42.0,
                "OffFg3": 40.0,  # Strong 3pt offense
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 31.0,  # Strong 3pt defense
                "RankOffFt": 150,
                "RankOffFg2": 150,
                "RankOffFg3": 30,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Weak 3PT", "Team Strong 3PT", 2025)

        # Team 2 should have 3pt advantage (negative value for team 1)
        assert matchup.three_point_advantage < 0


class TestPointDistributionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_exploitable_areas(self):
        """Test when no clear exploitable areas exist."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Even",
                "OffFt": 20.0,
                "OffFg2": 48.0,  # Below exploitable thresholds
                "OffFg3": 32.0,
                "DefFt": 20.0,
                "DefFg2": 46.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
            {
                "TeamName": "Team Solid",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 20.0,
                "DefFg2": 45.0,
                "DefFg3": 31.0,  # Solid defense (below 33%)
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Even", "Team Solid", 2025)

        # Should return default message
        assert "No clear exploitable areas" in matchup.team1_exploitable_areas

    def test_very_close_matchup(self):
        """Test matchup with minimal differences."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Twin A",
                "OffFt": 20.0,
                "OffFg2": 50.0,
                "OffFg3": 30.0,
                "DefFt": 20.5,
                "DefFg2": 49.0,
                "DefFg3": 32.0,
                "RankOffFt": 100,
                "RankOffFg2": 100,
                "RankOffFg3": 100,
            },
            {
                "TeamName": "Team Twin B",
                "OffFt": 20.1,
                "OffFg2": 50.1,
                "OffFg3": 29.8,
                "DefFt": 20.4,
                "DefFg2": 49.1,
                "DefFg3": 32.1,
                "RankOffFt": 101,
                "RankOffFg2": 101,
                "RankOffFg3": 102,
            },
        ]

        mock_api = Mock()
        mock_api.get_point_distribution.return_value = mock_response

        analyzer = PointDistributionAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Twin A", "Team Twin B", 2025)

        # Advantages should be very small
        assert abs(matchup.three_point_advantage) < 3.0
        assert abs(matchup.two_point_advantage) < 3.0
        assert abs(matchup.free_throw_advantage) < 3.0
