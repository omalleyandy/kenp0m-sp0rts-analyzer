"""Comprehensive tests for defensive analysis and scheme classification."""

import pytest
from unittest.mock import Mock

from kenp0m_sp0rts_analyzer.defensive_analysis import (
    DefensiveAnalyzer,
    DefensiveMatchup,
    DefensiveProfile,
)


class TestDefensiveProfile:
    """Test defensive profile generation and scheme classification."""

    def test_rim_protection_scheme(self):
        """Test classification of rim protection scheme (high blocks, low opp 2pt%)."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Houston",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 44.0,  # Elite interior defense (<46%)
                "RankOppFG2Pct": 5,
                "BlockPct": 12.5,  # High block rate (>10%)
                "RankBlockPct": 3,
                "StlRate": 8.0,
                "RankStlRate": 150,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 50.0,
                "RankOppARate": 100,
            }
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        profile = analyzer.get_defensive_profile("Houston", 2025)

        assert profile.team_name == "Houston"
        assert profile.defensive_scheme == "rim_protection"
        assert "Elite rim protection" in profile.primary_strength
        assert profile.block_pct >= 10.0
        assert profile.opp_fg2_pct <= 46.0

    def test_pressure_scheme(self):
        """Test classification of pressure scheme (high steal rate)."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "VCU",
                "OppFG3Pct": 33.0,
                "RankOppFG3Pct": 150,
                "OppFG2Pct": 48.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 150,
                "StlRate": 11.5,  # High steal rate (>9%)
                "RankStlRate": 5,
                "OppNSTRate": 12.0,
                "RankOppNSTRate": 50,
                "OppARate": 48.0,
                "RankOppARate": 50,
            }
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        profile = analyzer.get_defensive_profile("VCU", 2025)

        assert profile.defensive_scheme == "pressure"
        assert "Aggressive pressure" in profile.primary_strength
        assert profile.stl_rate >= 9.0

    def test_versatile_scheme(self):
        """Test classification of versatile elite defense (good at everything)."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Virginia",
                "OppFG3Pct": 30.0,  # Elite perimeter (<31%)
                "RankOppFG3Pct": 5,
                "OppFG2Pct": 44.0,  # Elite interior (<46%)
                "RankOppFG2Pct": 10,
                "BlockPct": 9.5,
                "RankBlockPct": 20,
                "StlRate": 8.5,  # Good steals (>8%)
                "RankStlRate": 30,
                "OppNSTRate": 11.0,
                "RankOppNSTRate": 20,
                "OppARate": 48.0,
                "RankOppARate": 15,
            }
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        profile = analyzer.get_defensive_profile("Virginia", 2025)

        assert profile.defensive_scheme == "versatile"
        assert "Multi-dimensional" in profile.primary_strength
        assert profile.opp_fg3_pct <= 31.0
        assert profile.opp_fg2_pct <= 46.0
        assert profile.stl_rate >= 8.0

    def test_balanced_scheme(self):
        """Test classification of balanced fundamental defense."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Duke",
                "OppFG3Pct": 32.5,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.5,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.5,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            }
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        profile = analyzer.get_defensive_profile("Duke", 2025)

        assert profile.defensive_scheme == "balanced"
        assert "Solid fundamental defense" in profile.primary_strength

    def test_balanced_scheme_weakness_identification(self):
        """Test identification of weakest area in balanced defense."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team A",
                "OppFG3Pct": 35.0,  # Highest = weakest area
                "RankOppFG3Pct": 200,
                "OppFG2Pct": 33.0,  # Lower than OppFG3Pct
                "RankOppFG2Pct": 50,
                "BlockPct": 9.0,
                "RankBlockPct": 50,
                "StlRate": 8.5,  # Below 9.0 to keep balanced, above 7.5 to not be weakness
                "RankStlRate": 50,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 50.0,
                "RankOppARate": 80,
            }
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        profile = analyzer.get_defensive_profile("Team A", 2025)

        assert profile.primary_weakness == "three-point defense"

    def test_team_not_found_raises_error(self):
        """Test that ValueError is raised when team is not found."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Duke",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            }
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)

        with pytest.raises(ValueError, match="not found"):
            analyzer.get_defensive_profile("NonexistentTeam", 2025)


class TestDefensiveMatchup:
    """Test defensive matchup analysis."""

    def test_perimeter_defense_advantage(self):
        """Test identification of perimeter defense advantage."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Elite 3PT Def",
                "OppFG3Pct": 30.0,  # Better perimeter defense
                "RankOppFG3Pct": 10,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 50.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Weak 3PT Def",
                "OppFG3Pct": 35.0,  # Worse perimeter defense
                "RankOppFG3Pct": 250,
                "OppFG2Pct": 46.0,
                "RankOppFG2Pct": 50,
                "BlockPct": 9.0,
                "RankBlockPct": 50,
                "StlRate": 8.0,
                "RankStlRate": 80,
                "OppNSTRate": 10.5,
                "RankOppNSTRate": 90,
                "OppARate": 51.0,
                "RankOppARate": 90,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Elite 3PT Def", "Weak 3PT Def", 2025)

        assert matchup.perimeter_defense_advantage == "Elite 3PT Def"

    def test_interior_defense_advantage(self):
        """Test identification of interior defense advantage."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Elite Paint Def",
                "OppFG3Pct": 33.0,
                "RankOppFG3Pct": 150,
                "OppFG2Pct": 44.0,  # Better interior defense
                "RankOppFG2Pct": 10,
                "BlockPct": 11.0,
                "RankBlockPct": 5,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 50.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Weak Paint Def",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 50.0,  # Worse interior defense
                "RankOppFG2Pct": 250,
                "BlockPct": 7.0,
                "RankBlockPct": 200,
                "StlRate": 8.0,
                "RankStlRate": 80,
                "OppNSTRate": 10.5,
                "RankOppNSTRate": 90,
                "OppARate": 51.0,
                "RankOppARate": 90,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Elite Paint Def", "Weak Paint Def", 2025)

        assert matchup.interior_defense_advantage == "Elite Paint Def"

    def test_pressure_defense_advantage(self):
        """Test identification of pressure defense advantage."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Pressure Team",
                "OppFG3Pct": 33.0,
                "RankOppFG3Pct": 150,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 10.5,  # Higher steal rate
                "RankStlRate": 10,
                "OppNSTRate": 11.0,
                "RankOppNSTRate": 50,
                "OppARate": 48.0,
                "RankOppARate": 50,
            },
            {
                "TeamName": "Passive Team",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 46.0,
                "RankOppFG2Pct": 50,
                "BlockPct": 9.0,
                "RankBlockPct": 50,
                "StlRate": 6.5,  # Lower steal rate
                "RankStlRate": 200,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 150,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Pressure Team", "Passive Team", 2025)

        assert matchup.pressure_defense_advantage == "Pressure Team"

    def test_defensive_advantage_score_calculation(self):
        """Test calculation of overall defensive advantage score."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Elite Defense",
                "OppFG3Pct": 30.0,  # 3% better than opponent
                "RankOppFG3Pct": 10,
                "OppFG2Pct": 44.0,  # 4% better than opponent
                "RankOppFG2Pct": 5,
                "BlockPct": 10.0,
                "RankBlockPct": 15,
                "StlRate": 9.0,  # 2% better than opponent
                "RankStlRate": 25,
                "OppNSTRate": 11.0,
                "RankOppNSTRate": 30,
                "OppARate": 49.0,
                "RankOppARate": 40,
            },
            {
                "TeamName": "Average Defense",
                "OppFG3Pct": 33.0,
                "RankOppFG3Pct": 150,
                "OppFG2Pct": 48.0,
                "RankOppFG2Pct": 150,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.0,
                "RankStlRate": 150,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Elite Defense", "Average Defense", 2025)

        # Elite Defense should have advantage (score > 5.0)
        assert matchup.defensive_advantage_score > 5.0
        assert matchup.better_defense == "Elite Defense"

    def test_defensive_advantage_score_bounds(self):
        """Test that defensive advantage score stays within 0-10 bounds."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team A",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Team B",
                "OppFG3Pct": 32.5,
                "RankOppFG3Pct": 110,
                "OppFG2Pct": 47.5,
                "RankOppFG2Pct": 110,
                "BlockPct": 7.8,
                "RankBlockPct": 110,
                "StlRate": 7.3,
                "RankStlRate": 110,
                "OppNSTRate": 10.2,
                "RankOppNSTRate": 110,
                "OppARate": 52.5,
                "RankOppARate": 110,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team A", "Team B", 2025)

        # Score should be bounded
        assert 0.0 <= matchup.defensive_advantage_score <= 10.0

    def test_defensive_keys_generation_rim_protection(self):
        """Test defensive keys for rim protection scheme."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Rim Protector",
                "OppFG3Pct": 33.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 44.0,
                "RankOppFG2Pct": 5,
                "BlockPct": 12.0,
                "RankBlockPct": 3,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 50.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Opponent",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 80,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Rim Protector", "Opponent", 2025)

        # Should include rim protection strategy
        assert any("rim" in key.lower() for key in matchup.team1_defensive_keys)

    def test_defensive_keys_generation_pressure(self):
        """Test defensive keys for pressure scheme."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Pressure Defense",
                "OppFG3Pct": 33.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 11.0,
                "RankStlRate": 5,
                "OppNSTRate": 12.0,
                "RankOppNSTRate": 30,
                "OppARate": 48.0,
                "RankOppARate": 40,
            },
            {
                "TeamName": "Opponent",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 80,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Pressure Defense", "Opponent", 2025)

        # Should include pressure strategy
        assert any("pressure" in key.lower() for key in matchup.team1_defensive_keys)

    def test_matchup_recommendation_significant_advantage(self):
        """Test recommendation for significant defensive advantage."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Elite Defense",
                "OppFG3Pct": 29.0,
                "RankOppFG3Pct": 5,
                "OppFG2Pct": 43.0,
                "RankOppFG2Pct": 3,
                "BlockPct": 11.0,
                "RankBlockPct": 8,
                "StlRate": 10.0,
                "RankStlRate": 15,
                "OppNSTRate": 11.5,
                "RankOppNSTRate": 20,
                "OppARate": 47.0,
                "RankOppARate": 25,
            },
            {
                "TeamName": "Poor Defense",
                "OppFG3Pct": 36.0,
                "RankOppFG3Pct": 300,
                "OppFG2Pct": 51.0,
                "RankOppFG2Pct": 320,
                "BlockPct": 6.0,
                "RankBlockPct": 280,
                "StlRate": 5.5,
                "RankStlRate": 300,
                "OppNSTRate": 9.0,
                "RankOppNSTRate": 280,
                "OppARate": 55.0,
                "RankOppARate": 290,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Elite Defense", "Poor Defense", 2025)

        # Should indicate significant advantage
        assert "significant" in matchup.matchup_recommendation.lower()
        assert "Elite Defense" in matchup.matchup_recommendation

    def test_matchup_recommendation_even(self):
        """Test recommendation for evenly matched defenses."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Even A",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Team Even B",
                "OppFG3Pct": 32.2,
                "RankOppFG3Pct": 105,
                "OppFG2Pct": 47.1,
                "RankOppFG2Pct": 105,
                "BlockPct": 7.9,
                "RankBlockPct": 105,
                "StlRate": 7.4,
                "RankStlRate": 105,
                "OppNSTRate": 10.1,
                "RankOppNSTRate": 105,
                "OppARate": 52.1,
                "RankOppARate": 105,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Even A", "Team Even B", 2025)

        # Should indicate even matchup
        assert "evenly matched" in matchup.matchup_recommendation.lower()


class TestDefensiveEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_defensive_keys_fallback(self):
        """Test fallback when no specific defensive keys apply."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Generic Defense",
                "OppFG3Pct": 32.0,  # Not elite enough to trigger keys
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,  # Not high enough
                "RankBlockPct": 100,
                "StlRate": 7.5,  # Not high enough
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Opponent",
                "OppFG3Pct": 32.5,
                "RankOppFG3Pct": 110,
                "OppFG2Pct": 47.5,
                "RankOppFG2Pct": 110,
                "BlockPct": 7.8,
                "RankBlockPct": 110,
                "StlRate": 7.3,
                "RankStlRate": 110,
                "OppNSTRate": 10.2,
                "RankOppNSTRate": 110,
                "OppARate": 52.5,
                "RankOppARate": 110,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Generic Defense", "Opponent", 2025)

        # Should have fallback keys
        assert "Play fundamental defense" in matchup.team1_defensive_keys

    def test_neutral_defensive_score(self):
        """Test that nearly identical defenses produce neutral score (~5.0)."""
        mock_response = Mock()
        mock_response.data = [
            {
                "TeamName": "Team Twin A",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
            {
                "TeamName": "Team Twin B",
                "OppFG3Pct": 32.0,
                "RankOppFG3Pct": 100,
                "OppFG2Pct": 47.0,
                "RankOppFG2Pct": 100,
                "BlockPct": 8.0,
                "RankBlockPct": 100,
                "StlRate": 7.5,
                "RankStlRate": 100,
                "OppNSTRate": 10.0,
                "RankOppNSTRate": 100,
                "OppARate": 52.0,
                "RankOppARate": 100,
            },
        ]

        mock_api = Mock()
        mock_api.get_misc_stats.return_value = mock_response

        analyzer = DefensiveAnalyzer(api=mock_api)
        matchup = analyzer.analyze_matchup("Team Twin A", "Team Twin B", 2025)

        # Should be very close to neutral (5.0)
        assert 4.5 <= matchup.defensive_advantage_score <= 5.5
