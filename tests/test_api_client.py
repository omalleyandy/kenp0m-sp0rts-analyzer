"""Tests for the KenPom API client."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import directly from the api_client module to avoid package dependencies
_api_client_path = (
    Path(__file__).parent.parent / "src/kenp0m_sp0rts_analyzer/api_client.py"
)
_spec = importlib.util.spec_from_file_location("api_client", _api_client_path)
_api_client = importlib.util.module_from_spec(_spec)
# Register module in sys.modules before execution (needed for dataclasses)
sys.modules["api_client"] = _api_client
_spec.loader.exec_module(_api_client)

APIResponse = _api_client.APIResponse
AuthenticationError = _api_client.AuthenticationError
KenPomAPI = _api_client.KenPomAPI
ValidationError = _api_client.ValidationError


@pytest.fixture
def mock_api():
    """Create a KenPomAPI instance with mocked HTTP client."""
    with patch.object(KenPomAPI, "__init__", lambda self, **kwargs: None):
        api = KenPomAPI()
        api.api_key = "test-key"
        api._client = MagicMock()
        return api


class TestKenPomAPIInit:
    """Tests for KenPomAPI initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("httpx.Client"):
            api = KenPomAPI(api_key="test-key")
            assert api.api_key == "test-key"

    def test_init_from_env_var(self, monkeypatch):
        """Test initialization from environment variable."""
        monkeypatch.setenv("KENPOM_API_KEY", "env-key")
        with patch("httpx.Client"):
            api = KenPomAPI()
            assert api.api_key == "env-key"

    def test_init_no_key_raises_error(self, monkeypatch):
        """Test that missing API key raises AuthenticationError."""
        monkeypatch.delenv("KENPOM_API_KEY", raising=False)
        with pytest.raises(AuthenticationError, match="API key not found"):
            KenPomAPI()


class TestGetHeight:
    """Tests for the get_height method."""

    def test_get_height_by_year(self, mock_api):
        """Test getting height data by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "TeamName": "Duke",
                "Season": 2025,
                "AvgHgt": 77.5,
                "AvgHgtRank": 10,
                "HgtEff": 78.2,
                "HgtEffRank": 15,
                "Hgt5": 82.0,
                "Hgt5Rank": 5,
                "Hgt4": 79.5,
                "Hgt4Rank": 12,
                "Hgt3": 77.0,
                "Hgt3Rank": 20,
                "Hgt2": 75.5,
                "Hgt2Rank": 25,
                "Hgt1": 73.0,
                "Hgt1Rank": 30,
                "Exp": 2.5,
                "ExpRank": 50,
                "Bench": 0.8,
                "BenchRank": 75,
                "Continuity": 0.65,
                "RankContinuity": 100,
                "ConfShort": "ACC",
                "DataThrough": "2025-03-01",
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_height(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "height"
        assert call_args[1]["params"]["y"] == 2025
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["TeamName"] == "Duke"
        assert result.data[0]["AvgHgt"] == 77.5

    def test_get_height_by_team_id(self, mock_api):
        """Test getting height data by team_id for historical data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "Season": 2024, "AvgHgt": 76.5, "Exp": 2.2},
            {"TeamName": "Duke", "Season": 2025, "AvgHgt": 77.5, "Exp": 2.5},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_height(team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["team_id"] == 73
        assert "y" not in call_args[1]["params"]
        assert len(result.data) == 2

    def test_get_height_with_conference(self, mock_api):
        """Test getting height data filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Gonzaga", "ConfShort": "WCC", "AvgHgt": 78.0},
            {"TeamName": "Saint Mary's", "ConfShort": "WCC", "AvgHgt": 77.0},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_height(year=2025, conference="WCC")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["c"] == "WCC"
        assert len(result.data) == 2

    def test_get_height_conference_requires_year(self, mock_api):
        """Test that conference filter requires year parameter."""
        with pytest.raises(
            ValidationError, match="'year' parameter is required when filtering"
        ):
            mock_api.get_height(conference="WCC")

    def test_get_height_no_params_raises_error(self, mock_api):
        """Test that missing year and team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_height()

    def test_get_height_all_params(self, mock_api):
        """Test with year, team_id, and conference combined."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"TeamName": "Duke"}]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_height(year=2025, team_id=73, conference="ACC")

        call_args = mock_api._client.get.call_args
        params = call_args[1]["params"]
        assert params["y"] == 2025
        assert params["team_id"] == 73
        assert params["c"] == "ACC"
        assert len(result.data) == 1

    def test_get_height_to_dataframe(self, mock_api):
        """Test converting height response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "AvgHgt": 77.5, "Exp": 2.5},
            {"TeamName": "Kansas", "AvgHgt": 76.8, "Exp": 2.8},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_height(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert list(df.columns) == ["TeamName", "AvgHgt", "Exp"]
        assert df.iloc[0]["TeamName"] == "Duke"


class TestGetMiscStats:
    """Tests for the get_misc_stats method."""

    def test_get_misc_stats_by_year(self, mock_api):
        """Test getting misc stats by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "TeamName": "Duke",
                "Season": 2025,
                "ConfShort": "ACC",
                "DataThrough": "2025-03-01",
                "ConfOnly": "false",
                "FG3Pct": 36.5,
                "RankFG3Pct": 50,
                "FG2Pct": 52.1,
                "RankFG2Pct": 25,
                "FTPct": 75.2,
                "RankFTPct": 100,
                "BlockPct": 10.5,
                "RankBlockPct": 30,
                "StlRate": 9.2,
                "RankStlRate": 75,
                "NSTRate": 11.3,
                "RankNSTRate": 150,
                "ARate": 58.5,
                "RankARate": 40,
                "F3GRate": 35.0,
                "RankF3GRate": 120,
                "AdjOE": 118.5,
                "RankAdjOE": 10,
                "OppFG3Pct": 32.1,
                "RankOppFG3Pct": 60,
                "OppFG2Pct": 48.5,
                "RankOppFG2Pct": 35,
                "OppFTPct": 70.0,
                "RankOppFTPct": 200,
                "OppBlockPct": 8.5,
                "RankOppBlockPct": 150,
                "OppStlRate": 8.0,
                "RankOppStlRate": 180,
                "OppNSTRate": 12.0,
                "RankOppNSTRate": 50,
                "OppARate": 52.0,
                "RankOppARate": 100,
                "OppF3GRate": 33.5,
                "RankOppF3GRate": 90,
                "AdjDE": 95.2,
                "RankAdjDE": 15,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_misc_stats(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "misc-stats"
        assert call_args[1]["params"]["y"] == 2025
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["TeamName"] == "Duke"
        assert result.data[0]["FG3Pct"] == 36.5

    def test_get_misc_stats_by_team_id(self, mock_api):
        """Test getting misc stats by team_id for historical data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "Season": 2024, "FG3Pct": 35.0, "AdjOE": 115.0},
            {"TeamName": "Duke", "Season": 2025, "FG3Pct": 36.5, "AdjOE": 118.5},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_misc_stats(team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["team_id"] == 73
        assert "y" not in call_args[1]["params"]
        assert len(result.data) == 2

    def test_get_misc_stats_with_conference(self, mock_api):
        """Test getting misc stats filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "ConfShort": "ACC", "FG3Pct": 36.5},
            {"TeamName": "North Carolina", "ConfShort": "ACC", "FG3Pct": 34.2},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_misc_stats(year=2025, conference="ACC")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["c"] == "ACC"
        assert len(result.data) == 2

    def test_get_misc_stats_conference_requires_year(self, mock_api):
        """Test that conference filter requires year parameter."""
        with pytest.raises(
            ValidationError, match="'year' parameter is required when filtering"
        ):
            mock_api.get_misc_stats(conference="ACC")

    def test_get_misc_stats_with_conf_only(self, mock_api):
        """Test getting conference-only misc stats."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "ConfOnly": "true", "FG3Pct": 37.0},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_misc_stats(year=2025, conf_only=True)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["conf_only"] == "true"
        assert result.data[0]["ConfOnly"] == "true"

    def test_get_misc_stats_no_params_raises_error(self, mock_api):
        """Test that missing year and team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_misc_stats()

    def test_get_misc_stats_conf_only_alone_raises_error(self, mock_api):
        """Test that conf_only without year/team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_misc_stats(conf_only=True)

    def test_get_misc_stats_all_params(self, mock_api):
        """Test with all optional params combined."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"TeamName": "Duke"}]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_misc_stats(
            year=2025, team_id=73, conference="ACC", conf_only=True
        )

        call_args = mock_api._client.get.call_args
        params = call_args[1]["params"]
        assert params["y"] == 2025
        assert params["team_id"] == 73
        assert params["c"] == "ACC"
        assert params["conf_only"] == "true"

    def test_get_misc_stats_to_dataframe(self, mock_api):
        """Test converting misc stats response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "FG3Pct": 36.5, "AdjOE": 118.5},
            {"TeamName": "Kansas", "FG3Pct": 35.2, "AdjOE": 116.0},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_misc_stats(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert list(df.columns) == ["TeamName", "FG3Pct", "AdjOE"]
        assert df.iloc[0]["TeamName"] == "Duke"


class TestGetPointDistribution:
    """Tests for the get_point_distribution method."""

    def test_get_point_distribution_by_year(self, mock_api):
        """Test getting point distribution by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "TeamName": "Duke",
                "Season": 2025,
                "OffFt": 20.5,
                "RankOffFt": 150,
                "OffFg2": 45.3,
                "RankOffFg2": 200,
                "OffFg3": 34.2,
                "RankOffFg3": 50,
                "DefFt": 18.1,
                "RankDefFt": 100,
                "DefFg2": 48.0,
                "RankDefFg2": 75,
                "DefFg3": 33.9,
                "RankDefFg3": 120,
                "ConfShort": "ACC",
                "DataThrough": "2025-03-01",
                "ConfOnly": "false",
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_point_distribution(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "pointdist"
        assert call_args[1]["params"]["y"] == 2025
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["TeamName"] == "Duke"

    def test_get_point_distribution_by_team_id(self, mock_api):
        """Test getting point distribution by team_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "Season": 2024, "OffFg3": 32.1},
            {"TeamName": "Duke", "Season": 2025, "OffFg3": 34.2},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_point_distribution(team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["team_id"] == 73
        assert "y" not in call_args[1]["params"]
        assert len(result.data) == 2

    def test_get_point_distribution_with_conference(self, mock_api):
        """Test getting point distribution filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "ConfShort": "ACC"},
            {"TeamName": "North Carolina", "ConfShort": "ACC"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_point_distribution(year=2025, conference="ACC")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["c"] == "ACC"

    def test_get_point_distribution_conference_requires_year(self, mock_api):
        """Test that conference filter requires year parameter."""
        with pytest.raises(
            ValidationError, match="'year' parameter is required when filtering"
        ):
            mock_api.get_point_distribution(conference="ACC")

    def test_get_point_distribution_with_conf_only(self, mock_api):
        """Test getting conference-only point distribution stats."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "ConfOnly": "true"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_point_distribution(year=2025, conf_only=True)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["conf_only"] == "true"
        assert result.data[0]["ConfOnly"] == "true"

    def test_get_point_distribution_no_params_raises_error(self, mock_api):
        """Test that missing year and team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_point_distribution()

    def test_get_point_distribution_conf_only_alone_raises_error(self, mock_api):
        """Test that conf_only without year/team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_point_distribution(conf_only=True)

    def test_get_point_distribution_all_params(self, mock_api):
        """Test with all optional params combined."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"TeamName": "Duke"}]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_point_distribution(
            year=2025, conference="B10", conf_only=True
        )

        call_args = mock_api._client.get.call_args
        params = call_args[1]["params"]
        assert params["y"] == 2025
        assert params["c"] == "B10"
        assert params["conf_only"] == "true"

    def test_get_point_distribution_to_dataframe(self, mock_api):
        """Test converting point distribution response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "OffFg3": 34.2},
            {"TeamName": "Kansas", "OffFg3": 32.1},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_point_distribution(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert list(df.columns) == ["TeamName", "OffFg3"]
        assert df.iloc[0]["TeamName"] == "Duke"


class TestGetTeams:
    """Tests for the get_teams method."""

    def test_get_teams_by_year(self, mock_api):
        """Test getting teams by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "TeamName": "Duke",
                "TeamID": 73,
                "ConfShort": "ACC",
                "Coach": "Jon Scheyer",
                "Arena": "Cameron Indoor Stadium",
                "ArenaCity": "Durham",
                "ArenaState": "NC",
            },
            {
                "Season": 2025,
                "TeamName": "North Carolina",
                "TeamID": 216,
                "ConfShort": "ACC",
                "Coach": "Hubert Davis",
                "Arena": "Dean E. Smith Center",
                "ArenaCity": "Chapel Hill",
                "ArenaState": "NC",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_teams(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "teams"
        assert call_args[1]["params"]["y"] == 2025
        assert "c" not in call_args[1]["params"]
        assert isinstance(result, APIResponse)
        assert len(result.data) == 2
        assert result.data[0]["TeamName"] == "Duke"
        assert result.data[0]["TeamID"] == 73

    def test_get_teams_with_conference(self, mock_api):
        """Test getting teams filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "TeamName": "Villanova",
                "TeamID": 347,
                "ConfShort": "BE",
                "Coach": "Kyle Neptune",
                "Arena": "Finneran Pavilion",
                "ArenaCity": "Villanova",
                "ArenaState": "PA",
            },
            {
                "Season": 2025,
                "TeamName": "UConn",
                "TeamID": 63,
                "ConfShort": "BE",
                "Coach": "Dan Hurley",
                "Arena": "Harry A. Gampel Pavilion",
                "ArenaCity": "Storrs",
                "ArenaState": "CT",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_teams(year=2025, conference="BE")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["c"] == "BE"
        assert len(result.data) == 2
        assert all(team["ConfShort"] == "BE" for team in result.data)

    def test_get_teams_all_response_fields(self, mock_api):
        """Test that all documented response fields are present."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "TeamName": "Kansas",
                "TeamID": 140,
                "ConfShort": "B12",
                "Coach": "Bill Self",
                "Arena": "Allen Fieldhouse",
                "ArenaCity": "Lawrence",
                "ArenaState": "KS",
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_teams(year=2025)
        team = result.data[0]

        # Verify all documented fields
        assert "Season" in team
        assert "TeamName" in team
        assert "TeamID" in team
        assert "ConfShort" in team
        assert "Coach" in team
        assert "Arena" in team
        assert "ArenaCity" in team
        assert "ArenaState" in team

        # Verify types
        assert isinstance(team["Season"], int)
        assert isinstance(team["TeamName"], str)
        assert isinstance(team["TeamID"], int)
        assert isinstance(team["ConfShort"], str)
        assert isinstance(team["Coach"], str)
        assert isinstance(team["Arena"], str)
        assert isinstance(team["ArenaCity"], str)
        assert isinstance(team["ArenaState"], str)

    def test_get_teams_to_dataframe(self, mock_api):
        """Test converting teams response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "TeamName": "Duke",
                "TeamID": 73,
                "ConfShort": "ACC",
                "Coach": "Jon Scheyer",
                "Arena": "Cameron Indoor Stadium",
                "ArenaCity": "Durham",
                "ArenaState": "NC",
            },
            {
                "Season": 2025,
                "TeamName": "Kentucky",
                "TeamID": 143,
                "ConfShort": "SEC",
                "Coach": "Mark Pope",
                "Arena": "Rupp Arena",
                "ArenaCity": "Lexington",
                "ArenaState": "KY",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_teams(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert "Season" in df.columns
        assert "TeamName" in df.columns
        assert "TeamID" in df.columns
        assert "ConfShort" in df.columns
        assert "Coach" in df.columns
        assert "Arena" in df.columns
        assert "ArenaCity" in df.columns
        assert "ArenaState" in df.columns
        assert df.iloc[0]["TeamName"] == "Duke"
        assert df.iloc[1]["TeamName"] == "Kentucky"

    def test_get_teams_iteration(self, mock_api):
        """Test that APIResponse supports iteration for teams."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "TeamID": 73, "ConfShort": "ACC"},
            {"TeamName": "UNC", "TeamID": 216, "ConfShort": "ACC"},
            {"TeamName": "NC State", "TeamID": 217, "ConfShort": "ACC"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_teams(year=2025)

        # Test iteration
        team_ids = [team["TeamID"] for team in result]
        assert team_ids == [73, 216, 217]

        # Test len
        assert len(result) == 3

    def test_get_teams_find_by_name(self, mock_api):
        """Test finding a team by name from teams response."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "TeamID": 73, "ConfShort": "ACC"},
            {"TeamName": "Kentucky", "TeamID": 143, "ConfShort": "SEC"},
            {"TeamName": "Kansas", "TeamID": 140, "ConfShort": "B12"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_teams(year=2025)

        # Find Duke as shown in docstring example
        duke = [t for t in result.data if t["TeamName"] == "Duke"][0]
        assert duke["TeamID"] == 73
        assert duke["ConfShort"] == "ACC"


class TestGetFanmatch:
    """Tests for the get_fanmatch method."""

    def test_get_fanmatch_with_string_date(self, mock_api):
        """Test getting fanmatch predictions with string date."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "GameID": 12345,
                "DateOfGame": "2025-03-15",
                "Visitor": "Duke",
                "Home": "North Carolina",
                "HomeRank": 5,
                "VisitorRank": 3,
                "HomePred": 78.5,
                "VisitorPred": 75.2,
                "HomeWP": 55.3,
                "PredTempo": 68.5,
                "ThrillScore": 82.1,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-03-15")

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "fanmatch"
        assert call_args[1]["params"]["d"] == "2025-03-15"
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["Season"] == 2025
        assert result.data[0]["GameID"] == 12345
        assert result.data[0]["Visitor"] == "Duke"
        assert result.data[0]["Home"] == "North Carolina"

    def test_get_fanmatch_with_date_object(self, mock_api):
        """Test getting fanmatch predictions with date object."""
        from datetime import date

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "GameID": 12346,
                "DateOfGame": "2025-03-16",
                "Visitor": "Kansas",
                "Home": "Kentucky",
                "HomeRank": 10,
                "VisitorRank": 8,
                "HomePred": 72.0,
                "VisitorPred": 70.5,
                "HomeWP": 52.1,
                "PredTempo": 65.2,
                "ThrillScore": 75.0,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch(date(2025, 3, 16))

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["d"] == "2025-03-16"
        assert result.data[0]["DateOfGame"] == "2025-03-16"

    def test_get_fanmatch_all_response_fields(self, mock_api):
        """Test that all documented response fields are present."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "GameID": 12347,
                "DateOfGame": "2025-03-17",
                "Visitor": "Gonzaga",
                "Home": "UCLA",
                "HomeRank": 15,
                "VisitorRank": 12,
                "HomePred": 68.3,
                "VisitorPred": 65.8,
                "HomeWP": 58.7,
                "PredTempo": 70.1,
                "ThrillScore": 68.9,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-03-17")
        game = result.data[0]

        # Verify all documented fields
        assert "Season" in game
        assert "GameID" in game
        assert "DateOfGame" in game
        assert "Visitor" in game
        assert "Home" in game
        assert "HomeRank" in game
        assert "VisitorRank" in game
        assert "HomePred" in game
        assert "VisitorPred" in game
        assert "HomeWP" in game
        assert "PredTempo" in game
        assert "ThrillScore" in game

        # Verify types
        assert isinstance(game["Season"], int)
        assert isinstance(game["GameID"], int)
        assert isinstance(game["DateOfGame"], str)
        assert isinstance(game["Visitor"], str)
        assert isinstance(game["Home"], str)
        assert isinstance(game["HomeRank"], int)
        assert isinstance(game["VisitorRank"], int)
        assert isinstance(game["HomePred"], float)
        assert isinstance(game["VisitorPred"], float)
        assert isinstance(game["HomeWP"], float)
        assert isinstance(game["PredTempo"], float)
        assert isinstance(game["ThrillScore"], float)

    def test_get_fanmatch_multiple_games(self, mock_api):
        """Test getting multiple games for a date."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "GameID": 12348,
                "DateOfGame": "2025-03-18",
                "Visitor": "Auburn",
                "Home": "Tennessee",
                "HomeRank": 4,
                "VisitorRank": 6,
                "HomePred": 75.0,
                "VisitorPred": 72.0,
                "HomeWP": 62.5,
                "PredTempo": 66.0,
                "ThrillScore": 79.5,
            },
            {
                "Season": 2025,
                "GameID": 12349,
                "DateOfGame": "2025-03-18",
                "Visitor": "Arizona",
                "Home": "Oregon",
                "HomeRank": 20,
                "VisitorRank": 18,
                "HomePred": 70.0,
                "VisitorPred": 68.5,
                "HomeWP": 54.0,
                "PredTempo": 72.5,
                "ThrillScore": 65.0,
            },
            {
                "Season": 2025,
                "GameID": 12350,
                "DateOfGame": "2025-03-18",
                "Visitor": "Purdue",
                "Home": "Indiana",
                "HomeRank": 25,
                "VisitorRank": 2,
                "HomePred": 65.0,
                "VisitorPred": 78.0,
                "HomeWP": 28.5,
                "PredTempo": 64.0,
                "ThrillScore": 72.0,
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-03-18")

        assert len(result.data) == 3
        assert result.data[0]["Visitor"] == "Auburn"
        assert result.data[1]["Visitor"] == "Arizona"
        assert result.data[2]["Visitor"] == "Purdue"

    def test_get_fanmatch_find_close_games(self, mock_api):
        """Test finding close games (HomeWP between 40-60%)."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "GameID": 12351,
                "DateOfGame": "2025-03-19",
                "Visitor": "Team A",
                "Home": "Team B",
                "HomeRank": 30,
                "VisitorRank": 35,
                "HomePred": 70.0,
                "VisitorPred": 68.0,
                "HomeWP": 55.0,  # Close game
                "PredTempo": 68.0,
                "ThrillScore": 70.0,
            },
            {
                "Season": 2025,
                "GameID": 12352,
                "DateOfGame": "2025-03-19",
                "Visitor": "Team C",
                "Home": "Team D",
                "HomeRank": 5,
                "VisitorRank": 100,
                "HomePred": 85.0,
                "VisitorPred": 60.0,
                "HomeWP": 92.0,  # Not close
                "PredTempo": 70.0,
                "ThrillScore": 45.0,
            },
            {
                "Season": 2025,
                "GameID": 12353,
                "DateOfGame": "2025-03-19",
                "Visitor": "Team E",
                "Home": "Team F",
                "HomeRank": 40,
                "VisitorRank": 42,
                "HomePred": 65.0,
                "VisitorPred": 64.5,
                "HomeWP": 50.5,  # Close game
                "PredTempo": 65.0,
                "ThrillScore": 60.0,
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-03-19")

        # Filter for close games as shown in docstring example
        close_games = [g for g in result.data if 40 <= g["HomeWP"] <= 60]

        assert len(close_games) == 2
        assert close_games[0]["GameID"] == 12351
        assert close_games[1]["GameID"] == 12353

    def test_get_fanmatch_to_dataframe(self, mock_api):
        """Test converting fanmatch response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "GameID": 12354,
                "DateOfGame": "2025-03-20",
                "Visitor": "Duke",
                "Home": "UNC",
                "HomeRank": 10,
                "VisitorRank": 8,
                "HomePred": 75.0,
                "VisitorPred": 73.0,
                "HomeWP": 54.0,
                "PredTempo": 68.0,
                "ThrillScore": 85.0,
            },
            {
                "Season": 2025,
                "GameID": 12355,
                "DateOfGame": "2025-03-20",
                "Visitor": "Kansas",
                "Home": "Missouri",
                "HomeRank": 50,
                "VisitorRank": 15,
                "HomePred": 65.0,
                "VisitorPred": 72.0,
                "HomeWP": 35.0,
                "PredTempo": 66.0,
                "ThrillScore": 70.0,
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-03-20")
        df = result.to_dataframe()

        assert len(df) == 2
        assert "Season" in df.columns
        assert "GameID" in df.columns
        assert "Visitor" in df.columns
        assert "Home" in df.columns
        assert "HomeWP" in df.columns
        assert "ThrillScore" in df.columns
        assert df.iloc[0]["Visitor"] == "Duke"
        assert df.iloc[1]["Visitor"] == "Kansas"

    def test_get_fanmatch_empty_date(self, mock_api):
        """Test getting fanmatch for a date with no games."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-07-15")

        assert len(result.data) == 0
        assert isinstance(result.data, list)

    def test_get_fanmatch_iteration(self, mock_api):
        """Test that APIResponse supports iteration."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"Season": 2025, "GameID": 1, "Visitor": "A", "Home": "B", "HomeWP": 50.0},
            {"Season": 2025, "GameID": 2, "Visitor": "C", "Home": "D", "HomeWP": 60.0},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_fanmatch("2025-03-21")

        # Test iteration
        game_ids = [game["GameID"] for game in result]
        assert game_ids == [1, 2]

        # Test len
        assert len(result) == 2
