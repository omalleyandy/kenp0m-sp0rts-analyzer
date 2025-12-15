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
    with patch.object(KenPomAPI, "__init__", lambda self, **kwargs: None):  # noqa: ARG005
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

        mock_api.get_misc_stats(
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

        mock_api.get_point_distribution(year=2025, conference="ACC")

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

        mock_api.get_point_distribution(year=2025, conference="B10", conf_only=True)

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


class TestGetConferences:
    """Tests for the get_conferences method."""

    def test_get_conferences_by_year(self, mock_api):
        """Test getting conferences by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "ConfID": 1,
                "ConfShort": "ACC",
                "ConfLong": "Atlantic Coast Conference",
            },
            {
                "Season": 2025,
                "ConfID": 2,
                "ConfShort": "B10",
                "ConfLong": "Big Ten Conference",
            },
            {
                "Season": 2025,
                "ConfID": 3,
                "ConfShort": "B12",
                "ConfLong": "Big 12 Conference",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "conferences"
        assert call_args[1]["params"]["y"] == 2025
        assert isinstance(result, APIResponse)
        assert len(result.data) == 3
        assert result.data[0]["ConfShort"] == "ACC"
        assert result.data[1]["ConfShort"] == "B10"
        assert result.data[2]["ConfShort"] == "B12"

    def test_get_conferences_all_response_fields(self, mock_api):
        """Test that all documented response fields are present."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "ConfID": 4,
                "ConfShort": "SEC",
                "ConfLong": "Southeastern Conference",
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2025)
        conf = result.data[0]

        # Verify all documented fields
        assert "Season" in conf
        assert "ConfID" in conf
        assert "ConfShort" in conf
        assert "ConfLong" in conf

        # Verify types
        assert isinstance(conf["Season"], int)
        assert isinstance(conf["ConfID"], int)
        assert isinstance(conf["ConfShort"], str)
        assert isinstance(conf["ConfLong"], str)

    def test_get_conferences_to_dataframe(self, mock_api):
        """Test converting conferences response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "ConfID": 1,
                "ConfShort": "ACC",
                "ConfLong": "Atlantic Coast Conference",
            },
            {
                "Season": 2025,
                "ConfID": 2,
                "ConfShort": "BE",
                "ConfLong": "Big East Conference",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert "Season" in df.columns
        assert "ConfID" in df.columns
        assert "ConfShort" in df.columns
        assert "ConfLong" in df.columns
        assert df.iloc[0]["ConfShort"] == "ACC"
        assert df.iloc[1]["ConfShort"] == "BE"

    def test_get_conferences_iteration(self, mock_api):
        """Test that APIResponse supports iteration for conferences."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"Season": 2025, "ConfID": 1, "ConfShort": "ACC", "ConfLong": "ACC"},
            {"Season": 2025, "ConfID": 2, "ConfShort": "SEC", "ConfLong": "SEC"},
            {"Season": 2025, "ConfID": 3, "ConfShort": "B10", "ConfLong": "Big Ten"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2025)

        # Test iteration
        conf_shorts = [conf["ConfShort"] for conf in result]
        assert conf_shorts == ["ACC", "SEC", "B10"]

        # Test len
        assert len(result) == 3

    def test_get_conferences_find_by_short_name(self, mock_api):
        """Test finding a conference by short name from response."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "ConfID": 1,
                "ConfShort": "ACC",
                "ConfLong": "Atlantic Coast Conference",
            },
            {
                "Season": 2025,
                "ConfID": 2,
                "ConfShort": "BE",
                "ConfLong": "Big East Conference",
            },
            {
                "Season": 2025,
                "ConfID": 3,
                "ConfShort": "WCC",
                "ConfLong": "West Coast Conference",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2025)

        # Find specific conference as shown in docstring example pattern
        wcc = [c for c in result.data if c["ConfShort"] == "WCC"][0]
        assert wcc["ConfID"] == 3
        assert wcc["ConfLong"] == "West Coast Conference"

    def test_get_conferences_all_major_conferences(self, mock_api):
        """Test response includes major conferences with correct data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "ConfID": 1,
                "ConfShort": "ACC",
                "ConfLong": "Atlantic Coast Conference",
            },
            {
                "Season": 2025,
                "ConfID": 2,
                "ConfShort": "B10",
                "ConfLong": "Big Ten Conference",
            },
            {
                "Season": 2025,
                "ConfID": 3,
                "ConfShort": "B12",
                "ConfLong": "Big 12 Conference",
            },
            {
                "Season": 2025,
                "ConfID": 4,
                "ConfShort": "SEC",
                "ConfLong": "Southeastern Conference",
            },
            {
                "Season": 2025,
                "ConfID": 5,
                "ConfShort": "BE",
                "ConfLong": "Big East Conference",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2025)

        # Verify major conferences are present
        conf_shorts = {c["ConfShort"] for c in result.data}
        assert "ACC" in conf_shorts
        assert "B10" in conf_shorts
        assert "B12" in conf_shorts
        assert "SEC" in conf_shorts
        assert "BE" in conf_shorts

    def test_get_conferences_different_season(self, mock_api):
        """Test getting conferences for different seasons."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2024,
                "ConfID": 1,
                "ConfShort": "ACC",
                "ConfLong": "Atlantic Coast Conference",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_conferences(year=2024)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2024
        assert result.data[0]["Season"] == 2024

    def test_get_conferences_docstring_example(self, mock_api):
        """Test the example shown in the docstring works correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "ConfID": 1,
                "ConfShort": "ACC",
                "ConfLong": "Atlantic Coast Conference",
            },
            {
                "Season": 2025,
                "ConfID": 2,
                "ConfShort": "SEC",
                "ConfLong": "Southeastern Conference",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        # Replicate docstring example
        conferences = mock_api.get_conferences(year=2025)
        output_lines = []
        for conf in conferences.data:
            output_lines.append(f"{conf['ConfShort']}: {conf['ConfLong']}")

        assert output_lines == [
            "ACC: Atlantic Coast Conference",
            "SEC: Southeastern Conference",
        ]


class TestGetArchive:
    """Tests for the get_archive method - Ratings Archive API endpoint."""

    def test_get_archive_by_date(self, mock_api):
        """Test getting archived ratings by date."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "ArchiveDate": "2025-02-15",
                "Season": 2025,
                "Preseason": "false",
                "TeamName": "Auburn",
                "Seed": None,
                "Event": None,
                "ConfShort": "SEC",
                "AdjEM": 30.5,
                "RankAdjEM": 1,
                "AdjOE": 118.5,
                "RankAdjOE": 5,
                "AdjDE": 88.0,
                "RankAdjDE": 1,
                "AdjTempo": 68.5,
                "RankAdjTempo": 150,
                "AdjEMFinal": 32.5,
                "RankAdjEMFinal": 1,
                "AdjOEFinal": 120.5,
                "RankAdjOEFinal": 5,
                "AdjDEFinal": 88.0,
                "RankAdjDEFinal": 1,
                "AdjTempoFinal": 69.2,
                "RankAdjTempoFinal": 140,
                "RankChg": 0,
                "AdjEMChg": 2.0,
                "AdjTChg": 0.7,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(archive_date="2025-02-15")

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "archive"
        assert call_args[1]["params"]["d"] == "2025-02-15"
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["TeamName"] == "Auburn"
        assert result.data[0]["ArchiveDate"] == "2025-02-15"

    def test_get_archive_with_date_object(self, mock_api):
        """Test getting archived ratings with date object."""
        from datetime import date

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"ArchiveDate": "2025-03-01", "TeamName": "Duke", "AdjEM": 28.5}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(archive_date=date(2025, 3, 1))

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["d"] == "2025-03-01"
        assert result.data[0]["ArchiveDate"] == "2025-03-01"

    def test_get_archive_preseason(self, mock_api):
        """Test getting preseason ratings."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "ArchiveDate": "Preseason",
                "Season": 2025,
                "Preseason": "true",
                "TeamName": "Kansas",
                "AdjEM": 25.5,
                "RankAdjEM": 3,
                "AdjEMFinal": 26.1,
                "RankAdjEMFinal": 5,
                "RankChg": -2,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(preseason=True, year=2025)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["preseason"] == "true"
        assert call_args[1]["params"]["y"] == 2025
        assert result.data[0]["Preseason"] == "true"

    def test_get_archive_preseason_requires_year(self, mock_api):
        """Test that preseason mode requires year parameter."""
        with pytest.raises(
            ValidationError, match="'year' parameter is required when preseason=True"
        ):
            mock_api.get_archive(preseason=True)

    def test_get_archive_with_team_id(self, mock_api):
        """Test getting archived ratings filtered by team_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"ArchiveDate": "2025-03-01", "TeamName": "Duke", "AdjEM": 28.5}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(archive_date="2025-03-01", team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["d"] == "2025-03-01"
        assert call_args[1]["params"]["team_id"] == 73
        assert len(result.data) == 1

    def test_get_archive_with_conference(self, mock_api):
        """Test getting archived ratings filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"ArchiveDate": "2025-02-15", "TeamName": "Auburn", "ConfShort": "SEC"},
            {"ArchiveDate": "2025-02-15", "TeamName": "Tennessee", "ConfShort": "SEC"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(archive_date="2025-02-15", conference="SEC")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["c"] == "SEC"
        assert len(result.data) == 2
        assert all(team["ConfShort"] == "SEC" for team in result.data)

    def test_get_archive_no_params_raises_error(self, mock_api):
        """Test that missing date and preseason raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Either 'archive_date' or 'preseason=True' with 'year' is required",
        ):
            mock_api.get_archive()

    def test_get_archive_all_response_fields(self, mock_api):
        """Test that all documented response fields are present."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "ArchiveDate": "2025-02-15",
                "Season": 2025,
                "Preseason": "false",
                "TeamName": "Houston",
                "Seed": 2,
                "Event": "Elite Eight",
                "ConfShort": "B12",
                "AdjEM": 30.2,
                "RankAdjEM": 2,
                "AdjOE": 118.5,
                "RankAdjOE": 10,
                "AdjDE": 88.3,
                "RankAdjDE": 2,
                "AdjTempo": 66.1,
                "RankAdjTempo": 270,
                "AdjEMFinal": 30.5,
                "RankAdjEMFinal": 2,
                "AdjOEFinal": 119.0,
                "RankAdjOEFinal": 8,
                "AdjDEFinal": 88.5,
                "RankAdjDEFinal": 2,
                "AdjTempoFinal": 66.5,
                "RankAdjTempoFinal": 265,
                "RankChg": 0,
                "AdjEMChg": 0.3,
                "AdjTChg": 0.4,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(archive_date="2025-02-15")
        team = result.data[0]

        # Verify all documented fields
        assert team["ArchiveDate"] == "2025-02-15"
        assert team["Season"] == 2025
        assert team["Preseason"] == "false"
        assert team["TeamName"] == "Houston"
        assert team["Seed"] == 2
        assert team["Event"] == "Elite Eight"
        assert team["ConfShort"] == "B12"
        assert team["AdjEM"] == 30.2
        assert team["RankAdjEM"] == 2
        assert team["AdjOE"] == 118.5
        assert team["RankAdjOE"] == 10
        assert team["AdjDE"] == 88.3
        assert team["RankAdjDE"] == 2
        assert team["AdjTempo"] == 66.1
        assert team["RankAdjTempo"] == 270
        assert team["AdjEMFinal"] == 30.5
        assert team["RankAdjEMFinal"] == 2
        assert team["AdjOEFinal"] == 119.0
        assert team["RankAdjOEFinal"] == 8
        assert team["AdjDEFinal"] == 88.5
        assert team["RankAdjDEFinal"] == 2
        assert team["AdjTempoFinal"] == 66.5
        assert team["RankAdjTempoFinal"] == 265
        assert team["RankChg"] == 0
        assert team["AdjEMChg"] == 0.3
        assert team["AdjTChg"] == 0.4

    def test_get_archive_to_dataframe(self, mock_api):
        """Test converting archive response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "ArchiveDate": "2025-02-15",
                "TeamName": "Auburn",
                "AdjEM": 30.5,
                "RankChg": 0,
            },
            {
                "ArchiveDate": "2025-02-15",
                "TeamName": "Houston",
                "AdjEM": 30.2,
                "RankChg": 1,
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(archive_date="2025-02-15")
        df = result.to_dataframe()

        assert len(df) == 2
        assert "ArchiveDate" in df.columns
        assert "TeamName" in df.columns
        assert "AdjEM" in df.columns
        assert "RankChg" in df.columns
        assert df.iloc[0]["TeamName"] == "Auburn"

    def test_get_archive_preseason_compare_to_final(self, mock_api):
        """Test comparing preseason rankings to final rankings."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "TeamName": "Kansas",
                "RankAdjEM": 3,
                "RankAdjEMFinal": 5,
                "RankChg": -2,
            },
            {
                "TeamName": "Duke",
                "RankAdjEM": 5,
                "RankAdjEMFinal": 3,
                "RankChg": 2,
            },
            {
                "TeamName": "Auburn",
                "RankAdjEM": 10,
                "RankAdjEMFinal": 1,
                "RankChg": 9,
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_archive(preseason=True, year=2025)

        # Verify we can compare preseason to final as shown in docstring
        for team in result.data[:10]:
            preseason_rank = team["RankAdjEM"]
            final_rank = team["RankAdjEMFinal"]
            change = team["RankChg"]
            assert change == preseason_rank - final_rank


class TestGetFourFactors:
    """Tests for the get_four_factors method - Four Factors API endpoint."""

    def test_get_four_factors_by_year(self, mock_api):
        """Test getting Four Factors by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "DataThrough": "2025-03-01",
                "ConfOnly": "false",
                "TeamName": "Duke",
                "Season": 2025,
                "eFG_Pct": 55.2,
                "RankeFG_Pct": 15,
                "TO_Pct": 16.5,
                "RankTO_Pct": 50,
                "OR_Pct": 32.1,
                "RankOR_Pct": 75,
                "FT_Rate": 38.5,
                "RankFT_Rate": 25,
                "DeFG_Pct": 46.2,
                "RankDeFG_Pct": 20,
                "DTO_Pct": 19.5,
                "RankDTO_Pct": 30,
                "DOR_Pct": 25.8,
                "RankDOR_Pct": 100,
                "DFT_Rate": 28.5,
                "RankDFT_Rate": 80,
                "OE": 115.5,
                "RankOE": 20,
                "DE": 95.2,
                "RankDE": 15,
                "Tempo": 68.5,
                "RankTempo": 150,
                "AdjOE": 118.5,
                "RankAdjOE": 10,
                "AdjDE": 92.5,
                "RankAdjDE": 8,
                "AdjTempo": 69.2,
                "RankAdjTempo": 140,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "four-factors"
        assert call_args[1]["params"]["y"] == 2025
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["TeamName"] == "Duke"
        assert result.data[0]["eFG_Pct"] == 55.2

    def test_get_four_factors_by_team_id(self, mock_api):
        """Test getting Four Factors by team_id for historical data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "Season": 2024, "eFG_Pct": 53.5, "AdjOE": 115.0},
            {"TeamName": "Duke", "Season": 2025, "eFG_Pct": 55.2, "AdjOE": 118.5},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["team_id"] == 73
        assert "y" not in call_args[1]["params"]
        assert len(result.data) == 2

    def test_get_four_factors_with_conference(self, mock_api):
        """Test getting Four Factors filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Houston", "ConfShort": "B12", "eFG_Pct": 54.1},
            {"TeamName": "Kansas", "ConfShort": "B12", "eFG_Pct": 52.8},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(year=2025, conference="B12")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["c"] == "B12"
        assert len(result.data) == 2

    def test_get_four_factors_conference_requires_year(self, mock_api):
        """Test that conference filter requires year parameter."""
        with pytest.raises(
            ValidationError, match="'year' parameter is required when filtering"
        ):
            mock_api.get_four_factors(conference="A10")

    def test_get_four_factors_with_conf_only(self, mock_api):
        """Test getting conference-only Four Factors stats."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "ConfOnly": "true", "eFG_Pct": 56.0},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(year=2025, conf_only=True)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["conf_only"] == "true"
        assert result.data[0]["ConfOnly"] == "true"

    def test_get_four_factors_no_params_raises_error(self, mock_api):
        """Test that missing year and team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_four_factors()

    def test_get_four_factors_conf_only_alone_raises_error(self, mock_api):
        """Test that conf_only without year/team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_four_factors(conf_only=True)

    def test_get_four_factors_all_response_fields(self, mock_api):
        """Test that all documented response fields are present."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "DataThrough": "2025-03-15",
                "ConfOnly": "false",
                "TeamName": "Auburn",
                "Season": 2025,
                "eFG_Pct": 56.5,
                "RankeFG_Pct": 5,
                "TO_Pct": 14.2,
                "RankTO_Pct": 25,
                "OR_Pct": 35.5,
                "RankOR_Pct": 20,
                "FT_Rate": 42.1,
                "RankFT_Rate": 10,
                "DeFG_Pct": 44.5,
                "RankDeFG_Pct": 5,
                "DTO_Pct": 21.2,
                "RankDTO_Pct": 15,
                "DOR_Pct": 23.5,
                "RankDOR_Pct": 50,
                "DFT_Rate": 25.8,
                "RankDFT_Rate": 40,
                "OE": 118.5,
                "RankOE": 5,
                "DE": 88.0,
                "RankDE": 1,
                "Tempo": 70.5,
                "RankTempo": 100,
                "AdjOE": 120.5,
                "RankAdjOE": 5,
                "AdjDE": 88.0,
                "RankAdjDE": 1,
                "AdjTempo": 71.2,
                "RankAdjTempo": 95,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(year=2025)
        team = result.data[0]

        # Verify all documented fields
        assert team["DataThrough"] == "2025-03-15"
        assert team["ConfOnly"] == "false"
        assert team["TeamName"] == "Auburn"
        assert team["Season"] == 2025
        # Offensive Four Factors
        assert team["eFG_Pct"] == 56.5
        assert team["RankeFG_Pct"] == 5
        assert team["TO_Pct"] == 14.2
        assert team["RankTO_Pct"] == 25
        assert team["OR_Pct"] == 35.5
        assert team["RankOR_Pct"] == 20
        assert team["FT_Rate"] == 42.1
        assert team["RankFT_Rate"] == 10
        # Defensive Four Factors
        assert team["DeFG_Pct"] == 44.5
        assert team["RankDeFG_Pct"] == 5
        assert team["DTO_Pct"] == 21.2
        assert team["RankDTO_Pct"] == 15
        assert team["DOR_Pct"] == 23.5
        assert team["RankDOR_Pct"] == 50
        assert team["DFT_Rate"] == 25.8
        assert team["RankDFT_Rate"] == 40
        # Efficiency and Tempo
        assert team["OE"] == 118.5
        assert team["RankOE"] == 5
        assert team["DE"] == 88.0
        assert team["RankDE"] == 1
        assert team["Tempo"] == 70.5
        assert team["RankTempo"] == 100
        assert team["AdjOE"] == 120.5
        assert team["RankAdjOE"] == 5
        assert team["AdjDE"] == 88.0
        assert team["RankAdjDE"] == 1
        assert team["AdjTempo"] == 71.2
        assert team["RankAdjTempo"] == 95

    def test_get_four_factors_all_params(self, mock_api):
        """Test with all optional params combined."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"TeamName": "Duke"}]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        mock_api.get_four_factors(
            year=2025, team_id=73, conference="ACC", conf_only=True
        )

        call_args = mock_api._client.get.call_args
        params = call_args[1]["params"]
        assert params["y"] == 2025
        assert params["team_id"] == 73
        assert params["c"] == "ACC"
        assert params["conf_only"] == "true"

    def test_get_four_factors_to_dataframe(self, mock_api):
        """Test converting Four Factors response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "eFG_Pct": 55.2, "AdjOE": 118.5},
            {"TeamName": "Kansas", "eFG_Pct": 52.8, "AdjOE": 116.0},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert list(df.columns) == ["TeamName", "eFG_Pct", "AdjOE"]
        assert df.iloc[0]["TeamName"] == "Duke"

    def test_get_four_factors_find_best_shooting(self, mock_api):
        """Test finding best shooting teams by eFG% rank."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "eFG_Pct": 55.2, "RankeFG_Pct": 15},
            {"TeamName": "Auburn", "eFG_Pct": 56.5, "RankeFG_Pct": 5},
            {"TeamName": "Kansas", "eFG_Pct": 52.8, "RankeFG_Pct": 50},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_four_factors(year=2025)
        df = result.to_dataframe()

        # Replicate docstring example: find best shooting teams
        best_shooting = df.nsmallest(10, "RankeFG_Pct")

        assert len(best_shooting) == 3
        assert best_shooting.iloc[0]["TeamName"] == "Auburn"  # Rank 5
        assert best_shooting.iloc[1]["TeamName"] == "Duke"  # Rank 15


class TestGetRatings:
    """Tests for the get_ratings method - Ratings API endpoint."""

    def test_get_ratings_by_year(self, mock_api):
        """Test getting ratings by year."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "DataThrough": "Games through 2025-03-01",
                "Season": 2025,
                "TeamName": "Auburn",
                "Seed": 1,
                "ConfShort": "SEC",
                "Coach": "Bruce Pearl",
                "Wins": 25,
                "Losses": 3,
                "AdjEM": 32.5,
                "RankAdjEM": 1,
                "Pythag": 0.975,
                "RankPythag": 1,
                "AdjOE": 120.5,
                "RankAdjOE": 5,
                "OE": 118.2,
                "RankOE": 8,
                "AdjDE": 88.0,
                "RankAdjDE": 1,
                "DE": 90.1,
                "RankDE": 3,
                "Tempo": 68.5,
                "RankTempo": 150,
                "AdjTempo": 69.2,
                "RankAdjTempo": 140,
                "Luck": 0.02,
                "RankLuck": 100,
                "SOS": 8.5,
                "RankSOS": 15,
                "SOSO": 5.2,
                "RankSOSO": 20,
                "SOSD": 3.3,
                "RankSOSD": 12,
                "NCSOS": 4.1,
                "RankNCSOS": 50,
                "Event": None,
                "APL_Off": 16.8,
                "RankAPL_Off": 200,
                "APL_Def": 17.2,
                "RankAPL_Def": 180,
                "ConfAPL_Off": 16.5,
                "RankConfAPL_Off": 5,
                "ConfAPL_Def": 17.0,
                "RankConfAPL_Def": 8,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025)

        mock_api._client.get.assert_called_once()
        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["endpoint"] == "ratings"
        assert call_args[1]["params"]["y"] == 2025
        assert isinstance(result, APIResponse)
        assert len(result.data) == 1
        assert result.data[0]["TeamName"] == "Auburn"
        assert result.data[0]["AdjEM"] == 32.5

    def test_get_ratings_by_team_id(self, mock_api):
        """Test getting historical ratings by team_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"Season": 2023, "TeamName": "Duke", "AdjEM": 28.5, "RankAdjEM": 3},
            {"Season": 2024, "TeamName": "Duke", "AdjEM": 25.2, "RankAdjEM": 8},
            {"Season": 2025, "TeamName": "Duke", "AdjEM": 27.1, "RankAdjEM": 5},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["team_id"] == 73
        assert "y" not in call_args[1]["params"]
        assert len(result.data) == 3
        # Historical data should span multiple seasons
        seasons = [r["Season"] for r in result.data]
        assert seasons == [2023, 2024, 2025]

    def test_get_ratings_by_year_and_team_id(self, mock_api):
        """Test getting ratings with both year and team_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"Season": 2025, "TeamName": "Duke", "AdjEM": 27.1, "RankAdjEM": 5}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025, team_id=73)

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["team_id"] == 73
        assert len(result.data) == 1

    def test_get_ratings_with_conference(self, mock_api):
        """Test getting ratings filtered by conference."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Auburn", "ConfShort": "SEC", "AdjEM": 32.5},
            {"TeamName": "Tennessee", "ConfShort": "SEC", "AdjEM": 28.1},
            {"TeamName": "Kentucky", "ConfShort": "SEC", "AdjEM": 25.5},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025, conference="SEC")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["y"] == 2025
        assert call_args[1]["params"]["c"] == "SEC"
        assert len(result.data) == 3
        assert all(team["ConfShort"] == "SEC" for team in result.data)

    def test_get_ratings_conference_requires_year(self, mock_api):
        """Test that conference filter requires year parameter."""
        with pytest.raises(
            ValidationError, match="'year' parameter is required when filtering"
        ):
            mock_api.get_ratings(conference="B12")

    def test_get_ratings_no_params_raises_error(self, mock_api):
        """Test that missing year and team_id raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Either 'year' or 'team_id' parameter is required"
        ):
            mock_api.get_ratings()

    def test_get_ratings_all_response_fields(self, mock_api):
        """Test that all documented response fields are properly returned."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                # Core info
                "DataThrough": "Games through 2025-03-15",
                "Season": 2025,
                "TeamName": "Houston",
                "Seed": 2,
                "ConfShort": "B12",
                "Coach": "Kelvin Sampson",
                "Wins": 28,
                "Losses": 4,
                # Efficiency metrics
                "AdjEM": 30.2,
                "RankAdjEM": 2,
                "Pythag": 0.965,
                "RankPythag": 2,
                # Offensive efficiency
                "AdjOE": 118.5,
                "RankAdjOE": 10,
                "OE": 115.8,
                "RankOE": 15,
                # Defensive efficiency
                "AdjDE": 88.3,
                "RankAdjDE": 2,
                "DE": 89.5,
                "RankDE": 5,
                # Tempo
                "Tempo": 65.2,
                "RankTempo": 280,
                "AdjTempo": 66.1,
                "RankAdjTempo": 270,
                # Luck
                "Luck": -0.01,
                "RankLuck": 180,
                # Strength of schedule
                "SOS": 10.2,
                "RankSOS": 5,
                "SOSO": 6.5,
                "RankSOSO": 8,
                "SOSD": 3.7,
                "RankSOSD": 10,
                "NCSOS": 5.5,
                "RankNCSOS": 25,
                # Event
                "Event": "Elite Eight",
                # Average possession length
                "APL_Off": 15.5,
                "RankAPL_Off": 50,
                "APL_Def": 18.2,
                "RankAPL_Def": 300,
                "ConfAPL_Off": 15.8,
                "RankConfAPL_Off": 3,
                "ConfAPL_Def": 17.8,
                "RankConfAPL_Def": 5,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025)
        team = result.data[0]

        # Verify all documented fields are present
        # Core info
        assert team["DataThrough"] == "Games through 2025-03-15"
        assert team["Season"] == 2025
        assert team["TeamName"] == "Houston"
        assert team["Seed"] == 2
        assert team["ConfShort"] == "B12"
        assert team["Coach"] == "Kelvin Sampson"
        assert team["Wins"] == 28
        assert team["Losses"] == 4

        # Efficiency metrics
        assert team["AdjEM"] == 30.2
        assert team["RankAdjEM"] == 2
        assert team["Pythag"] == 0.965
        assert team["RankPythag"] == 2

        # Offensive efficiency
        assert team["AdjOE"] == 118.5
        assert team["RankAdjOE"] == 10
        assert team["OE"] == 115.8
        assert team["RankOE"] == 15

        # Defensive efficiency
        assert team["AdjDE"] == 88.3
        assert team["RankAdjDE"] == 2
        assert team["DE"] == 89.5
        assert team["RankDE"] == 5

        # Tempo
        assert team["Tempo"] == 65.2
        assert team["RankTempo"] == 280
        assert team["AdjTempo"] == 66.1
        assert team["RankAdjTempo"] == 270

        # Luck
        assert team["Luck"] == -0.01
        assert team["RankLuck"] == 180

        # Strength of schedule
        assert team["SOS"] == 10.2
        assert team["RankSOS"] == 5
        assert team["SOSO"] == 6.5
        assert team["RankSOSO"] == 8
        assert team["SOSD"] == 3.7
        assert team["RankSOSD"] == 10
        assert team["NCSOS"] == 5.5
        assert team["RankNCSOS"] == 25

        # Event
        assert team["Event"] == "Elite Eight"

        # Average possession length
        assert team["APL_Off"] == 15.5
        assert team["RankAPL_Off"] == 50
        assert team["APL_Def"] == 18.2
        assert team["RankAPL_Def"] == 300
        assert team["ConfAPL_Off"] == 15.8
        assert team["RankConfAPL_Off"] == 3
        assert team["ConfAPL_Def"] == 17.8
        assert team["RankConfAPL_Def"] == 5

    def test_get_ratings_to_dataframe(self, mock_api):
        """Test converting ratings response to DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Season": 2025,
                "TeamName": "Auburn",
                "AdjEM": 32.5,
                "RankAdjEM": 1,
                "AdjOE": 120.5,
                "AdjDE": 88.0,
            },
            {
                "Season": 2025,
                "TeamName": "Houston",
                "AdjEM": 30.2,
                "RankAdjEM": 2,
                "AdjOE": 118.5,
                "AdjDE": 88.3,
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025)
        df = result.to_dataframe()

        assert len(df) == 2
        assert "Season" in df.columns
        assert "TeamName" in df.columns
        assert "AdjEM" in df.columns
        assert "RankAdjEM" in df.columns
        assert df.iloc[0]["TeamName"] == "Auburn"
        assert df.iloc[1]["TeamName"] == "Houston"

    def test_get_ratings_iteration(self, mock_api):
        """Test that APIResponse supports iteration for ratings."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Auburn", "RankAdjEM": 1},
            {"TeamName": "Houston", "RankAdjEM": 2},
            {"TeamName": "Duke", "RankAdjEM": 3},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025)

        # Test iteration
        rankings = [team["RankAdjEM"] for team in result]
        assert rankings == [1, 2, 3]

        # Test len
        assert len(result) == 3

    def test_get_ratings_multiple_conferences(self, mock_api):
        """Test getting ratings for different conference filters."""
        # Test B12 conference
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Houston", "ConfShort": "B12", "AdjEM": 30.2},
            {"TeamName": "Kansas", "ConfShort": "B12", "AdjEM": 26.1},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025, conference="B12")

        call_args = mock_api._client.get.call_args
        assert call_args[1]["params"]["c"] == "B12"
        assert len(result.data) == 2

    def test_get_ratings_all_params(self, mock_api):
        """Test with year, team_id, and conference combined."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Duke", "Season": 2025, "ConfShort": "ACC"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        result = mock_api.get_ratings(year=2025, team_id=73, conference="ACC")

        call_args = mock_api._client.get.call_args
        params = call_args[1]["params"]
        assert params["y"] == 2025
        assert params["team_id"] == 73
        assert params["c"] == "ACC"
        assert len(result.data) == 1

    def test_get_ratings_docstring_example(self, mock_api):
        """Test the examples shown in the docstring work correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"TeamName": "Auburn", "RankAdjEM": 1, "AdjEM": 32.5},
            {"TeamName": "Houston", "RankAdjEM": 2, "AdjEM": 30.2},
            {"TeamName": "Duke", "RankAdjEM": 3, "AdjEM": 29.8},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_api._client.get.return_value = mock_response

        # Replicate docstring example: get top_10
        ratings = mock_api.get_ratings(year=2025)
        top_10 = ratings.data[:10]

        assert len(top_10) == 3  # Only 3 in mock
        assert top_10[0]["TeamName"] == "Auburn"
        assert top_10[0]["RankAdjEM"] == 1
