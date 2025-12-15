"""Tests for the KenPom API client."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import directly from the api_client module to avoid package dependencies
_api_client_path = Path(__file__).parent.parent / "src/kenp0m_sp0rts_analyzer/api_client.py"
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
