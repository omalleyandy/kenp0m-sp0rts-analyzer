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
