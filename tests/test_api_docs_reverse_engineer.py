"""Tests for API documentation reverse engineering module."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

# Import directly from the module to avoid package dependencies
_api_docs_path = (
    Path(__file__).parent.parent
    / "src/kenp0m_sp0rts_analyzer/api_docs_reverse_engineer.py"
)
_spec = importlib.util.spec_from_file_location("api_docs_reverse_engineer", _api_docs_path)
_api_docs = importlib.util.module_from_spec(_spec)
# Register module in sys.modules before execution (needed for dataclasses)
sys.modules["api_docs_reverse_engineer"] = _api_docs
_spec.loader.exec_module(_api_docs)

APIDocsReverseEngineer = _api_docs.APIDocsReverseEngineer
EndpointDiscovery = _api_docs.EndpointDiscovery

# Import comparison script functions
_compare_path = Path(__file__).parent.parent / "scripts/compare_api_docs.py"
_compare_spec = importlib.util.spec_from_file_location("compare_api_docs", _compare_path)
_compare_module = importlib.util.module_from_spec(_compare_spec)
_compare_spec.loader.exec_module(_compare_module)

normalize_endpoint_name = _compare_module.normalize_endpoint_name


class TestParameterTableParsing:
    """Tests for parameter table parsing logic."""

    def test_parse_parameter_table_basic(self):
        """Test parsing a basic parameter table."""
        html = """
        <table class="param-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Type</th>
                    <th>Required</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>y</code></td>
                    <td>integer</td>
                    <td>Conditional*</td>
                    <td>Year/Season ending year</td>
                </tr>
                <tr>
                    <td><code>team_id</code></td>
                    <td>integer</td>
                    <td>Conditional*</td>
                    <td>Team ID</td>
                </tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        params = engineer._parse_parameter_table(table)
        
        assert len(params) == 2
        assert params[0]["name"] == "y"
        assert params[0]["type"] == "integer"
        assert params[0]["required"] is True  # "Conditional" should be treated as required
        assert params[0]["description"] == "Year/Season ending year"
        
        assert params[1]["name"] == "team_id"
        assert params[1]["type"] == "integer"

    def test_parse_parameter_table_without_code_tag(self):
        """Test parsing parameter name without code tag."""
        html = """
        <table class="param-table">
            <tbody>
                <tr>
                    <td>y</td>
                    <td>integer</td>
                    <td>Yes</td>
                    <td>Year</td>
                </tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        params = engineer._parse_parameter_table(table)
        
        assert len(params) == 1
        assert params[0]["name"] == "y"
        assert params[0]["required"] is True  # "Yes" should be treated as required

    def test_parse_parameter_table_optional_parameter(self):
        """Test parsing optional parameters."""
        html = """
        <table class="param-table">
            <tbody>
                <tr>
                    <td><code>c</code></td>
                    <td>string</td>
                    <td>No</td>
                    <td>Conference</td>
                </tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        params = engineer._parse_parameter_table(table)
        
        assert len(params) == 1
        assert params[0]["name"] == "c"
        assert params[0]["required"] is False

    def test_parse_parameter_table_empty_rows(self):
        """Test handling empty or invalid rows."""
        html = """
        <table class="param-table">
            <tbody>
                <tr>
                    <td><code>y</code></td>
                    <td>integer</td>
                </tr>
                <tr></tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        params = engineer._parse_parameter_table(table)
        
        # Should only parse valid rows
        assert len(params) == 1
        assert params[0]["name"] == "y"


class TestResponseTableParsing:
    """Tests for response field table parsing logic."""

    def test_parse_response_table_basic(self):
        """Test parsing a basic response table."""
        html = """
        <table class="response-table">
            <thead>
                <tr>
                    <th>Field</th>
                    <th>Type</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>DataThrough</code></td>
                    <td>string</td>
                    <td>Date through which data is current</td>
                </tr>
                <tr>
                    <td><code>Season</code></td>
                    <td>integer</td>
                    <td>Season year</td>
                </tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        fields = engineer._parse_response_table(table)
        
        assert len(fields) == 2
        assert fields[0]["name"] == "DataThrough"
        assert fields[0]["type"] == "string"
        assert fields[0]["description"] == "Date through which data is current"
        
        assert fields[1]["name"] == "Season"
        assert fields[1]["type"] == "integer"
        assert fields[1]["description"] == "Season year"

    def test_parse_response_table_without_code_tag(self):
        """Test parsing response field without code tag."""
        html = """
        <table class="response-table">
            <tbody>
                <tr>
                    <td>TeamName</td>
                    <td>string</td>
                    <td>Team name</td>
                </tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        fields = engineer._parse_response_table(table)
        
        assert len(fields) == 1
        assert fields[0]["name"] == "TeamName"
        assert fields[0]["type"] == "string"

    def test_parse_response_table_missing_columns(self):
        """Test parsing response table with missing type/description."""
        html = """
        <table class="response-table">
            <tbody>
                <tr>
                    <td><code>Field1</code></td>
                    <td>string</td>
                </tr>
                <tr>
                    <td><code>Field2</code></td>
                </tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        engineer = APIDocsReverseEngineer()
        fields = engineer._parse_response_table(table)
        
        assert len(fields) == 2
        assert fields[0]["name"] == "Field1"
        assert fields[0]["type"] == "string"
        assert fields[0]["description"] == ""
        
        assert fields[1]["name"] == "Field2"
        assert fields[1]["type"] == ""
        assert fields[1]["description"] == ""


class TestEndpointNameNormalization:
    """Tests for endpoint name normalization in comparison script."""

    def test_normalize_endpoint_name_basic(self):
        """Test basic normalization."""
        assert normalize_endpoint_name("ratings") == "ratings"
        assert normalize_endpoint_name("RATINGS") == "ratings"

    def test_normalize_endpoint_name_with_hyphen(self):
        """Test normalization handles hyphens."""
        assert normalize_endpoint_name("point-dist") == "pointdist"
        assert normalize_endpoint_name("four-factors") == "fourfactors"
        assert normalize_endpoint_name("misc-stats") == "miscstats"

    def test_normalize_endpoint_name_with_underscore(self):
        """Test normalization handles underscores."""
        assert normalize_endpoint_name("point_dist") == "pointdist"
        assert normalize_endpoint_name("misc_stats") == "miscstats"

    def test_normalize_endpoint_name_with_dot(self):
        """Test normalization handles dots."""
        assert normalize_endpoint_name("point.dist") == "pointdist"

    def test_normalize_endpoint_name_mixed(self):
        """Test normalization handles mixed separators."""
        assert normalize_endpoint_name("point-dist") == normalize_endpoint_name("pointdist")
        assert normalize_endpoint_name("point_dist") == normalize_endpoint_name("pointdist")
        assert normalize_endpoint_name("point.dist") == normalize_endpoint_name("pointdist")


class TestEndpointDiscovery:
    """Tests for EndpointDiscovery dataclass."""

    def test_endpoint_discovery_creation(self):
        """Test creating an EndpointDiscovery instance."""
        endpoint = EndpointDiscovery(
            name="ratings",
            parameters=[{"name": "y", "type": "integer"}],
            response_fields=[{"name": "TeamName", "type": "string"}],
            source="documentation",
        )
        
        assert endpoint.name == "ratings"
        assert len(endpoint.parameters) == 1
        assert len(endpoint.response_fields) == 1
        assert endpoint.source == "documentation"

    def test_endpoint_discovery_defaults(self):
        """Test EndpointDiscovery with default values."""
        endpoint = EndpointDiscovery(name="test")
        
        assert endpoint.name == "test"
        assert endpoint.parameters == []
        assert endpoint.response_fields == []
        assert endpoint.examples == []
        assert endpoint.source == "documentation"


class TestHTMLParsing:
    """Tests for HTML parsing with mock data."""

    def test_parse_section_with_endpoint(self):
        """Test parsing an endpoint section from HTML."""
        html = """
        <section id="ratings">
            <h3 class="endpoint-header">Ratings</h3>
            <p>Description</p>
            <h4>Parameters</h4>
            <div class="table-wrapper">
                <table class="param-table">
                    <tbody>
                        <tr>
                            <td><code>y</code></td>
                            <td>integer</td>
                            <td>Conditional*</td>
                            <td>Year</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <h4>Response Fields</h4>
            <div class="table-wrapper">
                <table class="response-table">
                    <tbody>
                        <tr>
                            <td><code>TeamName</code></td>
                            <td>string</td>
                            <td>Team name</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>
        """
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find("section", id="ratings")
        
        # Create a minimal engineer instance for testing
        engineer = APIDocsReverseEngineer.__new__(APIDocsReverseEngineer)
        
        # Test parameter table extraction
        param_table = section.find("table", class_="param-table")
        assert param_table is not None
        params = engineer._parse_parameter_table(param_table)
        assert len(params) == 1
        assert params[0]["name"] == "y"
        
        # Test response table extraction
        response_table = section.find("table", class_="response-table")
        assert response_table is not None
        fields = engineer._parse_response_table(response_table)
        assert len(fields) == 1
        assert fields[0]["name"] == "TeamName"

    def test_parse_multiple_endpoints(self):
        """Test parsing multiple endpoint sections."""
        html = """
        <section id="ratings">
            <h3 class="endpoint-header">Ratings</h3>
        </section>
        <section id="archive">
            <h3 class="endpoint-header">Archive</h3>
        </section>
        <section id="teams-endpoint">
            <h3 class="endpoint-header">Teams</h3>
        </section>
        """
        soup = BeautifulSoup(html, "html.parser")
        sections = soup.find_all("section", id=True)
        
        endpoint_names = [s.get("id") for s in sections]
        assert "ratings" in endpoint_names
        assert "archive" in endpoint_names
        assert "teams-endpoint" in endpoint_names

