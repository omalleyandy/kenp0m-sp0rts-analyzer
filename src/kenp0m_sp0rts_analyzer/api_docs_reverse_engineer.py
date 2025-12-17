"""Reverse engineer KenPom API documentation using Chrome DevTools Protocol.

This module uses CDP to access the KenPom API documentation page, monitor
network requests, and extract endpoint information for validation and discovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

# Handle imports for both module and script execution
# Import directly from module files to avoid __init__.py dependency chain
import sys
from pathlib import Path
import importlib.util

try:
    from .browser import StealthBrowser
    from .scraper import KENPOM_BASE_URL, KenPomScraper
except ImportError:
    # Allow running as script - import directly from module files
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Import browser module directly
    browser_spec = importlib.util.spec_from_file_location(
        "browser",
        Path(__file__).parent / "browser.py"
    )
    browser_module = importlib.util.module_from_spec(browser_spec)
    browser_spec.loader.exec_module(browser_module)
    StealthBrowser = browser_module.StealthBrowser
    
    # Import utils first (needed by scraper)
    utils_spec = importlib.util.spec_from_file_location(
        "kenp0m_sp0rts_analyzer.utils",
        Path(__file__).parent / "utils.py"
    )
    utils_module = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_module)
    
    # Import browser module (needed by scraper)
    browser_spec_for_scraper = importlib.util.spec_from_file_location(
        "kenp0m_sp0rts_analyzer.browser",
        Path(__file__).parent / "browser.py"
    )
    browser_module_for_scraper = importlib.util.module_from_spec(browser_spec_for_scraper)
    browser_spec_for_scraper.loader.exec_module(browser_module_for_scraper)
    
    # Create a mock package structure for scraper's relative imports
    class MockPackage:
        utils = utils_module
        browser = browser_module_for_scraper
    
    # Import scraper module directly
    scraper_spec = importlib.util.spec_from_file_location(
        "kenp0m_sp0rts_analyzer.scraper",
        Path(__file__).parent / "scraper.py"
    )
    scraper_module = importlib.util.module_from_spec(scraper_spec)
    # Set up the parent package for relative imports
    scraper_module.__package__ = "kenp0m_sp0rts_analyzer"
    # Inject dependencies
    import sys
    sys.modules['kenp0m_sp0rts_analyzer'] = MockPackage()
    sys.modules['kenp0m_sp0rts_analyzer.utils'] = utils_module
    sys.modules['kenp0m_sp0rts_analyzer.browser'] = browser_module_for_scraper
    scraper_spec.loader.exec_module(scraper_module)
    KENPOM_BASE_URL = scraper_module.KENPOM_BASE_URL
    KenPomScraper = scraper_module.KenPomScraper

logger = logging.getLogger(__name__)

KENPOM_API_DOCS_URL = f"{KENPOM_BASE_URL}/api-documentation.php"
KENPOM_API_BASE = f"{KENPOM_BASE_URL}/api.php"


@dataclass
class NetworkRequest:
    """Captured network request from CDP."""

    url: str
    method: str
    headers: dict[str, str]
    post_data: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Parse URL parameters."""
        parsed = urlparse(self.url)
        self.params = parse_qs(parsed.query)
        self.endpoint = self.params.get("endpoint", [None])[0] if self.params else None


@dataclass
class NetworkResponse:
    """Captured network response from CDP."""

    request_id: str
    url: str
    status: int
    headers: dict[str, str]
    body: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EndpointDiscovery:
    """Discovered endpoint information."""

    name: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    response_fields: list[dict[str, Any]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source: str = "documentation"  # "documentation", "network", or "both"


@dataclass
class APIReverseEngineeringResults:
    """Results from reverse engineering the API documentation."""

    endpoints_discovered: list[EndpointDiscovery] = field(default_factory=list)
    network_requests: list[NetworkRequest] = field(default_factory=list)
    network_responses: list[NetworkResponse] = field(default_factory=list)
    page_html: str = ""
    page_text: str = ""
    documentation_tables: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class APIDocsReverseEngineer:
    """Reverse engineer KenPom API documentation using Chrome DevTools Protocol.

    This class uses CDP to:
    1. Access the API documentation page (with login if needed)
    2. Monitor all network requests/responses
    3. Extract documentation structure from HTML
    4. Parse endpoint information from tables and text
    5. Compare with existing implementation

    Example:
        ```python
        async with APIDocsReverseEngineer() as engineer:
            results = await engineer.reverse_engineer()
            print(f"Discovered {len(results.endpoints_discovered)} endpoints")
        ```
    """

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        headless: bool = False,
        save_output: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize the reverse engineer.

        Args:
            email: KenPom email for login. Uses env var if not provided.
            password: KenPom password for login. Uses env var if not provided.
            headless: Run browser in headless mode.
            save_output: Save captured data to files.
            output_dir: Directory to save output files.
        """
        self._scraper = KenPomScraper(email=email, password=password, headless=headless)
        self._page: Any = None
        self._cdp_session: Any = None
        self.save_output = save_output
        self.output_dir = output_dir or Path("reports/api_reverse_engineering")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Network monitoring storage
        self._network_requests: dict[str, NetworkRequest] = {}
        self._network_responses: dict[str, NetworkResponse] = {}
        self._request_response_map: dict[str, str] = {}
        # Queue for request IDs that need response bodies fetched
        self._pending_response_bodies: set[str] = set()

    async def __aenter__(self) -> "APIDocsReverseEngineer":
        """Async context manager entry."""
        await self._scraper.start()
        # Use the scraper's page or create a new one
        self._page = self._scraper._page or await self._scraper._browser.new_page()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self._scraper.close()

    async def _setup_cdp_monitoring(self) -> None:
        """Set up Chrome DevTools Protocol monitoring for network requests."""
        if not self._page:
            raise RuntimeError("Page not initialized")

        # Get CDP session from the browser
        self._cdp_session = await self._scraper._browser.get_cdp_session(self._page)

        # Enable Network domain
        await self._cdp_session.send("Network.enable")
        await self._cdp_session.send("Page.enable")
        await self._cdp_session.send("Runtime.enable")

        # Set up request interception (optional - for modification)
        # await self._cdp_session.send("Network.setRequestInterception", {
        #     "patterns": [{"urlPattern": "*api.php*"}]
        # })

        # Listen for network events
        self._cdp_session.on(
            "Network.requestWillBeSent",
            self._on_request_will_be_sent,
        )
        self._cdp_session.on(
            "Network.responseReceived",
            self._on_response_received,
        )
        self._cdp_session.on(
            "Network.loadingFinished",
            self._on_loading_finished,
        )

        logger.info("CDP network monitoring enabled")

    def _on_request_will_be_sent(self, params: dict[str, Any]) -> None:
        """Handle Network.requestWillBeSent event."""
        request = params.get("request", {})
        request_id = params.get("requestId", "")

        network_request = NetworkRequest(
            url=request.get("url", ""),
            method=request.get("method", "GET"),
            headers=request.get("headers", {}),
            post_data=request.get("postData"),
        )

        self._network_requests[request_id] = network_request

        # Log API requests
        if "api.php" in network_request.url:
            logger.info(
                f"API Request: {network_request.method} {network_request.url}"
            )

    def _on_response_received(self, params: dict[str, Any]) -> None:
        """Handle Network.responseReceived event."""
        response = params.get("response", {})
        request_id = params.get("requestId", "")

        network_response = NetworkResponse(
            request_id=request_id,
            url=response.get("url", ""),
            status=response.get("status", 0),
            headers=response.get("headers", {}),
        )

        self._network_responses[request_id] = network_response
        self._request_response_map[request_id] = request_id

    def _on_loading_finished(self, params: dict[str, Any]) -> None:
        """Handle Network.loadingFinished event - queue response body fetch.
        
        Note: This must be synchronous because CDP event handlers don't await
        async callbacks. We queue the request_id and fetch bodies later.
        """
        request_id = params.get("requestId", "")

        if request_id in self._network_responses:
            # Queue for async fetching later
            self._pending_response_bodies.add(request_id)
    
    async def _fetch_response_bodies(self) -> None:
        """Fetch response bodies for all pending requests.
        
        This is called after page load to retrieve response bodies
        for all queued requests.
        """
        if not self._pending_response_bodies:
            return
        
        logger.debug(f"Fetching {len(self._pending_response_bodies)} response bodies...")
        
        for request_id in list(self._pending_response_bodies):
            if request_id in self._network_responses:
                try:
                    # Get response body via CDP
                    response_body = await self._cdp_session.send(
                        "Network.getResponseBody", {"requestId": request_id}
                    )
                    if response_body and "body" in response_body:
                        self._network_responses[request_id].body = response_body["body"]
                except Exception as e:
                    logger.debug(f"Could not get response body for {request_id}: {e}")
        
        self._pending_response_bodies.clear()

    async def reverse_engineer(self) -> APIReverseEngineeringResults:
        """Perform reverse engineering of the API documentation.

        Returns:
            APIReverseEngineeringResults with all discovered information.
        """
        logger.info("Starting API documentation reverse engineering...")

        # Set up CDP monitoring
        await self._setup_cdp_monitoring()

        # Navigate directly to API documentation page first
        # This page will redirect to login if authentication is needed
        logger.info(f"Navigating to {KENPOM_API_DOCS_URL}")
        try:
            await self._page.goto(KENPOM_API_DOCS_URL, wait_until="domcontentloaded", timeout=60000)
            await self._page.wait_for_load_state("networkidle", timeout=60000)
        except Exception as e:
            logger.warning(f"Page load timeout: {e}. Continuing with current state...")
            # Try to wait for at least domcontentloaded
            try:
                await self._page.wait_for_load_state("domcontentloaded", timeout=30000)
            except Exception:
                pass

        # Check if we were redirected to login page
        current_url = self._page.url.lower()
        if "login" in current_url:
            logger.info("Redirected to login page, attempting to login...")
            await self._scraper.login()
            # Navigate back to API documentation after login
            logger.info(f"Navigating to {KENPOM_API_DOCS_URL} after login...")
            try:
                await self._page.goto(KENPOM_API_DOCS_URL, wait_until="domcontentloaded", timeout=60000)
                await self._page.wait_for_load_state("networkidle", timeout=60000)
            except Exception as e:
                logger.warning(f"Page load timeout after login: {e}. Continuing...")
                try:
                    await self._page.wait_for_load_state("domcontentloaded", timeout=30000)
                except Exception:
                    pass

        # Wait a bit for any dynamic content to load
        await asyncio.sleep(3)

        # Fetch response bodies for all captured requests
        await self._fetch_response_bodies()

        # Extract page content
        page_html = await self._page.content()
        page_text = await self._page.inner_text("body")

        # Parse documentation
        endpoints = await self._parse_documentation(page_html, page_text)

        # Collect network requests/responses
        api_requests = [
            req
            for req in self._network_requests.values()
            if "api.php" in req.url
        ]
        api_responses = [
            resp
            for resp in self._network_responses.values()
            if "api.php" in resp.url
        ]

        # Extract documentation tables
        tables = self._extract_documentation_tables(page_html)

        results = APIReverseEngineeringResults(
            endpoints_discovered=endpoints,
            network_requests=api_requests,
            network_responses=api_responses,
            page_html=page_html,
            page_text=page_text,
            documentation_tables=tables,
        )

        # Save results if requested
        if self.save_output:
            await self._save_results(results)

        logger.info(
            f"Reverse engineering complete: {len(endpoints)} endpoints discovered, "
            f"{len(api_requests)} API requests captured"
        )

        return results

    async def _parse_documentation(
        self, html: str, text: str
    ) -> list[EndpointDiscovery]:
        """Parse documentation HTML to extract endpoint information.

        Args:
            html: Page HTML content.
            text: Page text content.

        Returns:
            List of discovered endpoints.
        """
        endpoints: list[EndpointDiscovery] = []
        soup = BeautifulSoup(html, "html.parser")

        # Find all endpoint sections - look for <section> elements with id attributes
        # or <h3> elements with class "endpoint-header"
        sections = soup.find_all("section", id=True)
        endpoint_headers = soup.find_all("h3", class_="endpoint-header")
        
        # Also look for h3 elements that might be endpoint headers
        all_h3 = soup.find_all("h3")
        
        # Combine all potential endpoint markers
        endpoint_markers = []
        for section in sections:
            # Get the h3 inside the section
            h3 = section.find("h3", class_="endpoint-header")
            if h3:
                endpoint_markers.append((section, h3))
        
        for h3 in endpoint_headers:
            # Find parent section
            section = h3.find_parent("section")
            if section and (section, h3) not in endpoint_markers:
                endpoint_markers.append((section, h3))
        
        # Process each endpoint section
        for section, h3 in endpoint_markers:
            heading_text = h3.get_text().strip()
            
            # Extract endpoint name from heading or section id
            endpoint_name = None
            if section and section.get("id"):
                endpoint_name = section.get("id").lower()
            else:
                # Try to extract from heading text
                endpoint_pattern = re.compile(
                    r"(ratings|archive|teams|conferences|fanmatch|four-factors|"
                    r"misc-stats|height|pointdist|point.dist|point-dist)",
                    re.IGNORECASE,
                )
                match = endpoint_pattern.search(heading_text)
                if match:
                    endpoint_name = match.group(1).lower()
                    endpoint_name = endpoint_name.replace(" ", "-")
            
            if not endpoint_name:
                continue
            
            # Normalize endpoint name
            endpoint_name = endpoint_name.replace("_", "-").replace(".", "-")
            
            # Create endpoint discovery
            endpoint = EndpointDiscovery(name=endpoint_name, source="documentation")
            
            # Find parameter table (look for table with class "param-table" in this section)
            param_table = section.find("table", class_="param-table") if section else None
            if not param_table and h3:
                # Try finding it after the h3
                current = h3.find_next_sibling()
                while current and current.name != "section":
                    if current.name == "table" and "param-table" in current.get("class", []):
                        param_table = current
                        break
                    # Check inside divs
                    if current.name == "div":
                        param_table = current.find("table", class_="param-table")
                        if param_table:
                            break
                    current = current.find_next_sibling()
            
            if param_table:
                endpoint.parameters = self._parse_parameter_table(param_table)
            
            # Find response fields table (look for table with class "response-table")
            response_table = section.find("table", class_="response-table") if section else None
            if not response_table and h3:
                # Try finding it after the h3
                current = h3.find_next_sibling()
                while current and current.name != "section":
                    if current.name == "table" and "response-table" in current.get("class", []):
                        response_table = current
                        break
                    # Check inside divs
                    if current.name == "div":
                        response_table = current.find("table", class_="response-table")
                        if response_table:
                            break
                    current = current.find_next_sibling()
            
            if response_table:
                endpoint.response_fields = self._parse_response_table(response_table)
            
            # Look for example code blocks
            if section:
                code_blocks = section.find_all(["code", "pre"])
            else:
                code_blocks = h3.find_all_next(["code", "pre"], limit=10)
            
            for code in code_blocks:
                code_text = code.get_text().strip()
                if "api.php" in code_text or endpoint_name in code_text.lower():
                    endpoint.examples.append(code_text)
            
            endpoints.append(endpoint)
        
        # Fallback: if no sections found, try the old method
        if not endpoints:
            endpoint_pattern = re.compile(
                r"(ratings|archive|teams|conferences|fanmatch|four-factors|"
                r"misc-stats|height|pointdist|point.dist|point-dist)",
                re.IGNORECASE,
            )
            headings = soup.find_all(["h1", "h2", "h3", "h4"])
            for heading in headings:
                heading_text = heading.get_text().strip()
                if endpoint_pattern.search(heading_text):
                    endpoint_name = endpoint_pattern.search(heading_text).group(1).lower()
                    endpoint_name = endpoint_name.replace(" ", "-")
                    
                    endpoint = EndpointDiscovery(name=endpoint_name, source="documentation")
                    
                    # Look for tables after this heading
                    current = heading.find_next_sibling()
                    while current:
                        if current.name == "table":
                            if "param-table" in current.get("class", []):
                                endpoint.parameters = self._parse_parameter_table(current)
                            elif "response-table" in current.get("class", []):
                                endpoint.response_fields = self._parse_response_table(current)
                        elif current.name == "div":
                            param_table = current.find("table", class_="param-table")
                            if param_table:
                                endpoint.parameters = self._parse_parameter_table(param_table)
                            response_table = current.find("table", class_="response-table")
                            if response_table:
                                endpoint.response_fields = self._parse_response_table(response_table)
                        current = current.find_next_sibling()
                        if current and current.name in ["h1", "h2", "h3", "h4"]:
                            break
                    
                    endpoints.append(endpoint)

        # Also check network requests for endpoints we might have missed
        for request in self._network_requests.values():
            if request.endpoint and request.endpoint not in [
                e.name for e in endpoints
            ]:
                endpoint = EndpointDiscovery(
                    name=request.endpoint, source="network"
                )
                # Extract parameters from URL
                if request.params:
                    for param, values in request.params.items():
                        if param != "endpoint":
                            endpoint.parameters.append(
                                {
                                    "name": param,
                                    "value": values[0] if values else None,
                                    "source": "network_request",
                                }
                            )
                endpoints.append(endpoint)

        return endpoints

    def _parse_parameter_table(self, table: Any) -> list[dict[str, Any]]:
        """Parse a parameter documentation table.

        Args:
            table: BeautifulSoup table element.

        Returns:
            List of parameter dictionaries.
        """
        parameters = []
        
        # Find tbody if it exists, otherwise use all rows
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")
        
        # Skip header row (first row in thead or first row overall)
        thead = table.find("thead")
        if thead:
            # Skip thead rows
            rows = [r for r in rows if r not in thead.find_all("tr")]
        else:
            # Skip first row if it looks like a header
            if rows and rows[0].find("th"):
                rows = rows[1:]

        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                # Extract parameter name (may be in <code> tag)
                param_name_cell = cells[0]
                param_name_elem = param_name_cell.find("code")
                if param_name_elem:
                    param_name = param_name_elem.get_text().strip()
                else:
                    param_name = param_name_cell.get_text().strip()
                
                param_type = cells[1].get_text().strip() if len(cells) > 1 else ""
                param_required = (
                    cells[2].get_text().strip() if len(cells) > 2 else ""
                )
                param_desc = (
                    cells[3].get_text().strip() if len(cells) > 3 else ""
                )

                if param_name:  # Only add if we have a parameter name
                    parameters.append(
                        {
                            "name": param_name,
                            "type": param_type,
                            "required": "required" in param_required.lower()
                            or "yes" in param_required.lower()
                            or "conditional" in param_required.lower(),
                            "description": param_desc,
                            "source": "documentation_table",
                        }
                    )

        return parameters

    def _parse_response_table(self, table: Any) -> list[dict[str, Any]]:
        """Parse a response fields documentation table.

        Args:
            table: BeautifulSoup table element.

        Returns:
            List of response field dictionaries with name, type, and description.
        """
        fields = []
        
        # Find tbody if it exists, otherwise use all rows
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")
        
        # Skip header row (first row in thead or first row overall)
        thead = table.find("thead")
        if thead:
            # Skip thead rows
            rows = [r for r in rows if r not in thead.find_all("tr")]
        else:
            # Skip first row if it looks like a header
            if rows and rows[0].find("th"):
                rows = rows[1:]

        for row in rows:
            cells = row.find_all(["td", "th"])
            if cells and len(cells) >= 1:
                # Extract field name (may be in <code> tag)
                field_name_cell = cells[0]
                field_name_elem = field_name_cell.find("code")
                if field_name_elem:
                    field_name = field_name_elem.get_text().strip()
                else:
                    field_name = field_name_cell.get_text().strip()
                
                # Extract type (second column)
                field_type = cells[1].get_text().strip() if len(cells) > 1 else ""
                
                # Extract description (third column)
                field_desc = cells[2].get_text().strip() if len(cells) > 2 else ""
                
                if field_name:
                    fields.append({
                        "name": field_name,
                        "type": field_type,
                        "description": field_desc,
                        "source": "documentation_table",
                    })

        return fields

    def _extract_documentation_tables(self, html: str) -> list[dict[str, Any]]:
        """Extract all documentation tables from HTML.

        Args:
            html: Page HTML content.

        Returns:
            List of table data dictionaries.
        """
        soup = BeautifulSoup(html, "html.parser")
        tables = []

        for table in soup.find_all("table"):
            table_data = {"headers": [], "rows": []}

            # Extract headers
            header_row = table.find("tr")
            if header_row:
                headers = header_row.find_all(["th", "td"])
                table_data["headers"] = [h.get_text().strip() for h in headers]

            # Extract rows
            for row in table.find_all("tr")[1:]:
                cells = row.find_all(["td", "th"])
                row_data = [cell.get_text().strip() for cell in cells]
                if row_data:
                    table_data["rows"].append(row_data)

            if table_data["headers"] or table_data["rows"]:
                tables.append(table_data)

        return tables

    async def _save_results(self, results: APIReverseEngineeringResults) -> None:
        """Save reverse engineering results to files.

        Args:
            results: Results to save.
        """
        timestamp_str = results.timestamp.strftime("%Y%m%d_%H%M%S")

        # Save HTML
        html_file = self.output_dir / f"api_docs_{timestamp_str}.html"
        html_file.write_text(results.page_html, encoding="utf-8")
        logger.info(f"Saved HTML to {html_file}")

        # Save text
        text_file = self.output_dir / f"api_docs_{timestamp_str}.txt"
        text_file.write_text(results.page_text, encoding="utf-8")
        logger.info(f"Saved text to {text_file}")

        # Save JSON results
        json_file = self.output_dir / f"api_reverse_engineering_{timestamp_str}.json"

        # Convert to JSON-serializable format
        json_data = {
            "timestamp": results.timestamp.isoformat(),
            "endpoints_discovered": [
                {
                    "name": e.name,
                    "parameters": e.parameters,
                    "response_fields": e.response_fields,
                    "examples": e.examples,
                    "source": e.source,
                }
                for e in results.endpoints_discovered
            ],
            "network_requests": [
                {
                    "url": r.url,
                    "method": r.method,
                    "headers": r.headers,
                    "endpoint": r.endpoint,
                    "params": {k: v for k, v in r.params.items()},
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results.network_requests
            ],
            "network_responses": [
                {
                    "url": r.url,
                    "status": r.status,
                    "headers": r.headers,
                    "body_length": len(r.body) if r.body else 0,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results.network_responses
            ],
            "documentation_tables_count": len(results.documentation_tables),
        }

        json_file.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Saved JSON results to {json_file}")

        # Save network requests separately
        if results.network_requests:
            network_file = self.output_dir / f"network_requests_{timestamp_str}.json"
            network_data = [
                {
                    "url": r.url,
                    "method": r.method,
                    "endpoint": r.endpoint,
                    "params": {k: v for k, v in r.params.items()},
                }
                for r in results.network_requests
            ]
            network_file.write_text(
                json.dumps(network_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Saved network requests to {network_file}")


async def reverse_engineer_api_docs(
    email: str | None = None,
    password: str | None = None,
    headless: bool = False,
) -> APIReverseEngineeringResults:
    """Convenience function to reverse engineer API documentation.

    Args:
        email: KenPom email for login.
        password: KenPom password for login.
        headless: Run browser in headless mode.

    Returns:
        APIReverseEngineeringResults with discovered information.
    """
    async with APIDocsReverseEngineer(
        email=email, password=password, headless=headless
    ) as engineer:
        return await engineer.reverse_engineer()


if __name__ == "__main__":
    """Allow running this module directly for testing.
    
    Note: It's recommended to use the CLI script instead:
        python scripts/reverse_engineer_api_docs.py --email <email> --password <password>
    """
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main() -> None:
        """Main entry point when run as script."""
        email = os.getenv("KENPOM_EMAIL")
        password = os.getenv("KENPOM_PASSWORD")

        if not email or not password:
            print(
                "❌ Error: KENPOM_EMAIL and KENPOM_PASSWORD environment variables required."
            )
            print(
                "\n💡 Recommended: Use the CLI script instead:\n"
                "   python scripts/reverse_engineer_api_docs.py --email <email> --password <password>\n"
                "\nOr set environment variables:\n"
                "   export KENPOM_EMAIL=your-email@example.com\n"
                "   export KENPOM_PASSWORD=your-password\n"
                "   python -m kenp0m_sp0rts_analyzer.api_docs_reverse_engineer"
            )
            return

        print("🚀 Starting API documentation reverse engineering...")
        print("   (Use Ctrl+C to cancel)\n")
        
        try:
            results = await reverse_engineer_api_docs(
                email=email, password=password, headless=False
            )

            print(f"\n✅ Reverse engineering complete!")
            print(f"   📊 Endpoints discovered: {len(results.endpoints_discovered)}")
            print(f"   🌐 Network requests captured: {len(results.network_requests)}")
            print(f"   💾 Results saved to: reports/api_reverse_engineering/")
            print(f"\n💡 Next step: Run comparison script")
            print(f"   python scripts/compare_api_docs.py")
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Reverse engineering failed")

    asyncio.run(main())

