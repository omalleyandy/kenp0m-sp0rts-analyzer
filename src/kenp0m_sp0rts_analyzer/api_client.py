"""Official KenPom API client.

This module provides a Python wrapper for the official KenPom API,
which requires a separate API key purchase from kenpom.com.

API Documentation discovered endpoints:
- ratings: Team ratings with efficiency metrics
- teams: Team list with IDs, coaches, arenas
- conferences: Conference list
- fanmatch: Game predictions
- four-factors: Four Factors analysis
- misc-stats: Miscellaneous team statistics
- height: Height and experience data
- pointdist: Point distribution data

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

    # Using environment variable KENPOM_API_KEY
    api = KenPomAPI()

    # Get current season ratings
    ratings = api.get_ratings(year=2025)

    # Get Duke's historical data
    duke_history = api.get_ratings(team_id=73)

    # Get game predictions
    games = api.get_fanmatch(date="2025-03-15")
    ```
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class KenPomAPIError(Exception):
    """Base exception for KenPom API errors."""

    pass


class AuthenticationError(KenPomAPIError):
    """Raised when API authentication fails."""

    pass


class EndpointNotFoundError(KenPomAPIError):
    """Raised when an invalid endpoint is requested."""

    pass


class ValidationError(KenPomAPIError):
    """Raised when required parameters are missing."""

    pass


@dataclass
class APIResponse:
    """Container for API response data."""

    data: list[dict[str, Any]]
    endpoint: str
    params: dict[str, Any]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def to_dataframe(self):
        """Convert response to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.data)


class KenPomAPI:
    """Official KenPom API client.

    This client wraps the official KenPom API endpoints. It requires
    an API key, which can be purchased separately from a KenPom subscription
    at https://kenpom.com/register-api.php.

    Attributes:
        BASE_URL: The KenPom API base URL.
        ENDPOINTS: List of valid API endpoints.

    Example:
        ```python
        # Initialize with API key from environment
        api = KenPomAPI()

        # Or provide key directly
        api = KenPomAPI(api_key="your-api-key")

        # Get 2025 ratings
        ratings = api.get_ratings(year=2025)
        print(f"Top team: {ratings.data[0]['TeamName']}")

        # Convert to DataFrame
        df = ratings.to_dataframe()
        ```
    """

    BASE_URL = "https://kenpom.com/api.php"

    ENDPOINTS = [
        "ratings",
        "teams",
        "conferences",
        "fanmatch",
        "four-factors",
        "misc-stats",
        "height",
        "pointdist",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the KenPom API client.

        Args:
            api_key: KenPom API key. If not provided, reads from
                KENPOM_API_KEY environment variable.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.

        Raises:
            AuthenticationError: If no API key is provided or found.
        """
        self.api_key = api_key or os.getenv("KENPOM_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "KenPom API key not found. Set KENPOM_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.timeout = timeout
        self.max_retries = max_retries

        # Configure HTTP client with retry transport
        transport = httpx.HTTPTransport(retries=max_retries)
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "KenPomSportsAnalyzer/1.0",
            },
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> KenPomAPI:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Make a request to the KenPom API.

        Args:
            endpoint: API endpoint name.
            params: Query parameters.

        Returns:
            APIResponse containing the response data.

        Raises:
            AuthenticationError: If authentication fails.
            EndpointNotFoundError: If the endpoint doesn't exist.
            ValidationError: If required parameters are missing.
            KenPomAPIError: For other API errors.
        """
        request_params = {"endpoint": endpoint}
        if params:
            request_params.update(params)

        logger.debug(f"API request: {endpoint} with params {params}")

        try:
            response = self._client.get(self.BASE_URL, params=request_params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise KenPomAPIError(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise KenPomAPIError(f"Request failed: {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            raise KenPomAPIError(f"Invalid JSON response: {e}") from e

        # Handle API error responses
        if isinstance(data, dict) and "error" in data:
            error_msg = data["error"]
            if "Unauthorized" in error_msg:
                raise AuthenticationError(error_msg)
            elif "Endpoint Not Found" in error_msg:
                raise EndpointNotFoundError(f"Invalid endpoint: {endpoint}")
            elif "parameter is required" in error_msg:
                raise ValidationError(error_msg)
            else:
                raise KenPomAPIError(error_msg)

        return APIResponse(data=data, endpoint=endpoint, params=request_params)

    # ==================== Ratings Endpoints ====================

    def get_ratings(
        self,
        year: int | None = None,
        team_id: int | None = None,
    ) -> APIResponse:
        """Get team ratings.

        Either `year` or `team_id` must be provided. If `year` is provided,
        returns ratings for all teams in that season. If `team_id` is provided,
        returns historical ratings for that team across all seasons.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season).
            team_id: Team ID for historical data. Use get_teams() to find IDs.

        Returns:
            APIResponse with team ratings including:
            - AdjEM: Adjusted Efficiency Margin
            - AdjOE: Adjusted Offensive Efficiency
            - AdjDE: Adjusted Defensive Efficiency
            - AdjTempo: Adjusted Tempo
            - Pythag: Pythagorean win expectation
            - Luck: Luck factor
            - SOS/SOSO/SOSD: Strength of schedule metrics
            - NCSOS: Non-conference SOS
            - APL_Off/APL_Def: Average Possession Length
            - Seed: NCAA Tournament seed (if applicable)

        Raises:
            ValidationError: If neither year nor team_id is provided.

        Example:
            ```python
            # Get 2025 season ratings
            ratings = api.get_ratings(year=2025)
            top_10 = ratings.data[:10]

            # Get Duke's historical ratings
            duke = api.get_ratings(team_id=73)
            ```
        """
        params: dict[str, Any] = {}
        if year is not None:
            params["y"] = year
        if team_id is not None:
            params["team_id"] = team_id

        if not params:
            raise ValidationError(
                "Either 'year' or 'team_id' parameter is required for ratings endpoint"
            )

        return self._request("ratings", params)

    # ==================== Teams & Conferences ====================

    def get_teams(self, year: int) -> APIResponse:
        """Get list of all teams for a season.

        Args:
            year: Season year.

        Returns:
            APIResponse with team data including:
            - TeamID: Unique team identifier
            - TeamName: Team name
            - ConfShort: Conference abbreviation
            - Coach: Head coach name
            - Arena: Home arena name
            - ArenaCity/ArenaState: Arena location

        Example:
            ```python
            teams = api.get_teams(year=2025)
            duke = [t for t in teams.data if t['TeamName'] == 'Duke'][0]
            print(f"Duke's TeamID: {duke['TeamID']}")  # 73
            ```
        """
        return self._request("teams", {"y": year})

    def get_conferences(self, year: int) -> APIResponse:
        """Get list of all conferences for a season.

        Args:
            year: Season year.

        Returns:
            APIResponse with conference data including:
            - ConfID: Unique conference identifier
            - ConfShort: Conference abbreviation (e.g., 'ACC', 'B10')
            - ConfLong: Full conference name

        Example:
            ```python
            conferences = api.get_conferences(year=2025)
            for conf in conferences.data:
                print(f"{conf['ConfShort']}: {conf['ConfLong']}")
            ```
        """
        return self._request("conferences", {"y": year})

    # ==================== Game Predictions ====================

    def get_fanmatch(self, game_date: str | date) -> APIResponse:
        """Get game predictions for a specific date.

        Args:
            game_date: Date string in YYYY-MM-DD format or date object.

        Returns:
            APIResponse with game predictions including:
            - GameID: Unique game identifier
            - DateOfGame: Game date
            - Visitor/Home: Team names
            - VisitorRank/HomeRank: KenPom rankings
            - VisitorPred/HomePred: Predicted scores
            - HomeWP: Home team win probability (%)
            - PredTempo: Predicted game tempo
            - ThrillScore: Expected game excitement (0-100)

        Example:
            ```python
            from datetime import date

            # Using string
            games = api.get_fanmatch("2025-03-15")

            # Using date object
            games = api.get_fanmatch(date(2025, 3, 15))

            # Find close games
            close_games = [g for g in games.data if 40 <= g['HomeWP'] <= 60]
            ```
        """
        if isinstance(game_date, date):
            date_str = game_date.strftime("%Y-%m-%d")
        else:
            date_str = game_date

        return self._request("fanmatch", {"d": date_str})

    # ==================== Statistical Analysis ====================

    def get_four_factors(self, year: int) -> APIResponse:
        """Get Four Factors analysis for all teams.

        The Four Factors (Dean Oliver) are the key stats that determine
        basketball success: shooting, turnovers, rebounding, and free throws.

        Args:
            year: Season year.

        Returns:
            APIResponse with Four Factors data including:
            - eFG_Pct: Effective Field Goal Percentage
            - TO_Pct: Turnover Percentage
            - OR_Pct: Offensive Rebound Percentage
            - FT_Rate: Free Throw Rate
            - DeFG_Pct/DTO_Pct/DOR_Pct/DFT_Rate: Defensive versions
            - AdjOE/AdjDE: Adjusted efficiencies
            - Rank* columns for each metric

        Example:
            ```python
            ff = api.get_four_factors(year=2025)
            df = ff.to_dataframe()

            # Find best shooting teams
            best_shooting = df.nsmallest(10, 'RankeFG_Pct')
            ```
        """
        return self._request("four-factors", {"y": year})

    def get_misc_stats(self, year: int) -> APIResponse:
        """Get miscellaneous team statistics.

        Args:
            year: Season year.

        Returns:
            APIResponse with misc stats including:
            - FG3Pct/FG2Pct/FTPct: Shooting percentages
            - BlockPct: Block percentage
            - StlRate: Steal rate
            - NSTRate: Non-steal turnover rate
            - ARate: Assist rate
            - F3GRate: Three-point attempt rate
            - Opp* versions for opponent stats

        Example:
            ```python
            stats = api.get_misc_stats(year=2025)
            df = stats.to_dataframe()

            # Find best 3-point shooting teams
            best_3pt = df.nsmallest(10, 'RankFG3Pct')
            ```
        """
        return self._request("misc-stats", {"y": year})

    def get_height(self, year: int) -> APIResponse:
        """Get team height and experience data.

        Args:
            year: Season year.

        Returns:
            APIResponse with height/experience data including:
            - AvgHgt: Average height (inches)
            - HgtEff: Height effectiveness
            - Experience metrics

        Example:
            ```python
            height = api.get_height(year=2025)
            df = height.to_dataframe()

            # Tallest teams
            tallest = df.nsmallest(10, 'AvgHgtRank')
            ```
        """
        return self._request("height", {"y": year})

    def get_point_distribution(self, year: int) -> APIResponse:
        """Get point distribution data for all teams.

        Args:
            year: Season year.

        Returns:
            APIResponse with point distribution data showing
            how teams score their points (2PT, 3PT, FT breakdown).

        Example:
            ```python
            dist = api.get_point_distribution(year=2025)
            df = dist.to_dataframe()
            ```
        """
        return self._request("pointdist", {"y": year})

    # ==================== Convenience Methods ====================

    def get_team_by_name(self, name: str, year: int) -> dict[str, Any] | None:
        """Find a team by name.

        Args:
            name: Team name (case-insensitive partial match).
            year: Season year.

        Returns:
            Team dict if found, None otherwise.

        Example:
            ```python
            duke = api.get_team_by_name("Duke", 2025)
            if duke:
                print(f"TeamID: {duke['TeamID']}")
            ```
        """
        teams = self.get_teams(year)
        name_lower = name.lower()
        for team in teams.data:
            if name_lower in team["TeamName"].lower():
                return team
        return None

    def get_team_ratings_by_name(
        self,
        name: str,
        year: int | None = None,
    ) -> APIResponse | None:
        """Get ratings for a team by name.

        Args:
            name: Team name.
            year: Season year. If None, returns historical data.

        Returns:
            APIResponse with ratings, or None if team not found.

        Example:
            ```python
            # Current season
            duke_ratings = api.get_team_ratings_by_name("Duke", 2025)

            # Historical
            duke_history = api.get_team_ratings_by_name("Duke")
            ```
        """
        # Determine which year to use for team lookup
        lookup_year = year or 2025
        team = self.get_team_by_name(name, lookup_year)
        if not team:
            return None

        if year:
            # Filter from season ratings
            ratings = self.get_ratings(year=year)
            team_data = [r for r in ratings.data if r["TeamName"] == team["TeamName"]]
            return APIResponse(
                data=team_data,
                endpoint="ratings",
                params={"y": year, "team": name},
            )
        else:
            # Get historical data
            return self.get_ratings(team_id=team["TeamID"])


# Convenience function for quick API access
def get_api(api_key: str | None = None) -> KenPomAPI:
    """Get a KenPom API client instance.

    Args:
        api_key: Optional API key. Uses KENPOM_API_KEY env var if not provided.

    Returns:
        KenPomAPI client instance.

    Example:
        ```python
        from kenp0m_sp0rts_analyzer.api_client import get_api

        api = get_api()
        ratings = api.get_ratings(year=2025)
        ```
    """
    return KenPomAPI(api_key=api_key)
