"""Official KenPom API client.

This module provides a Python wrapper for the official KenPom API,
which requires a separate API key purchase from kenpom.com.

API Documentation discovered endpoints:
- ratings: Team ratings with efficiency metrics
- archive: Historical ratings from specific dates or preseason
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

    # Get archived ratings from a specific date
    archive = api.get_archive(archive_date="2025-02-15")

    # Get preseason ratings
    preseason = api.get_archive(preseason=True, year=2025)
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
        "archive",
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
        conference: str | None = None,
    ) -> APIResponse:
        """Get team ratings.

        Either `year` or `team_id` must be provided. If `year` is provided,
        returns ratings for all teams in that season. If `team_id` is provided,
        returns historical ratings for that team across all seasons.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season).
            team_id: Team ID for historical data. Use get_teams() to find IDs.
            conference: Conference abbreviation to filter results (e.g., 'B12', 'ACC',
                'SEC', 'B10'). Requires `year` to be specified. Use get_conferences()
                to find valid abbreviations.

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
            ValidationError: If neither year nor team_id is provided, or if
                conference is specified without year.

        Example:
            ```python
            # Get 2025 season ratings
            ratings = api.get_ratings(year=2025)
            top_10 = ratings.data[:10]

            # Get Duke's historical ratings
            duke = api.get_ratings(team_id=73)

            # Get Big 12 conference teams for 2025
            big12 = api.get_ratings(year=2025, conference="B12")

            # Get SEC teams for 2025
            sec = api.get_ratings(year=2025, conference="SEC")
            ```
        """
        params: dict[str, Any] = {}
        if year is not None:
            params["y"] = year
        if team_id is not None:
            params["team_id"] = team_id
        if conference is not None:
            if year is None:
                raise ValidationError(
                    "The 'year' parameter is required when filtering by conference"
                )
            params["c"] = conference

        if not params:
            raise ValidationError(
                "Either 'year' or 'team_id' parameter is required for ratings endpoint"
            )

        return self._request("ratings", params)

    def get_archive(
        self,
        archive_date: str | date | None = None,
        year: int | None = None,
        preseason: bool = False,
        team_id: int | None = None,
        conference: str | None = None,
    ) -> APIResponse:
        """Get historical team ratings from a specific date or preseason.

        Retrieve archived ratings data, allowing you to see team ratings at any
        point during the season or before the season starts.

        Args:
            archive_date: Date in YYYY-MM-DD format or date object to retrieve
                archived ratings for. Required unless using preseason mode.
            year: Season year (e.g., 2025 for 2024-25 season). Required when
                using preseason=True.
            preseason: If True, retrieves preseason ratings for the specified year.
                Requires the year parameter to be set.
            team_id: Team ID to filter results. Use get_teams() to find IDs.
            conference: Conference abbreviation to filter results (e.g., 'B12', 'ACC').
                Use get_conferences() to find valid abbreviations.

        Returns:
            APIResponse with archived ratings including:
            - ArchiveDate: Date of the archived ratings
            - Season: Ending year of the season
            - Preseason: Whether this is preseason data ("true" or "false")
            - TeamName: Team name
            - Seed: NCAA tournament seed (if applicable)
            - Event: Tournament event description
            - ConfShort: Conference short name
            - AdjEM/RankAdjEM: Adjusted efficiency margin on archive date
            - AdjOE/RankAdjOE: Adjusted offensive efficiency on archive date
            - AdjDE/RankAdjDE: Adjusted defensive efficiency on archive date
            - AdjTempo/RankAdjTempo: Adjusted tempo on archive date
            - AdjEMFinal/RankAdjEMFinal: Final adjusted efficiency margin
            - AdjOEFinal/RankAdjOEFinal: Final adjusted offensive efficiency
            - AdjDEFinal/RankAdjDEFinal: Final adjusted defensive efficiency
            - AdjTempoFinal/RankAdjTempoFinal: Final adjusted tempo
            - RankChg: Change in efficiency margin rank from archive date to final
            - AdjEMChg: Change in efficiency margin from archive date to final
            - AdjTChg: Change in tempo from archive date to final

        Raises:
            ValidationError: If neither archive_date nor (preseason + year) is provided,
                or if preseason=True but year is not specified.

        Example:
            ```python
            from datetime import date

            # Get ratings from a specific date
            ratings = api.get_archive(archive_date="2025-02-15")

            # Using date object
            ratings = api.get_archive(archive_date=date(2025, 2, 15))

            # Get preseason ratings
            preseason = api.get_archive(preseason=True, year=2025)

            # Filter by team
            duke_history = api.get_archive(archive_date="2025-03-01", team_id=73)

            # Filter by conference
            sec_archive = api.get_archive(archive_date="2025-02-15", conference="SEC")

            # Compare preseason to final rankings
            preseason = api.get_archive(preseason=True, year=2025)
            for team in preseason.data[:10]:
                print(f"{team['TeamName']}: {team['RankAdjEM']} -> {team['RankAdjEMFinal']} ({team['RankChg']:+d})")
            ```
        """
        params: dict[str, Any] = {}

        # Handle date parameter
        if archive_date is not None:
            if isinstance(archive_date, date):
                params["d"] = archive_date.strftime("%Y-%m-%d")
            else:
                params["d"] = archive_date

        # Handle preseason parameters
        if preseason:
            if year is None:
                raise ValidationError(
                    "The 'year' parameter is required when preseason=True"
                )
            params["preseason"] = "true"
            params["y"] = year

        # Validate that we have either date or preseason+year
        if "d" not in params and "preseason" not in params:
            raise ValidationError(
                "Either 'archive_date' or 'preseason=True' with 'year' is required "
                "for the archive endpoint"
            )

        # Optional filters
        if team_id is not None:
            params["team_id"] = team_id
        if conference is not None:
            params["c"] = conference

        return self._request("archive", params)

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

    def get_four_factors(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
        conf_only: bool = False,
    ) -> APIResponse:
        """Get Four Factors analysis for teams.

        The Four Factors (Dean Oliver) are the key stats that determine
        basketball success: shooting, turnovers, rebounding, and free throws.

        Either `year` or `team_id` must be provided. If `year` is provided,
        returns Four Factors for all teams in that season. If `team_id` is
        provided, returns historical Four Factors for that team across all seasons.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season).
            team_id: Team ID for historical data. Use get_teams() to find IDs.
            conference: Conference abbreviation to filter results (e.g., 'B12', 'ACC',
                'SEC', 'A10'). Requires `year` to be specified. Use get_conferences()
                to find valid abbreviations.
            conf_only: If True, returns conference-only statistics instead of all
                games. Defaults to False.

        Returns:
            APIResponse with Four Factors data including:
            - DataThrough: Date through which data is current
            - ConfOnly: Whether this is conference-only data ("true" or "false")
            - TeamName: Team name
            - Season: Ending year of the season
            - eFG_Pct/RankeFG_Pct: Effective Field Goal Percentage (offense)
            - TO_Pct/RankTO_Pct: Turnover Percentage (offense)
            - OR_Pct/RankOR_Pct: Offensive Rebound Percentage
            - FT_Rate/RankFT_Rate: Free Throw Rate (offense)
            - DeFG_Pct/RankDeFG_Pct: Effective FG% allowed (defense)
            - DTO_Pct/RankDTO_Pct: Turnover Percentage forced (defense)
            - DOR_Pct/RankDOR_Pct: Defensive Rebound Percentage
            - DFT_Rate/RankDFT_Rate: Free Throw Rate allowed (defense)
            - OE/RankOE: Offensive Efficiency
            - DE/RankDE: Defensive Efficiency
            - Tempo/RankTempo: Tempo (possessions per 40 minutes)
            - AdjOE/RankAdjOE: Adjusted Offensive Efficiency
            - AdjDE/RankAdjDE: Adjusted Defensive Efficiency
            - AdjTempo/RankAdjTempo: Adjusted Tempo

        Raises:
            ValidationError: If neither year nor team_id is provided, or if
                conference is specified without year.

        Example:
            ```python
            # Get 2025 season Four Factors
            ff = api.get_four_factors(year=2025)
            df = ff.to_dataframe()

            # Find best shooting teams
            best_shooting = df.nsmallest(10, 'RankeFG_Pct')

            # Get Duke's historical Four Factors
            duke_ff = api.get_four_factors(team_id=73)

            # Get Big 12 conference teams
            big12_ff = api.get_four_factors(year=2025, conference="B12")

            # Get conference-only stats
            conf_stats = api.get_four_factors(year=2025, conf_only=True)
            ```
        """
        params: dict[str, Any] = {}
        if year is not None:
            params["y"] = year
        if team_id is not None:
            params["team_id"] = team_id
        if conference is not None:
            if year is None:
                raise ValidationError(
                    "The 'year' parameter is required when filtering by conference"
                )
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"

        if not params or (conf_only and len(params) == 1):
            raise ValidationError(
                "Either 'year' or 'team_id' parameter is required for "
                "four-factors endpoint"
            )

        return self._request("four-factors", params)

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

    def get_height(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
    ) -> APIResponse:
        """Get team height and experience data.

        Retrieve team height statistics including average height, effective height,
        position-specific heights, team experience, bench strength, and continuity.

        Either `year` or `team_id` must be provided. If `year` is provided,
        returns height data for all teams in that season. If `team_id` is
        provided, returns historical height data for that team across all seasons.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season).
            team_id: Team ID for historical data. Use get_teams() to find IDs.
            conference: Conference abbreviation to filter results (e.g., 'WCC', 'ACC',
                'SEC', 'B10'). Requires `year` to be specified. Use get_conferences()
                to find valid abbreviations.

        Returns:
            APIResponse with height/experience data including:
            - DataThrough: Date through which data is current
            - Season: Ending year of the season
            - TeamName: Team name
            - ConfShort: Conference short name
            - AvgHgt/AvgHgtRank: Average team height in inches
            - HgtEff/HgtEffRank: Effective height
            - Hgt5/Hgt5Rank: Center position height
            - Hgt4/Hgt4Rank: Power forward position height
            - Hgt3/Hgt3Rank: Small forward position height
            - Hgt2/Hgt2Rank: Shooting guard position height
            - Hgt1/Hgt1Rank: Point guard position height
            - Exp/ExpRank: Experience rating
            - Bench/BenchRank: Bench strength rating
            - Continuity/RankContinuity: Team continuity rating

        Raises:
            ValidationError: If neither year nor team_id is provided, or if
                conference is specified without year.

        Example:
            ```python
            # Get 2025 season height data
            height = api.get_height(year=2025)
            df = height.to_dataframe()

            # Tallest teams
            tallest = df.nsmallest(10, 'AvgHgtRank')

            # Most experienced teams
            experienced = df.nsmallest(10, 'ExpRank')

            # Get Duke's historical height data
            duke_height = api.get_height(team_id=73)

            # Get WCC conference teams
            wcc_height = api.get_height(year=2025, conference="WCC")
            ```
        """
        params: dict[str, Any] = {}
        if year is not None:
            params["y"] = year
        if team_id is not None:
            params["team_id"] = team_id
        if conference is not None:
            if year is None:
                raise ValidationError(
                    "The 'year' parameter is required when filtering by conference"
                )
            params["c"] = conference

        if not params:
            raise ValidationError(
                "Either 'year' or 'team_id' parameter is required for height endpoint"
            )

        return self._request("height", params)

    def get_point_distribution(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
        conf_only: bool = False,
    ) -> APIResponse:
        """Get point distribution data for teams.

        Retrieve the percentage of points scored from free throws, two-point
        field goals, and three-point field goals for both offense and defense.

        Either `year` or `team_id` must be provided. If `year` is provided,
        returns point distribution for all teams in that season. If `team_id`
        is provided, returns historical point distribution for that team.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season).
            team_id: Team ID for historical data. Use get_teams() to find IDs.
            conference: Conference abbreviation to filter results (e.g., 'B12', 'ACC',
                'SEC', 'B10'). Requires `year` to be specified. Use get_conferences()
                to find valid abbreviations.
            conf_only: If True, returns conference-only statistics instead of all
                games. Defaults to False.

        Returns:
            APIResponse with point distribution data including:
            - DataThrough: Date through which data is current
            - ConfOnly: Whether this is conference-only data ("true" or "false")
            - Season: Ending year of the season
            - TeamName: Team name
            - ConfShort: Conference short name
            - OffFt/RankOffFt: Percentage of points from free throws (offense)
            - OffFg2/RankOffFg2: Percentage of points from 2-point FG (offense)
            - OffFg3/RankOffFg3: Percentage of points from 3-point FG (offense)
            - DefFt/RankDefFt: Percentage of points allowed from FT (defense)
            - DefFg2/RankDefFg2: Percentage of points allowed from 2-point FG (defense)
            - DefFg3/RankDefFg3: Percentage of points allowed from 3-point FG (defense)

        Raises:
            ValidationError: If neither year nor team_id is provided, or if
                conference is specified without year.

        Example:
            ```python
            # Get 2025 season point distribution
            dist = api.get_point_distribution(year=2025)
            df = dist.to_dataframe()

            # Find teams that rely heavily on 3-pointers
            three_heavy = df.nsmallest(10, 'RankOffFg3')

            # Get Duke's historical point distribution
            duke_dist = api.get_point_distribution(team_id=73)

            # Get Big 10 conference teams
            big10_dist = api.get_point_distribution(year=2025, conference="B10")

            # Get conference-only stats
            conf_stats = api.get_point_distribution(year=2025, conf_only=True)
            ```
        """
        params: dict[str, Any] = {}
        if year is not None:
            params["y"] = year
        if team_id is not None:
            params["team_id"] = team_id
        if conference is not None:
            if year is None:
                raise ValidationError(
                    "The 'year' parameter is required when filtering by conference"
                )
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"

        if not params or (conf_only and len(params) == 1):
            raise ValidationError(
                "Either 'year' or 'team_id' parameter is required for "
                "pointdist endpoint"
            )

        return self._request("pointdist", params)

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
