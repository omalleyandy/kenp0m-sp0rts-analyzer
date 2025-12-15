"""KenPom client wrapper for simplified data access."""

import logging
from typing import Any

import pandas as pd
from kenpompy.misc import (
    get_arenas,
    get_current_season,
    get_hca,
    get_pomeroy_ratings,
    get_trends,
)
from kenpompy.summary import (
    get_efficiency,
    get_fourfactors,
    get_height,
    get_playerstats,
    get_pointdist,
    get_teamstats,
)
from kenpompy.team import get_schedule, get_scouting_report, get_valid_teams
from kenpompy.utils import login

from .utils import get_credentials, normalize_team_name

logger = logging.getLogger(__name__)


class KenPomClient:
    """Client wrapper for kenpompy library with caching and convenience methods."""

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        auto_login: bool = True,
    ) -> None:
        """Initialize the KenPom client.

        Args:
            email: KenPom subscription email. If not provided, reads from env.
            password: KenPom subscription password. If not provided, reads from env.
            auto_login: Whether to automatically login on initialization.
        """
        self._email = email
        self._password = password
        self._browser: Any = None
        self._current_season: int | None = None

        if auto_login:
            self._login()

    def _login(self) -> None:
        """Authenticate with KenPom."""
        if self._browser is not None:
            return

        email = self._email
        password = self._password

        if not email or not password:
            email, password = get_credentials()

        logger.info("Logging in to KenPom...")
        self._browser = login(email, password)
        logger.info("Successfully authenticated with KenPom")

    @property
    def browser(self) -> Any:
        """Get the authenticated browser session."""
        if self._browser is None:
            self._login()
        return self._browser

    @property
    def current_season(self) -> int:
        """Get the current season year."""
        if self._current_season is None:
            self._current_season = get_current_season(self.browser)
        return self._current_season

    # ==================== Summary Module ====================

    def get_efficiency(self, season: int | None = None) -> pd.DataFrame:
        """Get efficiency and tempo statistics for all teams.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with efficiency data.
        """
        return get_efficiency(self.browser, season=season)

    def get_fourfactors(self, season: int | None = None) -> pd.DataFrame:
        """Get Four Factors analysis for all teams.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with Four Factors data.
        """
        return get_fourfactors(self.browser, season=season)

    def get_height(self, season: int | None = None) -> pd.DataFrame:
        """Get height and experience data for all teams.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with height/experience data.
        """
        return get_height(self.browser, season=season)

    def get_playerstats(
        self, season: int | None = None, metric: str = "EFG"
    ) -> pd.DataFrame:
        """Get player statistics leaders.

        Args:
            season: Season year. Defaults to current season.
            metric: Stat type (EFG, eFG, ORtg, TO, etc.).

        Returns:
            DataFrame with player stats.
        """
        return get_playerstats(self.browser, season=season, metric=metric)

    def get_pointdist(self, season: int | None = None) -> pd.DataFrame:
        """Get team points distribution data.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with points distribution.
        """
        return get_pointdist(self.browser, season=season)

    def get_teamstats(
        self, season: int | None = None, defense: bool = False
    ) -> pd.DataFrame:
        """Get miscellaneous team statistics.

        Args:
            season: Season year. Defaults to current season.
            defense: If True, returns defensive stats.

        Returns:
            DataFrame with team stats.
        """
        return get_teamstats(self.browser, season=season, defense=defense)

    # ==================== Misc Module ====================

    def get_pomeroy_ratings(self, season: int | None = None) -> pd.DataFrame:
        """Get full Pomeroy ratings table.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with Pomeroy ratings.
        """
        return get_pomeroy_ratings(self.browser, season=season)

    def get_hca(self, season: int | None = None) -> pd.DataFrame:
        """Get home court advantage data.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with HCA data.
        """
        return get_hca(self.browser, season=season)

    def get_arenas(self, season: int | None = None) -> pd.DataFrame:
        """Get arena information.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with arena data.
        """
        return get_arenas(self.browser, season=season)

    def get_trends(self) -> pd.DataFrame:
        """Get statistical trends over time.

        Returns:
            DataFrame with trend data.
        """
        return get_trends(self.browser)

    # ==================== Team Module ====================

    def get_schedule(self, team: str, season: int | None = None) -> pd.DataFrame:
        """Get a team's schedule with results.

        Args:
            team: Team name (will be normalized).
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with schedule data.
        """
        team = normalize_team_name(team)
        return get_schedule(self.browser, team=team, season=season)

    def get_scouting_report(
        self, team: str, season: int | None = None
    ) -> dict[str, Any]:
        """Get comprehensive scouting report for a team.

        Args:
            team: Team name (will be normalized).
            season: Season year. Defaults to current season.

        Returns:
            Dictionary with scouting report data.
        """
        team = normalize_team_name(team)
        return get_scouting_report(self.browser, team=team, season=season)

    def get_valid_teams(self, season: int | None = None) -> list[str]:
        """Get list of valid team names for a season.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            List of valid team names.
        """
        return get_valid_teams(self.browser, season=season)

    # ==================== Convenience Methods ====================

    def get_team_rank(self, team: str, season: int | None = None) -> int:
        """Get a team's current KenPom ranking.

        Args:
            team: Team name.
            season: Season year. Defaults to current season.

        Returns:
            Team's KenPom rank.
        """
        team = normalize_team_name(team)
        ratings = self.get_pomeroy_ratings(season=season)
        team_row = ratings[ratings["Team"] == team]
        if team_row.empty:
            raise ValueError(f"Team '{team}' not found")
        return int(team_row["Rk"].iloc[0])

    def compare_teams(
        self, team1: str, team2: str, season: int | None = None
    ) -> pd.DataFrame:
        """Compare two teams side-by-side.

        Args:
            team1: First team name.
            team2: Second team name.
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with comparison data.
        """
        team1 = normalize_team_name(team1)
        team2 = normalize_team_name(team2)

        efficiency = self.get_efficiency(season=season)

        team1_eff = efficiency[efficiency["Team"] == team1]
        team2_eff = efficiency[efficiency["Team"] == team2]

        if team1_eff.empty:
            raise ValueError(f"Team '{team1}' not found")
        if team2_eff.empty:
            raise ValueError(f"Team '{team2}' not found")

        comparison = pd.concat([team1_eff, team2_eff]).reset_index(drop=True)
        return comparison
