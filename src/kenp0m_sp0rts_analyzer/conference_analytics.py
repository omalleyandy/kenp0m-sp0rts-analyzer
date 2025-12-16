"""Conference power ratings and strength analytics.

This module provides tools to:
1. Calculate aggregate conference power ratings
2. Analyze conference depth and quality
3. Estimate tournament bids per conference
4. Compare conference strength head-to-head (future)

The power rating system uses a weighted formula combining:
- Average AdjEM (40%)
- Top team strength (20%)
- Depth - teams above average (20%)
- Top 25 representation (10%)
- Non-conference strength of schedule (10%)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .api_client import KenPomAPI


@dataclass
class ConferencePowerRating:
    """Aggregate power rating for a conference.

    Attributes:
        conference: Full conference name
        conf_short: Conference abbreviation
        num_teams: Number of teams in conference
        avg_adj_em: Average AdjEM across all teams
        median_adj_em: Median AdjEM (better for skewed distributions)
        top_team_adj_em: Best team's AdjEM
        bottom_team_adj_em: Worst team's AdjEM
        top_25_teams: Number of teams in top 25 by AdjEM
        top_50_teams: Number of teams in top 50 by AdjEM
        above_average: Teams with positive AdjEM (above national average)
        avg_adj_oe: Average adjusted offensive efficiency
        avg_adj_de: Average adjusted defensive efficiency
        avg_tempo: Average tempo across conference
        avg_sos: Average strength of schedule
        avg_ncsos: Average non-conference SOS
        estimated_bids: Estimated NCAA tournament bids
        power_score: Calculated power rating (0-100 scale)
        power_rank: Rank among all conferences
    """

    conference: str
    conf_short: str
    num_teams: int
    avg_adj_em: float
    median_adj_em: float
    top_team_adj_em: float
    bottom_team_adj_em: float
    top_25_teams: int
    top_50_teams: int
    above_average: int
    avg_adj_oe: float
    avg_adj_de: float
    avg_tempo: float
    avg_sos: float
    avg_ncsos: float
    estimated_bids: int
    power_score: float
    power_rank: int


@dataclass
class ConferenceHeadToHead:
    """Head-to-head comparison between two conferences.

    Note: This requires schedule/game data which is not currently
    available via the KenPom API. This is a placeholder for future
    implementation using kenpompy library.

    Attributes:
        conf1: First conference abbreviation
        conf2: Second conference abbreviation
        games_played: Total games between conferences
        conf1_wins: Wins by conf1 teams
        conf2_wins: Wins by conf2 teams
        win_percentage: Conf1's winning percentage
        avg_margin: Conf1's average margin of victory
        top_vs_top: Record when both teams in top 50
        best_win_conf1: Best win by conf1 team
        worst_loss_conf1: Worst loss by conf1 team
    """

    conf1: str
    conf2: str
    games_played: int
    conf1_wins: int
    conf2_wins: int
    win_percentage: float
    avg_margin: float
    top_vs_top: str
    best_win_conf1: str
    worst_loss_conf1: str


class ConferenceAnalytics:
    """Advanced conference strength and comparison analytics.

    Example:
        >>> from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
        >>> from kenp0m_sp0rts_analyzer.conference_analytics import ConferenceAnalytics
        >>>
        >>> api = KenPomAPI()
        >>> analytics = ConferenceAnalytics(api)
        >>>
        >>> # Get power ratings for all conferences
        >>> power_ratings = analytics.calculate_conference_power_ratings(2025)
        >>> print(power_ratings[["conference", "power_score", "power_rank"]].head())
        >>>
        >>> # Analyze specific conference tournament outlook
        >>> sec_outlook = analytics.get_conference_tournament_outlook("SEC", 2025)
        >>> print(sec_outlook)
    """

    # Thresholds for team quality classification
    TOP_25_THRESHOLD = 15.0  # AdjEM threshold for likely top 25
    TOP_50_THRESHOLD = 10.0  # AdjEM threshold for likely top 50
    TOURNAMENT_LOCK_THRESHOLD = 15.0  # AdjEM for tournament lock
    TOURNAMENT_BUBBLE_THRESHOLD = 10.0  # AdjEM for bubble team

    # Power score weights (must sum to 100%)
    WEIGHT_AVG_EM = 0.40  # Average conference strength
    WEIGHT_TOP_TEAM = 0.20  # Elite team strength
    WEIGHT_DEPTH = 0.20  # Number of above-average teams
    WEIGHT_TOP_25 = 0.10  # Top 25 representation
    WEIGHT_NCSOS = 0.10  # Non-conference schedule strength

    def __init__(self, api: KenPomAPI | None = None) -> None:
        """Initialize conference analytics.

        Args:
            api: KenPomAPI instance. If None, creates new instance.
        """
        self.api = api or KenPomAPI()

    def calculate_conference_power_ratings(self, year: int) -> pd.DataFrame:
        """Calculate comprehensive power ratings for all conferences.

        The power rating formula combines multiple strength metrics:
        - Average AdjEM (40%): Overall conference strength
        - Top team strength (20%): Presence of elite teams
        - Depth (20%): Number of teams above national average
        - Top 25 representation (10%): National prominence
        - Non-conference SOS (10%): Quality of non-conference play

        Args:
            year: Season year (e.g., 2025 for 2024-25 season)

        Returns:
            DataFrame with conference power ratings, sorted by power_score.
            Columns include all ConferencePowerRating attributes.

        Example:
            >>> analytics = ConferenceAnalytics()
            >>> ratings = analytics.calculate_conference_power_ratings(2025)
            >>> top_5 = ratings.head(5)
            >>> print(top_5[["conference", "avg_adj_em", "power_score"]])
        """
        # Get all conferences
        conferences = self.api.get_conferences(year=year)

        # Calculate rating for each conference
        all_ratings = []
        for conf in conferences.data:
            try:
                rating = self._calculate_single_conference_rating(
                    conf["ConfShort"], conf["ConfLong"], year
                )
                all_ratings.append(rating)
            except Exception:
                # Skip conferences with no data (e.g., dissolved conferences)
                # This is expected for conferences like Pac-12 after realignment
                continue

        if not all_ratings:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in all_ratings])

        # Calculate power scores
        df = self._calculate_power_scores(df)

        # Rank conferences
        df["power_rank"] = df["power_score"].rank(ascending=False).astype(int)

        # Update power_rank in the dataframe
        for idx, row in df.iterrows():
            df.at[idx, "power_rank"] = int(row["power_rank"])

        return df.sort_values("power_score", ascending=False).reset_index(drop=True)

    def _calculate_single_conference_rating(
        self, conf_short: str, conf_name: str, year: int
    ) -> ConferencePowerRating:
        """Calculate rating for a single conference.

        Args:
            conf_short: Conference abbreviation (e.g., "ACC")
            conf_name: Full conference name
            year: Season year

        Returns:
            ConferencePowerRating with all metrics calculated
        """
        # Get all teams in conference
        teams_response = self.api.get_ratings(year=year, conference=conf_short)
        teams = teams_response.data

        if not teams:
            raise ValueError(f"No teams found for conference {conf_short}")

        # Extract metrics
        adj_ems = [float(t["AdjEM"]) for t in teams]
        adj_oes = [float(t["AdjOE"]) for t in teams]
        adj_des = [float(t["AdjDE"]) for t in teams]
        tempos = [float(t["AdjTempo"]) for t in teams]
        sos_vals = [float(t["SOS"]) for t in teams]
        ncsos_vals = [float(t["NCSOS"]) for t in teams]

        # Depth metrics
        top_25 = sum(1 for em in adj_ems if em >= self.TOP_25_THRESHOLD)
        top_50 = sum(1 for em in adj_ems if em >= self.TOP_50_THRESHOLD)
        above_avg = sum(1 for em in adj_ems if em > 0)

        # Tournament bid estimation
        estimated_bids = self._estimate_tournament_bids(teams)

        return ConferencePowerRating(
            conference=conf_name,
            conf_short=conf_short,
            num_teams=len(teams),
            avg_adj_em=sum(adj_ems) / len(adj_ems),
            median_adj_em=sorted(adj_ems)[len(adj_ems) // 2],
            top_team_adj_em=max(adj_ems),
            bottom_team_adj_em=min(adj_ems),
            top_25_teams=top_25,
            top_50_teams=top_50,
            above_average=above_avg,
            avg_adj_oe=sum(adj_oes) / len(adj_oes),
            avg_adj_de=sum(adj_des) / len(adj_des),
            avg_tempo=sum(tempos) / len(tempos),
            avg_sos=sum(sos_vals) / len(sos_vals),
            avg_ncsos=sum(ncsos_vals) / len(ncsos_vals),
            estimated_bids=estimated_bids,
            power_score=0.0,  # Calculated after all conferences
            power_rank=0,  # Assigned after ranking
        )

    def _estimate_tournament_bids(self, teams: list[dict[str, Any]]) -> int:
        """Estimate NCAA tournament bids for conference.

        Estimation logic:
        - AdjEM > 15: Tournament lock
        - AdjEM 10-15: Bubble team (50% chance)
        - AdjEM < 10: Unlikely unless auto-bid
        - Always include at least 1 (conference tournament auto-bid)

        Args:
            teams: List of team dictionaries with AdjEM values

        Returns:
            Estimated number of tournament bids
        """
        likely = sum(
            1 for t in teams if float(t["AdjEM"]) > self.TOURNAMENT_LOCK_THRESHOLD
        )
        bubble = sum(
            1
            for t in teams
            if self.TOURNAMENT_BUBBLE_THRESHOLD
            <= float(t["AdjEM"])
            <= self.TOURNAMENT_LOCK_THRESHOLD
        )

        estimated = likely + (bubble // 2)
        return max(estimated, 1)  # At least auto-bid

    def _calculate_power_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized power scores (0-100) for conferences.

        Args:
            df: DataFrame with conference metrics

        Returns:
            DataFrame with power_score column added
        """
        # Normalize each metric to 0-1 scale, then multiply by weight
        df["em_score"] = (
            self._normalize_column(df["avg_adj_em"]) * self.WEIGHT_AVG_EM * 100
        )
        df["top_score"] = (
            self._normalize_column(df["top_team_adj_em"]) * self.WEIGHT_TOP_TEAM * 100
        )
        df["depth_score"] = (
            self._normalize_column(df["above_average"]) * self.WEIGHT_DEPTH * 100
        )
        df["top25_score"] = (
            self._normalize_column(df["top_25_teams"]) * self.WEIGHT_TOP_25 * 100
        )
        df["ncsos_score"] = (
            self._normalize_column(df["avg_ncsos"]) * self.WEIGHT_NCSOS * 100
        )

        # Total power score (0-100)
        df["power_score"] = (
            df["em_score"]
            + df["top_score"]
            + df["depth_score"]
            + df["top25_score"]
            + df["ncsos_score"]
        )

        return df

    @staticmethod
    def _normalize_column(series: pd.Series) -> pd.Series:
        """Normalize series to 0-1 range.

        Args:
            series: Pandas Series to normalize

        Returns:
            Normalized series (0-1 range)
        """
        min_val = series.min()
        max_val = series.max()

        # Handle case where all values are the same
        if max_val == min_val:
            return pd.Series([1.0] * len(series), index=series.index)

        return (series - min_val) / (max_val - min_val)

    def get_conference_tournament_outlook(
        self, conf_short: str, year: int
    ) -> pd.DataFrame:
        """Analyze conference tournament implications and NCAA bid probabilities.

        Args:
            conf_short: Conference abbreviation (e.g., "SEC", "B12")
            year: Season year

        Returns:
            DataFrame with teams sorted by AdjEM, including:
            - Team: Team name
            - AdjEM: Adjusted efficiency margin
            - RankAdjEM: National rank
            - NCAA_Probability: Estimated tournament probability
            - Bubble: Tournament outlook (Lock/Bubble/NIT or CBI)

        Example:
            >>> analytics = ConferenceAnalytics()
            >>> sec = analytics.get_conference_tournament_outlook("SEC", 2025)
            >>> locks = sec[sec["Bubble"] == "Lock"]
            >>> print(f"SEC has {len(locks)} tournament locks")
        """
        teams_response = self.api.get_ratings(year=year, conference=conf_short)
        teams = teams_response.data

        results = []
        for team in teams:
            adj_em = float(team["AdjEM"])

            # Estimate NCAA tournament probability based on AdjEM
            if adj_em > 15:
                ncaa_prob = 0.95
                bubble_status = "Lock"
            elif adj_em > 10:
                ncaa_prob = 0.7
                bubble_status = "Bubble"
            elif adj_em > 5:
                ncaa_prob = 0.4
                bubble_status = "Bubble"
            elif adj_em > 0:
                ncaa_prob = 0.15
                bubble_status = "NIT/CBI"
            else:
                ncaa_prob = 0.05
                bubble_status = "NIT/CBI"

            results.append(
                {
                    "Team": team["TeamName"],
                    "AdjEM": adj_em,
                    "RankAdjEM": team["RankAdjEM"],
                    "NCAA_Probability": ncaa_prob,
                    "Bubble": bubble_status,
                }
            )

        df = pd.DataFrame(results)
        return df.sort_values("AdjEM", ascending=False).reset_index(drop=True)

    def compare_conferences_head_to_head(
        self, conf1: str, conf2: str, year: int
    ) -> ConferenceHeadToHead:
        """Compare two conferences in non-conference games.

        Note: This feature requires game-level schedule data which is not
        currently available via the KenPom API. This is a placeholder for
        future implementation using kenpompy library's schedule functions.

        Args:
            conf1: First conference abbreviation
            conf2: Second conference abbreviation
            year: Season year

        Raises:
            NotImplementedError: Always - feature not yet implemented

        Future Implementation:
            Will use kenpompy.team.get_schedule() to find all non-conference
            games between teams from the two conferences and calculate:
            - Win/loss records
            - Average margins
            - Quality of matchups (top vs top)
        """
        raise NotImplementedError(
            "Head-to-head comparison requires schedule data. "
            "Use kenpompy library's get_schedule() for this feature."
        )
