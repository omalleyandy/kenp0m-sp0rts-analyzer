"""Analytics functions for KenPom basketball data."""

import logging

import pandas as pd

from .client import KenPomClient
from .models import MatchupAnalysis
from .utils import calculate_expected_score, normalize_team_name

logger = logging.getLogger(__name__)


def analyze_matchup(
    team1: str,
    team2: str,
    season: int | None = None,
    neutral_site: bool = True,
    home_team: str | None = None,
    client: KenPomClient | None = None,
) -> MatchupAnalysis:
    """Analyze a head-to-head matchup between two teams.

    Args:
        team1: First team name.
        team2: Second team name.
        season: Season year. Defaults to current season.
        neutral_site: Whether the game is at a neutral site.
        home_team: If not neutral, which team is home (team1 or team2).
        client: Optional KenPomClient instance. Will create one if not provided.

    Returns:
        MatchupAnalysis object with prediction details.
    """
    if client is None:
        client = KenPomClient()

    team1 = normalize_team_name(team1)
    team2 = normalize_team_name(team2)

    # Get efficiency data
    efficiency = client.get_efficiency(season=season)

    team1_data = efficiency[efficiency["Team"] == team1]
    team2_data = efficiency[efficiency["Team"] == team2]

    if team1_data.empty:
        raise ValueError(f"Team '{team1}' not found")
    if team2_data.empty:
        raise ValueError(f"Team '{team2}' not found")

    # Extract key metrics
    team1_row = team1_data.iloc[0]
    team2_row = team2_data.iloc[0]

    team1_rank = int(team1_row.get("Rk", 0))
    team2_rank = int(team2_row.get("Rk", 0))

    team1_adj_em = float(team1_row.get("AdjEM", 0))
    team2_adj_em = float(team2_row.get("AdjEM", 0))

    team1_tempo = float(team1_row.get("AdjT", 68))
    team2_tempo = float(team2_row.get("AdjT", 68))

    # Calculate predictions
    is_home_team1 = not neutral_site and home_team == team1
    predictions = calculate_expected_score(
        adj_em1=team1_adj_em,
        adj_em2=team2_adj_em,
        adj_tempo1=team1_tempo,
        adj_tempo2=team2_tempo,
        is_home_team1=is_home_team1,
        neutral_site=neutral_site,
    )

    # Determine winner
    margin = predictions["predicted_margin"]
    predicted_winner = team1 if margin > 0 else team2

    # Determine pace advantage
    avg_tempo = (team1_tempo + team2_tempo) / 2
    if team1_tempo > avg_tempo and team1_adj_em > team2_adj_em:
        pace_advantage = team1
    elif team2_tempo > avg_tempo and team2_adj_em > team1_adj_em:
        pace_advantage = team2
    else:
        pace_advantage = "Neutral"

    return MatchupAnalysis(
        team1=team1,
        team2=team2,
        team1_rank=team1_rank,
        team2_rank=team2_rank,
        team1_adj_em=team1_adj_em,
        team2_adj_em=team2_adj_em,
        em_difference=round(team1_adj_em - team2_adj_em, 2),
        predicted_winner=predicted_winner,
        predicted_margin=abs(margin),
        predicted_total=predictions["predicted_total"],
        team1_tempo=team1_tempo,
        team2_tempo=team2_tempo,
        expected_tempo=predictions["expected_possessions"],
        pace_advantage=pace_advantage,
        neutral_site=neutral_site,
    )


def find_value_games(
    min_em_diff: float = 5.0,  # noqa: ARG001 - TODO: implement filtering by EM diff
    season: int | None = None,
    client: KenPomClient | None = None,
) -> pd.DataFrame:
    """Find potential value games based on efficiency differentials.

    Identifies matchups where there's a significant efficiency gap
    that may not be reflected in betting lines.

    Args:
        min_em_diff: Minimum efficiency margin difference to flag.
        season: Season year. Defaults to current season.
        client: Optional KenPomClient instance.

    Returns:
        DataFrame with potential value games.
    """
    if client is None:
        client = KenPomClient()

    efficiency = client.get_efficiency(season=season)

    # Calculate potential mismatches
    efficiency = efficiency.copy()
    efficiency["em_rank_diff"] = abs(
        efficiency["AdjEM"].rank(ascending=False) - efficiency["Rk"]
    )

    # Teams whose efficiency doesn't match their rank well
    undervalued = efficiency[efficiency["em_rank_diff"] > 20].copy()
    undervalued = undervalued.sort_values("AdjEM", ascending=False)

    return undervalued[["Team", "Conf", "Rk", "AdjEM", "AdjO", "AdjD", "em_rank_diff"]]


def get_conference_standings(
    conference: str,
    season: int | None = None,
    client: KenPomClient | None = None,
) -> pd.DataFrame:
    """Get KenPom rankings filtered by conference.

    Args:
        conference: Conference abbreviation (e.g., 'ACC', 'SEC', 'Big 12').
        season: Season year. Defaults to current season.
        client: Optional KenPomClient instance.

    Returns:
        DataFrame with conference teams ranked by KenPom.
    """
    if client is None:
        client = KenPomClient()

    efficiency = client.get_efficiency(season=season)

    conf_teams = efficiency[efficiency["Conf"] == conference].copy()
    conf_teams = conf_teams.sort_values("Rk")

    return conf_teams


def analyze_team_trends(
    team: str,
    seasons: list[int] | None = None,
    client: KenPomClient | None = None,
) -> pd.DataFrame:
    """Analyze a team's historical trends over multiple seasons.

    Args:
        team: Team name.
        seasons: List of seasons to analyze. Defaults to last 5 seasons.
        client: Optional KenPomClient instance.

    Returns:
        DataFrame with team's metrics across seasons.
    """
    if client is None:
        client = KenPomClient()

    team = normalize_team_name(team)

    if seasons is None:
        current = client.current_season
        seasons = list(range(current - 4, current + 1))

    results = []
    for season in seasons:
        try:
            efficiency = client.get_efficiency(season=season)
            team_data = efficiency[efficiency["Team"] == team]
            if not team_data.empty:
                row = team_data.iloc[0].to_dict()
                row["Season"] = season
                results.append(row)
        except Exception as e:
            logger.warning(f"Could not get data for {team} in {season}: {e}")

    if not results:
        raise ValueError(f"No data found for team '{team}'")

    df = pd.DataFrame(results)
    df = df.sort_values("Season")

    return df


def calculate_tournament_seed_line(
    season: int | None = None,
    num_teams: int = 68,
    client: KenPomClient | None = None,
) -> pd.DataFrame:
    """Calculate projected tournament seed lines based on KenPom rankings.

    Args:
        season: Season year. Defaults to current season.
        num_teams: Number of teams in tournament field.
        client: Optional KenPomClient instance.

    Returns:
        DataFrame with projected seed lines.
    """
    if client is None:
        client = KenPomClient()

    efficiency = client.get_efficiency(season=season)

    # Get top teams
    top_teams = efficiency.head(num_teams).copy()

    # Assign projected seeds (rough approximation)
    seeds = []
    for i, _ in enumerate(top_teams.itertuples()):
        if i < 4:
            seeds.append(1)
        elif i < 8:
            seeds.append(2)
        elif i < 12:
            seeds.append(3)
        elif i < 16:
            seeds.append(4)
        elif i < 20:
            seeds.append(5)
        elif i < 24:
            seeds.append(6)
        elif i < 28:
            seeds.append(7)
        elif i < 32:
            seeds.append(8)
        elif i < 36:
            seeds.append(9)
        elif i < 40:
            seeds.append(10)
        elif i < 44:
            seeds.append(11)
        elif i < 52:
            seeds.append(12)
        elif i < 56:
            seeds.append(13)
        elif i < 60:
            seeds.append(14)
        elif i < 64:
            seeds.append(15)
        else:
            seeds.append(16)

    top_teams["Projected_Seed"] = seeds

    return top_teams[["Team", "Conf", "Rk", "AdjEM", "Projected_Seed"]]
