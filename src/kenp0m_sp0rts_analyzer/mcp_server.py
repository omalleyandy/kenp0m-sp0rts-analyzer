"""MCP Server for KenPom basketball analytics.

This module provides an MCP (Model Context Protocol) server that exposes
KenPom basketball analytics tools for use with Claude and other MCP clients.

The server supports two authentication methods:
1. Official API (KENPOM_API_KEY) - Recommended, faster, more reliable
2. Scraper-based (KENPOM_EMAIL/KENPOM_PASSWORD) - Falls back if no API key

Run as module:
    python -m kenp0m_sp0rts_analyzer.mcp_server

Or use with Claude Code by configuring in .claude/mcp.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("kenpom-analyzer")


def get_current_season() -> int:
    """Get the current basketball season year.

    The college basketball season spans two calendar years (e.g., 2024-25).
    KenPom uses the ending year (2025 for 2024-25 season).
    Season typically starts in November.
    """
    now = datetime.now()
    # If we're past August, we're in the next season
    if now.month >= 8:
        return now.year + 1
    return now.year


def _get_api_client():
    """Get the KenPom API client if API key is available."""
    api_key = os.getenv("KENPOM_API_KEY")
    if api_key:
        from .api_client import KenPomAPI

        return KenPomAPI(api_key=api_key)
    return None


def _get_scraper_client():
    """Get the kenpompy-based client if credentials are available."""
    email = os.getenv("KENPOM_EMAIL")
    password = os.getenv("KENPOM_PASSWORD")
    if email and password:
        try:
            from .client import KenPomClient

            return KenPomClient()
        except ImportError:
            logger.warning("kenpompy not installed, scraper client unavailable")
    return None


def _format_dataframe_as_text(df, max_rows: int = 25) -> str:
    """Format a pandas DataFrame as readable text."""
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True
    else:
        truncated = False

    result = df.to_string(index=False)
    if truncated:
        result += f"\n\n... (showing top {max_rows} of {len(df)} results)"
    return result


def _format_dict_as_text(data: dict[str, Any]) -> str:
    """Format a dictionary as readable text."""
    lines = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_list_as_text(data: list[dict[str, Any]], max_items: int = 25) -> str:
    """Format a list of dicts as readable text."""
    if not data:
        return "No data available"

    if len(data) > max_items:
        data = data[:max_items]
        truncated = True
    else:
        truncated = False

    # Get all keys
    keys = list(data[0].keys())

    # Format as table-like output
    lines = []
    for item in data:
        line_parts = [f"{k}: {item.get(k, 'N/A')}" for k in keys[:6]]  # Limit columns
        lines.append(" | ".join(line_parts))

    result = "\n".join(lines)
    if truncated:
        result += f"\n\n... (showing top {max_items} results)"
    return result


# ==================== Tool Definitions ====================


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available KenPom analytics tools."""
    return [
        Tool(
            name="get_team_efficiency",
            description="Get adjusted offensive and defensive efficiency ratings for NCAA basketball teams. Returns AdjEM, AdjO, AdjD, tempo, and rankings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    }
                },
            },
        ),
        Tool(
            name="get_four_factors",
            description="Get Four Factors analysis (eFG%, TO%, OR%, FTRate) for NCAA basketball teams. These are Dean Oliver's key stats that determine basketball success.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    }
                },
            },
        ),
        Tool(
            name="get_team_schedule",
            description="Get a team's schedule with results, opponent rankings, and KenPom metrics for each game.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Team name (e.g., 'Duke', 'North Carolina', 'Kansas')",
                    },
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    },
                },
                "required": ["team"],
            },
        ),
        Tool(
            name="get_scouting_report",
            description="Get comprehensive scouting report for a team including offensive/defensive tendencies, key stats, and rankings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Team name (e.g., 'Duke', 'North Carolina', 'Kansas')",
                    },
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    },
                },
                "required": ["team"],
            },
        ),
        Tool(
            name="get_pomeroy_ratings",
            description="Get full KenPom ratings table with all teams, including efficiency metrics, tempo, luck, and strength of schedule.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    }
                },
            },
        ),
        Tool(
            name="analyze_matchup",
            description="Analyze a head-to-head matchup between two teams. Returns predicted winner, margin, tempo, and key statistical comparisons.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team1": {
                        "type": "string",
                        "description": "First team name",
                    },
                    "team2": {
                        "type": "string",
                        "description": "Second team name",
                    },
                    "neutral_site": {
                        "type": "boolean",
                        "description": "Whether the game is at a neutral site (default: true)",
                    },
                },
                "required": ["team1", "team2"],
            },
        ),
        Tool(
            name="get_home_court_advantage",
            description="Get home court advantage data for all teams, showing how much better teams perform at home.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    }
                },
            },
        ),
        Tool(
            name="get_game_predictions",
            description="Get KenPom game predictions for a specific date, including win probabilities and predicted scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (e.g., '2025-03-15')",
                    }
                },
                "required": ["date"],
            },
        ),
        Tool(
            name="analyze_tempo_matchup",
            description="Analyze tempo and pace advantages in a matchup. Returns detailed tempo profiles, style mismatches, pace control, APL (Average Possession Length) insights, and tempo impact on game outcome.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team1": {
                        "type": "string",
                        "description": "First team name",
                    },
                    "team2": {
                        "type": "string",
                        "description": "Second team name",
                    },
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    },
                },
                "required": ["team1", "team2"],
            },
        ),
        Tool(
            name="get_player_depth_chart",
            description="Get team depth chart with player valuations ranked by contribution. Shows each player's estimated value in AdjEM points and value over replacement (VOR).",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Team name (e.g., 'Duke', 'North Carolina')",
                    },
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025 for 2024-25 season). Defaults to current season.",
                    },
                },
                "required": ["team"],
            },
        ),
        Tool(
            name="calculate_player_value",
            description="Calculate individual player's value and contribution to team performance. Returns estimated AdjEM points contributed, offensive/defensive contributions, and value over replacement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player": {
                        "type": "string",
                        "description": "Player name",
                    },
                    "team": {
                        "type": "string",
                        "description": "Team name",
                    },
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025). Defaults to current season.",
                    },
                },
                "required": ["player", "team"],
            },
        ),
        Tool(
            name="estimate_injury_impact",
            description="Estimate impact of player injury/absence on team performance. Returns adjusted team ratings (AdjEM, AdjOE, AdjDE), confidence intervals, and severity classification (Minor/Moderate/Major/Devastating).",
            inputSchema={
                "type": "object",
                "properties": {
                    "player": {
                        "type": "string",
                        "description": "Player name",
                    },
                    "team": {
                        "type": "string",
                        "description": "Team name",
                    },
                    "injury_severity": {
                        "type": "string",
                        "description": "Injury status: 'out' (100% impact), 'doubtful' (75%), or 'questionable' (50%). Defaults to 'out'.",
                        "enum": ["out", "doubtful", "questionable"],
                    },
                    "season": {
                        "type": "integer",
                        "description": "Season year (e.g., 2025). Defaults to current season.",
                    },
                },
                "required": ["player", "team"],
            },
        ),
    ]


# ==================== Tool Handlers ====================


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result = await _handle_tool(name, arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.exception(f"Error handling tool {name}")
        return [TextContent(type="text", text=f"Error: {e!s}")]


async def _handle_tool(name: str, arguments: dict[str, Any]) -> str:
    """Route tool calls to appropriate handlers."""
    handlers = {
        "get_team_efficiency": _handle_get_team_efficiency,
        "get_four_factors": _handle_get_four_factors,
        "get_team_schedule": _handle_get_team_schedule,
        "get_scouting_report": _handle_get_scouting_report,
        "get_pomeroy_ratings": _handle_get_pomeroy_ratings,
        "analyze_matchup": _handle_analyze_matchup,
        "get_home_court_advantage": _handle_get_home_court_advantage,
        "get_game_predictions": _handle_get_game_predictions,
        "analyze_tempo_matchup": _handle_analyze_tempo_matchup,
        "get_player_depth_chart": _handle_get_player_depth_chart,
        "calculate_player_value": _handle_calculate_player_value,
        "estimate_injury_impact": _handle_estimate_injury_impact,
    }

    handler = handlers.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")

    return await handler(arguments)


async def _handle_get_team_efficiency(arguments: dict[str, Any]) -> str:
    """Get team efficiency ratings."""
    season = arguments.get("season", get_current_season())

    api = _get_api_client()
    if api:
        try:
            response = api.get_ratings(year=season)
            df = response.to_dataframe()
            # Select key columns
            cols = [
                "TeamName",
                "AdjEM",
                "AdjOE",
                "AdjDE",
                "AdjTempo",
                "Luck",
                "SOS",
            ]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df = df[available_cols]
            return f"Team Efficiency Ratings ({season} Season)\n\n{_format_dataframe_as_text(df)}"
        except Exception as e:
            logger.warning(f"API client failed: {e}, trying scraper")

    client = _get_scraper_client()
    if client:
        df = client.get_efficiency(season=season)
        return f"Team Efficiency Ratings ({season} Season)\n\n{_format_dataframe_as_text(df)}"

    return "Error: No authentication configured. Set KENPOM_API_KEY or KENPOM_EMAIL/KENPOM_PASSWORD environment variables."


async def _handle_get_four_factors(arguments: dict[str, Any]) -> str:
    """Get Four Factors analysis."""
    season = arguments.get("season", get_current_season())

    api = _get_api_client()
    if api:
        try:
            response = api.get_four_factors(year=season)
            df = response.to_dataframe()
            return f"Four Factors Analysis ({season} Season)\n\n{_format_dataframe_as_text(df)}"
        except Exception as e:
            logger.warning(f"API client failed: {e}, trying scraper")

    client = _get_scraper_client()
    if client:
        df = client.get_fourfactors(season=season)
        return f"Four Factors Analysis ({season} Season)\n\n{_format_dataframe_as_text(df)}"

    return "Error: No authentication configured. Set KENPOM_API_KEY or KENPOM_EMAIL/KENPOM_PASSWORD environment variables."


async def _handle_get_team_schedule(arguments: dict[str, Any]) -> str:
    """Get team schedule."""
    team = arguments.get("team")
    if not team:
        return "Error: 'team' parameter is required"

    season = arguments.get("season", get_current_season())

    client = _get_scraper_client()
    if client:
        try:
            df = client.get_schedule(team=team, season=season)
            return f"{team} Schedule ({season} Season)\n\n{_format_dataframe_as_text(df, max_rows=40)}"
        except Exception as e:
            logger.warning(f"Scraper client failed: {e}")

    # Schedule not available via API, only scraper
    return f"Error: Team schedule requires scraper authentication (KENPOM_EMAIL/KENPOM_PASSWORD). Could not retrieve schedule for {team}."


async def _handle_get_scouting_report(arguments: dict[str, Any]) -> str:
    """Get scouting report for a team."""
    team = arguments.get("team")
    if not team:
        return "Error: 'team' parameter is required"

    season = arguments.get("season", get_current_season())

    # Try API first for ratings and four factors
    api = _get_api_client()
    if api:
        try:
            # Get team data
            team_data = api.get_team_by_name(team, season)
            if not team_data:
                return f"Error: Team '{team}' not found"

            team_name = team_data["TeamName"]

            # Get ratings
            ratings = api.get_ratings(year=season)
            team_ratings = [r for r in ratings.data if r["TeamName"] == team_name]

            # Get four factors
            ff = api.get_four_factors(year=season)
            team_ff = [f for f in ff.data if f["TeamName"] == team_name]

            # Get misc stats
            misc = api.get_misc_stats(year=season)
            team_misc = [m for m in misc.data if m["TeamName"] == team_name]

            report = [f"Scouting Report: {team_name} ({season} Season)", "=" * 50]

            if team_ratings:
                r = team_ratings[0]
                report.extend(
                    [
                        "",
                        "EFFICIENCY METRICS",
                        f"  Adjusted Efficiency Margin: {r.get('AdjEM', 'N/A')} (Rank: {r.get('RankAdjEM', 'N/A')})",
                        f"  Adjusted Offense: {r.get('AdjOE', 'N/A')} (Rank: {r.get('RankAdjOE', 'N/A')})",
                        f"  Adjusted Defense: {r.get('AdjDE', 'N/A')} (Rank: {r.get('RankAdjDE', 'N/A')})",
                        f"  Adjusted Tempo: {r.get('AdjTempo', 'N/A')} (Rank: {r.get('RankAdjTempo', 'N/A')})",
                        f"  Strength of Schedule: {r.get('SOS', 'N/A')} (Rank: {r.get('RankSOS', 'N/A')})",
                    ]
                )

            if team_ff:
                f = team_ff[0]
                report.extend(
                    [
                        "",
                        "FOUR FACTORS (Offense)",
                        f"  eFG%: {f.get('eFG_Pct', 'N/A')} (Rank: {f.get('RankeFG_Pct', 'N/A')})",
                        f"  TO%: {f.get('TO_Pct', 'N/A')} (Rank: {f.get('RankTO_Pct', 'N/A')})",
                        f"  OR%: {f.get('OR_Pct', 'N/A')} (Rank: {f.get('RankOR_Pct', 'N/A')})",
                        f"  FT Rate: {f.get('FT_Rate', 'N/A')} (Rank: {f.get('RankFT_Rate', 'N/A')})",
                        "",
                        "FOUR FACTORS (Defense)",
                        f"  Opp eFG%: {f.get('DeFG_Pct', 'N/A')} (Rank: {f.get('RankDeFG_Pct', 'N/A')})",
                        f"  Opp TO%: {f.get('DTO_Pct', 'N/A')} (Rank: {f.get('RankDTO_Pct', 'N/A')})",
                        f"  Opp OR%: {f.get('DOR_Pct', 'N/A')} (Rank: {f.get('RankDOR_Pct', 'N/A')})",
                        f"  Opp FT Rate: {f.get('DFT_Rate', 'N/A')} (Rank: {f.get('RankDFT_Rate', 'N/A')})",
                    ]
                )

            if team_misc:
                m = team_misc[0]
                report.extend(
                    [
                        "",
                        "SHOOTING",
                        f"  3P%: {m.get('FG3Pct', 'N/A')} (Rank: {m.get('RankFG3Pct', 'N/A')})",
                        f"  2P%: {m.get('FG2Pct', 'N/A')} (Rank: {m.get('RankFG2Pct', 'N/A')})",
                        f"  FT%: {m.get('FTPct', 'N/A')} (Rank: {m.get('RankFTPct', 'N/A')})",
                    ]
                )

            return "\n".join(report)
        except Exception as e:
            logger.warning(f"API scouting report failed: {e}")

    # Try scraper client
    client = _get_scraper_client()
    if client:
        try:
            report_data = client.get_scouting_report(team=team, season=season)
            return f"Scouting Report: {team} ({season} Season)\n\n{_format_dict_as_text(report_data)}"
        except Exception as e:
            logger.warning(f"Scraper scouting report failed: {e}")

    return "Error: No authentication configured. Set KENPOM_API_KEY or KENPOM_EMAIL/KENPOM_PASSWORD environment variables."


async def _handle_get_pomeroy_ratings(arguments: dict[str, Any]) -> str:
    """Get full Pomeroy ratings."""
    season = arguments.get("season", get_current_season())

    api = _get_api_client()
    if api:
        try:
            response = api.get_ratings(year=season)
            df = response.to_dataframe()
            return f"KenPom Ratings ({season} Season)\n\n{_format_dataframe_as_text(df, max_rows=50)}"
        except Exception as e:
            logger.warning(f"API client failed: {e}, trying scraper")

    client = _get_scraper_client()
    if client:
        df = client.get_pomeroy_ratings(season=season)
        return f"KenPom Ratings ({season} Season)\n\n{_format_dataframe_as_text(df, max_rows=50)}"

    return "Error: No authentication configured. Set KENPOM_API_KEY or KENPOM_EMAIL/KENPOM_PASSWORD environment variables."


async def _handle_analyze_matchup(arguments: dict[str, Any]) -> str:
    """Analyze matchup between two teams."""
    team1 = arguments.get("team1")
    team2 = arguments.get("team2")
    neutral_site = arguments.get("neutral_site", True)

    if not team1 or not team2:
        return "Error: Both 'team1' and 'team2' parameters are required"

    season = get_current_season()

    # Try using analysis module with API data
    api = _get_api_client()
    if api:
        try:
            # Get ratings for both teams
            ratings = api.get_ratings(year=season)

            team1_data = None
            team2_data = None

            for r in ratings.data:
                name = r["TeamName"].lower()
                if team1.lower() in name or name in team1.lower():
                    team1_data = r
                if team2.lower() in name or name in team2.lower():
                    team2_data = r

            if not team1_data:
                return f"Error: Team '{team1}' not found in {season} ratings"
            if not team2_data:
                return f"Error: Team '{team2}' not found in {season} ratings"

            # Calculate matchup
            t1_name = team1_data["TeamName"]
            t2_name = team2_data["TeamName"]
            t1_em = float(team1_data.get("AdjEM", 0))
            t2_em = float(team2_data.get("AdjEM", 0))
            t1_tempo = float(team1_data.get("AdjTempo", 67))
            t2_tempo = float(team2_data.get("AdjTempo", 67))

            # HCA adjustment (if not neutral)
            hca = 0 if neutral_site else 3.5  # Average HCA is about 3.5 points

            em_diff = t1_em - t2_em + hca
            expected_tempo = (t1_tempo + t2_tempo) / 2

            # Predicted scores
            avg_efficiency = 100  # League average
            t1_pred = (avg_efficiency + (t1_em + em_diff / 2)) * expected_tempo / 100
            t2_pred = (avg_efficiency + (t2_em - em_diff / 2)) * expected_tempo / 100

            # Win probability (simplified)
            import math

            win_prob = 1 / (1 + math.exp(-em_diff / 10))  # Logistic function

            predicted_winner = t1_name if em_diff > 0 else t2_name
            predicted_margin = abs(em_diff)

            report = [
                f"Matchup Analysis: {t1_name} vs {t2_name}",
                "=" * 50,
                f"Location: {'Neutral Site' if neutral_site else f'{t1_name} Home'}",
                "",
                f"{t1_name}:",
                f"  KenPom Rank: {team1_data.get('RankAdjEM', 'N/A')}",
                f"  Adj. Efficiency Margin: {t1_em:+.1f}",
                f"  Adj. Offense: {team1_data.get('AdjOE', 'N/A')}",
                f"  Adj. Defense: {team1_data.get('AdjDE', 'N/A')}",
                f"  Adj. Tempo: {t1_tempo:.1f}",
                "",
                f"{t2_name}:",
                f"  KenPom Rank: {team2_data.get('RankAdjEM', 'N/A')}",
                f"  Adj. Efficiency Margin: {t2_em:+.1f}",
                f"  Adj. Offense: {team2_data.get('AdjOE', 'N/A')}",
                f"  Adj. Defense: {team2_data.get('AdjDE', 'N/A')}",
                f"  Adj. Tempo: {t2_tempo:.1f}",
                "",
                "PREDICTION",
                f"  Predicted Winner: {predicted_winner}",
                f"  Predicted Margin: {predicted_margin:.1f} points",
                f"  {t1_name} Win Probability: {win_prob * 100:.1f}%",
                f"  Expected Tempo: {expected_tempo:.1f} possessions",
                f"  Predicted Score: {t1_name} {t1_pred:.0f} - {t2_name} {t2_pred:.0f}",
            ]

            return "\n".join(report)
        except Exception as e:
            logger.warning(f"API matchup analysis failed: {e}")

    # Try using the analysis module with scraper
    client = _get_scraper_client()
    if client:
        try:
            from .analysis import analyze_matchup

            result = analyze_matchup(
                team1=team1, team2=team2, neutral_site=neutral_site, client=client
            )
            return f"Matchup Analysis\n\n{_format_dict_as_text(result.__dict__)}"
        except Exception as e:
            logger.warning(f"Scraper matchup analysis failed: {e}")

    return "Error: No authentication configured. Set KENPOM_API_KEY or KENPOM_EMAIL/KENPOM_PASSWORD environment variables."


async def _handle_get_home_court_advantage(arguments: dict[str, Any]) -> str:
    """Get home court advantage data."""
    season = arguments.get("season", get_current_season())

    # HCA not directly available via API, try scraper
    client = _get_scraper_client()
    if client:
        try:
            df = client.get_hca(season=season)
            return f"Home Court Advantage ({season} Season)\n\n{_format_dataframe_as_text(df)}"
        except Exception as e:
            logger.warning(f"Scraper HCA failed: {e}")

    # API doesn't have HCA endpoint, provide alternative
    api = _get_api_client()
    if api:
        return (
            "Home court advantage data requires scraper authentication "
            "(KENPOM_EMAIL/KENPOM_PASSWORD). The API doesn't provide HCA directly.\n\n"
            "General HCA info: Average HCA in college basketball is about 3.5 points. "
            "Elite home courts can be worth 5+ points."
        )

    return "Error: No authentication configured. Set KENPOM_API_KEY or KENPOM_EMAIL/KENPOM_PASSWORD environment variables."


async def _handle_get_game_predictions(arguments: dict[str, Any]) -> str:
    """Get game predictions for a date."""
    date_str = arguments.get("date")
    if not date_str:
        return "Error: 'date' parameter is required (format: YYYY-MM-DD)"

    api = _get_api_client()
    if api:
        try:
            response = api.get_fanmatch(date_str)
            if not response.data:
                return f"No games scheduled for {date_str}"

            lines = [f"Game Predictions for {date_str}", "=" * 50, ""]

            for game in response.data:
                visitor = game.get("Visitor", "Unknown")
                home = game.get("Home", "Unknown")
                v_rank = game.get("VisitorRank", "?")
                h_rank = game.get("HomeRank", "?")
                v_pred = game.get("VisitorPred", 0)
                h_pred = game.get("HomePred", 0)
                home_wp = game.get("HomeWP", 50)
                thrill = game.get("ThrillScore", 0)

                lines.extend(
                    [
                        f"#{v_rank} {visitor} @ #{h_rank} {home}",
                        f"  Predicted: {visitor} {v_pred:.0f} - {home} {h_pred:.0f}",
                        f"  {home} Win Prob: {home_wp:.1f}%",
                        f"  Thrill Score: {thrill:.1f}",
                        "",
                    ]
                )

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"API fanmatch failed: {e}")
            return f"Error retrieving predictions: {e}"

    return "Error: Game predictions require API authentication (KENPOM_API_KEY)."


async def _handle_analyze_tempo_matchup(arguments: dict[str, Any]) -> str:
    """Analyze tempo and pace matchup between two teams."""
    team1 = arguments.get("team1")
    team2 = arguments.get("team2")

    if not team1 or not team2:
        return "Error: Both 'team1' and 'team2' parameters are required"

    season = arguments.get("season", get_current_season())

    api = _get_api_client()
    if api:
        try:
            from .tempo_analysis import TempoMatchupAnalyzer

            analyzer = TempoMatchupAnalyzer(api)

            # Get team stats
            team1_stats = api.get_team_by_name(team1, season)
            team2_stats = api.get_team_by_name(team2, season)

            if not team1_stats:
                return f"Error: Team '{team1}' not found"
            if not team2_stats:
                return f"Error: Team '{team2}' not found"

            # Analyze tempo matchup
            analysis = analyzer.analyze_pace_matchup(team1_stats, team2_stats)

            # Format comprehensive report
            report = [
                f"# Tempo/Pace Matchup: {team1} vs {team2}",
                "=" * 70,
                "",
                "## Team Profiles",
                "",
                f"### {analysis.team1_profile.team_name}",
                f"- Adjusted Tempo: {analysis.team1_profile.adj_tempo} (Rank: {analysis.team1_profile.rank_tempo})",
                f"- Pace Style: {analysis.team1_profile.pace_style}",
                f"- Offensive Style: {analysis.team1_profile.off_style} (APL: {analysis.team1_profile.apl_off}s)",
                f"- Defensive Style: {analysis.team1_profile.def_style} (APL: {analysis.team1_profile.apl_def}s)",
                "",
                f"### {analysis.team2_profile.team_name}",
                f"- Adjusted Tempo: {analysis.team2_profile.adj_tempo} (Rank: {analysis.team2_profile.rank_tempo})",
                f"- Pace Style: {analysis.team2_profile.pace_style}",
                f"- Offensive Style: {analysis.team2_profile.off_style} (APL: {analysis.team2_profile.apl_off}s)",
                f"- Defensive Style: {analysis.team2_profile.def_style} (APL: {analysis.team2_profile.apl_def}s)",
                "",
                "## Matchup Analysis",
                "",
                f"- **Tempo Differential**: {analysis.tempo_differential:+.1f} possessions ({team1} {'faster' if analysis.tempo_differential > 0 else 'slower'})",
                f"- **Expected Game Pace**: {analysis.expected_possessions} possessions",
                f"- **Style Mismatch Score**: {analysis.style_mismatch_score}/10",
                f"- **Pace Control**: {analysis.pace_advantage}",
                f"- **Tempo Control Factor**: {analysis.tempo_control_factor:+.2f}",
                "",
                "## APL Matchup Insights",
                "",
                f"### {team1} Offensive Disruption: {analysis.offensive_disruption_team1}",
                f"  - {team1} offense ({analysis.team1_profile.apl_off}s) vs {team2} defense ({analysis.team2_profile.apl_def}s)",
                f"  - Mismatch: {analysis.apl_off_mismatch_team1:+.1f} seconds",
                "",
                f"### {team2} Offensive Disruption: {analysis.offensive_disruption_team2}",
                f"  - {team2} offense ({analysis.team2_profile.apl_off}s) vs {team1} defense ({analysis.team1_profile.apl_def}s)",
                f"  - Mismatch: {analysis.apl_off_mismatch_team2:+.1f} seconds",
                "",
                "## Impact Estimates",
                "",
                f"- **Tempo Impact on Margin**: {analysis.tempo_impact_on_margin:+.2f} points",
                f"- **Confidence Adjustment**: {analysis.confidence_adjustment:.3f}x variance",
                f"- **Optimal Pace ({team1})**: {analysis.optimal_pace_team1} possessions",
                f"- **Optimal Pace ({team2})**: {analysis.optimal_pace_team2} possessions",
                "",
                "## Pace Scenario Analysis",
                "",
                f"- **Fast Pace Favors**: {analysis.fast_pace_favors}",
                f"- **Slow Pace Favors**: {analysis.slow_pace_favors}",
            ]

            return "\n".join(report)
        except Exception as e:
            logger.warning(f"Tempo analysis failed: {e}")
            return f"Error analyzing tempo matchup: {e}"

    return "Error: Tempo analysis requires API authentication (KENPOM_API_KEY)."


async def _handle_get_player_depth_chart(arguments: dict[str, Any]) -> str:
    """Get team depth chart with player valuations."""
    team = arguments.get("team")
    if not team:
        return "Error: 'team' parameter is required"

    season = arguments.get("season", get_current_season())

    # Need both API for team stats and scraper for player stats
    api = _get_api_client()
    client = _get_scraper_client()

    if not api or not client:
        return "Error: Player depth chart requires both API key (KENPOM_API_KEY) and scraper credentials (KENPOM_EMAIL/KENPOM_PASSWORD)."

    try:
        from .player_impact import PlayerImpactModel

        # Get team stats from API
        team_data = api.get_team_by_name(team, season)
        if not team_data:
            return f"Error: Team '{team}' not found"

        # Get player stats from scraper
        player_stats_df = client.get_playerstats(season=season)

        # Create depth chart
        model = PlayerImpactModel()
        depth_chart = model.get_team_depth_chart(team, player_stats_df, team_data)

        if depth_chart.empty:
            return f"Error: No players found for {team}"

        # Format as text
        report = [
            f"# {team} Depth Chart ({season} Season)",
            "=" * 70,
            "",
            f"Players ranked by estimated value (AdjEM points contributed):",
            "",
        ]

        for idx, row in depth_chart.iterrows():
            report.append(
                f"{idx + 1}. {row['Player']} ({row['yr']}, {row['ht']})"
            )
            report.append(f"   Poss%: {row['Poss%']:.1f}% | ORtg: {row['ORtg']:.1f}")
            report.append(
                f"   Value: {row['EstimatedValue']:+.2f} AdjEM pts | VOR: {row['VOR']:+.2f}"
            )
            report.append("")

        return "\n".join(report)

    except Exception as e:
        logger.warning(f"Depth chart generation failed: {e}")
        return f"Error generating depth chart: {e}"


async def _handle_calculate_player_value(arguments: dict[str, Any]) -> str:
    """Calculate individual player's value."""
    player = arguments.get("player")
    team = arguments.get("team")

    if not player or not team:
        return "Error: Both 'player' and 'team' parameters are required"

    season = arguments.get("season", get_current_season())

    # Need both API and scraper
    api = _get_api_client()
    client = _get_scraper_client()

    if not api or not client:
        return "Error: Player value calculation requires both API key (KENPOM_API_KEY) and scraper credentials (KENPOM_EMAIL/KENPOM_PASSWORD)."

    try:
        from .player_impact import PlayerImpactModel

        # Get team stats
        team_data = api.get_team_by_name(team, season)
        if not team_data:
            return f"Error: Team '{team}' not found"

        # Get player stats
        player_stats_df = client.get_playerstats(season=season)
        player_row = player_stats_df[
            (player_stats_df["Team"] == team) & (player_stats_df["Player"] == player)
        ]

        if player_row.empty:
            return f"Error: Player '{player}' not found on {team}"

        # Calculate value
        model = PlayerImpactModel()
        value = model.calculate_player_value(
            player_row.iloc[0].to_dict(), team_data
        )

        # Format report
        report = [
            f"# Player Value: {value.player_name}",
            f"**Team**: {value.team} ({season})",
            "=" * 70,
            "",
            "## Usage Metrics",
            f"- Possession %: {value.possession_pct:.1f}%",
            f"- Estimated Minutes %: {value.minutes_pct:.1f}%",
            "",
            "## Efficiency Metrics",
            f"- Offensive Rating: {value.offensive_rating:.1f}",
            f"- Effective FG%: {value.effective_fg_pct:.1f}%",
            f"- True Shooting%: {value.true_shooting_pct:.1f}%",
            "",
            "## Estimated Value",
            f"- **Total Value**: {value.estimated_value:+.2f} AdjEM points",
            f"- Offensive Contribution: {value.offensive_contribution:+.2f} points",
            f"- Defensive Contribution: {value.defensive_contribution:+.2f} points",
            f"- Replacement Level: {value.replacement_level:+.2f} points",
            f"- **Value Over Replacement**: {value.value_over_replacement:+.2f} points",
        ]

        return "\n".join(report)

    except Exception as e:
        logger.warning(f"Player value calculation failed: {e}")
        return f"Error calculating player value: {e}"


async def _handle_estimate_injury_impact(arguments: dict[str, Any]) -> str:
    """Estimate impact of player injury."""
    player = arguments.get("player")
    team = arguments.get("team")

    if not player or not team:
        return "Error: Both 'player' and 'team' parameters are required"

    injury_severity = arguments.get("injury_severity", "out")
    season = arguments.get("season", get_current_season())

    # Need both API and scraper
    api = _get_api_client()
    client = _get_scraper_client()

    if not api or not client:
        return "Error: Injury impact estimation requires both API key (KENPOM_API_KEY) and scraper credentials (KENPOM_EMAIL/KENPOM_PASSWORD)."

    try:
        from .player_impact import PlayerImpactModel

        # Get team stats
        team_data = api.get_team_by_name(team, season)
        if not team_data:
            return f"Error: Team '{team}' not found"

        # Get player stats
        player_stats_df = client.get_playerstats(season=season)
        player_row = player_stats_df[
            (player_stats_df["Team"] == team) & (player_stats_df["Player"] == player)
        ]

        if player_row.empty:
            return f"Error: Player '{player}' not found on {team}"

        # Calculate value and injury impact
        model = PlayerImpactModel()
        value = model.calculate_player_value(
            player_row.iloc[0].to_dict(), team_data
        )
        injury = model.estimate_injury_impact(value, team_data, injury_severity)

        # Format report
        severity_label = injury_severity.upper()
        report = [
            f"# Injury Impact: {value.player_name} ({severity_label})",
            f"**Team**: {value.team} ({season})",
            "=" * 70,
            "",
            "## Player Value",
            f"- Estimated Value: {value.estimated_value:+.2f} AdjEM points",
            f"- Value Over Replacement: {value.value_over_replacement:+.2f} points",
            "",
            "## Injury Impact",
            f"- **Severity**: {injury.severity}",
            f"- **Estimated Loss**: {injury.estimated_adj_em_loss:.2f} AdjEM points",
            f"- **Confidence Interval**: [{injury.confidence_interval[0]:.1f}, {injury.confidence_interval[1]:.1f}]",
            "",
            "## Adjusted Team Ratings",
            f"- Baseline AdjEM: {injury.team_adj_em_baseline:.2f}",
            f"- **Adjusted AdjEM**: {injury.adjusted_adj_em:.2f} ({injury.adjusted_adj_em - injury.team_adj_em_baseline:+.2f})",
            f"- Adjusted AdjOE: {injury.adjusted_adj_oe:.2f}",
            f"- Adjusted AdjDE: {injury.adjusted_adj_de:.2f}",
            "",
            f"**Note**: '{injury_severity}' severity means {{'out': '100%', 'doubtful': '75%', 'questionable': '50%'}}[injury_severity] of value lost to replacement.",
        ]

        return "\n".join(report)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.warning(f"Injury impact estimation failed: {e}")
        return f"Error estimating injury impact: {e}"


# ==================== Server Entry Point ====================


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
