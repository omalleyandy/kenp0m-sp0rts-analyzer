"""Tournament simulation system for NCAA basketball.

Provides Monte Carlo simulation for tournament brackets with upset probability
calculations and expected advancement metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .api_client import KenPomAPI
from .helpers import normalize_team_name


@dataclass
class TournamentTeam:
    """Team in a tournament simulation."""

    name: str
    seed: int
    adj_em: float
    adj_oe: float
    adj_de: float
    adj_tempo: float
    rank: int = 0


@dataclass
class GameSimulation:
    """Result of a single game simulation."""

    team1: str
    team2: str
    winner: str
    team1_win_prob: float
    margin: float
    upset: bool


@dataclass
class TournamentProbabilities:
    """Tournament advancement probabilities for a team."""

    team: str
    seed: int
    round_of_64: float = 1.0
    round_of_32: float = 0.0
    sweet_16: float = 0.0
    elite_8: float = 0.0
    final_4: float = 0.0
    championship: float = 0.0
    winner: float = 0.0


@dataclass
class UpsetPick:
    """Potential upset pick recommendation."""

    team: str
    seed: int
    opponent: str
    opponent_seed: int
    upset_probability: float
    value_rating: float
    reasoning: str


@dataclass
class BracketRecommendation:
    """Tournament bracket recommendations."""

    final_four: list[str]
    champion: str
    best_upset_picks: list[UpsetPick]
    sleeper_picks: list[str]
    avoid_picks: list[str]


class TournamentSimulator:
    """Monte Carlo tournament simulator.

    Uses KenPom efficiency ratings to simulate tournament brackets
    and calculate advancement probabilities.
    """

    # Home court advantage in neutral site tournament
    HCA = 0.0  # Neutral site

    # Standard deviation for score predictions
    SCORE_STD = 11.0

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize tournament simulator."""
        self.api = api or KenPomAPI()

    def simulate_game(
        self,
        team1: TournamentTeam,
        team2: TournamentTeam,
        neutral_site: bool = True,
    ) -> GameSimulation:
        """Simulate a single game between two teams."""
        # Calculate expected margin
        em_diff = team1.adj_em - team2.adj_em

        # Tempo adjustment
        avg_tempo = (team1.adj_tempo + team2.adj_tempo) / 2
        league_avg = 68.0
        tempo_factor = avg_tempo / league_avg

        expected_margin = em_diff * tempo_factor

        # Win probability using logistic function
        win_prob = 1 / (1 + 10 ** (-expected_margin / 10))

        # Simulate outcome
        random_margin = random.gauss(expected_margin, self.SCORE_STD)
        winner = team1 if random_margin > 0 else team2

        # Check for upset (lower seed beats higher seed)
        upset = (
            winner.seed > min(team1.seed, team2.seed)
            if team1.seed != team2.seed
            else False
        )

        return GameSimulation(
            team1=team1.name,
            team2=team2.name,
            winner=winner.name,
            team1_win_prob=win_prob,
            margin=abs(random_margin),
            upset=upset,
        )

    def get_win_probability(
        self,
        team1: TournamentTeam,
        team2: TournamentTeam,
    ) -> float:
        """Calculate win probability for team1 vs team2."""
        em_diff = team1.adj_em - team2.adj_em
        avg_tempo = (team1.adj_tempo + team2.adj_tempo) / 2
        tempo_factor = avg_tempo / 68.0
        expected_margin = em_diff * tempo_factor
        return 1 / (1 + 10 ** (-expected_margin / 10))

    def simulate_tournament(
        self,
        teams: list[TournamentTeam],
        n_simulations: int = 10000,
    ) -> dict[str, TournamentProbabilities]:
        """Simulate entire tournament multiple times.

        Args:
            teams: List of 64 tournament teams with seeds.
            n_simulations: Number of Monte Carlo simulations.

        Returns:
            Dictionary mapping team names to advancement probabilities.
        """
        results: dict[str, dict[str, int]] = {
            team.name: {
                "r64": 0, "r32": 0, "s16": 0, "e8": 0, "f4": 0, "nc": 0, "champ": 0
            }
            for team in teams
        }

        for _ in range(n_simulations):
            bracket = self._simulate_single_tournament(teams)
            for round_name, winners in bracket.items():
                for winner in winners:
                    results[winner][round_name] += 1

        # Convert counts to probabilities
        probs = {}
        for team in teams:
            name = team.name
            counts = results[name]
            probs[name] = TournamentProbabilities(
                team=name,
                seed=team.seed,
                round_of_64=1.0,
                round_of_32=counts["r32"] / n_simulations,
                sweet_16=counts["s16"] / n_simulations,
                elite_8=counts["e8"] / n_simulations,
                final_4=counts["f4"] / n_simulations,
                championship=counts["nc"] / n_simulations,
                winner=counts["champ"] / n_simulations,
            )

        return probs

    def _simulate_single_tournament(
        self,
        teams: list[TournamentTeam],
    ) -> dict[str, list[str]]:
        """Simulate a single tournament."""
        # Group by seed for simplified bracket
        by_seed = {team.seed: team for team in teams}

        results = {
            "r32": [], "s16": [], "e8": [], "f4": [], "nc": [], "champ": []
        }

        # Round of 64: 1v16, 2v15, etc.
        matchups_r64 = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

        r32_teams = []
        for s1, s2 in matchups_r64:
            if s1 in by_seed and s2 in by_seed:
                game = self.simulate_game(by_seed[s1], by_seed[s2])
                winner = by_seed[s1] if game.winner == by_seed[s1].name else by_seed[s2]
                r32_teams.append(winner)
                results["r32"].append(winner.name)

        # Continue through rounds (simplified)
        current_round = r32_teams
        round_names = ["s16", "e8", "f4", "nc", "champ"]

        for round_name in round_names:
            next_round = []
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    game = self.simulate_game(current_round[i], current_round[i + 1])
                    winner = (
                        current_round[i]
                        if game.winner == current_round[i].name
                        else current_round[i + 1]
                    )
                    next_round.append(winner)
                    results[round_name].append(winner.name)
            current_round = next_round

        return results

    def find_upset_picks(
        self,
        teams: list[TournamentTeam],
        min_seed_diff: int = 4,
        min_upset_prob: float = 0.25,
    ) -> list[UpsetPick]:
        """Find valuable upset picks for bracket pools."""
        upset_picks = []

        # Check first round matchups
        by_seed = {team.seed: team for team in teams}
        matchups = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]

        for fav_seed, dog_seed in matchups:
            if fav_seed not in by_seed or dog_seed not in by_seed:
                continue

            favorite = by_seed[fav_seed]
            underdog = by_seed[dog_seed]
            seed_diff = dog_seed - fav_seed

            if seed_diff < min_seed_diff:
                continue

            upset_prob = 1 - self.get_win_probability(favorite, underdog)

            if upset_prob >= min_upset_prob:
                # Value rating: probability * seed_diff (higher seeds = more value)
                value = upset_prob * seed_diff

                upset_picks.append(UpsetPick(
                    team=underdog.name,
                    seed=underdog.seed,
                    opponent=favorite.name,
                    opponent_seed=favorite.seed,
                    upset_probability=upset_prob,
                    value_rating=value,
                    reasoning=f"#{dog_seed} has {upset_prob:.1%} chance vs #{fav_seed}",
                ))

        return sorted(upset_picks, key=lambda x: x.value_rating, reverse=True)

    def generate_bracket_recommendations(
        self,
        teams: list[TournamentTeam],
        n_simulations: int = 10000,
    ) -> BracketRecommendation:
        """Generate bracket recommendations based on simulations."""
        probs = self.simulate_tournament(teams, n_simulations)
        upset_picks = self.find_upset_picks(teams)

        # Final Four: top 4 by Final Four probability
        final_four = sorted(
            probs.items(),
            key=lambda x: x[1].final_4,
            reverse=True
        )[:4]

        # Champion: highest championship probability
        champion = max(probs.items(), key=lambda x: x[1].winner)

        # Sleepers: teams with good value (high prob relative to seed)
        sleepers = [
            name for name, p in probs.items()
            if p.seed >= 5 and p.sweet_16 > 0.4
        ][:3]

        # Avoid: high seeds with low advancement probability
        avoid = [
            name for name, p in probs.items()
            if p.seed <= 4 and p.sweet_16 < 0.5
        ][:3]

        return BracketRecommendation(
            final_four=[name for name, _ in final_four],
            champion=champion[0],
            best_upset_picks=upset_picks[:5],
            sleeper_picks=sleepers,
            avoid_picks=avoid,
        )
