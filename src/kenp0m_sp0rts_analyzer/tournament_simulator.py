"""NCAA Tournament Monte Carlo Simulator.

Simulates NCAA tournament brackets using KenPom efficiency metrics and variance
modeling to generate probabilistic predictions for round-by-round advancement.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .api_client import KenPomAPI


@dataclass
class TournamentTeam:
    """Team in tournament with seed and metrics."""

    name: str
    seed: int
    region: str
    adj_em: float
    adj_tempo: float
    adj_o: float
    adj_d: float


@dataclass
class GameSimulation:
    """Result of single game simulation."""

    winner: str
    loser: str
    predicted_margin: float
    upset: bool  # True if lower seed won


@dataclass
class TournamentProbabilities:
    """Tournament advancement probabilities for all teams."""

    round_64: dict[str, float]  # First round (all teams start at 1.0)
    round_32: dict[str, float]  # Second round
    sweet_16: dict[str, float]  # Sweet 16
    elite_8: dict[str, float]  # Elite 8
    final_4: dict[str, float]  # Final Four
    championship: dict[str, float]  # Championship game
    winner: dict[str, float]  # Tournament champion


@dataclass
class UpsetPick:
    """High-value upset opportunity."""

    matchup: str
    favorite: str
    underdog: str
    favorite_seed: int
    underdog_seed: int
    upset_probability: float
    expected_value: float  # For bracket pools


@dataclass
class BracketRecommendation:
    """Recommended bracket picks based on simulation."""

    champion: str
    champion_probability: float
    final_four: list[str]
    upset_picks: list[UpsetPick]
    confidence_score: float


class TournamentSimulator:
    """Monte Carlo simulator for NCAA tournament."""

    def __init__(self, api_key: str | None = None, random_seed: int | None = None):
        """Initialize tournament simulator.

        Args:
            api_key: KenPom API key (uses env var if not provided)
            random_seed: Random seed for reproducible simulations
        """
        self.api = KenPomAPI(api_key)

        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_tournament(
        self,
        bracket: dict[str, list[TournamentTeam]],
        num_simulations: int = 10000,
        verbose: bool = False,
    ) -> TournamentProbabilities:
        """Run Monte Carlo simulation of tournament.

        Args:
            bracket: Dictionary mapping regions to list of teams (seeded 1-16)
            num_simulations: Number of simulations to run
            verbose: Print progress updates

        Returns:
            Round-by-round advancement probabilities
        """
        # Initialize results tracking
        results: dict[str, dict[str, int]] = {
            "round_64": defaultdict(int),
            "round_32": defaultdict(int),
            "sweet_16": defaultdict(int),
            "elite_8": defaultdict(int),
            "final_4": defaultdict(int),
            "championship": defaultdict(int),
            "winner": defaultdict(int),
        }

        # All teams start in round of 64
        for region_teams in bracket.values():
            for team in region_teams:
                results["round_64"][team.name] = num_simulations

        # Run simulations
        for sim in range(num_simulations):
            if verbose and sim % 1000 == 0:
                print(f"Simulation {sim}/{num_simulations}")

            bracket_result = self._simulate_single_bracket(bracket)
            self._accumulate_results(bracket_result, results)

        # Convert counts to probabilities
        probabilities = TournamentProbabilities(
            round_64={
                team: 1.0
                for teams in bracket.values()
                for team in [t.name for t in teams]
            },
            round_32={
                team: count / num_simulations
                for team, count in results["round_32"].items()
            },
            sweet_16={
                team: count / num_simulations
                for team, count in results["sweet_16"].items()
            },
            elite_8={
                team: count / num_simulations
                for team, count in results["elite_8"].items()
            },
            final_4={
                team: count / num_simulations
                for team, count in results["final_4"].items()
            },
            championship={
                team: count / num_simulations
                for team, count in results["championship"].items()
            },
            winner={
                team: count / num_simulations
                for team, count in results["winner"].items()
            },
        )

        return probabilities

    def _simulate_single_bracket(
        self, bracket: dict[str, list[TournamentTeam]]
    ) -> dict[str, list[str]]:
        """Simulate one complete tournament bracket.

        Returns:
            Dictionary mapping round names to teams advancing
        """
        result = {
            "round_32": [],
            "sweet_16": [],
            "elite_8": [],
            "final_4": [],
            "championship": [],
            "winner": [],
        }

        # Simulate each region independently through Elite 8
        regional_winners = []

        for _region, teams in bracket.items():
            # Round of 64 (first round)
            round_32_teams = self._simulate_round(teams, 8)
            result["round_32"].extend([t.name for t in round_32_teams])

            # Round of 32 (second round)
            sweet_16_teams = self._simulate_round(round_32_teams, 4)
            result["sweet_16"].extend([t.name for t in sweet_16_teams])

            # Sweet 16 (regional semis)
            elite_8_teams = self._simulate_round(sweet_16_teams, 2)
            result["elite_8"].extend([t.name for t in elite_8_teams])

            # Elite 8 (regional final)
            final_4_team = self._simulate_round(elite_8_teams, 1)[0]
            regional_winners.append(final_4_team)

        result["final_4"] = [t.name for t in regional_winners]

        # Final Four semis
        championship_teams = self._simulate_round(regional_winners, 2)
        result["championship"] = [t.name for t in championship_teams]

        # Championship game
        champion = self._simulate_round(championship_teams, 1)[0]
        result["winner"] = [champion.name]

        return result

    def _simulate_round(
        self, teams: list[TournamentTeam], num_winners: int
    ) -> list[TournamentTeam]:
        """Simulate a round of games.

        Args:
            teams: List of teams in round (must be even number)
            num_winners: Number of teams advancing

        Returns:
            List of winning teams
        """
        if len(teams) != num_winners * 2:
            raise ValueError(f"Expected {num_winners * 2} teams, got {len(teams)}")

        winners = []

        # Simulate matchups
        for i in range(0, len(teams), 2):
            team1 = teams[i]
            team2 = teams[i + 1]

            winner = self._simulate_game(team1, team2)
            winners.append(winner)

        return winners

    def _simulate_game(
        self, team1: TournamentTeam, team2: TournamentTeam
    ) -> TournamentTeam:
        """Simulate single game with variance.

        Uses efficiency margin and tempo to model game outcomes with
        appropriate variance based on number of possessions.

        Args:
            team1: First team
            team2: Second team

        Returns:
            Winning team
        """
        # Efficiency margin difference (expected margin per 100 possessions)
        em_diff = team1.adj_em - team2.adj_em

        # Average tempo (possessions per 40 minutes)
        avg_tempo = (team1.adj_tempo + team2.adj_tempo) / 2

        # Calculate game variance based on tempo
        # More possessions = more predictable (law of large numbers)
        variance = self._calculate_game_variance(avg_tempo)

        # Sample from normal distribution
        # Convert per-100 margin to actual game margin
        expected_margin = em_diff * (avg_tempo / 100)
        actual_margin = np.random.normal(expected_margin, variance)

        return team1 if actual_margin > 0 else team2

    def _calculate_game_variance(self, tempo: float) -> float:
        """Calculate game outcome variance based on tempo.

        Higher tempo = lower variance per possession (law of large numbers).
        Empirically calibrated to ~11 point standard deviation for average tempo.

        Args:
            tempo: Average possessions per 40 minutes

        Returns:
            Standard deviation in points
        """
        # Baseline: 68 possessions (average college basketball tempo)
        # Standard deviation: ~11 points
        baseline_tempo = 68.0
        baseline_std = 11.0

        # Scale by square root (variance scales with sample size)
        return baseline_std * np.sqrt(baseline_tempo / tempo)

    def _accumulate_results(
        self, bracket_result: dict[str, list[str]], results: dict[str, dict[str, int]]
    ) -> None:
        """Accumulate results from single simulation."""
        for round_name, teams in bracket_result.items():
            for team in teams:
                results[round_name][team] += 1

    def identify_upset_picks(
        self,
        probabilities: TournamentProbabilities,
        bracket: dict[str, list[TournamentTeam]],
        min_upset_prob: float = 0.35,
        max_seed_diff: int = 3,
    ) -> list[UpsetPick]:
        """Identify high-value upset opportunities for bracket pools.

        Args:
            probabilities: Tournament simulation results
            bracket: Original bracket with seeds
            min_upset_prob: Minimum upset probability to consider (default: 35%)
            max_seed_diff: Maximum seed difference (default: 3 seeds apart)

        Returns:
            List of recommended upset picks sorted by expected value
        """
        upset_picks = []

        # Build seed lookup
        seed_lookup = {}
        for region_teams in bracket.values():
            for team in region_teams:
                seed_lookup[team.name] = team.seed

        # Analyze round of 64 for first-round upsets
        for team, prob in probabilities.round_32.items():
            seed = seed_lookup.get(team)
            if seed is None or seed <= 8:  # Skip favorites
                continue

            # Find their opponent (opposite seed in matchup)
            opponent_seed = 17 - seed  # 9 plays 8, 10 plays 7, etc.

            if abs(seed - opponent_seed) > max_seed_diff:
                continue

            if prob >= min_upset_prob:
                # Calculate expected value for bracket pools
                # Higher seeds picked less often, so more valuable
                pick_rate = self._estimate_public_pick_rate(opponent_seed, seed)
                expected_value = prob * (1.0 - pick_rate)

                upset_picks.append(
                    UpsetPick(
                        matchup=f"{opponent_seed} vs {seed}",
                        favorite=f"(Seed {opponent_seed})",
                        underdog=team,
                        favorite_seed=opponent_seed,
                        underdog_seed=seed,
                        upset_probability=prob,
                        expected_value=expected_value,
                    )
                )

        # Sort by expected value
        upset_picks.sort(key=lambda x: x.expected_value, reverse=True)

        return upset_picks

    def _estimate_public_pick_rate(
        self, favorite_seed: int, underdog_seed: int
    ) -> float:
        """Estimate what percentage of public picks favorite.

        Based on historical bracket pool data.
        """
        seed_diff = underdog_seed - favorite_seed

        if seed_diff <= 1:
            return 0.50  # Toss-up games
        elif seed_diff <= 3:
            return 0.70  # Moderate favorite
        elif seed_diff <= 5:
            return 0.85  # Strong favorite
        else:
            return 0.95  # Heavy favorite

    def generate_bracket_recommendation(
        self,
        probabilities: TournamentProbabilities,
        bracket: dict[str, list[TournamentTeam]],
    ) -> BracketRecommendation:
        """Generate recommended bracket based on probabilities.

        Args:
            probabilities: Tournament simulation results
            bracket: Original bracket structure

        Returns:
            Recommended picks with confidence scores
        """
        # Find most likely champion
        champion_probs = sorted(
            probabilities.winner.items(), key=lambda x: x[1], reverse=True
        )
        champion, champion_prob = champion_probs[0]

        # Find most likely Final Four
        final_4_probs = sorted(
            probabilities.final_4.items(), key=lambda x: x[1], reverse=True
        )
        final_four = [team for team, _ in final_4_probs[:4]]

        # Identify upset picks
        upsets = self.identify_upset_picks(probabilities, bracket)

        # Calculate overall confidence (based on champion probability)
        if champion_prob > 0.30:
            confidence = 0.9
        elif champion_prob > 0.20:
            confidence = 0.7
        elif champion_prob > 0.10:
            confidence = 0.5
        else:
            confidence = 0.3

        return BracketRecommendation(
            champion=champion,
            champion_probability=champion_prob,
            final_four=final_four,
            upset_picks=upsets[:5],  # Top 5 upsets
            confidence_score=confidence,
        )

    def analyze_historical_seed_performance(
        self,
        probabilities: TournamentProbabilities,
        bracket: dict[str, list[TournamentTeam]],
    ) -> dict[int, dict[str, float]]:
        """Analyze how each seed performs in simulation.

        Args:
            probabilities: Simulation results
            bracket: Bracket structure with seeds

        Returns:
            Dictionary mapping seed to round-by-round advancement rates
        """
        # Build seed lookup
        seed_lookup = {}
        for region_teams in bracket.values():
            for team in region_teams:
                seed_lookup[team.name] = team.seed

        # Aggregate by seed
        seed_performance: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for team, prob in probabilities.round_32.items():
            seed = seed_lookup.get(team)
            if seed:
                seed_performance[seed]["round_32"].append(prob)

        for team, prob in probabilities.sweet_16.items():
            seed = seed_lookup.get(team)
            if seed:
                seed_performance[seed]["sweet_16"].append(prob)

        for team, prob in probabilities.elite_8.items():
            seed = seed_lookup.get(team)
            if seed:
                seed_performance[seed]["elite_8"].append(prob)

        for team, prob in probabilities.final_4.items():
            seed = seed_lookup.get(team)
            if seed:
                seed_performance[seed]["final_4"].append(prob)

        for team, prob in probabilities.winner.items():
            seed = seed_lookup.get(team)
            if seed:
                seed_performance[seed]["winner"].append(prob)

        # Calculate averages
        seed_averages = {}
        for seed, rounds in seed_performance.items():
            seed_averages[seed] = {
                round_name: float(np.mean(probs))
                for round_name, probs in rounds.items()
            }

        return seed_averages

    def load_bracket_from_kenpom(
        self,
        selection_sunday_date: str,
        regions: dict[str, list[tuple[str, int]]] | None = None,
    ) -> dict[str, list[TournamentTeam]]:
        """Load tournament bracket from KenPom ratings.

        Args:
            selection_sunday_date: Date string (YYYY-MM-DD) for Selection Sunday
            regions: Optional manual region assignments {region: [(team, seed), ...]}
                    If None, auto-assigns top 64 teams to regions

        Returns:
            Dictionary mapping regions to seeded teams
        """
        # Get ratings from Selection Sunday
        ratings_response = self.api.get_archive(archive_date=selection_sunday_date)
        ratings = ratings_response.data

        # If regions provided, use them
        if regions:
            bracket = {}
            for region, team_seeds in regions.items():
                region_teams = []
                for team_name, seed in team_seeds:
                    # Find team in ratings
                    team_data = next(
                        (t for t in ratings if t["TeamName"] == team_name), None
                    )
                    if team_data:
                        region_teams.append(
                            TournamentTeam(
                                name=team_name,
                                seed=seed,
                                region=region,
                                adj_em=team_data["AdjEM"],
                                adj_tempo=team_data["AdjTempo"],
                                adj_o=team_data["AdjOE"],
                                adj_d=team_data["AdjDE"],
                            )
                        )
                bracket[region] = region_teams
            return bracket

        # Auto-assign top 64 teams to 4 regions
        top_64 = sorted(ratings, key=lambda x: x["AdjEM"], reverse=True)[:64]

        # Assign to regions (simplified S-curve seeding)
        regions_list = ["East", "West", "South", "Midwest"]
        bracket = {region: [] for region in regions_list}

        for i, team_data in enumerate(top_64):
            seed = (i % 16) + 1
            region_idx = i // 16
            region = regions_list[region_idx]

            bracket[region].append(
                TournamentTeam(
                    name=team_data["TeamName"],
                    seed=seed,
                    region=region,
                    adj_em=team_data["AdjEM"],
                    adj_tempo=team_data["AdjTempo"],
                    adj_o=team_data["AdjOE"],
                    adj_d=team_data["AdjDE"],
                )
            )

        return bracket
