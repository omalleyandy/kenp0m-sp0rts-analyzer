"""Quick test script for KenPomService."""

from kenp0m_sp0rts_analyzer.kenpom.api import KenPomService


def main():
    # Initialize service
    service = KenPomService()

    # Get database stats
    stats = service.get_database_stats()
    print("Database Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get latest ratings
    ratings = service.get_latest_ratings()
    print(f"\nFound {len(ratings)} teams")

    if ratings:
        top_team = ratings[0]
        print(f"\nTop team: {top_team.team_name}")
        print(f"  Adj EM: {top_team.adj_em:.2f}")
        print(f"  Adj OE: {top_team.adj_oe:.2f}")
        print(f"  Adj DE: {top_team.adj_de:.2f}")


if __name__ == "__main__":
    main()
