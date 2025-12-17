#!/usr/bin/env python3
"""
Daily KenPom Data Collection Pipeline

Fetches all necessary data from KenPom API for predictions and analysis.

Usage:
    python scripts/collect_daily_data.py --season 2025
    python scripts/collect_daily_data.py --season 2025 --output data/kenpom.parquet
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


class DailyDataCollector:
    """Automated daily data collection from KenPom API."""

    def __init__(self, season: int, output_dir: Path = Path("data")):
        self.season = season
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.api = KenPomAPI()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def collect_all_data(self) -> pd.DataFrame:
        """
        Fetch all data from KenPom API and merge into single DataFrame.

        Returns:
            Merged DataFrame with all team data
        """
        print(f"\n{'='*70}")
        print(f"KenPom Data Collection - {self.season} Season")
        print(f"{'='*70}\n")

        # 1. Core Ratings
        print("[1/6] Fetching team ratings...")
        ratings_response = self.api.get_ratings(year=self.season)
        ratings_df = ratings_response.to_dataframe()
        print(f"      Retrieved {len(ratings_df)} teams")

        # 2. Four Factors
        print("[2/6] Fetching Four Factors...")
        ff_response = self.api.get_four_factors(year=self.season)
        ff_df = ff_response.to_dataframe()
        print(f"      Retrieved {len(ff_df)} teams")

        # 3. Point Distribution
        print("[3/6] Fetching point distribution...")
        pd_response = self.api.get_point_distribution(year=self.season)
        pd_df = pd_response.to_dataframe()
        print(f"      Retrieved {len(pd_df)} teams")

        # 4. Height & Experience
        print("[4/6] Fetching height/experience data...")
        height_response = self.api.get_height(year=self.season)
        height_df = height_response.to_dataframe()
        print(f"      Retrieved {len(height_df)} teams")

        # 5. Misc Stats
        print("[5/6] Fetching miscellaneous stats...")
        misc_response = self.api.get_misc_stats(year=self.season)
        misc_df = misc_response.to_dataframe()
        print(f"      Retrieved {len(misc_df)} teams")

        # 6. Historical Data (30 days ago for momentum)
        print("[6/6] Fetching historical data (30 days ago)...")
        try:
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime(
                "%Y-%m-%d"
            )
            archive_response = self.api.get_archive(archive_date=thirty_days_ago)
            archive_df = archive_response.to_dataframe()
            print(f"      Retrieved {len(archive_df)} teams from {thirty_days_ago}")
            has_archive = True
        except Exception as e:
            print(f"      Warning: Could not fetch archive data: {e}")
            archive_df = pd.DataFrame()
            has_archive = False

        # Merge all datasets
        print("\nMerging datasets...")
        merged = ratings_df.copy()

        # Merge Four Factors
        merged = merged.merge(
            ff_df, on="TeamName", how="left", suffixes=("", "_ff")
        )

        # Merge Point Distribution
        merged = merged.merge(pd_df, on="TeamName", how="left", suffixes=("", "_pd"))

        # Merge Height
        merged = merged.merge(
            height_df, on="TeamName", how="left", suffixes=("", "_ht")
        )

        # Merge Misc Stats
        merged = merged.merge(
            misc_df, on="TeamName", how="left", suffixes=("", "_misc")
        )

        # Merge Archive (if available)
        if has_archive and len(archive_df) > 0:
            # Rename archive columns to avoid conflicts
            archive_df = archive_df.rename(
                columns={
                    "AdjEM": "AdjEM_30d",
                    "AdjOE": "AdjOE_30d",
                    "AdjDE": "AdjDE_30d",
                    "AdjTempo": "AdjTempo_30d",
                    "RankAdjEM": "Rank_30d",
                }
            )
            merged = merged.merge(
                archive_df[
                    [
                        "TeamName",
                        "AdjEM_30d",
                        "AdjOE_30d",
                        "AdjDE_30d",
                        "AdjTempo_30d",
                        "Rank_30d",
                    ]
                ],
                on="TeamName",
                how="left",
                suffixes=("", "_archive"),
            )

            # Calculate momentum (change over 30 days)
            merged["AdjEM_Momentum"] = merged["AdjEM"] - merged["AdjEM_30d"]
            merged["Rank_Change"] = merged["Rank_30d"] - merged["RankAdjEM"]

        print(f"\nMerged dataset:")
        print(f"  Teams: {len(merged)}")
        print(f"  Features: {len(merged.columns)}")
        print(f"  Memory: {merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return merged

    def save_data(self, data: pd.DataFrame, output_path: Path | None = None):
        """
        Save data to parquet file.

        Args:
            data: DataFrame to save
            output_path: Custom output path (optional)
        """
        if output_path is None:
            output_path = (
                self.output_dir / f"kenpom_{self.season}_{self.timestamp}.parquet"
            )

        data.to_parquet(output_path, index=False, compression="snappy")
        print(f"\nSaved data to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")

        # Also save a "latest" copy for easy access
        latest_path = self.output_dir / f"kenpom_{self.season}_latest.parquet"
        data.to_parquet(latest_path, index=False, compression="snappy")
        print(f"Saved latest copy to: {latest_path}")

    def print_summary(self, data: pd.DataFrame):
        """Print data summary statistics."""
        print(f"\n{'='*70}")
        print("DATA SUMMARY")
        print(f"{'='*70}\n")

        # Top 10 teams by AdjEM
        print("Top 10 Teams (by AdjEM):")
        top10 = data.nlargest(10, "AdjEM")[["TeamName", "AdjEM", "RankAdjEM", "Wins", "Losses"]]
        for idx, row in top10.iterrows():
            print(
                f"  {row['RankAdjEM']:2.0f}. {row['TeamName']:25s} "
                f"AdjEM: {row['AdjEM']:+6.2f}  "
                f"Record: {row['Wins']:.0f}-{row['Losses']:.0f}"
            )

        # Key metrics
        print(f"\nKey Metrics:")
        print(f"  Highest AdjEM: {data['AdjEM'].max():.2f} ({data.loc[data['AdjEM'].idxmax(), 'TeamName']})")
        print(f"  Lowest AdjEM: {data['AdjEM'].min():.2f} ({data.loc[data['AdjEM'].idxmin(), 'TeamName']})")
        print(f"  Avg Tempo: {data['AdjTempo'].mean():.1f} possessions/game")
        print(f"  Fastest: {data['AdjTempo'].max():.1f} ({data.loc[data['AdjTempo'].idxmax(), 'TeamName']})")
        print(f"  Slowest: {data['AdjTempo'].min():.1f} ({data.loc[data['AdjTempo'].idxmin(), 'TeamName']})")

        # Momentum (if available)
        if "AdjEM_Momentum" in data.columns:
            print(f"\nMomentum (30-day change):")
            print(f"  Biggest Riser: {data.loc[data['AdjEM_Momentum'].idxmax(), 'TeamName']} "
                  f"({data['AdjEM_Momentum'].max():+.2f} AdjEM)")
            print(f"  Biggest Faller: {data.loc[data['AdjEM_Momentum'].idxmin(), 'TeamName']} "
                  f"({data['AdjEM_Momentum'].min():+.2f} AdjEM)")

        print(f"\n{'='*70}\n")

    def run(self, output_path: Path | None = None) -> pd.DataFrame:
        """
        Execute full data collection pipeline.

        Args:
            output_path: Custom output path (optional)

        Returns:
            Collected DataFrame
        """
        # Collect data
        data = self.collect_all_data()

        # Save data
        self.save_data(data, output_path)

        # Print summary
        self.print_summary(data)

        print(f"Data collection complete for {self.season} season!")
        return data


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect daily KenPom data for analysis and predictions"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season ending year (e.g., 2025 for 2024-25 season)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/kenpom_SEASON_TIMESTAMP.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)",
    )

    args = parser.parse_args()

    # Run collector
    collector = DailyDataCollector(season=args.season, output_dir=args.output_dir)
    collector.run(output_path=args.output)


if __name__ == "__main__":
    main()
