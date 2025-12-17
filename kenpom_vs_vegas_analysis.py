"""Compare KenPom predictions vs Vegas lines vs Actual results.

This script analyzes KenPom's predictive accuracy compared to Vegas spreads
and totals to identify systematic gaps and opportunities for improvement.
"""

from datetime import date

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


def analyze_kenpom_vs_vegas(
    game_date: str | date,
    actual_results: dict[str, tuple[int, int]],
    vegas_lines: dict[str, dict[str, float]],
) -> None:
    """Complete analysis of KenPom vs Vegas vs Actual results.

    Args:
        game_date: Date in YYYY-MM-DD format or date object.
        actual_results: Dictionary mapping "Visitor @ Home" to (visitor_score,
            home_score) tuple.
        vegas_lines: Dictionary mapping "Visitor @ Home" to dict with:
            - 'spread': Home team spread (negative = favorite)
            - 'total': Over/under total
            - 'home_ml': Home team moneyline
            - 'visitor_ml': Visitor team moneyline
    """
    print(f"\n{'='*90}")
    print(f"KenPom vs Vegas Baseline Analysis - {game_date}")
    print(f"{'='*90}\n")

    # Initialize API client
    try:
        api = KenPomAPI()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Fetch KenPom predictions
    print("Fetching KenPom FanMatch predictions...\n")
    try:
        response = api.get_fanmatch(game_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Convert to DataFrame
    df = response.to_dataframe()

    if df.empty:
        print(f"No games found for {game_date}")
        return

    # Build comparison dataset
    comparisons = []

    for _, game in df.iterrows():
        visitor = game["Visitor"]
        home = game["Home"]
        game_key = f"{visitor} @ {home}"

        # Check if we have actual results and Vegas lines
        if game_key not in actual_results or game_key not in vegas_lines:
            continue

        visitor_actual, home_actual = actual_results[game_key]
        vegas = vegas_lines[game_key]

        # KenPom predictions
        kenpom_visitor_pred = game["VisitorPred"]
        kenpom_home_pred = game["HomePred"]
        kenpom_margin = kenpom_home_pred - kenpom_visitor_pred
        kenpom_total = kenpom_visitor_pred + kenpom_home_pred
        kenpom_home_wp = game["HomeWP"]

        # Vegas lines
        vegas_spread = vegas["spread"]  # Negative = home favored
        vegas_total = vegas["total"]

        # Convert Vegas spread to predicted margin
        # Spread of -14.5 means home favored by 14.5, so predicted margin = +14.5
        vegas_predicted_margin = -vegas_spread

        # Actual results
        actual_margin = home_actual - visitor_actual
        actual_total = visitor_actual + home_actual

        # Calculate errors
        kenpom_margin_error = abs(kenpom_margin - actual_margin)
        vegas_margin_error = abs(vegas_predicted_margin - actual_margin)

        kenpom_total_error = abs(kenpom_total - actual_total)
        vegas_total_error = abs(vegas_total - actual_total)

        # ATS analysis
        # Did KenPom beat the spread?
        kenpom_ats_result = None
        if kenpom_margin > vegas_predicted_margin:
            # KenPom more bullish on home team
            kenpom_ats_result = "HOME" if actual_margin > vegas_predicted_margin else "VISITOR"
        elif kenpom_margin < vegas_predicted_margin:
            # KenPom more bearish on home team
            kenpom_ats_result = "VISITOR" if actual_margin < vegas_predicted_margin else "HOME"
        else:
            kenpom_ats_result = "PUSH"

        # Over/Under analysis
        kenpom_ou_result = "OVER" if actual_total > kenpom_total else "UNDER"
        vegas_ou_result = "OVER" if actual_total > vegas_total else "UNDER"

        # Who was more accurate?
        margin_more_accurate = (
            "KenPom" if kenpom_margin_error < vegas_margin_error else "Vegas"
        )
        total_more_accurate = (
            "KenPom" if kenpom_total_error < vegas_total_error else "Vegas"
        )

        # Calculate disagreement
        margin_disagreement = abs(kenpom_margin - vegas_predicted_margin)
        total_disagreement = abs(kenpom_total - vegas_total)

        comparisons.append(
            {
                "game": game_key,
                "visitor": visitor,
                "home": home,
                # KenPom
                "kenpom_margin": kenpom_margin,
                "kenpom_total": kenpom_total,
                "kenpom_home_wp": kenpom_home_wp,
                # Vegas
                "vegas_spread": vegas_spread,
                "vegas_total": vegas_total,
                # Actual
                "actual_margin": actual_margin,
                "actual_total": actual_total,
                # Errors
                "kenpom_margin_error": kenpom_margin_error,
                "vegas_margin_error": vegas_margin_error,
                "kenpom_total_error": kenpom_total_error,
                "vegas_total_error": vegas_total_error,
                # Analysis
                "margin_disagreement": margin_disagreement,
                "total_disagreement": total_disagreement,
                "margin_more_accurate": margin_more_accurate,
                "total_more_accurate": total_more_accurate,
                "kenpom_ats_result": kenpom_ats_result,
                "kenpom_ou_result": kenpom_ou_result,
                "vegas_ou_result": vegas_ou_result,
            }
        )

    if not comparisons:
        print("No matching games found for comparison.")
        api.close()
        return

    comp_df = pd.DataFrame(comparisons)

    # =========================================================================
    # SECTION 1: GAME-BY-GAME COMPARISON
    # =========================================================================
    print("=" * 90)
    print("GAME-BY-GAME COMPARISON: KenPom vs Vegas vs Actual")
    print("=" * 90 + "\n")

    for _, row in comp_df.iterrows():
        print(f"\n{row['game']}")
        # Reconstruct actual scores from margin and total
        # actual_total = visitor + home
        # actual_margin = home - visitor
        # Solve: visitor = (total - margin) / 2, home = (total + margin) / 2
        actual_visitor_score = (row['actual_total'] - row['actual_margin']) / 2
        actual_home_score = (row['actual_total'] + row['actual_margin']) / 2
        print(f"  Actual Result: {row['visitor']} {int(actual_visitor_score)}, "
              f"{row['home']} {int(actual_home_score)}")
        print(f"    Margin: {row['home']} by {row['actual_margin']:.1f}")
        print(f"    Total: {row['actual_total']:.0f} points")
        print()

        print(f"  KenPom Prediction:")
        print(f"    Margin: {row['home']} by {row['kenpom_margin']:.1f} "
              f"(Error: {row['kenpom_margin_error']:.1f})")
        print(f"    Total: {row['kenpom_total']:.1f} "
              f"(Error: {row['kenpom_total_error']:.1f})")
        print(f"    Home WP: {row['kenpom_home_wp']:.1f}%")
        print()

        print(f"  Vegas Line:")
        print(f"    Spread: {row['home']} {row['vegas_spread']:+.1f} "
              f"(Error: {row['vegas_margin_error']:.1f})")
        print(f"    Total: {row['vegas_total']:.1f} "
              f"(Error: {row['vegas_total_error']:.1f})")
        print()

        print(f"  Disagreement:")
        kenpom_sentiment = "more bullish" if row['kenpom_margin'] > row['vegas_spread'] else "more bearish"
        scoring_sentiment = "higher" if row['kenpom_total'] > row['vegas_total'] else "lower"
        print(f"    Margin: {row['margin_disagreement']:.1f} pts "
              f"(KenPom {kenpom_sentiment} on home)")
        print(f"    Total: {row['total_disagreement']:.1f} pts "
              f"(KenPom predicted {scoring_sentiment} scoring)")
        print()

        print(f"  More Accurate:")
        print(f"    Margin: {row['margin_more_accurate']}")
        print(f"    Total: {row['total_more_accurate']}")

        # Highlight significant disagreements
        if row["margin_disagreement"] >= 5.0:
            print(f"\n  *** MAJOR DISAGREEMENT: {row['margin_disagreement']:.1f} "
                  f"point difference ***")
            if row["margin_more_accurate"] == "KenPom":
                print(f"      -> KenPom was RIGHT to disagree!")
            else:
                print(f"      -> Vegas was RIGHT, KenPom missed something")

    # =========================================================================
    # SECTION 2: AGGREGATE ACCURACY COMPARISON
    # =========================================================================
    print("\n\n" + "=" * 90)
    print("AGGREGATE ACCURACY: KenPom vs Vegas")
    print("=" * 90 + "\n")

    print("MARGIN PREDICTION ACCURACY")
    print("-" * 90)
    print(f"  KenPom Average Error:  {comp_df['kenpom_margin_error'].mean():.2f} pts")
    print(f"  Vegas Average Error:   {comp_df['vegas_margin_error'].mean():.2f} pts")
    print(f"  KenPom Median Error:   {comp_df['kenpom_margin_error'].median():.2f} pts")
    print(f"  Vegas Median Error:    {comp_df['vegas_margin_error'].median():.2f} pts")
    print()

    kenpom_margin_wins = (
        comp_df["margin_more_accurate"] == "KenPom"
    ).sum()
    print(f"  KenPom More Accurate: {kenpom_margin_wins}/{len(comp_df)} games "
          f"({kenpom_margin_wins/len(comp_df)*100:.1f}%)")
    print(f"  Vegas More Accurate:  {len(comp_df) - kenpom_margin_wins}/{len(comp_df)} "
          f"games ({(len(comp_df) - kenpom_margin_wins)/len(comp_df)*100:.1f}%)")
    print()

    print("\nTOTAL POINTS PREDICTION ACCURACY")
    print("-" * 90)
    print(f"  KenPom Average Error:  {comp_df['kenpom_total_error'].mean():.2f} pts")
    print(f"  Vegas Average Error:   {comp_df['vegas_total_error'].mean():.2f} pts")
    print(f"  KenPom Median Error:   {comp_df['kenpom_total_error'].median():.2f} pts")
    print(f"  Vegas Median Error:    {comp_df['vegas_total_error'].median():.2f} pts")
    print()

    kenpom_total_wins = (comp_df["total_more_accurate"] == "KenPom").sum()
    print(f"  KenPom More Accurate: {kenpom_total_wins}/{len(comp_df)} games "
          f"({kenpom_total_wins/len(comp_df)*100:.1f}%)")
    print(f"  Vegas More Accurate:  {len(comp_df) - kenpom_total_wins}/{len(comp_df)} "
          f"games ({(len(comp_df) - kenpom_total_wins)/len(comp_df)*100:.1f}%)")

    # =========================================================================
    # SECTION 3: BETTING VALUE ANALYSIS
    # =========================================================================
    print("\n\n" + "=" * 90)
    print("BETTING VALUE ANALYSIS: Where KenPom Disagreed with Vegas")
    print("=" * 90 + "\n")

    # Games with significant disagreement
    significant_disagreement = comp_df[comp_df["margin_disagreement"] >= 3.0]

    if len(significant_disagreement) > 0:
        print(f"Games with 3+ point disagreement: {len(significant_disagreement)}\n")

        for _, row in significant_disagreement.iterrows():
            print(f"{row['game']}")
            print(f"  KenPom: {row['home']} by {row['kenpom_margin']:.1f}")
            print(f"  Vegas:  {row['home']} {row['vegas_spread']:+.1f}")
            print(f"  Disagreement: {row['margin_disagreement']:.1f} pts")
            print(f"  Actual: {row['home']} by {row['actual_margin']:.1f}")
            print(f"  Winner: {row['margin_more_accurate']} was more accurate")

            # Would betting with KenPom have been profitable?
            if row["kenpom_margin"] > row["vegas_spread"]:
                # KenPom more bullish on home
                home_ats = row["actual_margin"] - row["vegas_spread"]
                if home_ats > 0:
                    print(f"  Result: HOME covered spread by {home_ats:.1f} pts -> "
                          f"KenPom bet WINS")
                else:
                    print(f"  Result: HOME missed spread by {abs(home_ats):.1f} pts -> "
                          f"KenPom bet LOSES")
            else:
                # KenPom more bearish on home
                visitor_ats = -row["actual_margin"] + row["vegas_spread"]
                if visitor_ats > 0:
                    print(f"  Result: VISITOR covered spread by {visitor_ats:.1f} pts -> "
                          f"KenPom bet WINS")
                else:
                    print(f"  Result: VISITOR missed spread by {abs(visitor_ats):.1f} "
                          f"pts -> KenPom bet LOSES")

            print()

    # =========================================================================
    # SECTION 4: ERROR PATTERN ANALYSIS
    # =========================================================================
    print("\n" + "=" * 90)
    print("ERROR PATTERN ANALYSIS: What Did KenPom Miss?")
    print("=" * 90 + "\n")

    # Games where KenPom had biggest errors
    print("BIGGEST KENPOM MARGIN ERRORS (Top 3):")
    print("-" * 90)
    biggest_errors = comp_df.nlargest(3, "kenpom_margin_error")

    for _, row in biggest_errors.iterrows():
        print(f"\n{row['game']} - Error: {row['kenpom_margin_error']:.1f} pts")
        print(f"  KenPom predicted: {row['home']} by {row['kenpom_margin']:.1f}")
        print(f"  Vegas line: {row['home']} {row['vegas_spread']:+.1f}")
        print(f"  Actual: {row['home']} by {row['actual_margin']:.1f}")
        vegas_advantage = row['kenpom_margin_error'] - row['vegas_margin_error']
        print(f"  Vegas was closer by {vegas_advantage:.1f} pts")
        print()
        print(f"  Potential Factors KenPom Missed:")
        print(f"    - Check for injuries not in efficiency ratings")
        print(f"    - Look for motivational/situational factors")
        print(f"    - Analyze lineup changes or rotation patterns")
        print(f"    - Consider rest/schedule advantages")
        print(f"    - Review recent form vs season-long efficiency")

    # =========================================================================
    # SECTION 5: SYSTEMATIC BIASES
    # =========================================================================
    print("\n\n" + "=" * 90)
    print("SYSTEMATIC BIAS ANALYSIS")
    print("=" * 90 + "\n")

    # Does KenPom consistently over/underpredict margins?
    avg_kenpom_margin = comp_df["kenpom_margin"].mean()
    avg_actual_margin = comp_df["actual_margin"].mean()
    avg_vegas_spread = comp_df["vegas_spread"].mean()

    print("MARGIN PREDICTION BIAS:")
    print("-" * 90)
    print(f"  Average KenPom Margin: {avg_kenpom_margin:+.2f}")
    print(f"  Average Vegas Spread:  {avg_vegas_spread:+.2f}")
    print(f"  Average Actual Margin: {avg_actual_margin:+.2f}")
    print()

    if avg_kenpom_margin > avg_actual_margin:
        print(f"  -> KenPom tends to OVERPREDICT home margins by "
              f"{avg_kenpom_margin - avg_actual_margin:.2f} pts")
    else:
        print(f"  -> KenPom tends to UNDERPREDICT home margins by "
              f"{avg_actual_margin - avg_kenpom_margin:.2f} pts")

    print()

    # Total points bias
    avg_kenpom_total = comp_df["kenpom_total"].mean()
    avg_vegas_total = comp_df["vegas_total"].mean()
    avg_actual_total = comp_df["actual_total"].mean()

    print("\nTOTAL POINTS BIAS:")
    print("-" * 90)
    print(f"  Average KenPom Total: {avg_kenpom_total:.1f}")
    print(f"  Average Vegas Total:  {avg_vegas_total:.1f}")
    print(f"  Average Actual Total: {avg_actual_total:.1f}")
    print()

    if avg_kenpom_total < avg_actual_total:
        print(f"  -> KenPom UNDERPREDICTS scoring by "
              f"{avg_actual_total - avg_kenpom_total:.1f} pts")
    else:
        print(f"  -> KenPom OVERPREDICTS scoring by "
              f"{avg_kenpom_total - avg_actual_total:.1f} pts")

    print()

    # =========================================================================
    # SECTION 6: ACTIONABLE INSIGHTS
    # =========================================================================
    print("\n" + "=" * 90)
    print("ACTIONABLE INSIGHTS & OPPORTUNITIES")
    print("=" * 90 + "\n")

    print("KEY FINDINGS:")
    print("-" * 90)

    # 1. Overall accuracy comparison
    if comp_df["kenpom_margin_error"].mean() < comp_df["vegas_margin_error"].mean():
        print("1. KenPom margins are MORE accurate than Vegas on average")
        print("   -> Opportunity: Use KenPom as primary prediction model")
    else:
        print("1. Vegas margins are MORE accurate than KenPom on average")
        print("   -> Opportunity: Identify what Vegas knows that KenPom doesn't")

    print()

    # 2. Total points opportunity
    if comp_df["kenpom_total_error"].mean() > comp_df["vegas_total_error"].mean():
        print("2. KenPom totals are LESS accurate than Vegas")
        print("   -> Opportunity: Build enhanced total points model")
        improvement_needed = comp_df['kenpom_total_error'].mean() - comp_df['vegas_total_error'].mean()
        print(f"   -> Average improvement needed: {improvement_needed:.1f} pts")
    else:
        print("2. KenPom totals are comparable to Vegas")
        print("   -> Opportunity: Minor refinements could create edge")

    print()

    # 3. Disagreement opportunities
    avg_disagreement = comp_df["margin_disagreement"].mean()
    print(f"3. Average KenPom-Vegas disagreement: {avg_disagreement:.1f} pts")
    if avg_disagreement >= 3.0:
        print("   -> HIGH disagreement suggests betting opportunities")
        print("   -> Focus on games with 5+ point disagreement")
    else:
        print("   -> MODERATE disagreement")
        print("   -> Look for situational factors causing disagreement")

    print()

    # 4. Where to improve
    print("4. Priority Improvement Areas:")
    if comp_df["kenpom_total_error"].mean() > 10.0:
        print("   HIGH: Total points prediction (large errors)")
    if len(biggest_errors) > 0:
        print("   MEDIUM: Close game prediction (Louisville-Tennessee scenario)")
    if abs(avg_kenpom_total - avg_actual_total) > 5.0:
        print("   MEDIUM: Systematic scoring bias correction")

    print("\n" + "=" * 90)

    api.close()


if __name__ == "__main__":
    # Actual results from December 16, 2025
    actual_results = {
        "Abilene Christian @ Arizona": (62, 96),
        "Lipscomb @ Duke": (73, 97),
        "Butler @ Connecticut": (60, 79),
        "Toledo @ Michigan St.": (69, 92),
        "Pacific @ BYU": (57, 93),
        "Louisville @ Tennessee": (62, 83),
        "East Tennessee St. @ North Carolina": (58, 77),
        "Queens @ Arkansas": (80, 108),
        "Towson @ Kansas": (49, 73),
        "Northern Colorado @ Texas Tech": (90, 101),
        "DePaul @ St. John's": (66, 79),
    }

    # Vegas lines for December 16, 2025
    # Format: 'spread' is home team spread (negative = home favored)
    # We need to populate this with actual Vegas lines
    vegas_lines = {
        "Abilene Christian @ Arizona": {
            "spread": -33.5,  # Arizona -33.5
            "total": 155.5,
            "home_ml": -10000,
            "visitor_ml": 3500,
        },
        "Lipscomb @ Duke": {
            "spread": -28.5,  # Duke -28.5
            "total": 153.5,
            "home_ml": -8000,
            "visitor_ml": 2800,
        },
        "Butler @ Connecticut": {
            "spread": -15.5,  # UConn -15.5
            "total": 148.5,
            "home_ml": -1400,
            "visitor_ml": 850,
        },
        "Toledo @ Michigan St.": {
            "spread": -20.5,  # MSU -20.5
            "total": 151.5,
            "home_ml": -2500,
            "visitor_ml": 1300,
        },
        "Pacific @ BYU": {
            "spread": -21.5,  # BYU -21.5
            "total": 152.5,
            "home_ml": -2200,
            "visitor_ml": 1200,
        },
        "Louisville @ Tennessee": {
            "spread": -4.5,  # Tennessee -4.5
            "total": 156.5,
            "home_ml": -200,
            "visitor_ml": 170,
        },
        "East Tennessee St. @ North Carolina": {
            "spread": -17.5,  # UNC -17.5
            "total": 152.5,
            "home_ml": -2000,
            "visitor_ml": 1100,
        },
        "Queens @ Arkansas": {
            "spread": -25.5,  # Arkansas -25.5
            "total": 165.5,
            "home_ml": -5000,
            "visitor_ml": 2000,
        },
        "Towson @ Kansas": {
            "spread": -21.5,  # Kansas -21.5
            "total": 140.5,
            "home_ml": -3500,
            "visitor_ml": 1600,
        },
        "Northern Colorado @ Texas Tech": {
            "spread": -23.5,  # Texas Tech -23.5
            "total": 155.5,
            "home_ml": -3000,
            "visitor_ml": 1500,
        },
        "DePaul @ St. John's": {
            "spread": -14.5,  # St. John's -14.5
            "total": 148.5,
            "home_ml": -1000,
            "visitor_ml": 700,
        },
    }

    # Note: These Vegas lines are ESTIMATES for demonstration purposes.
    # For production use, fetch actual historical lines from:
    # - The Odds API (odds-api.com)
    # - Sports betting sites (DraftKings, FanDuel, BetMGM)
    # - Odds aggregators (Covers.com, ActionNetwork)

    print("\n" + "=" * 90)
    print("NOTE: Vegas lines in this analysis are estimates for demonstration.")
    print("For production analysis, fetch actual historical closing lines.")
    print("=" * 90)

    analyze_kenpom_vs_vegas("2025-12-16", actual_results, vegas_lines)
