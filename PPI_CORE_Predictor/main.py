"""
PPI Core Predictor - Main Entry Point

Orchestrates the full pipeline:
  1. Collect 20+ years of PPI Core & economic data from FRED
  2. Engineer features (lags, rolling stats, seasonality)
  3. Analyze historical correlations
  4. Train ML models:
     - Regression  (Gradient Boosting, Random Forest, Ridge)
     - Classification (GB / RF classifiers with probability output)
  5. Predict next PPI Core MoM change with probability distribution
  6. Generate PDF report in results/ folder
"""

import sys
from pathlib import Path

# Ensure PPI_CORE_Predictor root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_collector import DataCollector
from src.data_processor import DataProcessor
from src.analyzer import Analyzer
from src.predictor import Predictor
from src.report_generator import generate_report


def run(expected_ppi_core_mom: float | None = None):
    """
    Run the full PPI Core prediction pipeline.

    Parameters
    ----------
    expected_ppi_core_mom : float or None
        The market-consensus expected MoM % change for PPI Core.
        E.g. 0.2 means the market expects +0.2% MoM.
        If None, the user will be prompted to enter it.
    """

    print("=" * 60)
    print("  PPI CORE PREDICTOR  -  ML Pipeline (Regression + Classification)")
    print("=" * 60)

    # -- 1. Collect data ----------------------------
    print("\n>> STEP 1: Collecting data from FRED API...\n")
    collector = DataCollector()
    raw_df = collector.fetch_all()

    # -- 2. Process features ------------------------
    print("\n>> STEP 2: Engineering features...\n")
    processor = DataProcessor(raw_df)
    feature_df = processor.build_features()

    # -- 3. Analyze patterns ------------------------
    print("\n>> STEP 3: Analyzing historical patterns...\n")
    analyzer = Analyzer(feature_df)
    summary = analyzer.summary()

    print(f"  Total months of data : {summary['total_months']}")
    print(f"  Avg MoM change       : {summary['avg_mom_pct']}%")
    print(f"  Std MoM change       : {summary['std_mom_pct']}%")
    print(f"  Positive months      : {summary['positive_months_pct']}%")
    print(f"  Latest PPI Core level: {summary['latest_ppi_core']}")
    print(f"  Latest YoY change    : {summary['latest_yoy']}%")

    analyzer.seasonal_pattern()
    analyzer.year_over_year_trend()
    top_corr = analyzer.correlation_with_target()

    # -- 4. Add expected value ----------------------
    if expected_ppi_core_mom is None:
        try:
            expected_ppi_core_mom = float(input(
                "\nEnter the market-expected PPI Core MoM change (%) "
                "(e.g. 0.2): "
            ))
        except (ValueError, EOFError):
            expected_ppi_core_mom = 0.0
            print("  Using 0.0% as default expected value.")

    feature_df = processor.add_expected_value(feature_df, expected_ppi_core_mom)
    print(f"\n  Expected PPI Core MoM: {expected_ppi_core_mom:+.2f}%")

    # -- 5. Train models ----------------------------
    print("\n>> STEP 4: Training ML models (regression + classification)...\n")
    predictor = Predictor()
    metrics = predictor.train(feature_df)

    # Feature importance
    predictor.feature_importance()

    # -- 6. Predict ---------------------------------
    print("\n>> STEP 5: Generating predictions...\n")
    result = predictor.predict(feature_df)

    reg = result["regression"]
    ci = reg.get("confidence_interval", {})
    prob_dist = result["probability_distribution"]

    print("=" * 60)
    print("  PPI CORE PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n  Market expected MoM  : {expected_ppi_core_mom:+.4f}%")
    print(f"  {'─'*38}")
    for model_name, pred in reg.items():
        if model_name == "confidence_interval":
            continue
        label = model_name.replace("_", " ").title()
        print(f"  {label:<22}: {pred:+.4f}%")
    print(f"  {'─'*38}")
    if ci:
        print(f"  {ci.get('level',0.9)*100:.0f}% Confidence Range : [{ci.get('low',0):+.4f}%, {ci.get('high',0):+.4f}%]")

    # Interpretation
    ensemble = reg.get("ensemble", 0)
    diff = ensemble - expected_ppi_core_mom
    if abs(diff) < 0.05:
        interpretation = "Model closely agrees with market expectation."
    elif diff > 0:
        interpretation = f"Model predicts HIGHER than expected ({diff:+.3f}pp above consensus)."
    else:
        interpretation = f"Model predicts LOWER than expected ({diff:+.3f}pp below consensus)."
    print(f"\n  {interpretation}")

    # Print probability distribution
    print(f"\n  {'='*50}")
    print(f"  PROBABILITY DISTRIBUTION (Classification)")
    print(f"  {'='*50}")
    print(f"  {'PPI Core MoM':<14} {'Probability':>12}")
    print(f"  {'-'*30}")
    for label, prob in prob_dist:
        bar = '#' * int(prob / 2)
        print(f"  {label:<14} {prob:>8.1f}%   {bar}")
    print(f"  {'='*50}")

    # -- 7. Direct Impact Model ─────────────────────
    print("\n>> STEP 6: Training Direct Impact Model (external indicators only)...\n")
    direct_df = processor.filter_direct_impact(feature_df)

    direct_predictor = Predictor()
    direct_metrics = direct_predictor.train(direct_df)
    direct_predictor.feature_importance()

    print("\n>> STEP 7: Direct Impact Model predictions...\n")
    direct_result = direct_predictor.predict(direct_df)

    direct_reg = direct_result["regression"]
    direct_ci = direct_reg.get("confidence_interval", {})
    direct_prob_dist = direct_result["probability_distribution"]

    print("=" * 60)
    print("  DIRECT IMPACT MODEL RESULTS")
    print("  (external indicators only - no PPI Core self-history)")
    print("=" * 60)
    print(f"\n  Market expected MoM  : {expected_ppi_core_mom:+.4f}%")
    print(f"  {'─'*38}")
    for model_name, pred in direct_reg.items():
        if model_name == "confidence_interval":
            continue
        label = model_name.replace("_", " ").title()
        print(f"  {label:<22}: {pred:+.4f}%")
    print(f"  {'─'*38}")
    if direct_ci:
        print(f"  {direct_ci.get('level',0.9)*100:.0f}% Confidence Range : [{direct_ci.get('low',0):+.4f}%, {direct_ci.get('high',0):+.4f}%]")

    direct_ensemble = direct_reg.get("ensemble", 0)
    direct_diff = direct_ensemble - expected_ppi_core_mom
    if abs(direct_diff) < 0.05:
        direct_interp = "Direct impact model closely agrees with market expectation."
    elif direct_diff > 0:
        direct_interp = f"Direct impact model predicts HIGHER ({direct_diff:+.3f}pp above consensus)."
    else:
        direct_interp = f"Direct impact model predicts LOWER ({direct_diff:+.3f}pp below consensus)."
    print(f"\n  {direct_interp}")

    print(f"\n  {'='*50}")
    print(f"  DIRECT IMPACT - PROBABILITY DISTRIBUTION")
    print(f"  {'='*50}")
    print(f"  {'PPI Core MoM':<14} {'Probability':>12}")
    print(f"  {'-'*30}")
    for label, prob in direct_prob_dist:
        bar = '#' * int(prob / 2)
        print(f"  {label:<14} {prob:>8.1f}%   {bar}")
    print(f"  {'='*50}")

    # -- 8. Generate PDF report ---------------------
    print("\n>> STEP 8: Generating PDF report...\n")
    pdf_path = generate_report(
        result=result,
        expected_ppi_core_mom=expected_ppi_core_mom,
        summary=summary,
        metrics=metrics,
        direct_result=direct_result,
        direct_metrics=direct_metrics,
    )

    print(f"\n  Done! Report saved to: {pdf_path}\n")

    return {
        "predictions": result,
        "direct_predictions": direct_result,
        "expected": expected_ppi_core_mom,
        "summary": summary,
        "metrics": metrics,
        "direct_metrics": direct_metrics,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPI Core Predictor")
    parser.add_argument(
        "--expected", type=float, default=None,
        help="Market-expected PPI Core MoM change in %% (e.g. 0.2)",
    )
    args = parser.parse_args()

    run(expected_ppi_core_mom=args.expected)
