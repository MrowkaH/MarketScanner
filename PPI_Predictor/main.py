"""
PPI Predictor - Main Entry Point

Orchestrates the full pipeline:
  1. Collect 20+ years of PPI & economic data from FRED
  2. Engineer features (lags, rolling stats, seasonality)
  3. Analyze historical correlations
  4. Train ML models:
     - Regression  (Gradient Boosting, Random Forest, Ridge)
     - Classification (GB / RF classifiers with probability output)
  5. Predict next PPI MoM change with probability distribution
  6. Generate PDF report in results/ folder
"""

import sys
from pathlib import Path

# Ensure PPI_Predictor root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_collector import DataCollector
from src.data_processor import DataProcessor
from src.analyzer import Analyzer
from src.predictor import Predictor
from src.report_generator import generate_report


def run(expected_ppi_mom: float | None = None):
    """
    Run the full PPI prediction pipeline.

    Parameters
    ----------
    expected_ppi_mom : float or None
        The market-consensus expected MoM % change for PPI.
        E.g. 0.3 means the market expects +0.3% MoM.
        If None, the user will be prompted to enter it.
    """

    print("=" * 60)
    print("  PPI PREDICTOR  -  ML Pipeline (Regression + Classification)")
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
    print(f"  Latest PPI level     : {summary['latest_ppi']}")
    print(f"  Latest YoY change    : {summary['latest_yoy']}%")

    analyzer.seasonal_pattern()
    analyzer.year_over_year_trend()
    top_corr = analyzer.correlation_with_target()

    # -- 4. Add expected value ----------------------
    if expected_ppi_mom is None:
        try:
            expected_ppi_mom = float(input(
                "\nEnter the market-expected PPI MoM change (%) "
                "(e.g. 0.3): "
            ))
        except (ValueError, EOFError):
            expected_ppi_mom = 0.0
            print("  Using 0.0% as default expected value.")

    feature_df = processor.add_expected_value(feature_df, expected_ppi_mom)
    print(f"\n  Expected PPI MoM: {expected_ppi_mom:+.2f}%")

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
    print("  PPI PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n  Market expected MoM  : {expected_ppi_mom:+.4f}%")
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
    diff = ensemble - expected_ppi_mom
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
    print(f"  {'PPI MoM':<12} {'Probability':>12}")
    print(f"  {'-'*30}")
    for label, prob in prob_dist:
        bar = '#' * int(prob / 2)
        print(f"  {label:<12} {prob:>8.1f}%   {bar}")
    print(f"  {'='*50}")

    # -- 7. Generate PDF report ---------------------
    print("\n>> STEP 6: Generating PDF report...\n")
    pdf_path = generate_report(
        result=result,
        expected_ppi_mom=expected_ppi_mom,
        summary=summary,
        metrics=metrics,
    )

    print(f"\n  Done! Report saved to: {pdf_path}\n")

    return {
        "predictions": result,
        "expected": expected_ppi_mom,
        "summary": summary,
        "metrics": metrics,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPI Predictor")
    parser.add_argument(
        "--expected", type=float, default=None,
        help="Market-expected PPI MoM change in %% (e.g. 0.3)",
    )
    args = parser.parse_args()

    run(expected_ppi_mom=args.expected)
