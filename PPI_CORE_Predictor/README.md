# PPI Core Predictor

ML-powered predictor for **PPI Core (Finished Goods Less Food & Energy)** month-over-month changes.

Uses 20+ years of historical data from the FRED API to train regression and classification models, producing both point estimates and a full probability distribution across 0.1% bins.

## Target Series

- **WPSFD49116** - PPI: Finished Goods Less Food and Energy

## Features

- 19 economic indicators from FRED (PPI sub-components, energy, labor, money supply, production, prices, trade)
- Feature engineering: MoM changes, lags (1-12 months), rolling statistics, calendar features, YoY
- **Regression**: Gradient Boosting, Random Forest, Ridge + inverse-MAE weighted ensemble
- **Classification**: GB & RF classifiers producing probability distribution over discrete bins
- PDF report generation with charts and probability tables

## Usage

```bash
# From MarketScanner root (with venv activated)
python PPI_CORE_Predictor/main.py --expected 0.2

# Or without --expected (will prompt for input)
python PPI_CORE_Predictor/main.py
```

## Project Structure

```
PPI_CORE_Predictor/
    .env                        # FRED API key
    main.py                     # Pipeline orchestration
    config/
        settings.py             # All configuration parameters
    src/
        data_collector.py       # FRED API data fetching
        data_processor.py       # Feature engineering
        analyzer.py             # Correlation & pattern analysis
        predictor.py            # ML models (regression + classification)
        report_generator.py     # PDF report generation
    data/                       # Data artifacts
    results/                    # Generated PDF reports
```

## Requirements

- Python 3.10+
- fredapi, pandas, numpy, scikit-learn, matplotlib, fpdf2, python-dotenv
