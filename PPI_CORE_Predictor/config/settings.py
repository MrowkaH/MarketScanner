"""
PPI Core Predictor - Configuration Settings

Contains all configuration parameters for data collection,
feature engineering, and ML model training.

Target: PPI Finished Goods Less Food & Energy (Core PPI)
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from PPI_CORE_Predictor directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ──────────────────────────────────────────────
# FRED API
# ──────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ──────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────
# Number of years of historical data to fetch
HISTORY_YEARS = 20

# Primary PPI Core series (target variable)
PPI_CORE_TARGET_SERIES = "WPSFD49116"  # PPI: Finished Goods Less Food & Energy

# Additional economic indicators used as features
FEATURE_SERIES = {
    # PPI sub-components
    "PPIACO":   "PPI: All Commodities",
    "PPIFGS":   "PPI: Finished Goods",
    "PPIFCG":   "PPI: Finished Consumer Goods",
    "PPIITM":   "PPI: Intermediate Materials",
    "PPICPE":   "PPI: Capital Equipment",

    # Energy & commodities
    "DCOILWTICO": "Crude Oil WTI",
    "PPIENG":     "PPI: Energy",

    # Labor market
    "UNRATE":   "Unemployment Rate",
    "PAYEMS":   "Nonfarm Payrolls",
    "CES0500000003": "Average Hourly Earnings",

    # Money & credit
    "M2SL":     "M2 Money Supply",

    # Production & output
    "INDPRO":   "Industrial Production Index",
    "TCU":      "Capacity Utilization",

    # Prices & inflation
    "CPIAUCSL": "CPI All Urban Consumers",
    "CPILFESL": "CPI Core (Less Food & Energy)",
    "PCEPILFE": "PCE Core Price Index",

    # Trade
    "IR":       "Import Price Index",
    "IQ":       "Export Price Index",

    # Business activity
    "MANEMP":   "Manufacturing Employees",
}

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
# Lag periods (in months) to create for each feature
LAG_PERIODS = [1, 2, 3, 6, 12]

# Rolling window sizes for moving averages & std
ROLLING_WINDOWS = [3, 6, 12]

# ──────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────
# Train/test split ratio
TEST_SIZE = 0.15

# Random state for reproducibility
RANDOM_STATE = 42

# Cross-validation folds (TimeSeriesSplit)
CV_FOLDS = 5

# Models to evaluate
MODELS = {
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
    },
    "ridge": {
        "alpha": 1.0,
    },
}

# Primary model used for final predictions
PRIMARY_MODEL = "gradient_boosting"

# ──────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────
# Confidence interval percentile (for prediction range)
CONFIDENCE_LEVEL = 0.90

# Decimal places for output display
DISPLAY_DECIMALS = 4

# ──────────────────────────────────────────────
# CLASSIFICATION BINS
# ──────────────────────────────────────────────
# Bin step size for discretized PPI Core MoM (in percentage points)
BIN_STEP = 0.1

# Range of bins: (min, max) in percentage points
# Values outside this range go to "<min" and ">max" bins
BIN_RANGE = (-1.0, 1.0)
