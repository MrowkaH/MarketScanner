"""
PPI Predictor - Data Processor

Transforms raw economic data into ML-ready features:
  - Month-over-month & year-over-year percentage changes
  - Lagged values
  - Rolling statistics (mean, std)
  - Seasonal / calendar features
  - Expected value integration
"""

import numpy as np
import pandas as pd

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config.settings import LAG_PERIODS, ROLLING_WINDOWS


class DataProcessor:
    """Processes raw FRED data into feature matrix for ML models."""

    def __init__(self, raw_df: pd.DataFrame):
        """
        Parameters
        ----------
        raw_df : DataFrame with DatetimeIndex, must contain 'PPI' column.
        """
        if "PPI" not in raw_df.columns:
            raise ValueError("DataFrame must contain a 'PPI' column (target).")
        self.raw = raw_df.copy().sort_index()

    # ── public ────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        """
        Build the full feature matrix from raw data.
        Returns a DataFrame where every row is one month and columns
        are engineered features + target.
        """
        df = self.raw.copy()

        # 1) Percentage changes for every column
        df = self._add_pct_changes(df)

        # 2) Lag features for every column
        df = self._add_lags(df)

        # 3) Rolling statistics on PPI & key columns
        df = self._add_rolling_stats(df)

        # 4) Calendar / seasonal features
        df = self._add_calendar_features(df)

        # 5) Year-over-year change for PPI (target-adjacent)
        df = self._add_yoy(df)

        # 6) Target: next month PPI MoM change (what we predict)
        df["target_ppi_mom"] = df["PPI"].pct_change().shift(-1) * 100

        # Drop rows with NaN caused by lags / rolling
        df = df.dropna(subset=["target_ppi_mom"])

        print(f"[DataProcessor] Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def add_expected_value(self, df: pd.DataFrame, expected_ppi_mom: float) -> pd.DataFrame:
        """
        Add a column with the market-expected PPI MoM change
        for the prediction month. This is set on the *last* row
        (the row we want to predict).
        """
        df = df.copy()
        df["expected_ppi_mom"] = np.nan
        df.loc[df.index[-1], "expected_ppi_mom"] = expected_ppi_mom
        # Back-fill with 0 for training rows (no expectation available)
        df["expected_ppi_mom"] = df["expected_ppi_mom"].fillna(0.0)
        return df

    # ── private helpers ───────────────────────────────

    def _add_pct_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {f"{col}_mom": df[col].pct_change() * 100 for col in df.columns}
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        key_cols = [c for c in df.columns if c.endswith("_mom")]
        new_cols = {}
        for col in key_cols:
            for lag in LAG_PERIODS:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        ppi_mom = "PPI_mom"
        if ppi_mom not in df.columns:
            return df
        new_cols = {}
        for window in ROLLING_WINDOWS:
            new_cols[f"PPI_mom_ma{window}"] = df[ppi_mom].rolling(window).mean()
            new_cols[f"PPI_mom_std{window}"] = df[ppi_mom].rolling(window).std()
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {
            "month": df.index.month,
            "quarter": df.index.quarter,
            "month_sin": np.sin(2 * np.pi * df.index.month / 12),
            "month_cos": np.cos(2 * np.pi * df.index.month / 12),
        }
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_yoy(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {"PPI_yoy": df["PPI"].pct_change(12) * 100}
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
