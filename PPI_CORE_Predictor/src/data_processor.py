"""
PPI Core Predictor - Data Processor

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
        raw_df : DataFrame with DatetimeIndex, must contain 'PPI_CORE' column.
        """
        if "PPI_CORE" not in raw_df.columns:
            raise ValueError("DataFrame must contain a 'PPI_CORE' column (target).")
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

        # 3) Rolling statistics on PPI_CORE & key columns
        df = self._add_rolling_stats(df)

        # 4) Calendar / seasonal features
        df = self._add_calendar_features(df)

        # 5) Year-over-year change for PPI Core (target-adjacent)
        df = self._add_yoy(df)

        # 6) Target: next month PPI Core MoM change (what we predict)
        df["target_ppi_core_mom"] = df["PPI_CORE"].pct_change().shift(-1) * 100

        # Drop rows with NaN caused by lags / rolling
        df = df.dropna(subset=["target_ppi_core_mom"])

        print(f"[DataProcessor] Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
        return df

    def add_expected_value(self, df: pd.DataFrame, expected_ppi_core_mom: float) -> pd.DataFrame:
        """
        Add a column with the market-expected PPI Core MoM change
        for the prediction month. This is set on the *last* row
        (the row we want to predict).
        """
        df = df.copy()
        df["expected_ppi_core_mom"] = np.nan
        df.loc[df.index[-1], "expected_ppi_core_mom"] = expected_ppi_core_mom
        # Back-fill with 0 for training rows (no expectation available)
        df["expected_ppi_core_mom"] = df["expected_ppi_core_mom"].fillna(0.0)
        return df

    def filter_direct_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter feature DataFrame to keep only external economic indicators.
        Removes PPI Core's own historical values (level, MoM, lags, rolling stats, YoY)
        to focus on causal/leading indicators only.

        The target column (target_ppi_core_mom) is preserved.
        """
        exclude_exact = {"PPI_CORE", "PPI_CORE_mom", "PPI_CORE_yoy"}
        drop_cols = set()
        for col in df.columns:
            if col in exclude_exact:
                drop_cols.add(col)
            elif col.startswith("PPI_CORE_mom_"):  # PPI_CORE_mom_lag*, PPI_CORE_mom_ma*, PPI_CORE_mom_std*
                drop_cols.add(col)
        # Never drop target
        drop_cols.discard("target_ppi_core_mom")

        result = df.drop(columns=[c for c in drop_cols if c in df.columns])
        removed = len(drop_cols)
        print(f"[DataProcessor] Direct impact features: {result.shape[0]} rows x {result.shape[1]} columns")
        print(f"  (removed {removed} PPI Core self-referencing columns)")
        return result

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
        ppi_mom = "PPI_CORE_mom"
        if ppi_mom not in df.columns:
            return df
        new_cols = {}
        for window in ROLLING_WINDOWS:
            new_cols[f"PPI_CORE_mom_ma{window}"] = df[ppi_mom].rolling(window).mean()
            new_cols[f"PPI_CORE_mom_std{window}"] = df[ppi_mom].rolling(window).std()
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
        new_cols = {"PPI_CORE_yoy": df["PPI_CORE"].pct_change(12) * 100}
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
