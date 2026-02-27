"""
PPI Core Predictor - Data Collector

Fetches historical PPI Core (Less Food & Energy) data and related
economic indicators from the FRED (Federal Reserve Economic Data) API.
Covers 20+ years of monthly data.
"""

import sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fredapi import Fred

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config.settings import (
    FRED_API_KEY,
    HISTORY_YEARS,
    PPI_CORE_TARGET_SERIES,
    FEATURE_SERIES,
)


class DataCollector:
    """Collects PPI Core and related economic data from FRED API."""

    def __init__(self):
        if not FRED_API_KEY:
            raise ValueError(
                "FRED_API_KEY not found. Set it in PPI_CORE_Predictor/.env"
            )
        self.fred = Fred(api_key=FRED_API_KEY)
        self.end_date = datetime.now()
        self.start_date = self.end_date - relativedelta(years=HISTORY_YEARS)

    # ── public ────────────────────────────────────────

    def fetch_ppi_core_target(self) -> pd.Series:
        """Fetch the primary PPI Core series (target variable)."""
        print(f"[DataCollector] Fetching target series: {PPI_CORE_TARGET_SERIES}")
        series = self._fetch_series(PPI_CORE_TARGET_SERIES)
        series.name = "PPI_CORE"
        return series

    def fetch_feature_series(self) -> pd.DataFrame:
        """Fetch all additional economic indicator series."""
        frames: dict[str, pd.Series] = {}
        total = len(FEATURE_SERIES)

        for i, (series_id, label) in enumerate(FEATURE_SERIES.items(), 1):
            print(f"[DataCollector] ({i}/{total}) Fetching {label} ({series_id})")
            try:
                s = self._fetch_series(series_id)
                if s is not None and len(s) > 0:
                    frames[series_id] = s
                else:
                    print(f"  >> No data returned for {series_id}")
            except Exception as e:
                print(f"  >> Failed to fetch {series_id}: {e}")

        if not frames:
            raise RuntimeError("Could not fetch any feature series from FRED.")

        df = pd.DataFrame(frames)
        return df

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch target PPI Core + all feature series, merge into a single
        monthly DataFrame aligned by date index.
        """
        target = self.fetch_ppi_core_target()
        features = self.fetch_feature_series()

        # Merge on date index
        df = features.copy()
        df["PPI_CORE"] = target

        # Resample to month-end frequency to align mixed frequencies
        df = df.resample("ME").last()

        print(f"\n[DataCollector] Collected {len(df)} monthly observations")
        print(f"  Date range : {df.index.min().date()} -> {df.index.max().date()}")
        print(f"  Features   : {len(df.columns) - 1}")
        return df

    # ── private ───────────────────────────────────────

    def _fetch_series(self, series_id: str) -> pd.Series:
        """Fetch a single FRED series within the configured date range."""
        data = self.fred.get_series(
            series_id,
            observation_start=self.start_date,
            observation_end=self.end_date,
        )
        return data


if __name__ == "__main__":
    collector = DataCollector()
    df = collector.fetch_all()
    print(df.tail())
