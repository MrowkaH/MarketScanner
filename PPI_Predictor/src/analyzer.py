"""
PPI Predictor - Analyzer

Performs exploratory / correlation analysis on PPI features
to identify the strongest predictors and historical patterns.
"""

import numpy as np
import pandas as pd


class Analyzer:
    """Analyzes historical PPI data for correlations and patterns."""

    def __init__(self, feature_df: pd.DataFrame):
        """
        Parameters
        ----------
        feature_df : DataFrame produced by DataProcessor.build_features()
        """
        self.df = feature_df.copy()

    # ── public ────────────────────────────────────────

    def correlation_with_target(self, top_n: int = 25) -> pd.Series:
        """
        Compute Pearson correlation of every feature column
        with target_ppi_mom. Returns the top_n strongest (absolute).
        """
        numeric = self.df.select_dtypes(include=[np.number])
        corr = numeric.corr()["target_ppi_mom"].drop("target_ppi_mom", errors="ignore")
        ranked = corr.abs().sort_values(ascending=False).head(top_n)
        result = corr.loc[ranked.index]

        print(f"\n{'─'*55}")
        print(f" Top {top_n} features correlated with target PPI MoM")
        print(f"{'─'*55}")
        for feat, val in result.items():
            bar = "█" * int(abs(val) * 40)
            sign = "+" if val > 0 else "-"
            print(f"  {sign} {val:+.4f}  {bar}  {feat}")
        print(f"{'─'*55}\n")
        return result

    def seasonal_pattern(self) -> pd.DataFrame:
        """
        Show average PPI MoM change by calendar month
        over the entire history.
        """
        if "month" not in self.df.columns or "PPI_mom" not in self.df.columns:
            print("[Analyzer] Cannot compute seasonal pattern – missing columns.")
            return pd.DataFrame()

        seasonal = (
            self.df.groupby("month")["PPI_mom"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={
                "mean": "avg_mom_%",
                "median": "median_mom_%",
                "std": "volatility",
                "count": "observations",
            })
        )

        print(f"\n{'─'*60}")
        print(" PPI MoM (%) – Seasonal Pattern by Month")
        print(f"{'─'*60}")
        month_names = [
            "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        for month_num, row in seasonal.iterrows():
            bar_len = int(abs(row["avg_mom_%"]) * 30)
            direction = "▲" if row["avg_mom_%"] >= 0 else "▼"
            print(
                f"  {month_names[month_num]:>3}  {direction} {row['avg_mom_%']:+.3f}%  "
                f"(median {row['median_mom_%']:+.3f}%, vol {row['volatility']:.3f}, "
                f"n={int(row['observations'])})"
            )
        print(f"{'─'*60}\n")
        return seasonal

    def year_over_year_trend(self) -> pd.Series:
        """Return the latest 12 months of PPI YoY changes."""
        if "PPI_yoy" not in self.df.columns:
            return pd.Series(dtype=float)
        recent = self.df["PPI_yoy"].dropna().tail(12)

        print(f"\n{'─'*45}")
        print(" PPI Year-over-Year (%) – Last 12 Months")
        print(f"{'─'*45}")
        for date, val in recent.items():
            bar = "█" * int(abs(val) * 3)
            print(f"  {date.strftime('%Y-%m')}  {val:+.2f}%  {bar}")
        print(f"{'─'*45}\n")
        return recent

    def summary(self) -> dict:
        """Return a compact summary dict of key statistics."""
        ppi_mom = self.df.get("PPI_mom", pd.Series(dtype=float)).dropna()
        return {
            "total_months": len(self.df),
            "avg_mom_pct": round(ppi_mom.mean(), 4) if len(ppi_mom) else None,
            "std_mom_pct": round(ppi_mom.std(), 4) if len(ppi_mom) else None,
            "positive_months_pct": round((ppi_mom > 0).mean() * 100, 1) if len(ppi_mom) else None,
            "latest_ppi": round(self.df["PPI"].iloc[-1], 2) if "PPI" in self.df.columns else None,
            "latest_yoy": round(self.df["PPI_yoy"].iloc[-1], 2) if "PPI_yoy" in self.df.columns else None,
        }
