"""
PPI Predictor - PDF Report Generator

Generates a professional PDF report with:
  - Probability distribution table (each 0.1% bin with probability)
  - Bar chart of the probability distribution
  - Regression point estimates
  - Model performance metrics
  - Historical summary
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


class PPIReportPDF(FPDF):
    """Custom FPDF subclass with header/footer."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "PPI Predictor - Forecast Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(
    result: dict,
    expected_ppi_mom: float,
    summary: dict,
    metrics: dict,
) -> str:
    """
    Generate a PDF report and save it to PPI_Predictor/results/.

    Parameters
    ----------
    result : dict from Predictor.predict()
    expected_ppi_mom : market expectation
    summary : dict from Analyzer.summary()
    metrics : dict from Predictor.train()

    Returns
    -------
    str : path to generated PDF file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"PPI_Report_{timestamp}.pdf"
    chart_path = RESULTS_DIR / f"_temp_chart_{timestamp}.png"

    reg = result["regression"]
    prob_dist = result["probability_distribution"]

    # ── Generate probability distribution chart ───────
    _generate_chart(prob_dist, expected_ppi_mom, reg.get("ensemble", 0), chart_path)

    # ── Build PDF ─────────────────────────────────────
    pdf = PPIReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Section 1: Summary ────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Historical Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    summary_rows = [
        ("Data period (months)", str(summary.get("total_months", "N/A"))),
        ("Avg MoM change", f"{summary.get('avg_mom_pct', 'N/A')}%"),
        ("Std MoM change", f"{summary.get('std_mom_pct', 'N/A')}%"),
        ("Positive months", f"{summary.get('positive_months_pct', 'N/A')}%"),
        ("Latest PPI level", str(summary.get("latest_ppi", "N/A"))),
        ("Latest YoY change", f"{summary.get('latest_yoy', 'N/A')}%"),
    ]
    _add_key_value_table(pdf, summary_rows)
    pdf.ln(3)

    # ── Section 2: Regression Results ─────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Regression Point Estimates", ln=True)
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(0, 6, f"Market Expected MoM: {expected_ppi_mom:+.2f}%", ln=True)
    pdf.ln(2)

    reg_rows = []
    for name, val in reg.items():
        if name == "confidence_interval":
            continue
        label = name.replace("_", " ").title()
        reg_rows.append((label, f"{val:+.4f}%"))

    ci = reg.get("confidence_interval", {})
    if ci:
        reg_rows.append((
            f"{ci.get('level', 0.9)*100:.0f}% Confidence Range",
            f"[{ci.get('low', 0):+.4f}%, {ci.get('high', 0):+.4f}%]",
        ))
    _add_key_value_table(pdf, reg_rows)

    # Interpretation
    ensemble = reg.get("ensemble", 0)
    diff = ensemble - expected_ppi_mom
    if abs(diff) < 0.05:
        interp = "Model closely agrees with market expectation."
    elif diff > 0:
        interp = f"Model predicts HIGHER than expected ({diff:+.3f}pp above consensus)."
    else:
        interp = f"Model predicts LOWER than expected ({diff:+.3f}pp below consensus)."
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, interp, ln=True)
    pdf.ln(3)

    # ── Section 3: Probability Distribution ───────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Probability Distribution (Classification)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "Probability that PPI MoM change falls in each 0.1% bin:", ln=True)
    pdf.ln(2)

    _add_probability_table(pdf, prob_dist)
    pdf.ln(4)

    # ── Chart ─────────────────────────────────────────
    if chart_path.exists():
        # Calculate width to fit nicely
        img_w = 180
        pdf.image(str(chart_path), x=15, w=img_w)
        # Clean up temp chart
        try:
            os.remove(chart_path)
        except OSError:
            pass

    # ── Section 4: Model Performance ──────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Model Performance Metrics", ln=True)

    # Regression metrics
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Regression Models:", ln=True)
    pdf.set_font("Helvetica", "", 9)

    reg_metrics = metrics.get("regression", {})
    for model_name, m in reg_metrics.items():
        label = model_name.replace("_", " ").title()
        pdf.cell(0, 5,
            f"  {label}  -  MAE: {m['MAE']:.4f}  |  RMSE: {m['RMSE']:.4f}  |  R2: {m['R2']:.4f}",
            ln=True,
        )

    pdf.ln(3)

    # Classification metrics
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Classification Models:", ln=True)
    pdf.set_font("Helvetica", "", 9)

    clf_metrics = metrics.get("classification", {})
    for model_name, m in clf_metrics.items():
        label = model_name.replace("_", " ").title()
        pdf.cell(0, 5,
            f"  {label}  -  Accuracy: {m['accuracy']:.4f}  |  Log Loss: {m['log_loss']:.4f}",
            ln=True,
        )

    # ── Save ──────────────────────────────────────────
    pdf.output(str(pdf_path))
    print(f"\n  [PDF] Report saved to: {pdf_path}")
    return str(pdf_path)


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def _add_key_value_table(pdf: FPDF, rows: list[tuple[str, str]]):
    """Render a simple 2-column key-value table."""
    col_w = [90, 80]
    pdf.set_font("Helvetica", "", 9)
    for key, val in rows:
        pdf.cell(col_w[0], 5.5, f"  {key}", border=0)
        pdf.cell(col_w[1], 5.5, val, border=0, ln=True)


def _add_probability_table(pdf: FPDF, prob_dist: list[tuple[str, float]]):
    """
    Render probability distribution as a multi-column table.
    Each row: bin label | probability % | visual bar.
    """
    col_w = [25, 20, 100]
    max_prob = max((p for _, p in prob_dist), default=1)

    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(col_w[0], 5, "  PPI MoM", border="B")
    pdf.cell(col_w[1], 5, "Prob %", border="B", align="R")
    pdf.cell(col_w[2], 5, "  Distribution", border="B", ln=True)

    pdf.set_font("Helvetica", "", 8)
    for label, prob in prob_dist:
        # Color: green for positive bins, red for negative
        try:
            val = float(label.replace("%", "").replace(">", "").replace("<", ""))
        except ValueError:
            val = 0
        if val > 0:
            pdf.set_text_color(0, 120, 0)
        elif val < 0:
            pdf.set_text_color(180, 0, 0)
        else:
            pdf.set_text_color(0, 0, 180)

        pdf.cell(col_w[0], 4.5, f"  {label}", border=0)
        pdf.cell(col_w[1], 4.5, f"{prob:.1f}%", border=0, align="R")

        # Draw bar
        bar_max_w = col_w[2] - 5
        bar_w = (prob / max_prob * bar_max_w) if max_prob > 0 else 0
        x = pdf.get_x() + 3
        y = pdf.get_y() + 0.8
        if val > 0:
            pdf.set_fill_color(100, 180, 100)
        elif val < 0:
            pdf.set_fill_color(220, 100, 100)
        else:
            pdf.set_fill_color(100, 100, 220)
        pdf.rect(x, y, bar_w, 3, "F")
        pdf.ln(4.5)

    pdf.set_text_color(0, 0, 0)


def _generate_chart(
    prob_dist: list[tuple[str, float]],
    expected: float,
    ensemble: float,
    save_path: Path,
):
    """Generate a bar chart of the probability distribution."""
    labels = [l for l, _ in prob_dist]
    probs = [p for _, p in prob_dist]

    # Color bars
    colors = []
    for l in labels:
        try:
            val = float(l.replace("%", "").replace(">", "").replace("<", ""))
        except ValueError:
            val = 0
        if val > 0:
            colors.append("#4CAF50")
        elif val < 0:
            colors.append("#E53935")
        else:
            colors.append("#1E88E5")

    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, probs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Highlight the expected & ensemble
    ax.axvline(x=_find_closest_bin_idx(labels, expected), color="orange",
               linestyle="--", linewidth=2, label=f"Expected: {expected:+.1f}%")
    ax.axvline(x=_find_closest_bin_idx(labels, ensemble), color="purple",
               linestyle="--", linewidth=2, label=f"Ensemble: {ensemble:+.1f}%")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probability (%)", fontsize=10)
    ax.set_title("PPI MoM Change - Probability Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add percentage labels on top of bars > 3%
    for bar, prob in zip(bars, probs):
        if prob >= 3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{prob:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _find_closest_bin_idx(labels, value):
    """Find the index of the bin label closest to a given value."""
    best_idx = 0
    best_diff = float("inf")
    for i, label in enumerate(labels):
        try:
            v = float(label.replace("%", "").replace(">", "").replace("<", ""))
            if abs(v - value) < best_diff:
                best_diff = abs(v - value)
                best_idx = i
        except ValueError:
            continue
    return best_idx
