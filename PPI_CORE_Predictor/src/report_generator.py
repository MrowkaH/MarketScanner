"""
PPI Core Predictor - PDF Report Generator

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


class PPICoreReportPDF(FPDF):
    """Custom FPDF subclass with header/footer."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "PPI Core Predictor - Forecast Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(
    result: dict,
    expected_ppi_core_mom: float,
    summary: dict,
    metrics: dict,
    direct_result: dict | None = None,
    direct_metrics: dict | None = None,
) -> str:
    """
    Generate a PDF report and save it to PPI_CORE_Predictor/results/.

    Parameters
    ----------
    result : dict from Predictor.predict()
    expected_ppi_core_mom : market expectation
    summary : dict from Analyzer.summary()
    metrics : dict from Predictor.train()

    Returns
    -------
    str : path to generated PDF file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"PPI_CORE_Report_{timestamp}.pdf"
    chart_path = RESULTS_DIR / f"_temp_chart_{timestamp}.png"
    matrix_path = RESULTS_DIR / f"_temp_matrix_{timestamp}.png"

    reg = result["regression"]
    prob_dist = result["probability_distribution"]
    ensemble = reg.get("ensemble", 0)

    # ── Generate charts ───────────────────────────────
    _generate_chart(prob_dist, expected_ppi_core_mom, ensemble, chart_path)
    _generate_probability_matrix(prob_dist, ensemble, matrix_path, label="PPI Core")

    # Generate direct impact charts if available
    direct_chart_path = RESULTS_DIR / f"_temp_direct_chart_{timestamp}.png"
    direct_matrix_path = RESULTS_DIR / f"_temp_direct_matrix_{timestamp}.png"
    if direct_result is not None:
        d_reg = direct_result["regression"]
        d_prob_dist = direct_result["probability_distribution"]
        d_ensemble = d_reg.get("ensemble", 0)
        _generate_chart(d_prob_dist, expected_ppi_core_mom, d_ensemble, direct_chart_path,
                        title="PPI Core MoM - Direct Impact Model")
        _generate_probability_matrix(d_prob_dist, d_ensemble, direct_matrix_path,
                                     label="PPI Core Direct Impact")

    # ── Build PDF ─────────────────────────────────────
    pdf = PPICoreReportPDF(orientation="P", unit="mm", format="A4")
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
        ("Latest PPI Core level", str(summary.get("latest_ppi_core", "N/A"))),
        ("Latest YoY change", f"{summary.get('latest_yoy', 'N/A')}%"),
    ]
    _add_key_value_table(pdf, summary_rows)
    pdf.ln(3)

    # ── Section 2: Regression Results ─────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Regression Point Estimates", ln=True)
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(0, 6, f"Market Expected MoM: {expected_ppi_core_mom:+.2f}%", ln=True)
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
    diff = ensemble - expected_ppi_core_mom
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
    pdf.cell(0, 5, "Probability that PPI Core MoM change falls in each 0.1% bin:", ln=True)
    pdf.ln(2)

    _add_probability_table(pdf, prob_dist)
    pdf.ln(4)

    # ── Chart ─────────────────────────────────────────
    if chart_path.exists():
        img_w = 180
        pdf.image(str(chart_path), x=15, w=img_w)
        try:
            os.remove(chart_path)
        except OSError:
            pass

    # ── Section 4: Probability Matrix ─────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Outcome Probability Matrix", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Probability of outcome vs ensemble prediction ({ensemble:+.2f}%):", ln=True)
    pdf.ln(2)

    if matrix_path.exists():
        pdf.image(str(matrix_path), x=15, w=180)
        try:
            os.remove(matrix_path)
        except OSError:
            pass

    # ── Direct Impact Model Sections ──────────────────
    sec = 5
    if direct_result is not None:
        d_reg = direct_result["regression"]
        d_prob_dist = direct_result["probability_distribution"]
        d_ensemble = d_reg.get("ensemble", 0)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "DIRECT IMPACT MODEL", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, "(External economic indicators only - no PPI Core self-history)", ln=True, align="C")
        pdf.ln(4)

        # Direct Impact Regression
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{sec}. Direct Impact - Regression Estimates", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Market Expected MoM: {expected_ppi_core_mom:+.2f}%", ln=True)
        pdf.ln(2)

        dreg_rows = []
        for name, val in d_reg.items():
            if name == "confidence_interval":
                continue
            dreg_rows.append((name.replace("_", " ").title(), f"{val:+.4f}%"))
        dci = d_reg.get("confidence_interval", {})
        if dci:
            dreg_rows.append((
                f"{dci.get('level', 0.9)*100:.0f}% Confidence Range",
                f"[{dci.get('low', 0):+.4f}%, {dci.get('high', 0):+.4f}%]",
            ))
        _add_key_value_table(pdf, dreg_rows)

        ddiff = d_ensemble - expected_ppi_core_mom
        if abs(ddiff) < 0.05:
            dinterp = "Direct impact model closely agrees with market expectation."
        elif ddiff > 0:
            dinterp = f"Direct impact model predicts HIGHER ({ddiff:+.3f}pp above consensus)."
        else:
            dinterp = f"Direct impact model predicts LOWER ({ddiff:+.3f}pp below consensus)."
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, dinterp, ln=True)
        pdf.ln(3)
        sec += 1

        # Direct Impact Probability Distribution
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{sec}. Direct Impact - Probability Distribution", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, "Probability using only external indicators:", ln=True)
        pdf.ln(2)
        _add_probability_table(pdf, d_prob_dist)
        pdf.ln(4)

        if direct_chart_path.exists():
            pdf.image(str(direct_chart_path), x=15, w=180)
            try:
                os.remove(direct_chart_path)
            except OSError:
                pass
        sec += 1

        # Direct Impact Probability Matrix
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{sec}. Direct Impact - Outcome Probability Matrix", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Probability vs direct impact ensemble ({d_ensemble:+.2f}%):", ln=True)
        pdf.ln(2)

        if direct_matrix_path.exists():
            pdf.image(str(direct_matrix_path), x=15, w=180)
            try:
                os.remove(direct_matrix_path)
            except OSError:
                pass
        sec += 1

    # ── Model Performance ────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"{sec}. Model Performance Metrics", ln=True)

    # Full model metrics
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Full Model:", ln=True)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "  Regression:", ln=True)
    pdf.set_font("Helvetica", "", 9)

    reg_metrics = metrics.get("regression", {})
    for model_name, m in reg_metrics.items():
        label = model_name.replace("_", " ").title()
        pdf.cell(0, 5,
            f"    {label}  -  MAE: {m['MAE']:.4f}  |  RMSE: {m['RMSE']:.4f}  |  R2: {m['R2']:.4f}",
            ln=True,
        )

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "  Classification:", ln=True)
    pdf.set_font("Helvetica", "", 9)

    clf_metrics = metrics.get("classification", {})
    for model_name, m in clf_metrics.items():
        label = model_name.replace("_", " ").title()
        pdf.cell(0, 5,
            f"    {label}  -  Accuracy: {m['accuracy']:.4f}  |  Log Loss: {m['log_loss']:.4f}",
            ln=True,
        )

    # Direct impact model metrics
    if direct_metrics is not None:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Direct Impact Model:", ln=True)
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "  Regression:", ln=True)
        pdf.set_font("Helvetica", "", 9)

        dreg_metrics = direct_metrics.get("regression", {})
        for model_name, m in dreg_metrics.items():
            label = model_name.replace("_", " ").title()
            pdf.cell(0, 5,
                f"    {label}  -  MAE: {m['MAE']:.4f}  |  RMSE: {m['RMSE']:.4f}  |  R2: {m['R2']:.4f}",
                ln=True,
            )

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "  Classification:", ln=True)
        pdf.set_font("Helvetica", "", 9)

        dclf_metrics = direct_metrics.get("classification", {})
        for model_name, m in dclf_metrics.items():
            label = model_name.replace("_", " ").title()
            pdf.cell(0, 5,
                f"    {label}  -  Accuracy: {m['accuracy']:.4f}  |  Log Loss: {m['log_loss']:.4f}",
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
    pdf.cell(col_w[0], 5, "  PPI Core", border="B")
    pdf.cell(col_w[1], 5, "Prob %", border="B", align="R")
    pdf.cell(col_w[2], 5, "  Distribution", border="B", ln=True)

    pdf.set_font("Helvetica", "", 8)
    for label, prob in prob_dist:
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
    title: str = "PPI Core MoM Change - Probability Distribution",
):
    """Generate a bar chart of the probability distribution."""
    labels = [l for l, _ in prob_dist]
    probs = [p for _, p in prob_dist]

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

    ax.axvline(x=_find_closest_bin_idx(labels, expected), color="orange",
               linestyle="--", linewidth=2, label=f"Expected: {expected:+.1f}%")
    ax.axvline(x=_find_closest_bin_idx(labels, ensemble), color="purple",
               linestyle="--", linewidth=2, label=f"Ensemble: {ensemble:+.1f}%")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probability (%)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

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


def _generate_probability_matrix(
    prob_dist: list[tuple[str, float]],
    ensemble: float,
    save_path: Path,
    label: str = "PPI Core",
):
    """
    Generate a probability matrix chart showing:
      - P(Below prediction)
      - P(At prediction)  (the bin matching the ensemble)
      - P(Above prediction)
    """
    # Find the bin closest to the ensemble prediction
    best_idx = _find_closest_bin_idx([l for l, _ in prob_dist], ensemble)

    p_below = sum(p for _, p in prob_dist[:best_idx])
    p_at = prob_dist[best_idx][1] if best_idx < len(prob_dist) else 0.0
    p_above = sum(p for _, p in prob_dist[best_idx + 1:])

    # Normalize (should already sum to ~100 but ensure)
    total = p_below + p_at + p_above
    if total > 0:
        p_below = p_below / total * 100
        p_at = p_at / total * 100
        p_above = p_above / total * 100

    bin_label = prob_dist[best_idx][0] if best_idx < len(prob_dist) else f"{ensemble:+.1f}%"

    categories = [f"Below\n(< {bin_label})", f"At Prediction\n({bin_label})", f"Above\n(> {bin_label})"]
    values = [p_below, p_at, p_above]
    colors_bg = ["#E53935", "#1E88E5", "#4CAF50"]
    colors_text = ["white", "white", "white"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{label} MoM Outcome Probability Matrix  (vs Ensemble: {ensemble:+.2f}%)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, cat, val, bg, tc in zip(axes, categories, values, colors_bg, colors_text):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # Background rectangle
        from matplotlib.patches import FancyBboxPatch
        alpha = max(0.35, min(val / 100, 1.0))  # opacity scales with probability
        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                               boxstyle="round,pad=0.05",
                               facecolor=bg, alpha=alpha,
                               edgecolor="gray", linewidth=1.5)
        ax.add_patch(rect)

        # Probability value
        ax.text(0.5, 0.55, f"{val:.1f}%", ha="center", va="center",
                fontsize=28, fontweight="bold", color=tc,
                alpha=max(0.5, alpha))

        # Category label
        ax.text(0.5, 0.18, cat, ha="center", va="center",
                fontsize=10, color=tc, alpha=max(0.6, alpha))

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
