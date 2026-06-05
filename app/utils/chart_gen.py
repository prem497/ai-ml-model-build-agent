"""
Chart generation helpers — return base64-encoded PNG strings.
"""
from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for server use)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


# ─── Shared Style ─────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#c9d1d9"
ACCENT    = "#7c3aed"
ACCENT2   = "#3b82f6"
GREEN     = "#34d399"
ORANGE    = "#f59e0b"


def _apply_dark_style(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply a consistent dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor("#161b22")
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.7)


def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ─── Actual vs Predicted (Regression) ────────────────────────────────────────

def generate_actual_vs_predicted_chart(
    y_true: np.ndarray, y_pred: np.ndarray
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_style(fig, ax)

    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, color=ACCENT2, edgecolors="none")

    # Perfect-prediction diagonal
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#f87171", linewidth=1.5, label="Perfect fit")

    ax.set_xlabel("Actual Values",    fontsize=10, color=TEXT_CLR)
    ax.set_ylabel("Predicted Values", fontsize=10, color=TEXT_CLR)
    ax.set_title("Actual vs Predicted", fontsize=12, fontweight="bold", color=TEXT_CLR, pad=12)
    ax.legend(facecolor="#161b22", edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)

    return _fig_to_b64(fig)


# ─── Feature Importance (Bar) ─────────────────────────────────────────────────

def generate_feature_importance_chart(
    importances: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
) -> str:
    # Sort descending, take top N
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [x[0] for x in sorted_items][::-1]   # reverse for bottom-to-top bars
    values = [x[1] for x in sorted_items][::-1]

    fig, ax = plt.subplots(figsize=(6, max(4, len(names) * 0.45)))
    _apply_dark_style(fig, ax)

    # Gradient colour based on rank
    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]

    bars = ax.barh(names, values, color=colors, height=0.65, edgecolor="none")

    ax.set_xlabel("Importance", fontsize=10, color=TEXT_CLR)
    ax.set_title(f"Top {top_n} Important Features", fontsize=12,
                 fontweight="bold", color=TEXT_CLR, pad=12)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=7.5, color=TEXT_CLR,
        )

    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="y", visible=False)

    return _fig_to_b64(fig)


# ─── Confusion Matrix (Classification) ───────────────────────────────────────

def generate_confusion_matrix_chart(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None
) -> str:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu",
        ax=ax,
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        linewidths=0.5,
        linecolor=DARK_BG,
        annot_kws={"size": 11, "color": "white"},
    )

    ax.set_xlabel("Predicted Label", fontsize=10, color=TEXT_CLR)
    ax.set_ylabel("True Label",      fontsize=10, color=TEXT_CLR)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold", color=TEXT_CLR, pad=12)
    ax.tick_params(colors=TEXT_CLR)

    return _fig_to_b64(fig)


# ─── Metrics Bar (summary) ────────────────────────────────────────────────────

def generate_metrics_bar_chart(metrics: Dict[str, float]) -> str:
    names  = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _apply_dark_style(fig, ax)

    colors_list = [ACCENT, ACCENT2, GREEN, ORANGE] * 4
    bars = ax.bar(names, values, color=colors_list[:len(names)],
                  width=0.5, edgecolor="none")

    ax.set_title("Model Metrics Summary", fontsize=12, fontweight="bold",
                 color=TEXT_CLR, pad=12)
    ax.set_ylim(0, max(values) * 1.25)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, color=TEXT_CLR, fontweight="bold",
        )

    return _fig_to_b64(fig)
