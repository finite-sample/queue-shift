"""Shared plotting style for Queue Shift evidence."""

from __future__ import annotations

from typing import Any

BLUE = "#2166AC"
ORANGE = "#B35806"
GRAY = "#555555"
LIGHT_GRAY = "#D9D9D9"


def apply_plot_style(matplotlib_module: Any) -> None:
    """Apply the paper's restrained, print-safe plotting defaults."""
    matplotlib_module.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
