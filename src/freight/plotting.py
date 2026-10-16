"""Shared matplotlib styling so every report figure looks consistent."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2a30"
MUTED = "#5b6b72"
GRID = "#e3e8ea"
# Fixed categorical order (colour follows the entity, never its rank).
EQUIPMENT_COLORS = {"Dry Van": "#2a78d6", "Reefer": "#eb6834", "Flatbed": "#1baf7a"}
PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
NEUTRAL = "#8a9aa1"
ALERT = "#e34948"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#b9c4c8",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
