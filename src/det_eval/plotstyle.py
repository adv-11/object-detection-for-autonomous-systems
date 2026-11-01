"""Shared matplotlib style for report figures (light, print-oriented)."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Validated categorical order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

# Three super-families keep scatter charts within the 3-hue all-pairs limit.
GROUP_COLOR = {"CNN (torchvision)": SERIES[0], "Real-time (YOLO / RT-DETR)": SERIES[1], "Open-vocabulary": SERIES[2]}


def group_of(family: str) -> str:
    if family.startswith("open-vocabulary"):
        return "Open-vocabulary"
    if family.startswith("real-time"):
        return "Real-time (YOLO / RT-DETR)"
    return "CNN (torchvision)"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })
    return plt
