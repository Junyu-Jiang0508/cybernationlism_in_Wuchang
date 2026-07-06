# -*- coding: utf-8 -*-
"""
viz_style.py — Phase 1 论文/海报图共享样式（validated palette，light mode）

调色板来自经 CVD/对比度校验的参考实例（dataviz skill references/palette.md）：
分类色按固定槽位顺序使用、不循环；顺序色为单色蓝 ramp；图表 chrome 用中性 ink。
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# 分类槽位（light mode，固定顺序）
CAT = {
    "blue": "#2a78d6",     # slot 1 — 主系列（flat / before）
    "aqua": "#1baf7a",     # slot 2 — 对比系列（hierarchical / after）
    "yellow": "#eda100",   # slot 3
    "green": "#008300",    # slot 4
    "violet": "#4a3aa7",   # slot 5
    "red": "#e34948",      # slot 6
}
SERIES = list(CAT.values())

# 顺序 ramp（蓝，100→700）——热力图/混淆矩阵用
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

# Chrome & ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply_style():
    """全局 rcParams：细网格、隐右上轴、中性 ink、无衬线。"""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC", "sans-serif"],
        "axes.axisbelow": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
