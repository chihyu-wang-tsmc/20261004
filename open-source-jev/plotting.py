"""報表圖的共同樣式（三份報表共用）。圖上的字一律英文，所以不需要中文字型。

顏色照 dataviz 的參考調色盤（淺色版），類別色固定照順序配，不循環、不自動生成新色。
"""

import math

import matplotlib

matplotlib.use("Agg")  # 不開視窗，直接存 PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from common import rel  # noqa: E402

COLORS = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",   # 格線用實線細線；虛線會被讀成門檻
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
})


def new_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    style_axes(ax)
    return fig, ax


def style_axes(ax):
    """只留左、下兩條軸，格線退到背景。"""
    ax.set_facecolor(COLORS["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["muted"], labelcolor=COLORS["ink2"])


def pct_axis(axis):
    """百分比刻度：間距取 1 / 2 / 5 的倍數，小數位數跟著間距走（跨距很小時也不會每格印成同一個數）。"""
    axis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 5, 10]))

    def fmt(v, _):
        locs = axis.get_majorticklocs()
        step = float(np.min(np.diff(locs))) * 100 if len(locs) > 1 else 1.0
        digits = max(0, -math.floor(math.log10(step) + 1e-6))
        return f"{v * 100:.{digits}f}%"

    axis.set_major_formatter(FuncFormatter(fmt))


def top_ticks(ax, positions, labels, title=None, fontsize=8):
    """在上方加一排刻度（同一條 X 軸的另一種讀法，例如門檻 τ 落在哪個呼叫比例）。"""
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_facecolor("none")
    top.grid(False)
    for side in ("bottom", "left", "right"):
        top.spines[side].set_visible(False)
    top.spines["top"].set_color(COLORS["axis"])
    top.set_xticks(positions)
    top.set_xticklabels(labels)
    top.tick_params(axis="x", colors=COLORS["muted"], labelcolor=COLORS["ink2"], labelsize=fontsize, length=3)
    if title:
        top.set_xlabel(title, color=COLORS["ink2"], fontsize=fontsize + 0.5, labelpad=4)
    return top


LABEL_SPOTS = ((7, -11, "left"), (7, 5, "left"), (-7, -11, "right"), (-7, 5, "right"))
"""標籤相對於點的候選位置（點數）：先右下，再右上、左下、左上。"""


def label_points(ax, points, taken=None, fontsize=8):
    """在點旁邊標字 points = [(x, y, 文字)]。四個位置都會撞到已經標好的字、或超出繪圖區時就不標，
    寧可少一個數字也不要疊成一團。taken 是已經標好的字框，幾條線共用時由呼叫端傳同一個 list。
    要在軸的範圍定好（和 tight_layout）之後才呼叫，位置才算得準。回傳沒標字的個數。
    """
    taken = [] if taken is None else taken
    px = ax.get_figure().dpi / 72.0
    bounds = ax.get_window_extent()
    missed = 0
    for x, y, text in points:
        w, h = 0.62 * fontsize * len(text) * px, 1.25 * fontsize * px
        mx, my = ax.transData.transform((x, y))
        for dx, dy, ha in LABEL_SPOTS:
            x0 = mx + dx * px - (0 if ha == "left" else w)
            box = (x0, my + dy * px, x0 + w, my + dy * px + h)
            inside = bounds.x0 <= box[0] and box[2] <= bounds.x1 and bounds.y0 <= box[1] and box[3] <= bounds.y1
            free = all(box[2] <= t[0] or t[2] <= box[0] or box[3] <= t[1] or t[3] <= box[1] for t in taken)
            if inside and free:
                ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="baseline",
                            color=COLORS["ink2"], fontsize=fontsize, zorder=8)
                taken.append(box)
                break
        else:
            missed += 1
    return missed


def save(fig, path):
    fig.patch.set_facecolor(COLORS["surface"])
    fig.savefig(path, facecolor=COLORS["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"  → {rel(path)}")
