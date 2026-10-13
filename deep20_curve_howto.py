"""`python deep20_curve_howto.py` —— 用 10 題的示意資料，把 deep20_report 圖 1 / 圖 2 的藍線怎麼畫出來，
整條流程畫成一張圖（取代文字說明）：

    ① 依 route_p 由高到低排好                       → 最上面的長條
    ② 每題 fast / powerful 各自答對沒有（y = 1 / 0） → 中間的表格
    ③ 一題一題改送 powerful，每改一題正確率 +1/N、0 或 −1/N
    ④ 累加成藍線；每個點 k 同時對應
         上方 X 軸   門檻 τ_k = 第 k 題的 route_p（route_p ≥ τ_k 的前 k 題送 powerful）
         下方 X 軸   呼叫比例 k / N，以及這個作業點每千題的花費
         Y 軸        A(k) = A_F + (1/N) · Σ_{j≤k} (y_P − y_F)

公式和 deep20_report.py 的 Data.curve() 相同，只是題數少到每一階都看得見。
資料是虛構的（題數、對錯、花費都是），只用來說明畫法。

    python deep20_curve_howto.py            # → deep20_curve_howto.png
    python deep20_curve_howto.py --dark     # → deep20_curve_howto_dark.png
"""

import argparse
import os

import numpy as np

from deep20_report import HERE, PALETTE, _save, pct_axis, setup_font, style_axes

# ---- 示意資料：已依 route_p 由高到低排好 ----
#            route_p  fast 對  powerful 對  fast 花費  powerful 花費（$ / 千題）
QUESTIONS = [
    (0.92, 0, 1, 1.0, 4.0),
    (0.85, 1, 1, 1.0, 3.0),
    (0.74, 0, 1, 1.0, 3.0),
    (0.61, 0, 0, 1.0, 2.5),
    (0.55, 1, 0, 1.0, 2.0),
    (0.43, 0, 1, 1.0, 2.0),
    (0.31, 1, 1, 1.0, 1.5),
    (0.22, 1, 1, 1.0, 1.5),
    (0.12, 1, 0, 1.0, 1.5),
    (0.05, 1, 1, 1.0, 1.5),
]
TAU_NOW = 0.5   # 現行 argmax 等於門檻 0.5
K_DEMO = 5      # 拿來逐項拆解的那個點（= route_p ≥ 0.5 的題數）


def build():
    p, yf, yp, kf, kp = (np.array(col, float) for col in zip(*QUESTIONS))
    n = len(p)
    delta = yp - yf
    acc = (yf.sum() + np.concatenate([[0.0], np.cumsum(delta)])) / n            # A(k)
    cost = (np.concatenate([[0.0], np.cumsum(kp)])
            + kf.sum() - np.concatenate([[0.0], np.cumsum(kf)])) / n             # $(k)
    return dict(p=p, yf=yf, yp=yp, kf=kf, kp=kp, n=n, delta=delta, acc=acc, cost=cost)


def draw(d, c, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import FancyBboxPatch, Rectangle

    def tint(col, a):
        """把半透明色先和底色混好再用：不透明，壓在藍線上時文字才讀得清楚。"""
        fg, bg = np.array(to_rgb(col)), np.array(to_rgb(c["surface"]))
        return tuple(a * fg + (1 - a) * bg)

    n, k5 = d["n"], K_DEMO
    xs = np.arange(n + 1) / n                 # 點 k 的位置
    cx = (np.arange(n) + 0.5) / n             # 第 j 題的欄位中心：剛好在它造成的那一段上方
    up, down = c["cat"][2], c["cat"][1]

    fig = plt.figure(figsize=(12.5, 12.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 2.25, 5.0], hspace=0.13)
    ax_p = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharex=ax_p)
    ax = fig.add_subplot(gs[2], sharex=ax_p)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.94, bottom=0.08)
    fig.suptitle("藍線怎麼畫出來的（示意資料：N = 10 題）", x=0.02, ha="left",
                 color=c["ink"], fontsize=14)

    # ------------------------------------------------------------------ ① route_p
    style_axes(ax_p, c)
    ax_p.grid(False)
    ax_p.bar(cx, d["p"], width=0.62 / n, color=c["router"], alpha=0.35, linewidth=0)
    for x, v in zip(cx, d["p"]):  # 底色蓋住虛線，0.43 這種靠近門檻的數字才讀得到
        ax_p.text(x, v + 0.04, f"{v:.2f}", ha="center", va="bottom", color=c["ink"], fontsize=9,
                  bbox=dict(boxstyle="square,pad=0.1", fc=c["surface"], ec="none"), zorder=3)
    ax_p.axhline(TAU_NOW, color=c["ink2"], lw=1.0, ls=(0, (4, 3)))
    ax_p.text(1.0, TAU_NOW + 0.03, "τ = 0.5（現行 argmax）", ha="right", va="bottom",
              color=c["ink2"], fontsize=8.5)
    ax_p.set_ylim(0, 1.18)
    ax_p.set_yticks([])  # 每根長條都標了數字，不需要刻度
    ax_p.spines["left"].set_visible(False)
    ax_p.tick_params(labelbottom=False, colors=c["muted"], labelcolor=c["ink2"])
    for i, x in enumerate(cx, 1):
        ax_p.text(x, -0.02, f"Q{i}", ha="center", va="top", color=c["ink2"], fontsize=9,
                  transform=ax_p.get_xaxis_transform())
    ax_p.text(-0.025, 0.5, "① 依 route_p\n由高到低排好\n（jevk5 給 powerful\n的機率）",
              transform=ax_p.transAxes, ha="right", va="center", color=c["ink"], fontsize=9.5)

    # ------------------------------------------------------------------ ② ③ 表格
    rows = ["fast 答對 $y_F$", "powerful 答對 $y_P$", "③ 改送 powerful\n的變化 $y_P - y_F$",
            "花費 fast / powerful\n（$ / 千題）"]
    ax_t.set_xlim(-0.02, 1.02)
    ax_t.set_ylim(len(rows) - 0.5, -0.95)
    ax_t.axis("off")
    # k = K_DEMO 時：前 k 題送 powerful、其餘送 fast。底色標出來，算 A(k) 用到的格子加粗
    ax_t.add_patch(Rectangle((0, -0.5), k5 / n, 2.0, color=c["router"], alpha=0.08, lw=0))
    ax_t.add_patch(Rectangle((k5 / n, -0.5), 1 - k5 / n, 2.0, color=c["muted"], alpha=0.10,
                             lw=0))
    ax_t.text(k5 / n / 2, -0.62, f"以 k = {k5} 為例：前 {k5} 題送 powerful → 看 $y_P$",
              ha="center", va="bottom", color=c["ink"], fontsize=9)
    ax_t.text((1 + k5 / n) / 2, -0.62, f"後 {n - k5} 題送 fast → 看 $y_F$",
              ha="center", va="bottom", color=c["ink"], fontsize=9)
    for r, label in enumerate(rows):
        ax_t.text(-0.025, r, label, ha="right", va="center", color=c["ink"], fontsize=9.5,
                  transform=ax_t.get_yaxis_transform())
    for j in range(n):
        x = cx[j]
        for r, (y, used) in enumerate(((d["yf"][j], j >= k5), (d["yp"][j], j < k5))):
            ax_t.text(x, r, f"{int(y)}  {'對' if y else '錯'}", ha="center", va="center",
                      fontsize=10 if used else 9, fontweight="bold" if used else "normal",
                      color=c["ink"] if used else c["muted"])
        dv = d["delta"][j]
        col, txt = ((up, "+1/N") if dv > 0 else (down, "$-$1/N") if dv < 0
                    else (c["muted"], "0"))
        ax_t.add_patch(FancyBboxPatch((x - 0.036, 2 - 0.27), 0.072, 0.54,
                                      boxstyle="round,pad=0,rounding_size=0.012",
                                      color=tint(col, 0.22 if dv else 0.12), lw=0))
        ax_t.text(x, 2, txt, ha="center", va="center", color=c["ink"], fontsize=9.5)
        ax_t.text(x, 3, f"{d['kf'][j]:.1f} / {d['kp'][j]:.1f}", ha="center", va="center",
                  color=c["ink2"], fontsize=8.5)
    for b in xs:  # 欄與欄之間的分隔線 = 曲線上點 k 的位置
        ax_t.plot([b, b], [-0.5, 3.5], color=c["grid"], lw=0.8)
    ax_t.text(-0.025, -0.62, "② 每題兩個 model\n各自答對沒有", ha="right", va="bottom",
              color=c["ink"], fontsize=9.5, transform=ax_t.get_yaxis_transform())

    # ------------------------------------------------------------------ ④ 藍線
    style_axes(ax, c)
    acc, cost = d["acc"], d["cost"]
    ax.plot(xs, acc, color=c["router"], lw=2.4, zorder=3)
    ax.plot(xs, acc, "o", ms=8, color=c["router"], markeredgecolor=c["surface"],
            markeredgewidth=2, zorder=4)
    # 每一段標它是 +1/N、0 還是 −1/N，底色和 ③ 那一列的同一題一樣，看得出是哪一題造成的
    for j, dv in enumerate(d["delta"]):
        xm, ym = (xs[j] + xs[j + 1]) / 2, (acc[j] + acc[j + 1]) / 2
        col, txt, dy = ((up, "+1/N", 0.0) if dv > 0 else (down, "$-$1/N", 0.0) if dv < 0
                        else (c["muted"], "0", 0.022))
        ax.text(xm, ym + dy, txt, ha="center", va="center", color=c["ink"], fontsize=9, zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", ec="none",
                          fc=tint(col, 0.22 if dv else 0.12)))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.50, 0.95)
    pct_axis(ax.yaxis)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.0%}\n${v:.2f}" for x, v in zip(xs, cost)])
    ax.tick_params(colors=c["muted"], labelcolor=c["ink2"])
    ax.set_xlabel("下方 X 軸：呼叫比例 k / N（上行）　/　這個作業點每千題的花費（下行）",
                  color=c["ink2"], labelpad=8)
    ax.set_ylabel("Y 軸：正確率 A(k)", color=c["ink2"])

    axt = ax.twiny()  # 上方 X 軸：點 k 對應的門檻
    axt.set_xlim(ax.get_xlim())
    axt.grid(False)
    for side in ("bottom", "left", "right"):
        axt.spines[side].set_visible(False)
    axt.spines["top"].set_color(c["axis"])
    axt.set_xticks(xs)
    axt.set_xticklabels([r"$\infty$"] + [f"{v:.2f}" for v in d["p"]])
    axt.tick_params(colors=c["muted"], labelcolor=c["ink2"])
    axt.set_xlabel("上方 X 軸：門檻 τ_k = 第 k 題的 route_p（route_p ≥ τ_k 的前 k 題送 powerful）",
                   color=c["ink2"], labelpad=8)

    # 兩端點與 k = K_DEMO 的逐項拆解
    nf, npw = int(d["yf"].sum()), int(d["yp"].sum())
    ax.annotate(f"k = 0：τ = ∞，全部送 fast\nA(0) = A_F = {nf}/{n} = {acc[0]:.0%}",
                (0, acc[0]), xytext=(14, -40), textcoords="offset points",
                color=c["ink"], fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=c["muted"], lw=0.8))
    ax.annotate(f"k = N：τ ≤ {d['p'][-1]:.2f}，全部送 powerful\n"
                f"A(N) = A_P = {npw}/{n} = {acc[-1]:.0%}",
                (1, acc[-1]), xytext=(-12, -46), textcoords="offset points", ha="right",
                color=c["ink"], fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=c["muted"], lw=0.8))
    a_pow = int(d["yp"][:k5].sum())
    a_fast = int(d["yf"][k5:].sum())
    ax.plot([xs[k5]], [acc[k5]], "o", ms=13, markerfacecolor="none",
            markeredgecolor=c["ink"], markeredgewidth=1.4, zorder=5)
    ax.annotate(
        f"點 k = {k5}（τ = {d['p'][k5 - 1]:.2f}，等同現行 argmax 的 τ = 0.5）\n"
        f"上方 X：route_p ≥ {d['p'][k5 - 1]:.2f} 的 Q1–Q{k5} 送 powerful\n"
        f"Y：Q1–Q{k5} 的 $y_P$ 有 {a_pow} 題對 + Q{k5 + 1}–Q{n} 的 $y_F$ 有 {a_fast} 題對\n"
        f"　 = ({a_pow} + {a_fast}) / {n} = {acc[k5]:.0%}\n"
        f"下方 X：{k5}/{n} = {xs[k5]:.0%} 呼叫 powerful；"
        f"花費 = ({d['kp'][:k5].sum():.1f} + {d['kf'][k5:].sum():.1f}) / {n} = "
        f"\\${cost[k5]:.2f}",
        (xs[k5], acc[k5]), xytext=(18, -128), textcoords="offset points",
        color=c["ink"], fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", fc=c["surface"], ec=c["axis"], lw=0.8),
        arrowprops=dict(arrowstyle="-", color=c["ink2"], lw=0.9))

    ax.text(0.015, 0.975,
            "④ 從左到右一題一題改送 powerful，把 ③ 累加起來：\n"
            r"$A(k) = A_F + \frac{1}{N}\sum_{j \leq k}\,(y_P^{(j)} - y_F^{(j)})$"
            "　　每一項是 +1、0 或 $-$1\n"
            "第 j 題的欄位正好在它造成的那一段上方；欄與欄之間的分隔線就是點 k 的位置\n"
            "真實的圖題數多，上方只標 τ = 0.9 / 0.7 / 0.5 / 0.3 / 0.1，"
            "位置 = route_p ≥ τ 的題數 / N",
            transform=ax.transAxes, ha="left", va="top", color=c["ink"], fontsize=9.5,
            linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.5", fc=c["surface"], ec=c["axis"], lw=0.8))
    _save(fig, path, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", action="store_true", help="深色版（檔名加 _dark）")
    ap.add_argument("--out", default=HERE, help="輸出目錄（預設和本檔同一層）")
    args = ap.parse_args()
    setup_font()
    c = PALETTE["dark" if args.dark else "light"]
    path = os.path.join(args.out, f"deep20_curve_howto{'_dark' if args.dark else ''}.png")
    draw(build(), c, path)
    print(f"→ {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
