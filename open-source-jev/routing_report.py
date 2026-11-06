"""routing 成本／正確率評測：h2o-lightning-4b 的 model_route 決定每題送 fast 還是 powerful，看省多少錢、掉多少分。

    python routing_report.py status      [--fast M ...]                 資料齊不齊、缺的要跑什麼
    python routing_report.py report      [--fast M] [--bench ...]       一個 fast：印報表、畫圖、寫 Excel
    python routing_report.py multi-fast  --fast M M ... [--bench ...]   powerful 固定、比好幾個 fast

用到兩種結果：
    route_p   h2o 給每題「該送 powerful」的機率（model_route）。coding_report.py run --classifier h2o-lightning-4b
              會一起問，存在 results/coding/h2o-lightning-4b/<GPQA|AI2ARC|AIME|GSM8K>.json
    作答      fast 和 powerful 對每題的回答與 token 用量：run_bench.py 寫的 results/<benchmark>/<model>.jsonl
兩邊靠題目本身對起來，所以要讀 datasets（預設 ../datasets，可用 --data-dir 或環境變數 JEV_DATASETS_DIR 指定）。

門檻 τ：route_p ≥ τ 的題目送 powerful、其餘送 fast。τ 從 1 掃到 0，得到「powerful 呼叫比例 → end-to-end 正確率」
的曲線，報表把它和全送 fast、全送 powerful、oracle（事先知道哪個 model 答得對）放在一起比。
每題的花費 = 輸入 token × 輸入單價 + 輸出 token × 輸出單價，單價在 models.py；不含 router 本身的花費。

route_p 只看題目，和 fast 是哪個 model 無關，所以換 fast 不用重問 h2o，只要讓新的 fast 回答四個 benchmark。
--bench 可以只算幾個 benchmark，表上的「合計」也只含選到的那幾個。
結果寫到 results/routing/<fast>/（multi-fast 是 results/routing/multi-fast/）；
有 --bench 時目錄名稱後面加上選到的 benchmark。

報表裡的名詞：
    天花板    oracle 的最高正確率 = 兩個 model 任一答對的比例
    APGR      藍線在「隨機升級」與 oracle 之間補回了多少（0 = 和隨機一樣、1 = 和 oracle 一樣）
    CPT(95%)  藍線要達到「全送 powerful 的正確率 × 95%」，最少要送幾成給 powerful
"""

import argparse
import sys

import numpy as np

from benchmarks import BENCHMARKS
from common import (NA, RESULTS_DIR, Cell, F, P, Report, add_data_dir_arg, data_dir, prompt_key, read_json,
                    read_jsonl, rel, safe_name)
from metrics import auroc
from models import MODELS, cost

ROUTER = "h2o-lightning-4b"
DEFAULT_FAST = "gemma-4-31b"
DEFAULT_POWERFUL = "gemini-3.7-flash"
BENCH_ORDER = ("gpqa", "ai2arc", "aime", "gsm8k")
"""題目照這個順序排；route_p 同分的題目也照這個順序升級。"""
GRID = np.linspace(0.0, 1.0, 201)
"""算 APGR 時，曲線統一插值到這組呼叫比例上。"""
TAU_TICKS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
OVERLAY_TAUS = (0.9, 0.7, 0.5, 0.3, 0.1)


# ============================================================================
# 資料：每題的 route_p + 兩個 model 的對錯與花費
# ============================================================================
def pick_benches(names):
    if not names:
        return [BENCHMARKS[b] for b in BENCH_ORDER]
    wanted = [n.lower() for n in names]
    bad = [n for n in wanted if n not in BENCHMARKS]
    if bad:
        sys.exit(f"--bench 沒有這些：{' / '.join(bad)}（有 {' / '.join(BENCH_ORDER)}）")
    return [BENCHMARKS[b] for b in BENCH_ORDER if b in wanted]


def check_price(model):
    if model not in MODELS:
        sys.exit(f"models.py 沒有 {model} 的單價，先在 MODELS 加上它")


def answers(bench, model):
    """{題目 id: 有效作答}。"""
    return {i: r for i, r in read_jsonl(RESULTS_DIR / bench.name / f"{model}.jsonl").items() if r["ok"]}


def route_rows(bench):
    res = read_json(RESULTS_DIR / "coding" / ROUTER / f"{safe_name(bench.dataset)}.json")
    return {k: r for k, r in (res or {"rows": {}})["rows"].items() if "route_p" in r}


def load_items(benches, root, powerful, fasts):
    """每題一個 dict：bench、qid、route_p、powerful 的對錯和花費、每個 fast 的對錯和花費。

    只收 route_p、powerful 和每個 fast 都有有效作答的題目（不同 fast 用不同題目比，數字就不能互比）。
    回傳 (items, coverage)；coverage[bench] = (題數, 有 route_p, powerful 有答, {fast: 有答})。
    """
    items, coverage = [], {}
    for b in benches:
        questions = b.questions(root)
        routes = route_rows(b)
        p_ans = answers(b, powerful)
        f_ans = {m: answers(b, m) for m in fasts}
        with_route = [(q, routes.get(prompt_key(b.dataset, q["prompt"]))) for q in questions]
        with_route = [(q, r) for q, r in with_route if r]
        coverage[b.name] = (len(questions), len(with_route), sum(q["id"] in p_ans for q, _ in with_route),
                            {m: sum(q["id"] in f_ans[m] for q, _ in with_route) for m in fasts})
        for q, r in with_route:
            if q["id"] not in p_ans or any(q["id"] not in f_ans[m] for m in fasts):
                continue
            p = p_ans[q["id"]]
            item = {"bench": b.name, "qid": q["id"], "route_p": r["route_p"],
                    "c_pow": int(bool(p["correct"])),
                    "cost_pow": cost(powerful, p.get("input_tokens"), p.get("output_tokens"))}
            for m in fasts:
                f = f_ans[m][q["id"]]
                item[f"c_fast:{m}"] = int(bool(f["correct"]))
                item[f"cost_fast:{m}"] = cost(m, f.get("input_tokens"), f.get("output_tokens"))
            items.append(item)
    return items, coverage


def warn_missing_tokens(benches, models):
    for m in models:
        n = sum(r.get("input_tokens") is None or r.get("output_tokens") is None
                for b in benches for r in answers(b, m).values())
        if n:
            print(f"  ⚠ {m} 有 {n} 題作答沒有 token 用量（API 沒回 usage），這些題的花費算成 0")


# ============================================================================
# 分析
# ============================================================================
class Items:
    """一批題目（同一個 fast），攤成 numpy 陣列。"""

    def __init__(self, rows, fast):
        self.rows = rows
        self.bench = np.array([r["bench"] for r in rows])
        self.p = np.array([r["route_p"] for r in rows], float)
        self.cf = np.array([r[f"c_fast:{fast}"] for r in rows], float)
        self.cp = np.array([r["c_pow"] for r in rows], float)
        self.kf = np.array([r[f"cost_fast:{fast}"] for r in rows], float)
        self.kp = np.array([r["cost_pow"] for r in rows], float)
        self.n = len(rows)


def curve(it, order):
    """照 order 一題一題升級到 powerful：前 k 題送 powerful、其餘送 fast，k = 0…n。"""
    cf, cp, kf, kp = it.cf[order], it.cp[order], it.kf[order], it.kp[order]

    def mix(fast, pow_):
        return (np.concatenate([[0.0], np.cumsum(pow_)]) + fast.sum() - np.concatenate([[0.0], np.cumsum(fast)])) / it.n

    return {"call": np.arange(it.n + 1) / it.n, "acc": mix(cf, cp), "cost": mix(kf, kp),
            "tau": np.concatenate([[np.inf], it.p[order]])}


def oracle_curve(it):
    """預算「最多」k 次 powerful 時的最佳：先升級 fast 錯、powerful 對的題目（同樣的先升級便宜的）。

    最後取 running max：預算是上限不是下限，沒有人會被迫把「fast 對、powerful 錯」的題目也送去 powerful。
    """
    delta = it.cp - it.cf
    cur = curve(it, np.lexsort((it.kp, -delta)))
    best = np.maximum.accumulate(np.where(cur["acc"] >= np.maximum.accumulate(cur["acc"]) - 1e-12,
                                          np.arange(len(cur["acc"])), -1))
    cur["acc"], cur["cost"] = cur["acc"][best], cur["cost"][best]
    return cur


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.x 只有 trapz


def apgr(router, oracle, acc_fast, acc_pow):
    rand = acc_fast + GRID * (acc_pow - acc_fast)
    num = _trapezoid(np.interp(GRID, router["call"], router["acc"]) - rand, GRID)
    den = _trapezoid(np.interp(GRID, oracle["call"], oracle["acc"]) - rand, GRID)
    return float(num / den) if den > 1e-12 else None


def point(a, k):
    r = a["router"]
    return {"k": int(k), "call": float(r["call"][k]), "acc": float(r["acc"][k]), "cost": float(r["cost"][k]),
            "tau": float(r["tau"][k])}


def tau_point(a, tau):
    """門檻 τ 的作業點，外加升級的題目裡 +1（fast 錯、powerful 對）和 −1（fast 對、powerful 錯）各幾題。"""
    k = int((a["p_sorted"] >= tau).sum())
    d = a["delta_sorted"][:k]
    return {**point(a, k), "up": int((d > 0).sum()), "down": int((d < 0).sum())}


def analyse(it):
    order = np.argsort(-it.p, kind="stable")
    router = curve(it, order)
    acc_fast, acc_pow = float(it.cf.mean()), float(it.cp.mean())
    gold = np.where(it.cf > 0, "fast", np.where(it.cp > 0, "powerful", "none"))
    keep = gold != "none"
    a = {
        "n": it.n, "router": router, "oracle": oracle_curve(it),
        "acc_fast": acc_fast, "acc_pow": acc_pow, "ceiling": float(np.maximum(it.cf, it.cp).mean()),
        "cost_fast": float(it.kf.mean()), "cost_pow": float(it.kp.mean()),
        "p_sorted": it.p[order], "delta_sorted": (it.cp - it.cf)[order],
        "auroc": auroc(it.p[keep].tolist(), (gold[keep] == "powerful").tolist()),
        "quad": {"both": int(((it.cf == 1) & (it.cp == 1)).sum()),
                 "fast_only": int(((it.cf == 1) & (it.cp == 0)).sum()),
                 "pow_only": int(((it.cf == 0) & (it.cp == 1)).sum()),
                 "none": int(((it.cf == 0) & (it.cp == 0)).sum())},
        "gold": gold, "chose_pow": it.p >= 0.5,
    }
    a["apgr"] = apgr(router, a["oracle"], acc_fast, acc_pow)
    a["cpt"] = {q: next((float(c) for c, x in zip(router["call"], router["acc"]) if x >= q * acc_pow - 1e-12), None)
                for q in (0.90, 0.95, 0.99)}
    a["tau05"] = tau_point(a, 0.5)
    a["best"] = point(a, int(np.argmax(router["acc"])))  # 同分取呼叫比例最小的
    a["cheap95"] = None if a["cpt"][0.95] is None else point(a, int(round(a["cpt"][0.95] * it.n)))
    return a


def analyse_scopes(rows, fast, benches):
    """{"合計": 分析, bench 名稱: 分析}；只收有題目的 bench。"""
    it = Items(rows, fast)
    out = {"合計": analyse(it)}
    for b in benches:
        mask = it.bench == b.name
        if mask.any():
            out[b.name] = analyse(Items([r for r, m in zip(rows, mask) if m], fast))
    return out


# ============================================================================
# 主控台與 Excel 的表
# ============================================================================
def dollars(x):
    return NA if x is None else Cell(x * 1000, f"${x * 1000:,.2f}")


def scope_label(name):
    return name if name == "合計" else BENCHMARKS[name].title


def coverage_table(rep, coverage, powerful, fasts):
    body = []
    for b, (n, routed, p_ok, f_ok) in coverage.items():
        body.append([BENCHMARKS[b].title, n, routed, p_ok, *(f_ok[m] for m in fasts)])
    rep.table("資料", "每個 benchmark 有幾題拿得到 route_p、各 model 有幾題有效作答",
              ["benchmark", "題數", "有 route_p", f"{powerful} 有答", *(f"{m} 有答" for m in fasts)], body)
    hints = []
    for b, (n, routed, p_ok, f_ok) in coverage.items():
        if routed < n:
            hints.append(f"缺 route_p：python coding_report.py run --classifier {ROUTER} "
                         f"--only {BENCHMARKS[b].dataset}")
        for m, k in [(powerful, p_ok), *f_ok.items()]:
            if k < routed:
                hints.append(f"缺 {m} 的作答：python run_bench.py {b} answer --model {m}")
    if hints:
        rep.note("只比全部都有的題目。要補齊：\n" + "\n".join(f"  {h}" for h in dict.fromkeys(hints)))


def single_tables(rep, scopes, fast, powerful):
    a = scopes["合計"]
    rep.section("[R1] 作業點（合計）")
    ops = [("全送 fast", None, 0.0, a["acc_fast"], a["cost_fast"]),
           ("全送 powerful", None, 1.0, a["acc_pow"], a["cost_pow"]),
           ("τ = 0.5（model_route 的 argmax）", 0.5, a["tau05"]["call"], a["tau05"]["acc"], a["tau05"]["cost"])]
    if a["cheap95"]:
        c = a["cheap95"]
        ops.append(("最省、仍達全送 powerful 的 95%", c["tau"], c["call"], c["acc"], c["cost"]))
    b = a["best"]
    ops.append(("藍線正確率最高", b["tau"], b["call"], b["acc"], b["cost"]))
    rep.table("R1_作業點", "", ["作業點", "門檻 τ", "powerful 呼叫比例", "正確率", "相對全送 powerful", "$/千題",
                              "花費相對全送 powerful"],
              [[label, NA if t is None or np.isinf(t) else F(t), P(call), P(acc),
                P(acc / a["acc_pow"]) if a["acc_pow"] else NA, dollars(k),
                P(k / a["cost_pow"]) if a["cost_pow"] else NA] for label, t, call, acc, k in ops])
    cpts = "　".join(f"CPT({q:.0%}) {P(a['cpt'][q]).text}" for q in (0.90, 0.95, 0.99))
    rep.note(f"oracle 天花板 {a['ceiling']:.1%}（兩個 model 任一答對）　APGR {F(a['apgr']).text}　{cpts}")
    if a["best"]["acc"] > a["acc_pow"] + 1e-12:
        rep.note("藍線的最高正確率高過全送 powerful：router 把一些「fast 對、powerful 錯」的題目留在 fast")

    rep.section("[R2] 各 benchmark")
    rep.table("R2_各bench", "", ["範圍", "題數", "全送 fast", "全送 powerful", "天花板", "τ=0.5 正確率",
                               "τ=0.5 呼叫比例", "最高正確率", "在呼叫比例", "APGR", "CPT(95%)", "AUROC"],
              [[scope_label(s), x["n"], P(x["acc_fast"]), P(x["acc_pow"]), P(x["ceiling"]), P(x["tau05"]["acc"]),
                P(x["tau05"]["call"]), P(x["best"]["acc"]), P(x["best"]["call"]), F(x["apgr"], 2), P(x["cpt"][0.95]),
                F(x["auroc"])] for s, x in scopes.items()])
    rep.note("AUROC：route_p 分不分得出「fast 就答對」和「只有 powerful 答對」的題目（兩個都答錯的不算）")

    rep.section("[R3] 兩個 model 的能力重疊（曲線的形狀從這裡來）")
    rep.table("R3_能力重疊", "", ["範圍", "題數", "兩個都對", "只有 fast 對", "只有 powerful 對", "兩個都錯",
                                "全送 powerful 淨益"],
              [[scope_label(s), x["n"], x["quad"]["both"], x["quad"]["fast_only"], x["quad"]["pow_only"],
                x["quad"]["none"], P((x["quad"]["pow_only"] - x["quad"]["fast_only"]) / x["n"])]
               for s, x in scopes.items()])
    rep.note("升級「只有 powerful 對」的題目 +1 分、升級「只有 fast 對」的 −1 分；\n"
             "藍線的終點釘在全送 powerful，所以中途超過它的話，右半段一定會掉回來")

    rep.section("[R4] router 的選擇（τ = 0.5）× 標準答案（能答對的最便宜 model）")
    gold, chose = a["gold"], a["chose_pow"]
    body = []
    for label, mask in (("選 fast", ~chose), ("選 powerful", chose)):
        body.append([label, *(int(((gold == g) & mask).sum()) for g in ("fast", "powerful", "none")), int(mask.sum())])
    body.append(["合計", *(int((gold == g).sum()) for g in ("fast", "powerful", "none")), a["n"]])
    rep.table("R4_選擇", "", ["", "標準答案 fast", "標準答案 powerful", "兩個都答錯", "合計"], body)
    keep = gold != "none"
    right = float(((gold == "powerful") == chose)[keep].mean()) if keep.any() else None
    rep.note(f"選對的比例（不含兩個都答錯）{P(right).text}"
             f"　白用 powerful（fast 就夠卻選 powerful）{int(((gold == 'fast') & chose).sum())} 題"
             f"　該升沒升（只有 powerful 對卻選 fast）{int(((gold == 'powerful') & ~chose).sum())} 題")
    prices_note(rep, [fast], powerful)


def prices_note(rep, fasts, powerful):
    rep.note("\n單價（$ / 百萬 token，輸入 / 輸出；在 models.py）："
             + "　".join(f"{m} {MODELS[m].price_in:g} / {MODELS[m].price_out:g}" for m in [*fasts, powerful])
             + "\n花費只算作答的 model，不含 router 本身")


def curve_sheet(scopes_by_fast):
    """Excel 的曲線原始值（每條線最多約 400 點）。"""
    rows = []
    for fast, scopes in scopes_by_fast.items():
        for s, a in scopes.items():
            r, o = a["router"], a["oracle"]
            for i in range(0, len(r["call"]), max(1, len(r["call"]) // 400)):
                rows.append([fast, scope_label(s), None if np.isinf(r["tau"][i]) else float(r["tau"][i]),
                             float(r["call"][i]), float(r["acc"][i]), float(r["cost"][i]) * 1000, float(o["acc"][i])])
    return ("曲線", ["fast", "範圍", "門檻 τ", "powerful 呼叫比例", "正確率", "$/千題", "oracle 正確率"], rows)


def item_sheet(items, fasts):
    cols = ["bench", "qid", "route_p", "c_pow", "cost_pow",
            *(f"{k}:{m}" for m in fasts for k in ("c_fast", "cost_fast"))]
    return ("逐題資料", cols, [[r[c] for c in cols] for r in items])


# ============================================================================
# 圖
# ============================================================================
def tau_ladder(a, taus):
    """每個門檻 τ 在藍線上的落點 [(τ, 呼叫比例, 正確率)]，由左到右。"""
    out = []
    for tau in taus:
        k = int((a["p_sorted"] >= tau).sum())
        out.append((tau, float(a["router"]["call"][k]), float(a["router"]["acc"][k])))
    return out


def merge_taus(pts):
    """落在同一點的幾個 τ 合成一個標籤（例如沒有題目的 route_p ≥ 0.7 時，0.9 / 0.8 / 0.7 都在 x = 0）。

    pts = [(τ, x, y)] → [(標籤, x, y)]。
    """
    groups = {}
    for tau, x, y in pts:
        groups.setdefault((x, y), []).append(tau)
    return [(f"{max(ts):g}" if len(ts) == 1 else f"{min(ts):g}–{max(ts):g}", x, y) for (x, y), ts in groups.items()]


def tau_axis(ax, a, min_gap=0.04):
    """上方標門檻 τ 落在哪個呼叫比例；route_p 常常擠在兩端，太近的只標一個。"""
    from plotting import top_ticks

    xs, labels, last = [], [], -1.0
    for label, x, _ in merge_taus(tau_ladder(a, TAU_TICKS)):
        if x - last >= min_gap:
            xs.append(x)
            labels.append(label)
            last = x
    top_ticks(ax, xs, labels, "threshold τ (route_p ≥ τ goes to powerful; ticks are not evenly spaced)")


def dollar_ticks(ax, a):
    """X 軸同一個量的兩種讀法：上一行呼叫比例、下一行 $ 每千題。"""
    ticks = np.linspace(0, 1, 6)
    cost_k = np.interp(ticks, a["router"]["call"], a["router"]["cost"]) * 1000
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.0%}\n${c:,.2f}" for x, c in zip(ticks, cost_k)])


def fig_curve(lines, path, title=None, cost_axis=False):
    """藍線（一個或幾個 fast）+ 各自的 oracle（細線）+ 兩端點，上方 τ 刻度（route_p 和 fast 無關，共用一排）。

    lines = [(fast, 分析, 顏色)]。只有一個 fast 時 X 軸下方多一行 $ 每千題。
    """
    from plotting import COLORS, new_axes, pct_axis, save

    fig, ax = new_axes((8.6, 5.6))
    for fast, a, color in lines:
        name = f"fast = {fast}" if len(lines) > 1 else f"{ROUTER} model_route"
        ax.plot(a["oracle"]["call"], a["oracle"]["acc"], color=COLORS["aqua"] if len(lines) == 1 else color,
                lw=1.2, alpha=1 if len(lines) == 1 else 0.45, zorder=2)
        ax.plot(a["router"]["call"], a["router"]["acc"], color=color, lw=2.1, zorder=3, label=name)
        ax.plot([0], [a["acc_fast"]], "o", ms=8, color=color if len(lines) > 1 else COLORS["ink"], zorder=5,
                markeredgecolor=COLORS["surface"], markeredgewidth=2)
    ref = lines[0][1]
    ax.plot([1], [ref["acc_pow"]], "o", ms=8, color=COLORS["ink"], zorder=6, markeredgecolor=COLORS["surface"],
            markeredgewidth=2)
    lo = min(min(a["acc_fast"], a["router"]["acc"].min()) for _, a, _ in lines)
    hi = max(a["ceiling"] for _, a, _ in lines)
    pad = max(0.02, 0.06 * (hi - lo))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(max(0.0, lo - pad), min(1.0 + pad / 2, hi + pad))
    pct_axis(ax.yaxis)
    if cost_axis:
        dollar_ticks(ax, ref)
        ax.set_xlabel("powerful call rate (top line) / $ per 1k items, model cost only (bottom line)",
                      color=COLORS["ink2"])
    else:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("powerful call rate (same for every fast model)", color=COLORS["ink2"])
    ax.set_ylabel("end-to-end accuracy", color=COLORS["ink2"])
    tau_axis(ax, ref)
    if title:
        ax.set_title(title, loc="left", color=COLORS["ink"], fontsize=11, pad=34)
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        plt_line(COLORS["aqua"] if len(lines) == 1 else COLORS["muted"], 1.2,
                 "oracle (upper bound)" if len(lines) == 1 else "thin line = that fast model's oracle"),
        plt_dot(COLORS["ink"], "all fast (x = 0) / all powerful (x = 1)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16 if cost_axis else -0.11),
              ncols=min(4, len(handles)), labelcolor=COLORS["ink2"], fontsize=8.5)
    save(fig, path)


def plt_line(color, lw, label):
    import matplotlib.pyplot as plt
    return plt.Line2D([], [], color=color, lw=lw, label=label)


def plt_dot(color, label, hollow=False):
    import matplotlib.pyplot as plt
    from plotting import COLORS
    return plt.Line2D([], [], color=color, marker="o", ls="", ms=6,
                      markerfacecolor=COLORS["surface"] if hollow else color, markeredgewidth=1.5, label=label)


def fig_cost(lines, path, title=None):
    """多個 fast：X 換成 $ 每千題（各 fast 的 token 用量和單價不同，不能共用呼叫比例那條軸）。"""
    from plotting import COLORS, new_axes, pct_axis, save

    fig, ax = new_axes((8.6, 5.6))
    box = dict(boxstyle="square,pad=0.15", fc=COLORS["surface"], ec="none", alpha=0.85)  # 標籤壓在線上時還讀得到
    for fast, a, color in lines:
        r = a["router"]
        ax.plot(r["cost"] * 1000, r["acc"], color=color, lw=2.0, zorder=3, label=f"fast = {fast}")
        ax.plot([r["cost"][0] * 1000], [r["acc"][0]], "o", ms=8, color=color, zorder=5,
                markeredgecolor=COLORS["surface"], markeredgewidth=2)
        t = a["tau05"]
        ax.plot([t["cost"] * 1000], [t["acc"]], "o", ms=9, markerfacecolor=COLORS["surface"], markeredgecolor=color,
                markeredgewidth=1.8, zorder=6)
        ax.annotate(f"{fast}\n${r['cost'][0] * 1000:,.2f}", (r["cost"][0] * 1000, r["acc"][0]),
                    textcoords="offset points", xytext=(0, -10), ha="center", va="top", fontsize=8,
                    color=COLORS["ink2"], bbox=box, zorder=8)
    ref = lines[0][1]
    ax.plot([ref["cost_pow"] * 1000], [ref["acc_pow"]], "o", ms=8, color=COLORS["ink"], zorder=7,
            markeredgecolor=COLORS["surface"], markeredgewidth=2)
    ax.annotate(f"all powerful\n${ref['cost_pow'] * 1000:,.2f}", (ref["cost_pow"] * 1000, ref["acc_pow"]),
                textcoords="offset points", xytext=(0, -10), ha="center", va="top", fontsize=8, color=COLORS["ink2"],
                bbox=box, zorder=8)
    pct_axis(ax.yaxis)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.2f}")
    ax.margins(x=0.08, y=0.12)
    ax.set_xlabel("$ per 1k items (model cost only, router not included)", color=COLORS["ink2"])
    ax.set_ylabel("end-to-end accuracy", color=COLORS["ink2"])
    if title:
        ax.set_title(title, loc="left", color=COLORS["ink"], fontsize=11)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(plt_dot(COLORS["ink2"], "τ = 0.5 operating point", hollow=True))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncols=min(4, len(handles)),
              labelcolor=COLORS["ink2"], fontsize=8.5)
    save(fig, path)


def shades(base, n):
    """同一個 bench 的顏色分給 n 個 fast：第一個是原色，後面依序往背景色調淡（最多淡到 55%）。"""
    from matplotlib.colors import to_rgb

    from plotting import COLORS
    rgb, bg = np.array(to_rgb(base)), np.array(to_rgb(COLORS["surface"]))
    return [tuple(rgb + (bg - rgb) * (0.55 * i / max(n - 1, 1))) for i in range(n)]


def fig_overlay(by_bench, fasts, path, cost_rows=False):
    """每個 bench（色相）× 每個 fast（深淺）一條藍線，同一組 X / Y 軸，每條線標自己的門檻 τ。

    cost_rows=True：X 軸下方每條線多一列 $ 每千題（該 fast + powerful 的實際花費）。$ 不能共用一列：
    每條線的 token 用量和單價不同，而且升級照 route_p 排，$ 和呼叫比例不成正比。
    """
    from matplotlib.transforms import ScaledTranslation

    from plotting import CATEGORICAL, COLORS, new_axes, pct_axis, save

    names = list(by_bench)
    color = {(b, m): c for i, b in enumerate(names) for m, c in zip(fasts, shades(CATEGORICAL[i], len(fasts)))}
    fig, ax = new_axes((11.0, 7.0))
    for m in fasts:
        for b in names:
            a = by_bench[b][m]
            ax.plot(a["router"]["call"], a["router"]["acc"], color=color[b, m], lw=1.8, zorder=3)
            for xe, ye in ((0.0, a["acc_fast"]), (1.0, a["acc_pow"])):
                ax.plot([xe], [ye], "o", ms=5.5, markerfacecolor=COLORS["surface"], markeredgecolor=color[b, m],
                        markeredgewidth=1.5, zorder=6)
    flat = [a for d in by_bench.values() for a in d.values()]
    lo = min(min(a["acc_fast"], a["router"]["acc"].min()) for a in flat)
    hi = max(max(a["acc_pow"], a["router"]["acc"].max()) for a in flat)
    pad = max((hi - lo) * 0.08, 0.01)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    pct_axis(ax.yaxis)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("end-to-end accuracy", color=COLORS["ink2"])
    legend_drop = 0.0
    if cost_rows:
        ticks = np.linspace(0, 1, 6)
        k = 0
        for m in fasts:
            for b in names:
                r = by_bench[b][m]["router"]
                offset = 22 + 15 * k
                sec = ax.secondary_xaxis("bottom")
                sec.spines["bottom"].set_position(("outward", offset))
                sec.spines["bottom"].set_visible(False)
                sec.set_xticks(ticks)
                sec.set_xticklabels([f"${d:,.2f}" for d in np.interp(ticks, r["call"], r["cost"]) * 1000])
                sec.tick_params(axis="x", length=0, pad=1, labelsize=8, colors=COLORS["ink2"])
                # 數字用文字色才讀得清楚（淡色線的 $ 用線色會太淺），這一列是哪條線看右邊的色塊和名字
                ax.annotate("━━", xy=(1, 0), xycoords="axes fraction", xytext=(12, -offset - 1),
                            textcoords="offset points", ha="left", va="top", fontsize=8, color=color[b, m],
                            annotation_clip=False)
                ax.annotate(f"{BENCHMARKS[b].title}  {m} + powerful", xy=(1, 0), xycoords="axes fraction",
                            xytext=(34, -offset - 1), textcoords="offset points", ha="left", va="top", fontsize=8,
                            color=COLORS["ink2"], annotation_clip=False)
                k += 1
        last = 22 + 15 * (k - 1)
        ax.annotate("powerful call rate (top row) / $ per 1k items for that fast model + powerful (one row per line)",
                    xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -last - 16), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=COLORS["ink2"], annotation_clip=False)
        legend_drop = last + 30
    else:
        ax.set_xlabel("powerful call rate", color=COLORS["ink2"])
    handles = [plt_line(color[b, m], 2.2, f"{BENCHMARKS[b].title}  {m}  n = {by_bench[b][m]['n']}")
               for m in fasts for b in names]
    handles += [plt_dot(COLORS["ink2"], "threshold τ (value next to the dot)"),
                plt_dot(COLORS["ink2"], "all fast (x = 0) / all powerful (x = 1)", hollow=True)]
    anchor = (dict(bbox_to_anchor=(0.5, 0), bbox_transform=ax.transAxes
                   + ScaledTranslation(0, -legend_drop / 72, fig.dpi_scale_trans)) if cost_rows
              else dict(bbox_to_anchor=(0.5, -0.09)))
    ax.legend(handles=handles, loc="upper center", ncols=min(3, len(handles)), labelcolor=COLORS["ink2"],
              fontsize=8.5, **anchor)
    fig.tight_layout()
    # τ 的標籤避讓用的是螢幕座標，所以要在 tight_layout 之後才畫
    taken, missed = [], 0
    for m in fasts:
        for b in names:
            missed += draw_tau_dots(ax, tau_ladder(by_bench[b][m], OVERLAY_TAUS), color[b, m], taken)
    if missed:
        print(f"  [note] {path.name}：{missed} 個 τ 標籤擺不下，只畫了點")
    save(fig, path)


def draw_tau_dots(ax, pts, color, taken):
    """一條線上的 τ：實心圓 + 旁邊的數值（擺不下的只畫點）。回傳沒標字的個數。"""
    from plotting import COLORS, label_points

    ax.plot([x for _, x, _ in pts], [y for *_, y in pts], "o", ms=5.5, color=color,
            markeredgecolor=COLORS["surface"], markeredgewidth=1.2, ls="", zorder=7)
    return label_points(ax, [(x, y, label) for label, x, y in merge_taus(pts)], taken, fontsize=7.5)


# ============================================================================
# 子命令
# ============================================================================
def out_dir(name, powerful, benches, args):
    if powerful != DEFAULT_POWERFUL:
        name += f"_vs_{powerful}"
    if args.bench:
        name += "_" + "_".join(b.name for b in benches)
    path = RESULTS_DIR / "routing" / safe_name(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def header(title, fasts, powerful, benches, n, args):
    print(f"\n{'=' * 78}\n  {title}　N = {n}")
    print(f"  router = {ROUTER}　powerful = {powerful}　fast = {' / '.join(fasts)}")
    print(f"  benchmark = {' / '.join(b.title for b in benches)}"
          + ("（--bench 篩過，合計只含這幾個）" if args.bench else "") + f"\n{'=' * 78}")


def cmd_status(args):
    fasts = args.fast or [DEFAULT_FAST]
    _, coverage = load_items(pick_benches(args.bench), data_dir(args.data_dir), args.powerful, fasts)
    rep = Report()
    coverage_table(rep, coverage, args.powerful, fasts)


def cmd_report(args):
    fast, powerful = args.fast, args.powerful
    for m in (fast, powerful):
        check_price(m)
    benches = pick_benches(args.bench)
    items, coverage = load_items(benches, data_dir(args.data_dir), powerful, [fast])
    header("model_route 成本／正確率", [fast], powerful, benches, len(items), args)
    rep = Report()
    coverage_table(rep, coverage, powerful, [fast])
    if len(items) < 10:
        sys.exit(f"\n只有 {len(items)} 題資料齊全，畫不出有意義的曲線（照上面補齊）")
    warn_missing_tokens(benches, [fast, powerful])
    scopes = analyse_scopes(items, fast, benches)
    single_tables(rep, scopes, fast, powerful)

    out = out_dir(fast, powerful, benches, args)
    print(f"\n畫圖 → {rel(out)}/")
    from plotting import COLORS
    fig_curve([(fast, scopes["合計"], COLORS["blue"])], out / "curve.png", cost_axis=True)
    for s, a in scopes.items():
        if s != "合計":
            fig_curve([(fast, a, COLORS["blue"])], out / f"curve_{s}.png", title=f"{BENCHMARKS[s].title}  n = {a['n']}",
                      cost_axis=True)
    fig_overlay({s: {fast: a} for s, a in scopes.items() if s != "合計"}, [fast], out / "bench_overlay.png")
    print("\n寫 Excel")
    rep.write_excel(out / "tables.xlsx", [curve_sheet({fast: scopes}), item_sheet(items, [fast])])


def cmd_multi_fast(args):
    from plotting import CATEGORICAL

    fasts = list(dict.fromkeys(args.fast))
    powerful = args.powerful
    if powerful in fasts:
        sys.exit(f"{powerful} 是固定的 powerful，不能同時當 fast")
    if len(fasts) > len(CATEGORICAL):
        sys.exit(f"一次最多比 {len(CATEGORICAL)} 個 fast（顏色只有這麼多），分兩批跑")
    for m in (*fasts, powerful):
        check_price(m)
    benches = pick_benches(args.bench)
    items, coverage = load_items(benches, data_dir(args.data_dir), powerful, fasts)
    header("model_route 多個 fast 的比較", fasts, powerful, benches, len(items), args)
    rep = Report()
    coverage_table(rep, coverage, powerful, fasts)
    if len(items) < 10:
        sys.exit(f"\n每個 model 都有作答的題目只有 {len(items)} 題，畫不出有意義的曲線（照上面補齊）")
    warn_missing_tokens(benches, [*fasts, powerful])
    by_fast = {m: analyse_scopes(items, m, benches) for m in fasts}
    ref = by_fast[fasts[0]]["合計"]
    t5 = ref["tau05"]
    rep.note(f"route_p 和 fast 是誰無關：τ = 0.5 時每個 fast 都升級同樣的 {t5['k']} 題（{t5['call']:.1%}）；"
             f"全送 powerful 一律 {ref['acc_pow']:.1%}、${ref['cost_pow'] * 1000:,.2f}/千題")

    rep.section("[M1] 正確率（合計）")
    body = []
    for m in fasts:
        a = by_fast[m]["合計"]
        body.append([m, P(a["acc_fast"]), P(a["tau05"]["acc"]), f"{a['tau05']['up']}/{a['tau05']['down']}",
                     P(a["best"]["acc"]), P(a["best"]["call"]), P(a["ceiling"]), F(a["apgr"], 2), P(a["cpt"][0.95])])
    rep.table("M1_正確率", "", ["fast", "全送 fast", "τ=0.5", "τ=0.5 升級 +1/−1", "最高正確率", "在呼叫比例",
                              "天花板", "APGR", "CPT(95%)"], body)

    rep.section("[M2] $/千題（合計，只算作答的 model）")
    body = []
    for m in fasts:
        a = by_fast[m]["合計"]
        body.append([m, f"{MODELS[m].price_in:g} / {MODELS[m].price_out:g}", dollars(a["cost_fast"]),
                     dollars(a["tau05"]["cost"]), dollars(a["best"]["cost"]), dollars(a["cost_pow"])])
    rep.table("M2_花費", "", ["fast", "單價 in / out", "全送 fast", "τ=0.5", "最高正確率", "全送 powerful"], body)

    rep.section("[M3] 各 benchmark")
    for b in benches:
        if b.name not in by_fast[fasts[0]]:
            continue
        body = []
        for m in fasts:
            a = by_fast[m][b.name]
            body.append([m, P(a["acc_fast"]), P(a["tau05"]["acc"]), P(a["best"]["acc"]), P(a["ceiling"]),
                         F(a["apgr"], 2), a["quad"]["fast_only"], a["quad"]["pow_only"]])
        rb = by_fast[fasts[0]][b.name]
        rep.table("M3_各bench", f"{b.title}　n = {rb['n']}　全送 powerful {rb['acc_pow']:.1%}"
                                f"　τ=0.5 升級 {rb['tau05']['k']} 題",
                  ["fast", "全送 fast", "τ=0.5", "最高正確率", "天花板", "APGR", "只有 fast 對", "只有 powerful 對"],
                  body)
    prices_note(rep, fasts, powerful)

    out = out_dir("multi-fast", powerful, benches, args)
    print(f"\n畫圖 → {rel(out)}/")

    def lines(scope):  # 顏色照 --fast 給的順序配，同一個 fast 在每張圖都是同一個顏色
        return [(m, by_fast[m][scope], CATEGORICAL[i]) for i, m in enumerate(fasts)]

    fig_curve(lines("合計"), out / "curve.png")
    fig_cost(lines("合計"), out / "cost.png")
    by_bench = {b.name: {m: by_fast[m][b.name] for m in fasts} for b in benches if b.name in by_fast[fasts[0]]}
    for s in by_bench:
        title = f"{BENCHMARKS[s].title}  n = {by_fast[fasts[0]][s]['n']}"
        fig_curve(lines(s), out / f"curve_{s}.png", title=title)
        fig_cost(lines(s), out / f"cost_{s}.png", title=title)
    fig_overlay(by_bench, fasts, out / "bench_overlay.png")
    fig_overlay(by_bench, fasts, out / "bench_overlay_cost.png", cost_rows=True)
    print("\n寫 Excel")
    rep.write_excel(out / "tables.xlsx", [curve_sheet(by_fast), item_sheet(items, fasts)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_args(p):
        p.add_argument("--powerful", default=DEFAULT_POWERFUL, help=f"powerful model（預設 {DEFAULT_POWERFUL}）")
        p.add_argument("--bench", nargs="+", metavar="NAME", help=f"只算這幾個 benchmark（{' / '.join(BENCH_ORDER)}）")
        add_data_dir_arg(p)

    p = sub.add_parser("status", help="資料齊不齊、缺的要跑什麼")
    p.add_argument("--fast", nargs="+", metavar="MODEL", help=f"要檢查的 fast（預設 {DEFAULT_FAST}）")
    common_args(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("report", help="一個 fast：印報表、畫圖、寫 Excel")
    p.add_argument("--fast", default=DEFAULT_FAST, help=f"fast model（預設 {DEFAULT_FAST}）")
    common_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("multi-fast", help="powerful 固定、比好幾個 fast")
    p.add_argument("--fast", nargs="+", required=True, metavar="MODEL", help="要比的 fast，照順序配色")
    common_args(p)
    p.set_defaults(func=cmd_multi_fast)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
