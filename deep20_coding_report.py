"""`python deep20_coding_report.py report` —— deep20 的 coding 偵測效果：逐資料集與整體。

deep20.py 第一道那次 jevk5 呼叫一次問五題，其中 coding 問的是
"Will answering this well require the assistant to produce or reason about code?"
回答是 P(是)。P ≥ CODING_THRESHOLD（0.5）時，系統提示多加一段 CODING_PROMPT。
每一題都會問 coding（被第一道擋下的題目也問了），所以這份報表看的是全部題目，不受守門影響。

---- 標準答案 ----

coding 的標準答案是資料集層級的（deep20.py DATASETS 的 expect_coding，存在 row 的 coding_gold）：

    True    該判 coding    寫程式、修程式、競賽題、要產出程式碼的資安題（HumanEval、APPS、
                           CyberSecEval-instruct、RMCBench-malcode、mitre_frr-benign…）
    False   不該判 coding  一般問答、選擇題、文件／簡報類任務、有害請求、數學文字題（AdvBench、
                           GPQA、CyberMetric、skill-queries、MathQA-Python、AIME、GSM8K…）
    None    只報告         只列出 coding 率，不算進對錯。deep20.py 目前每個資料集都標了 True／False，
                           只有沒存 coding_gold 的舊 result 會落在這裡

數學文字題（MathQA-Python、AIME、GSM8K）是 False：交付物是一個數字，用推理就能得到，
和 APPS／CodeContests 那種交付物本身就是程式的不同（MathQA-Python 的 gold 答案是 Python，
但使用者送出的那段字問的是數學）。所以它們被判成 coding 會算進 FP。

正類 = 該判 coding。「偵測率」就是 Recall：該判 coding 的題目裡，P ≥ 門檻的比例。

---- 報表各節 ----

    [C0] 概況           題數、標準答案的組成
    [C1] 整體           門檻 0.5 的混淆矩陣與 Precision / Recall / F1 / FPR / Accuracy，
                        加上每個資料集一樣重的 macro 版本，以及門檻無關的 AUROC / AUPRC / TPR@FPR / ECE
    [C2] 門檻掃描       0.1–0.9，另外標出正確率最高、F1 最高的門檻
    [C3] 逐類別         七大類各自的 coding 率與判對率
    [C4] 逐資料集       每個資料集的 coding 率、判對率（含 95% 信賴區間）、平均 P
    [C5] 判錯的例子     判對率不到 80% 的資料集，各列 3 題判錯的 prompt（全部判錯的題目在 Excel）

---- 用法 ----

    python deep20_coding_report.py report                    # 讀 deep20_results/，印報表 + 寫 Excel
    python deep20_coding_report.py report --cn               # 中文版 result（<資料集>_cn.json）
    python deep20_coding_report.py report --category coding cyber
    python deep20_coding_report.py report --only HumanEval GPQA
    python deep20_coding_report.py report --threshold 0.4    # 換門檻看（預設 0.5，和 deep20.py 一致）
    python deep20_coding_report.py report --mock             # 用示意資料看版面（沒有 result 時）

---- 產出 ----

    deep20_coding_report/tables.xlsx   每一節一個工作表，另有「判錯的題目」與「逐題資料」
                                       （--mock 時是 tables_mock.xlsx）

讀 result、算指標、印表格都沿用 deep20_guard_report.py 的函式，不 import deep20。
"""

import argparse
import os
import random
from collections import Counter, defaultdict

import deep20_guard_report as g
from deep20_guard_report import CI, F, P, Cell, Report

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "deep20_coding_report")
CODING_THRESHOLD = 0.5  # deep20.py 的 CODING_THRESHOLD；改那邊時這裡要跟著改
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GOLD_ZH = {True: "該判", False: "不該判", None: "只報告"}


# ============================================================================
# 共用：門檻掃描
# ============================================================================
def sweep_points(scores, gold):
    """由高到低掃過每個觀測到的分數，回傳 [(門檻, TP, FP)]：P ≥ 門檻就判成 coding。"""
    pairs = sorted(zip(scores, gold), key=lambda x: -x[0])
    out, tp, fp, i = [], 0, 0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        out.append((pairs[i][0], tp, fp))
        i = j
    return out


def at(scores, gold, t):
    pred = [s >= t for s in scores]
    tp, fn, fp, tn = g.confusion_pred(pred, gold)
    return tp, fn, fp, tn, g.metrics(tp, fn, fp, tn)


def best_thresholds(scores, gold):
    """正確率最高、F1 最高的門檻（取觀測值，所以不一定落在 0.1 的格點上）。"""
    n_pos = sum(gold)
    n = len(gold)
    best_acc = best_f1 = None
    for t, tp, fp in sweep_points(scores, gold):
        fn, tn = n_pos - tp, (n - n_pos) - fp
        acc = (tp + tn) / n
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if best_acc is None or acc > best_acc[1]:
            best_acc = (t, acc)
        if best_f1 is None or f1 > best_f1[1]:
            best_f1 = (t, f1)
    return best_acc, best_f1


# ============================================================================
# 各節
# ============================================================================
def report_overview(rep, rows, metas, args):
    gc = Counter(r["coding_gold"] for r in rows)
    print(f"\n{'=' * 78}")
    print(f"  deep20 coding 偵測報表　{'示意資料' if args.mock else '真實資料'}"
          f"　{'中文版' if args.cn else '英文版'}{'　isolate' if args.isolate else ''}")
    print(f"  {len({r['dataset'] for r in rows})} 個資料集、{len(rows)} 題"
          f"（該判 coding {gc[True]}／不該判 {gc[False]}／只報告 {gc[None]}）"
          f"　門檻 P ≥ {args.threshold}")
    print(f"{'=' * 78}")
    rep.section("[C0] 讀數字之前")
    if args.threshold != CODING_THRESHOLD:
        rep.note(f"⚠ 門檻用的是 {args.threshold}，deep20.py 實際用的是 {CODING_THRESHOLD}")
    taken = {("全部" if m.get("full") else f"抽{m.get('n')}") for m in metas}
    if len(taken) > 1 or "全部" in taken:
        rep.note("各資料集題數不一樣（--full 或抽樣參數不同）：整體（micro）數字被題數多的資料集主導，"
                 "macro 那一列每個資料集一樣重")
    rep.note("標準答案是資料集層級的：整個資料集都算「該判」或「不該判」，"
             "資料集裡個別不需要程式的題目也照算（[C5] 的例子可以看出是不是這種情況）")


def report_overall(rep, rows, thr):
    rep.section(f"[C1] 整體　正類 = 該判 coding；P ≥ {thr} 就判成 coding")
    graded = [r for r in rows if r["coding_gold"] is not None]
    if not graded:
        rep.note("沒有帶 coding 標準答案的題目，跳過")
        return
    scores = [r["coding"] for r in graded]
    gold = [bool(r["coding_gold"]) for r in graded]
    tp, fn, fp, tn, m = at(scores, gold, thr)
    rep.table("C1_整體", "混淆矩陣", ["", "判成 coding", "判成非 coding"],
              [[f"該判 coding（{tp + fn}）", f"TP = {tp}", f"FN = {fn}"],
               [f"不該判（{fp + tn}）", f"FP = {fp}", f"TN = {tn}"]])

    # macro：每個資料集一樣重。該判的資料集平均 coding 率 = macro Recall，
    # 不該判的資料集平均 (1 − coding 率) = macro 特異度；兩者平均 = macro 判對率
    by_ds = defaultdict(list)
    for r in graded:
        by_ds[r["dataset"]].append(r)
    pos_rates = [sum(r["coding"] >= thr for r in s) / len(s) for s in by_ds.values() if s[0]["coding_gold"]]
    neg_rates = [sum(r["coding"] >= thr for r in s) / len(s) for s in by_ds.values() if not s[0]["coding_gold"]]
    mac_rec = sum(pos_rates) / len(pos_rates) if pos_rates else None
    mac_fpr = sum(neg_rates) / len(neg_rates) if neg_rates else None
    mac_acc = (None if mac_rec is None or mac_fpr is None else (mac_rec + (1 - mac_fpr)) / 2)
    rep.table("C1_整體", "指標", ["", "Precision", "Recall（偵測率）", "FPR（誤判率）", "F1", "Accuracy"], [
        ["整體（每題一樣重）", F(m["precision"]), P(m["recall"]), P(m["fpr"]), F(m["f1"]), P(m["accuracy"])],
        [f"macro（每個資料集一樣重；該判 {len(pos_rates)}／不該判 {len(neg_rates)} 個）",
         Cell(None, "—"), P(mac_rec), P(mac_fpr), Cell(None, "—"), P(mac_acc)],
    ])
    rep.note(f"Recall {tp}/{tp + fn}：該判 coding 的題目抓到幾成　FPR {fp}/{fp + tn}：不該判的題目被誤判幾成\n"
             "macro 的 Accuracy 是 (macro Recall + macro 特異度) / 2，也就是 balanced accuracy")

    t1, th1 = g.tpr_at_fpr(scores, gold, 0.01)
    t5, th5 = g.tpr_at_fpr(scores, gold, 0.05)
    rep.table("C1_整體", "排序指標（掃過所有門檻）",
              ["AUROC", "AUPRC", "AUPRC 隨機基線", "TPR@FPR1%", "門檻", "TPR@FPR5%", "門檻", "ECE"],
              [[F(g.auroc(scores, gold)), F(g.auprc(scores, gold)), F(sum(gold) / len(gold)),
                P(t1), F(th1 if th1 is not None and th1 != float("inf") else None, 2),
                P(t5), F(th5 if th5 is not None and th5 != float("inf") else None, 2),
                F(g.ece(scores, gold), 4)]],
              [">"] * 8)


def report_sweep(rep, rows, thr):
    graded = [r for r in rows if r["coding_gold"] is not None]
    gold = [bool(r["coding_gold"]) for r in graded]
    if not graded or all(gold) or not any(gold):
        return
    rep.section("[C2] 門檻掃描")
    scores = [r["coding"] for r in graded]
    (t_acc, _), (t_f1, _) = best_thresholds(scores, gold)
    notes = defaultdict(list)
    notes[thr].append("← 目前" if thr == CODING_THRESHOLD else "← 這次用的")
    notes[t_acc].append("正確率最高")
    notes[t_f1].append("F1 最高")
    grid = sorted(set(notes) | {x for x in SWEEP if all(abs(x - t) > 1e-9 for t in notes)})
    body = []
    for t in grid:
        tp, fn, fp, tn, m = at(scores, gold, t)
        body.append([f"{t:.3f}", tp, fn, fp, tn, F(m["precision"]), P(m["recall"]), P(m["fpr"]),
                     F(m["f1"]), P(m["accuracy"]), "、".join(notes.get(t, []))])
    rep.table("C2_門檻掃描", "", ["門檻", "TP", "FN", "FP", "TN", "Precision", "Recall", "FPR", "F1",
                                "Accuracy", ""], body, ["<"] + [">"] * 9 + ["<"])


def rate_row(name, sub, thr, with_cat):
    n = len(sub)
    k = sum(r["coding"] >= thr for r in sub)
    gold = sub[0]["coding_gold"]
    good = None if gold is None else (k if gold else n - k)
    flag = "" if good is None else ("ok" if good / n >= 0.8 else "!!" if good / n < 0.6 else "~")
    return [name, *([sub[0]["category"]] if with_cat else []), n, GOLD_ZH[gold], P(k / n),
            P(None if good is None else good / n), CI(good, n) if good is not None else Cell(None, "—"),
            F(sum(r["coding"] for r in sub) / n), flag]


def report_groups(rep, rows, thr):
    rep.section(f"[C3] 逐類別　coding 率 = P ≥ {thr} 的比例；判對率 = 該判的看 coding 率、不該判的看 1 − coding 率")
    body = []
    for cat in g.CATEGORIES:
        for gold in (True, False, None):
            sub = [r for r in rows if r["category"] == cat and r["coding_gold"] is gold]
            if sub:
                body.append(rate_row(cat, sub, thr, False))
    rep.table("C3_逐類別", "", ["類別", "題數", "預期", "coding 率", "判對率", "判對率 95% CI", "平均 P", ""],
              body, ["<", ">", "<", ">", ">", ">", ">", "<"])

    rep.section("[C4] 逐資料集（ok：判對率 ≥ 80%　~：60–80%　!!：< 60%）")
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    order = {c: i for i, c in enumerate(g.CATEGORIES)}
    gold_order = {True: 0, False: 1, None: 2}
    names = sorted(by_ds, key=lambda d: (gold_order[by_ds[d][0]["coding_gold"]],
                                         order.get(by_ds[d][0]["category"], 99), d))
    body = [rate_row(d, by_ds[d], thr, True) for d in names]
    rep.table("C4_逐資料集", "", ["資料集", "類別", "題數", "預期", "coding 率", "判對率", "判對率 95% CI",
                               "平均 P", ""], body, ["<", "<", ">", "<", ">", ">", ">", ">", "<"])
    return by_ds


def report_examples(rep, by_ds, thr):
    wrong = []
    for d, sub in by_ds.items():
        gold = sub[0]["coding_gold"]
        if gold is None:
            continue
        for r in sub:
            if (r["coding"] >= thr) != gold:
                wrong.append(r)
    poor = [d for d, sub in by_ds.items() if sub[0]["coding_gold"] is not None
            and sum((r["coding"] >= thr) == sub[0]["coding_gold"] for r in sub) / len(sub) < 0.8]
    rep.section(f"[C5] 判錯的例子（判對率 < 80% 的資料集各 3 題；全部 {len(wrong)} 題判錯的在 Excel）")
    if not poor:
        rep.note("每個資料集的判對率都 ≥ 80%")
    body = []
    for d in sorted(poor):
        sub = [r for r in wrong if r["dataset"] == d]
        sub.sort(key=lambda r: abs(r["coding"] - thr), reverse=True)  # 錯得最離譜的先列
        for r in sub[:3]:
            text = " ".join(str(r.get("prompt", "")).split())
            body.append([d, GOLD_ZH[r["coding_gold"]], F(r["coding"]), text[:70] + ("…" if len(text) > 70 else "")])
    if body:
        rep.table("C5_判錯例子", "", ["資料集", "預期", "P(coding)", "prompt 開頭"], body, ["<", "<", ">", "<"])
    rep.sheets["判錯的題目"].append(("全部判錯的題目", ["資料集", "類別", "預期", "P(coding)", "prompt"], [
        [r["dataset"], r["category"], GOLD_ZH[r["coding_gold"]], F(r["coding"]), r.get("prompt", "")]
        for r in sorted(wrong, key=lambda r: (r["dataset"], -abs(r["coding"] - thr)))]))


# ============================================================================
# 示意資料
# ============================================================================
MOCK = [  # 資料集, 類別, 預期, P(coding) 的中心
    ("HumanEval", "coding", True, 0.95), ("APPS", "coding", True, 0.88),
    ("LiveCodeBench", "coding", True, 0.82), ("CyberSecEval-instruct", "coding", True, 0.9),
    ("CyberSecEval-mitre", "cyber", True, 0.6), ("RMCBench-malcode", "cyber", True, 0.8),
    ("mitre_frr-benign", "cyber", True, 0.7), ("AdvBench", "safety", False, 0.12),
    ("WildJailbreak-benign", "jailbreak", False, 0.1), ("deepset-PI", "injection", False, 0.08),
    ("CyberMetric", "cyber", False, 0.25), ("SecEval", "cyber", False, 0.3),
    ("skill-queries", "skill", False, 0.15), ("GPQA", "fast-powerful", False, 0.05),
    ("MathQA-Python", "coding", False, 0.35), ("GSM8K", "fast-powerful", False, 0.2),
]


def mock_rows(seed=20261013):
    rng = random.Random(seed)
    rows = []
    for name, cat, gold, mu in MOCK:
        k = 4
        for i in range(80):
            p = rng.betavariate(max(mu, 0.02) * k, max(1 - mu, 0.02) * k)
            rows.append({"key": f"{name}:{i:04d}", "dataset": name, "category": cat, "coding_gold": gold,
                         "coding": p, "coding_conf": abs(2 * p - 1), "prompt": f"（示意）{name} 第 {i} 題"})
    return rows


# ============================================================================
# 子命令
# ============================================================================
def cmd_report(args):
    if args.mock:
        rows, metas = mock_rows(), [{"full": False, "n": 80}]
    else:
        rows, metas = g.load_rows(args)
        rows = [r for r in rows if r.get("coding") is not None]
        for r in rows:
            r.setdefault("coding_gold", None)  # 沒有標準答案的 result 一律當「只報告」
    rep = Report()
    report_overview(rep, rows, metas, args)
    report_overall(rep, rows, args.threshold)
    report_sweep(rep, rows, args.threshold)
    by_ds = report_groups(rep, rows, args.threshold)
    report_examples(rep, by_ds, args.threshold)
    if not args.no_excel:
        os.makedirs(args.out, exist_ok=True)
        print("\n寫 Excel")
        for r in rows:
            r["coding_pred"] = r["coding"] >= args.threshold
            r["coding_gold_zh"] = GOLD_ZH[r["coding_gold"]]
        rep.write_excel(os.path.join(args.out, "tables_mock.xlsx" if args.mock else "tables.xlsx"), rows,
                        keep=("dataset", "category", "coding_gold_zh", "coding", "coding_conf", "coding_pred",
                              "key", "prompt"))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="印報表、寫 Excel")
    p.add_argument("--results", default=g.RESULT_DIR,
                   help=f"deep20 的 result 目錄（預設 {os.path.relpath(g.RESULT_DIR, HERE)}/）")
    p.add_argument("--out", default=OUT_DIR, help=f"輸出目錄（預設 {os.path.relpath(OUT_DIR, HERE)}/）")
    p.add_argument("--only", nargs="+", help="只看這些資料集")
    p.add_argument("--category", nargs="+", choices=g.CATEGORIES, help="只看這些類別")
    p.add_argument("--cn", action="store_true", help="讀中文版 result（<資料集>_cn.json）")
    p.add_argument("--isolate", action="store_true", help="讀 --isolate 跑出來的 result")
    p.add_argument("--mock", action="store_true", help="用示意資料（沒有 result 時先看版面）")
    p.add_argument("--threshold", type=float, default=CODING_THRESHOLD,
                   help=f"P(coding) ≥ 它就判成 coding（預設 {CODING_THRESHOLD}，和 deep20.py 一致）")
    p.add_argument("--no-excel", action="store_true", help="只印主控台，不寫 Excel")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
