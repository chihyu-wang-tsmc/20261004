"""coding 偵測評測：21 個資料集，看 classifier 判「這題要不要寫程式」判得準不準。

    python coding_report.py run     [--classifier C] [-n N] [--only ...]   問每個資料集的題目
    python coding_report.py status  [--classifier C]                       看哪些資料集已經有結果
    python coding_report.py report  [--classifier C] [--threshold 0.5]     印報表、寫 Excel、畫 ROC
    python coding_report.py compare                                        各 classifier 的逐資料集 coding 率並排

classifier（預設 h2o-lightning-4b）：
    h2o-lightning-4b   本機 H2O-Lightning-4B 服務（要 GPU，見 README）。同一個請求順便問 model_route，
                       routing_report.py 要用
    gemini-3.7-flash   公司 AI gateway（.env 的 GEMINI_37_FLASH_BASE_URL / GEMINI_37_FLASH_TOKEN）

coding 問題是一題 Noul（文字見 classifiers.py 的 CODING）："Will answering this well require the assistant
to produce or reason about code?"，回答是 P(是)，P ≥ 0.5 就判成 coding。
資料集和標準答案見 prompt_datasets.py：17 個該判 coding、GPQA / AI2ARC / AIME / GSM8K 不該判。

結果存在 results/coding/<classifier>/<資料集>.json，報表（report.xlsx、roc.png）寫在同一個目錄。
run 預設問每個資料集的全部題目；-n 改成每個資料集固定 seed 抽幾題，試跑用。每問完 40 題存一次檔，
中斷後重跑同一個指令會從中斷處繼續；問過的題目會沿用，所以先用 -n 試、再跑全部，只會問新的題目。
"""

import argparse
import sys
from collections import Counter, defaultdict

from classifiers import CLASSIFIERS, CODING, MODEL_ROUTE, classify_many, create_classifier, user_state
from common import (CI, NA, RESULTS_DIR, Cell, F, P, Report, add_data_dir_arg, data_dir, load_results,
                    print_status, rel, run_datasets)
from metrics import auprc, auroc, confusion, ece, metrics, sweep, tpr_at_fpr
from prompt_datasets import CODING_REPORT, select

THRESHOLD = 0.5
"""P(coding) ≥ 它就判成 coding（和上線時加 coding 系統提示的門檻一樣）。"""
GOLD = {True: "coding", False: "非coding"}
LABELS = {
    # classifier → 逐資料集表格最後一欄的標題
    "h2o-lightning-4b": "Jev (開源) 判斷coding 率",
    "gemini-3.7-flash": "Gemini-3.7-flash 判斷coding 率",
}
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ORDER = {d.name: i for i, d in enumerate(CODING_REPORT)}


def result_dir(classifier):
    return RESULTS_DIR / "coding" / classifier


def questions_for(classifier):
    """h2o 一個請求多問一題幾乎不多花時間，順便問 model_route（routing_report.py 要用）；
    Gemini 一題就是一次請求，只問 coding。"""
    if classifier == "h2o-lightning-4b":
        return {"coding": CODING, "model_route": MODEL_ROUTE}
    return {"coding": CODING}


# ============================================================================
# run / status
# ============================================================================
def to_row(answers):
    row = {"coding": answers["coding"]["noul"]}
    if "model_route" in answers:
        row["route"] = answers["model_route"]["choice"]
        row["route_p"] = answers["model_route"]["probabilities"]["powerful"]
    return row


def cmd_run(args):
    datasets = select(args.only, None, CODING_REPORT)
    classifier = create_classifier(args.classifier)
    questions = questions_for(args.classifier)
    print(f"classifier：{args.classifier}　問：{'、'.join(questions)}　{len(datasets)} 個資料集"
          f"　{'每個抽 ' + str(args.n) + ' 題' if args.n else '全部題目'}")

    def ask(prompts):
        out = []
        for answers in classify_many(classifier, [(user_state(p), questions) for p in prompts], args.workers):
            try:
                out.append(answers if isinstance(answers, Exception) else to_row(answers))
            except (KeyError, TypeError) as e:
                out.append(ValueError(f"答案格式不對：{e!r}"))
        return out

    def meta(d):
        return {"dataset": d.name, "category": d.category, "coding_gold": d.coding,
                "classifier": args.classifier, "questions": list(questions)}

    run_datasets(datasets, result_dir(args.classifier), data_dir(args.data_dir), ask, meta,
                 n=args.n, refresh=args.refresh)


def cmd_status(args):
    print_status(CODING_REPORT, result_dir(args.classifier))


# ============================================================================
# report
# ============================================================================
def load_rows(classifier):
    rows = []
    for meta, rs in load_results(result_dir(classifier)):
        for r in rs:
            rows.append({**r, "category": meta["category"], "gold": bool(meta["coding_gold"])})
    return rows


def by_dataset(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["dataset"]].append(r)
    return dict(sorted(groups.items(), key=lambda kv: (ORDER.get(kv[0], len(ORDER)), kv[0])))


def report_datasets(rep, rows, classifier, thr):
    rep.section(f"[C1] 逐資料集　coding 率 = P ≥ {thr} 的比例；判對率 = 該判的看 coding 率、不該判的看 1 − coding 率")
    body = []
    for name, sub in by_dataset(rows).items():
        n = len(sub)
        k = sum(r["pred"] for r in sub)
        gold = sub[0]["gold"]
        good = k if gold else n - k
        body.append([name, sub[0]["category"], GOLD[gold], n, P(good / n), CI(good, n),
                     F(sum(r["coding"] for r in sub) / n), P(k / n)])
    rep.table("C1_逐資料集", "", ["資料集", "類別", "標準答案", "題數", "判對率", "判對率 95% CI", "平均 P(coding)",
                                LABELS.get(classifier, f"{classifier} 判斷coding 率")], body, "<<<>>>>>")


def report_overall(rep, rows, thr):
    rep.section(f"[C2] 整體　正類 = 該判 coding；P ≥ {thr} 就判成 coding")
    gold = [r["gold"] for r in rows]
    scores = [r["coding"] for r in rows]
    tp, fn, fp, tn = confusion([r["pred"] for r in rows], gold)
    m = metrics(tp, fn, fp, tn)
    rep.table("C2_整體", "混淆矩陣", ["", "判成 coding", "判成非 coding"],
              [[f"該判 coding（{tp + fn}）", f"TP = {tp}", f"FN = {fn}"],
               [f"不該判（{fp + tn}）", f"FP = {fp}", f"TN = {tn}"]])
    # macro：每個資料集一樣重。該判的資料集平均 coding 率 = macro Recall；不該判的平均 coding 率 = macro FPR
    rates = {name: (sub[0]["gold"], sum(r["pred"] for r in sub) / len(sub)) for name, sub in by_dataset(rows).items()}
    pos = [rate for g, rate in rates.values() if g]
    neg = [rate for g, rate in rates.values() if not g]
    m_rec = sum(pos) / len(pos) if pos else None
    m_fpr = sum(neg) / len(neg) if neg else None
    m_acc = None if m_rec is None or m_fpr is None else (m_rec + 1 - m_fpr) / 2
    rep.table("C2_整體", "指標", ["", "Precision", "Recall（偵測率）", "FPR（誤判率）", "F1", "Accuracy"], [
        ["每題一樣重（micro）", F(m["precision"]), P(m["recall"]), P(m["fpr"]), F(m["f1"]), P(m["accuracy"])],
        [f"每個資料集一樣重（macro；該判 {len(pos)}／不該判 {len(neg)} 個）", NA, P(m_rec), P(m_fpr), NA, P(m_acc)],
    ])
    rep.note("macro 的 Accuracy = (macro Recall + 1 − macro FPR) / 2，也就是 balanced accuracy；\n"
             "--full 時各資料集題數差很多（MultiPL-E 上萬題、AIME 60 題），micro 會被大資料集主導")
    t1, th1 = tpr_at_fpr(scores, gold, 0.01)
    t5, th5 = tpr_at_fpr(scores, gold, 0.05)
    rep.table("C2_整體", "排序指標（掃過所有門檻，和選哪個門檻無關）",
              ["AUROC", "AUPRC", "AUPRC 隨機基線", "TPR@FPR 1%", "門檻", "TPR@FPR 5%", "門檻", "ECE"],
              [[F(auroc(scores, gold)), F(auprc(scores, gold)), F(sum(gold) / len(gold)),
                P(t1), F(th1, 2), P(t5), F(th5, 2), F(ece(scores, gold), 4)]], ">" * 8)


def at(rows, t):
    tp, fn, fp, tn = confusion([r["coding"] >= t for r in rows], [r["gold"] for r in rows])
    return tp, fn, fp, tn, metrics(tp, fn, fp, tn)


def report_sweep(rep, rows, thr):
    gold = [r["gold"] for r in rows]
    if all(gold) or not any(gold):
        return
    rep.section("[C3] 門檻掃描")
    n_pos, n = sum(gold), len(gold)
    best_acc = best_f1 = (None, -1.0)
    for t, tp, fp in sweep([r["coding"] for r in rows], gold):
        fn, tn = n_pos - tp, n - n_pos - fp
        best_acc = max(best_acc, (t, (tp + tn) / n), key=lambda x: x[1])
        best_f1 = max(best_f1, (t, 2 * tp / (2 * tp + fp + fn)), key=lambda x: x[1])
    notes = defaultdict(list)
    notes[thr].append("← 這次用的")
    notes[best_acc[0]].append("正確率最高")
    notes[best_f1[0]].append("F1 最高")
    body = []
    for t in sorted(set(notes) | {x for x in SWEEP if all(abs(x - y) > 1e-9 for y in notes)}):
        tp, fn, fp, tn, m = at(rows, t)
        body.append([f"{t:.3f}", tp, fn, fp, tn, F(m["precision"]), P(m["recall"]), P(m["fpr"]), F(m["f1"]),
                     P(m["accuracy"]), "、".join(notes.get(t, []))])
    rep.table("C3_門檻掃描", "", ["門檻", "TP", "FN", "FP", "TN", "Precision", "Recall", "FPR", "F1", "Accuracy", ""],
              body, "<" + ">" * 9 + "<")


def report_wrong(rep, rows, thr):
    wrong = [r for r in rows if r["pred"] != r["gold"]]
    poor = [name for name, sub in by_dataset(rows).items()
            if sum(r["pred"] == r["gold"] for r in sub) / len(sub) < 0.8]
    rep.section(f"[C4] 判錯的例子（判對率 < 80% 的資料集各 3 題；全部 {len(wrong)} 題判錯的在 Excel）")
    if not poor:
        rep.note("每個資料集的判對率都 ≥ 80%")
        return wrong
    body = []
    for name in poor:
        sub = sorted((r for r in wrong if r["dataset"] == name), key=lambda r: -abs(r["coding"] - thr))
        for r in sub[:3]:  # 錯得最離譜的先列
            text = " ".join(r["prompt"].split())
            body.append([name, GOLD[r["gold"]], F(r["coding"]), text[:70] + ("…" if len(text) > 70 else "")])
    rep.table("C4_判錯例子", "", ["資料集", "標準答案", "P(coding)", "prompt 開頭"], body, "<<><")
    return wrong


def report_languages(rep, rows):
    sub = [r for r in rows if r["category"] == "coding"]
    rep.section("[C5] 逐程式語言（coding 類；語言是 run 的時候從原始資料集記下來的）")
    if not sub:
        rep.note("這次沒有 coding 類的資料集")
        return
    rep.note("coding 類全是該判 coding 的題目，所以 coding 率就是 Recall（漏判率 = 1 − coding 率）。\n"
             "mceval-* 是反編譯題，prompt 整篇是組語，記成組語；「不限」是沒指定語言的競賽題（CodeContests）")
    by_lang = defaultdict(list)
    for r in sub:
        by_lang[r.get("lang") or "未知"].append(r)
    body = [[lang, len(g), len({r["dataset"] for r in g}), P(len(g) / len(sub)),
             P(sum(r["pred"] for r in g) / len(g)), F(sum(r["coding"] for r in g) / len(g))]
            for lang, g in sorted(by_lang.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    rep.table("C5_逐語言", f"語言彙總（{len(by_lang)} 種、{len(sub)} 題）",
              ["語言", "題數", "資料集數", "占 coding 類", "coding 率（Recall）", "平均 P"], body)
    body = []
    for name, ds_rows in by_dataset(sub).items():
        langs = defaultdict(list)
        for r in ds_rows:
            langs[r.get("lang") or "未知"].append(r)
        for i, (lang, g) in enumerate(sorted(langs.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
            body.append([name if i == 0 else "", lang, len(g), P(len(g) / len(ds_rows)),
                         P(sum(r["pred"] for r in g) / len(g)), F(sum(r["coding"] for r in g) / len(g))])
    rep.table("C5_逐語言", "逐資料集 × 語言", ["資料集", "語言", "題數", "占該資料集", "coding 率（Recall）", "平均 P"],
              body, "<<>>>>")


ROC_MARKS = tuple(round(0.1 * i, 1) for i in range(11))


def report_roc(rep, rows, thr, path):
    rep.section(f"[C6] 門檻 trade-off　X = 誤判率 FPR、Y = 偵測率 Recall；這次用的門檻是 {thr}")
    gold = [r["gold"] for r in rows]
    n_pos, n_neg = sum(gold), len(gold) - sum(gold)
    if not n_pos or not n_neg:
        rep.note(f"只有一類題目（該判 {n_pos}／不該判 {n_neg}），畫不出曲線")
        return
    body = []
    for t in sorted(set(ROC_MARKS) | {thr}, reverse=True):
        tp, fn, fp, tn, m = at(rows, t)
        body.append([f"{t:.2f}", P(m["fpr"]), P(m["recall"]), tp, fn, fp, tn, F(m["precision"]), F(m["f1"]),
                     "← 這次用的" if abs(t - thr) < 1e-9 else ""])
    rep.table("C6_門檻曲線", "τ 由大到小；越小越容易判成 coding",
              ["門檻 τ", "誤判率 FPR", "偵測率 Recall", "TP", "FN", "FP", "TN", "Precision", "F1", ""],
              body, "<" + ">" * 8 + "<")
    neg = Counter(r["dataset"] for r in rows if not r["gold"])
    rep.note(f"不該判的 {n_neg} 題：{'、'.join(f'{d} {k}' for d, k in neg.most_common())}\n"
             "兩個軸都是在同一類題目裡算的比例，不受「該判 coding 的題目佔幾成」影響")
    fig_roc(rows, thr, path)


def fig_roc(rows, thr, path):
    from plotting import COLORS, label_points, new_axes, pct_axis, save

    scores = [r["coding"] for r in rows]
    gold = [r["gold"] for r in rows]
    n_pos, n_neg = sum(gold), len(gold) - sum(gold)
    pts = [(0.0, 0.0)] + [(fp / n_neg, tp / n_pos) for _, tp, fp in sweep(scores, gold)] + [(1.0, 1.0)]
    fig, ax = new_axes((7.2, 6.0))
    ax.plot([0, 1], [0, 1], color=COLORS["muted"], lw=1, ls="--", label="Random (AUROC 0.5)")
    ax.plot([x for x, _ in pts], [y for _, y in pts], color=COLORS["blue"], lw=2,
            label=f"Coding detection (AUROC {auroc(scores, gold):.3f})")
    ax.set_xlim(-0.02, 1.02)  # 先定好軸，下面才算得出每個標籤在圖上的位置
    ax.set_ylim(-0.02, 1.02)
    # 落在同一點的門檻合成一個標籤（例如 0.6–0.8），太靠近已經標過的就只畫點
    spots = {}
    for t in ROC_MARKS:
        *_, m = at(rows, t)
        spots.setdefault((m["fpr"], m["recall"]), []).append(t)
    for x, y in spots:
        ax.plot(x, y, "o", ms=5, color=COLORS["blue"], mew=0, zorder=5)
    label_points(ax, [(x, y, f"{min(ts):.1f}" if len(ts) == 1 else f"{min(ts):.1f}–{max(ts):.1f}")
                      for (x, y), ts in spots.items()])
    *_, cur = at(rows, thr)
    ax.plot(cur["fpr"], cur["recall"], "*", ms=15, color=COLORS["orange"], mew=0, zorder=6,
            label=f"Current threshold τ = {thr:g}")
    pct_axis(ax.xaxis)
    pct_axis(ax.yaxis)
    ax.set_xlabel("False positive rate: non-coding prompts flagged as coding", color=COLORS["ink2"])
    ax.set_ylabel("Recall: coding prompts detected", color=COLORS["ink2"])
    ax.legend(loc="lower right", labelcolor=COLORS["ink2"], fontsize=8.5)
    save(fig, path)


def cmd_report(args):
    rows = load_rows(args.classifier)
    if not rows:
        sys.exit(f"{rel(result_dir(args.classifier))}/ 還沒有結果：先跑 python coding_report.py run "
                 f"--classifier {args.classifier}")
    thr = args.threshold
    for r in rows:
        r["pred"] = r["coding"] >= thr
    gold = Counter(r["gold"] for r in rows)
    print(f"\n{'=' * 78}\n  coding 偵測報表　classifier = {args.classifier}\n"
          f"  {len({r['dataset'] for r in rows})} 個資料集、{len(rows)} 題"
          f"（該判 coding {gold[True]}／不該判 {gold[False]}）　門檻 P ≥ {thr}\n{'=' * 78}")
    if thr != THRESHOLD:
        print(f"  ⚠ 門檻用的是 {thr}，上線用的是 {THRESHOLD}")
    out = result_dir(args.classifier)
    rep = Report()
    report_datasets(rep, rows, args.classifier, thr)
    report_overall(rep, rows, thr)
    report_sweep(rep, rows, thr)
    wrong = report_wrong(rep, rows, thr)
    report_languages(rep, rows)
    report_roc(rep, rows, thr, out / "roc.png")
    print("\n寫 Excel")
    cols = ("dataset", "category", "gold", "lang", "coding", "pred", "route_p", "key", "prompt")
    wrong = sorted(wrong, key=lambda r: (r["dataset"], r["coding"]))
    rep.write_excel(out / "report.xlsx", [("判錯的題目", cols, [[r.get(c) for c in cols] for r in wrong]),
                                          ("逐題資料", cols, [[r.get(c) for c in cols] for r in rows])])


# ============================================================================
# compare：幾個 classifier 的逐資料集 coding 率並排
# ============================================================================
def cmd_compare(args):
    names = args.classifiers or [c for c in CLASSIFIERS if result_dir(c).is_dir()]
    if not names:
        sys.exit("還沒有任何 classifier 的結果：先跑 python coding_report.py run --classifier <classifier>")
    scores = {c: defaultdict(dict) for c in names}  # classifier → 資料集 → {key: P(coding)}
    for c in names:
        for meta, rs in load_results(result_dir(c)):
            scores[c][meta["dataset"]] = {r["key"]: r["coding"] for r in rs}
    labels = [LABELS.get(c, f"{c} 判斷coding 率") for c in names]
    print(f"\n各 classifier 的 coding 率（P ≥ {args.threshold}），每個資料集只比每個 classifier 都問過的題目")
    rep = Report()
    body, rates = [], defaultdict(list)
    for d in CODING_REPORT:
        have = [scores[c][d.name] for c in names if scores[c].get(d.name)]
        common = set.intersection(*(set(s) for s in have)) if have else set()
        row = [d.name, GOLD[d.coding], len(common) or NA]
        for c in names:
            s = scores[c].get(d.name)
            if s and common:
                rate = sum(s[k] >= args.threshold for k in common) / len(common)
                rates[c, d.coding].append(rate)
                row.append(P(rate))
            else:
                row.append(NA)
        body.append(row)
    for gold in (True, False):
        body.append([f"平均（標準答案 = {GOLD[gold]}）", "", Cell(None, ""),
                     *(P(sum(rates[c, gold]) / len(rates[c, gold])) if rates[c, gold] else NA for c in names)])
    rep.table("比較", "", ["資料集", "標準答案", "題數", *labels], body, "<<" + ">" * (len(names) + 1))
    rep.note("「—」是那個 classifier 還沒跑這個資料集；平均是每個資料集一樣重")
    print("\n寫 Excel")
    rep.write_excel(RESULTS_DIR / "coding" / "compare.xlsx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def classifier_arg(p):
        p.add_argument("--classifier", choices=CLASSIFIERS, default="h2o-lightning-4b",
                       help="問誰（預設 h2o-lightning-4b）")

    p = sub.add_parser("run", help="問每個資料集的題目")
    classifier_arg(p)
    p.add_argument("-n", type=int, help="每個資料集固定 seed 抽幾題（預設全部）")
    p.add_argument("--only", nargs="+", metavar="DATASET", help="只跑這些資料集")
    p.add_argument("--refresh", action="store_true", help="丟掉已經問過的答案，全部重問")
    p.add_argument("--workers", type=int, default=8, help="同時送出的請求數（預設 8）")
    add_data_dir_arg(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="看哪些資料集已經有結果")
    classifier_arg(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("report", help="印報表、寫 Excel、畫 ROC")
    classifier_arg(p)
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help=f"P(coding) ≥ 它就判成 coding（預設 {THRESHOLD}）")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="各 classifier 的逐資料集 coding 率並排")
    p.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, help="要比的 classifier（預設：有結果的全部）")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
