"""用 JevBench 的題目評測一個 Jev 相容的 classifier，不透過 `python -m jevbench.cli`。

evaluate_winnow.py、evaluate_jevk5.py 共用的部分；各自只決定用哪個 classifier、預設輸出到哪裡。
做的事跟下面兩行 CLI 一樣，但改用 LangChain 的 classifier（WinnowClassifier / JevK5Classifier …）：
    python -m jevbench.cli run --tasks datasets/public/hard.jsonl --adapter typesafe ...
    python -m jevbench.cli summarize --tasks datasets/public/hard.jsonl ...

1. 讀題目（JevBench 的 datasets/public/*.jsonl，或 ~/jevk5 底下自己產生的題目檔）
2. 每題轉成 Noul / Choice / Score，用 classifier 回答
3. 答案轉成選項機率後，用 jevbench 的 score_task 評分、summarize 彙總，數字算法跟 CLI 相同
4. 寫出 results.jsonl、summary.json，並印出中文報表
"""

import argparse
import json
import os
import sys
import time

from langchain_typesafe import Choice, Noul, Score
from langchain_typesafe.types import NoulCriteria

JEVBENCH_DIR = os.path.expanduser("~/jevbench")
sys.path.insert(0, JEVBENCH_DIR)  # 直接用 jevbench 原始碼裡的評分和彙總函式，不用另外安裝
from jevbench.scoring import score_task  # noqa: E402
from jevbench.summarize import public_export, summarize  # noqa: E402
from jevbench.tasks import dataset_hash, load_jsonl  # noqa: E402

TIER_FILES = {
    "easy": "easy.jsonl",
    "standard": "original.jsonl",
    "hard": "hard.jsonl",
    "hard_cn": "hard_cn.jsonl",  # hard 的繁體中文翻譯版，labels 和標準答案不變
    # 自己出的 model-routing 題（跟 deep1_router.py 的 fast / powerful 設定相同），放在這個檔案旁邊；
    # os.path.join 遇到絕對路徑會直接用它，所以不會去 jevbench 的資料夾找
    "routing": os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing.jsonl"),
    # GPQA-Diamond 的 routing ground truth，由 gpqa_routing.py report 產生
    "gpqa": os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpqa", "gpqa_routing.jsonl"),
    # AIME 的 routing ground truth，由 aime_routing.py report 產生
    "aime": os.path.join(os.path.dirname(os.path.abspath(__file__)), "aime", "aime_routing.jsonl"),
    # LiveCodeBench 的 routing ground truth，由 lcb_routing.py report 產生
    "lcb": os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb", "lcb_routing.jsonl"),
    # GSM8K 的 routing ground truth，由 gsm8k_routing.py report 產生
    "gsm8k": os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k", "gsm8k_routing.jsonl"),
    # AI2 ARC 的 routing ground truth，由 ai2arc_routing.py report 產生
    "ai2arc": os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai2arc", "ai2arc_routing.jsonl"),
    # 自己出的 tool-risk-gating 題（AutoModeMiddleware 的 is_risky），由 tool_risk_cases.py 產生
    "tool_risk": os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_risk.jsonl"),
}
JEVBENCH_TIERS = ["easy", "standard", "hard", "hard_cn"]  # --tier all 只跑官方的四個


def to_question(task):
    """JevBench 的題目格式 → langchain_typesafe 的 Noul / Choice / Score。"""
    q = task.question
    if q["type"] == "noul":
        criteria = NoulCriteria(**q["criteria"]) if q.get("criteria") else None
        return Noul(instructions=q["instructions"], criteria=criteria)
    if q["type"] == "choice":
        return Choice(instructions=q["instructions"], criteria=q["criteria"])
    return Score(instructions=q["instructions"], criteria=q["criteria"])


def to_probs(answer, task):
    """classifier 的答案 → 每個選項的機率，對應方式跟 jevbench 的 typesafe adapter 相同。"""
    if task.question["type"] == "noul":
        return {"yes": answer.noul, "no": 1.0 - answer.noul}
    # score 的機率 key 是整數等級，jevbench 的選項是字串 "0"、"1"…
    return {str(k): v for k, v in answer.probabilities.items()}


def run_one(classifier, task):
    """問一題並評分，回傳跟 jevbench runner 相同欄位的紀錄。"""
    started = time.perf_counter()
    try:
        response = classifier.invoke(
            {"state": task.state, "questions": {"decision": to_question(task)}}
        )
        answer = response.answers["decision"]
        probs, ok, error, model, usage = (
            to_probs(answer, task), True, None, response.model, response.usage.model_dump()
        )
    except Exception as e:  # 失敗算答錯，不重試（跟 jevbench 一樣）
        probs, ok, error, model, usage = None, False, f"{type(e).__name__}: {e}", "", {}
    latency = time.perf_counter() - started
    scored = (
        score_task(probs, task)
        if ok
        else {"valid": False, "strict_valid": False, "renormalized": False, "correct": False, "predicted": None}
    )
    return {
        "task_id": task.id, "family": task.family, "split": task.split, "group": task.group,
        "status": "ok" if ok else "failed", "ok": ok,
        "valid": scored["valid"], "correct": scored["correct"], "predicted": scored.get("predicted"),
        "ordinal_ev": scored.get("ordinal_ev"), "probs": scored.get("probs"), "probs_as_returned": probs,
        "strict_valid": scored.get("strict_valid", False), "renormalized": scored.get("renormalized", False),
        "probs_source": "native", "model": model, "error": error, "schema_error": scored.get("error"),
        "latency_s": latency, "usage": usage,
        "cost_usd": 0.0, "cost_basis": "self_hosted",  # 本機服務，不計費
    }


def print_report(tier, summary):
    s = summary
    print(f"\n========== {tier}：{s['n_attempted']}/{s['n_planned']} 題，答對 {s['n_correct']} 題（{s['accuracy']:.1%}）==========")
    print("\n[整體結果]")
    rows = [
        ("n_correct / n_scorable", f"{s['n_correct']} / {s['n_scorable']}", "答對幾題 / 總題數"),
        ("accuracy", f"{s['accuracy']:.3f}", "正確率，Intelligence 從這裡算"),
        ("macro_accuracy", f"{s['macro_accuracy']:.3f}", "每個題型各算正確率再平均"),
        ("operational_success / coverage / schema_validity",
         f"{s['operational_success']:.1f} / {s['coverage']:.1f} / {s['schema_validity']:.1f}", "有回應 / 有跑到 / 格式合法"),
        ("ece.ece", f"{s['ece']['ece']:.3f}" if s.get("ece") else "-", "校準誤差，越小越好"),
        ("brier_mean", f"{s['brier_mean']:.3f}" if s.get("brier_mean") is not None else "-", "機率預測誤差，越小越好"),
        ("ordinal_mae", f"{s['ordinal_mae']:.2f}" if s.get("ordinal_mae") is not None else "-", "score 題預測等級平均差幾級"),
        ("latency.p50_s / p95_s", f"{s['latency']['p50_s']:.2f} 秒 / {s['latency']['p95_s']:.2f} 秒", "一半 / 95% 的題目在這個時間內完成"),
    ]
    for name, value, meaning in rows:
        print(f"  {name:<50} {value:<18} {meaning}")
    pc = s["paraphrase_consistency"]
    if pc["agreement"] is None:
        value, meaning = "-", "沒有換句話說的題目組，不適用"
    else:
        value, meaning = f"{pc['agreement']:.3f}", f"{pc['pairs']} 組換句話說的題目，答案一致率"
    print(f"  {'paraphrase_consistency':<50} {value:<18} {meaning}")

    if s.get("ece"):
        print("\n[ece.bins：模型說的信心 vs 實際答對率]")
        print(f"  {'信心區間':<10}{'題數':>6}{'平均信心':>10}{'實際答對率':>12}")
        for b in reversed(s["ece"]["bins"]):
            if b["n"]:
                print(f"  {b['lo']:.1f}–{b['hi']:.1f}{b['n']:>10}{b['mean_confidence']:>12.3f}{b['accuracy']:>13.3f}")

    print("\n[per_family：每個題型的正確率（由低到高）]")
    for fam, v in sorted(s["per_family"].items(), key=lambda kv: kv[1]["accuracy"] or 0):
        print(f"  {fam:<18} {v['n_correct']:>3}/{v['n_scorable']:<3} {v['accuracy']:.2f}")


def main(description, make_classifier, default_out, base_url_help, add_arguments=lambda ap: None):
    """description：說明；make_classifier(args) 建立 classifier；default_out 例如 "~/jb_runs/winnow_{tier}"。"""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--tier", choices=[*TIER_FILES, "all"], default="hard", help="all = easy / standard / hard / hard_cn 依序跑")
    ap.add_argument(
        "--out", default=None,
        help=f"輸出資料夾，預設 {default_out.format(tier='<tier>')}；--tier all 時是上層資料夾",
    )
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題")
    ap.add_argument("--base-url", default=None, help=base_url_help)
    add_arguments(ap)
    args = ap.parse_args()

    tiers = JEVBENCH_TIERS if args.tier == "all" else [args.tier]
    out_dirs = {
        tier: os.path.expanduser(
            os.path.join(args.out, tier) if args.out and len(tiers) > 1
            else args.out or default_out.format(tier=tier)
        )
        for tier in tiers
    }
    # 先全部檢查過，免得跑到一半才發現後面的 tier 會撞到舊結果
    for out_dir in out_dirs.values():
        results_path = os.path.join(out_dir, "results.jsonl")
        if os.path.exists(results_path):
            sys.exit(f"{results_path} 已經存在；請換一個 --out，或先把舊的移走（不覆蓋之前的結果）")

    classifier = make_classifier(args)
    for tier in tiers:
        run_tier(classifier, tier, out_dirs[tier], args.limit)


def run_tier(classifier, tier, out_dir, limit):
    tasks_path = os.path.join(JEVBENCH_DIR, "datasets", "public", TIER_FILES[tier])
    tasks = load_jsonl(tasks_path)[:limit]
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    print(f"評測 {tier}：{len(tasks)} 題 → {out_dir}（{classifier.base_url}）")
    records = []
    with open(results_path, "x", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            record = run_one(classifier, task)
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(tasks):
                done = sum(r["correct"] for r in records)
                print(f"  {i}/{len(tasks)} 題，目前答對 {done} 題，失敗 {sum(not r['ok'] for r in records)} 題")

    summary = public_export(summarize(tasks, records), tasks, records)
    summary["dataset_hash"] = dataset_hash(tasks)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
    print_report(tier, summary)
    print(f"\n輸出：{results_path}\n      {os.path.join(out_dir, 'summary.json')}")

