"""用 GPQA / AIME / AI2 ARC / GSM8K 評測 model；routing_report.py 要用 fast 和 powerful 兩邊的作答。

    python run_bench.py <benchmark> answer  --model <model>     讓 model 回答每一題
    python run_bench.py <benchmark> regrade --model <model>     不重新呼叫 model，用存下來的回答重新評分
    python run_bench.py <benchmark> summary [--models ...]      各 model 的正確率、token 用量、花費

<benchmark> 是 gpqa、aime、ai2arc、gsm8k，或 all（四個依序跑）；<model> 見 models.py。
每個 model 的回答存在 results/<benchmark>/<model>.jsonl，一行一題。answer 中斷後重跑同一個指令，
答完的題目會跳過、失敗的會重試。題目從 datasets 讀（預設 ../datasets，可用 --data-dir 或環境變數
JEV_DATASETS_DIR 指定）。
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from benchmarks import BENCHMARKS
from common import RESULTS_DIR, add_data_dir_arg, data_dir, pad, read_jsonl, rel
from models import MODELS, cost, create_model


def answers_path(bench, model):
    return RESULTS_DIR / bench.name / f"{model}.jsonl"


def safe_grade(bench, text, q):
    """評分程式本身出錯時記成答錯，不要為此重新呼叫 model；修好後用 regrade 重新評分就好。"""
    try:
        return bench.grade(text, q)
    except Exception as e:  # noqa: BLE001
        return f"grader error: {type(e).__name__}: {e}", False


def answer_one(bench, llm, q):
    started = time.perf_counter()
    try:
        reply = llm.chat([{"role": "user", "content": q["prompt"]}])
    except Exception as e:  # noqa: BLE001 —— 記下來，下次重跑會再試
        return {"id": q["id"], "model": llm.name, "ok": False, "answer": None,
                "correct_answer": q["answer"], "correct": False, "error": f"{type(e).__name__}: {e}"}
    answer, correct = safe_grade(bench, reply.text, q)
    return {"id": q["id"], "model": llm.name, "ok": True, "answer": answer,
            "correct_answer": q["answer"], "correct": correct,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "finish_reason": reply.finish_reason, "latency_s": round(time.perf_counter() - started, 2),
            "response": reply.text}


def cmd_answer(bench, args):
    questions = bench.questions(data_dir(args.data_dir))[: args.limit]
    path = answers_path(bench, args.model)
    done = {i for i, r in read_jsonl(path).items() if r["ok"]}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{bench.title}：{len(questions)} 題，已答完 {len(questions) - len(todo)} 題，這次要答 {len(todo)} 題"
          f" → {rel(path)}", flush=True)
    if not todo:
        return
    llm = create_model(args.model, timeout=args.timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    n = n_correct = n_failed = 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(answer_one, bench, llm, q) for q in todo]):
            r = future.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                n_correct += r["correct"]
                n_failed += not r["ok"]
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}：答對 {n_correct}，失敗 {n_failed}", flush=True)
    if n_failed:
        print(f"  ⚠ {n_failed} 題失敗（原因在 error 欄位），重跑同一個指令會重試這些題目")


def cmd_regrade(bench, args):
    questions = {q["id"]: q for q in bench.questions(data_dir(args.data_dir))}
    path = answers_path(bench, args.model)
    records = read_jsonl(path)
    targets = [r for r in records.values() if r["ok"] and r["id"] in questions]
    before = sum(r["correct"] for r in targets)
    for r in targets:
        r["answer"], r["correct"] = safe_grade(bench, r["response"], questions[r["id"]])
    with open(path, "w", encoding="utf-8") as f:  # 重寫整個檔，每題只留最後一筆
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{bench.title}／{args.model}：重新評分 {len(targets)} 題，答對 {before} → "
          f"{sum(r['correct'] for r in targets)} 題 → {rel(path)}")


def cmd_summary(bench, args):
    n_questions = len(bench.questions(data_dir(args.data_dir)))
    folder = RESULTS_DIR / bench.name
    models = args.models or sorted(p.stem for p in folder.glob("*.jsonl"))
    print(f"\n{bench.title}（{n_questions} 題）")
    if not models:
        print(f"  還沒有作答：python run_bench.py {bench.name} answer --model <model>")
        return
    cols = [("model", 20, False), ("已答", 7, True), ("正確率", 9, True), ("失敗", 6, True), ("被截斷", 8, True),
            ("平均輸入", 10, True), ("平均輸出", 10, True), ("$/千題", 10, True)]
    print("  " + "".join(pad(c, w, r) for c, w, r in cols))
    for m in models:
        records = list(read_jsonl(answers_path(bench, m)).values())
        ok = [r for r in records if r["ok"]]
        if not ok:
            print("  " + pad(m, 20) + "（沒有有效作答）")
            continue
        n_in = sum(r.get("input_tokens") or 0 for r in ok) / len(ok)
        n_out = sum(r.get("output_tokens") or 0 for r in ok) / len(ok)
        dollars = f"${cost(m, n_in, n_out) * 1000:,.2f}" if m in MODELS else "?"
        vals = [m, f"{len(ok)}/{n_questions}", f"{sum(r['correct'] for r in ok) / len(ok):.1%}",
                len(records) - len(ok), sum(r.get("finish_reason") == "length" for r in ok),
                f"{n_in:,.0f}", f"{n_out:,.0f}", dollars]
        print("  " + "".join(pad(v, w, r) for v, (_, w, r) in zip(vals, cols)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=[*BENCHMARKS, "all"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("answer", help="讓 model 回答每一題")
    a.add_argument("--model", required=True, choices=MODELS)
    a.add_argument("--workers", type=int, default=8, help="同時送出的請求數（預設 8）")
    a.add_argument("--timeout", type=float, default=1200, help="每次呼叫的逾時秒數（預設 1200）")
    a.add_argument("--limit", type=int, help="只答前 N 題（試跑用）")
    a.set_defaults(func=cmd_answer)

    g = sub.add_parser("regrade", help="不重新呼叫 model，用存下來的回答重新評分")
    g.add_argument("--model", required=True, choices=MODELS)
    g.set_defaults(func=cmd_regrade)

    s = sub.add_parser("summary", help="各 model 的正確率、token 用量、花費")
    s.add_argument("--models", nargs="+", help="只看這些 model（預設 results/<benchmark>/ 底下全部）")
    s.set_defaults(func=cmd_summary)

    for p in (a, g, s):
        add_data_dir_arg(p)
    args = ap.parse_args()
    for name in BENCHMARKS if args.benchmark == "all" else [args.benchmark]:
        args.func(BENCHMARKS[name], args)


if __name__ == "__main__":
    main()
