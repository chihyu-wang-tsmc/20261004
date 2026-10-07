"""用 benchmark 題目做 model routing 的 ground truth，並統計 Jev router（Winnow / jevk5）省下多少次大 model。

gpqa_routing.py、aime_routing.py 共用的流程；各 benchmark 只要提供一個 Bench（怎麼出題、怎麼對答案）。
三個子命令：

  answer --model <名稱>   用指定 model 回答每一題 → <bench>/answers/<model>.jsonl
                          可以中斷後重跑：已經答完的題目會跳過，失敗的會重試
  regrade --model <名稱>  不重新呼叫 model，用已存的回答重新評分（評分方式改過之後用）
  route [--router R]     用 Winnow（預設）或 jevk5 當 router，每題選 fast / powerful
                          → <bench>/routes_<R>.jsonl
                          問題跟 deep1_router.py 的 ModelRouterMiddleware 完全相同；state 是 router
                          會看到的最後一則使用者訊息，也就是 answer 送給 model 的同一段 prompt
  report [--router R]    合併上面兩步，產生 ground truth 並統計
                          → <bench>/ground_truth.jsonl、<bench>/<bench>_routing.jsonl（JevBench 格式）

Ground truth 的定義（能答對的最便宜 model）：
    fast 答對                     → fast
    fast 答錯、powerful 答對       → powerful
    兩個都答錯                     → none（換 model 也沒用；不列入 router 準確率）
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage

import llm_models

HERE = os.path.dirname(os.path.abspath(__file__))

# 可以用 --model 指定的 model：名稱 → llm_models 裡的建立函式
MODELS = {
    "deepseek-flash": llm_models.deepseek,     # DEEPSEEK_V41_FLASH_MODEL
    "gemini-3.7-flash": llm_models.gemini,     # GEMINI_37_FLASH_MODEL
    "kimi-k3": llm_models.kimi,                # KIMI_K3_MODEL
    "glm-5.3": llm_models.glm,                 # GLM_53_MODEL
    "glm-4.7-flash": llm_models.glm,           # GLM_47_FLASH_MODEL
    "qwen3.8-27b": llm_models.qwen,            # QWEN_38_27B_MODEL
    "qwen3.8-flash": llm_models.qwen,          # QWEN_38_FLASH_NEXT_MODEL
}

# 跟 deep1_router.py 的 ModelRouterMiddleware 設定一字不差
ROUTER_INSTRUCTIONS = "Choose the least costly model that can complete the task safely."
ROUTER_CRITERIA = {
    "fast": "Direct lookups, extraction, and localized changes with explicit targets.",
    "powerful": "Architecture, novel root-cause reasoning, and high-stakes decisions.",
}


# 可以用 --router 指定的 router（Jev 相容的 classifier）
ROUTERS = ("winnow", "jevk5")


def make_router(name, base_url=None):
    if name == "winnow":
        from classifier_winnow import WinnowClassifier

        return WinnowClassifier(timeout=120, **({"base_url": base_url} if base_url else {}))
    from classifier_jevk5 import JEVK5_BASE_URL, JevK5Classifier

    return JevK5Classifier(base_url=base_url or JEVK5_BASE_URL, timeout=120)


@dataclass
class Bench:
    name: str
    """資料夾名稱，也是輸出檔名的前綴，例如 gpqa → gpqa/gpqa_routing.jsonl。"""

    title: str
    """報表標題。"""

    load_questions: Callable[[argparse.Namespace], list[dict]]
    """回傳題目 list，每題要有 id、domain、subdomain、prompt、correct。"""

    grade: Callable[[str, dict], tuple[object, bool]]
    """(model 的回答全文, 題目) → (抽出來的答案, 是否答對)。"""

    source: str
    """寫進 JevBench 題目 provenance 的資料來源。"""

    add_arguments: Callable[[argparse.ArgumentParser], None] = lambda parser: None
    """benchmark 自己的命令列參數（例如 AIME 的 --year），每個子命令都會加上。"""

    @property
    def dir(self):
        return os.path.join(HERE, self.name)

    @property
    def answers_dir(self):
        return os.path.join(self.dir, "answers")

    def routes_path(self, router):
        return os.path.join(self.dir, f"routes_{router}.jsonl")


def read_jsonl(path):
    """讀 jsonl；同一題有多筆時以最後一筆為準（重跑時會追加新紀錄）。"""
    records = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["id"]] = r
    return records


def response_text(message):
    # Gemini 的 content 可能是 parts 的 list；.text 會把文字部分接起來
    return str(message.text)


def safe_grade(bench, text, q):
    # 評分程式本身出錯時記成答錯，不要重新呼叫 model；修好後用 regrade 重新評分就好
    try:
        return bench.grade(text, q)
    except Exception as e:  # noqa: BLE001
        return f"grader error: {type(e).__name__}: {e}", False


def answer_one(bench, model, model_name, q, retries):
    last_error = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            message = model.invoke([HumanMessage(q["prompt"])])
        except Exception as e:  # noqa: BLE001 - 失敗要記下來，下次重跑會再試
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))
            continue
        latency = time.perf_counter() - started
        text = response_text(message)
        answer, correct = safe_grade(bench, text, q)
        usage = message.usage_metadata or {}
        meta = message.response_metadata or {}
        return {
            "id": q["id"], "model": model_name, "ok": True,
            "answer": answer, "correct_answer": q["correct"], "correct": correct,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            # length = 輸出被截斷（通常就抽不到答案）
            "finish_reason": meta.get("finish_reason"),
            "latency_s": round(latency, 2), "response": text,
        }
    return {
        "id": q["id"], "model": model_name, "ok": False, "answer": None,
        "correct_answer": q["correct"], "correct": False, "error": last_error,
    }


def cmd_answer(bench, args):
    questions = bench.load_questions(args)[: args.limit]
    os.makedirs(bench.answers_dir, exist_ok=True)
    path = os.path.join(bench.answers_dir, f"{args.model}.jsonl")
    done = {i for i, r in read_jsonl(path).items() if r["ok"]}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{args.model}：{len(questions)} 題，已完成 {len(questions) - len(todo)} 題，這次要跑 {len(todo)} 題 → {path}")
    if not todo:
        return

    # 沒設 timeout 的話，model 想很久或連線卡住時會一直等下去；逾時算這題失敗，下次重跑會再試
    model = MODELS[args.model](args.model, timeout=args.timeout)
    lock = threading.Lock()
    n = n_correct = n_failed = 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(answer_one, bench, model, args.model, q, args.retries) for q in todo]
        for future in as_completed(futures):
            r = future.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                n_correct += r["correct"]
                n_failed += not r["ok"]
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} 題，答對 {n_correct}，失敗 {n_failed}")
    if n_failed:
        print(f"有 {n_failed} 題失敗（見 error 欄位），再跑一次同樣的指令會重試這些題目")


def cmd_regrade(bench, args):
    """不重新呼叫 model，用已存的回答重新評分（評分方式改過之後用）。"""
    questions = {q["id"]: q for q in bench.load_questions(args)}
    path = os.path.join(bench.answers_dir, f"{args.model}.jsonl")
    records = read_jsonl(path)
    targets = [r for r in records.values() if r["ok"] and r["id"] in questions]
    print(f"{args.model}：重新評分 {len(targets)} 題 → {path}")
    before = sum(r["correct"] for r in targets)
    with ThreadPoolExecutor(args.workers) as pool:
        graded = pool.map(lambda r: safe_grade(bench, r["response"], questions[r["id"]]), targets)
        for r, (answer, correct) in zip(targets, graded):
            r["answer"], r["correct"] = answer, correct
    # 重寫整個檔案（每題只留最後一筆）
    with open(path, "w", encoding="utf-8") as f:
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  答對 {before} → {sum(r['correct'] for r in targets)} 題")


def cmd_route(bench, args):
    from langchain_typesafe import Choice

    from classifier_jevk5 import choice_confidence

    questions = bench.load_questions(args)[: args.limit]
    os.makedirs(bench.dir, exist_ok=True)
    routes_path = bench.routes_path(args.router)
    done = {i for i, r in read_jsonl(routes_path).items() if r["ok"]}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{args.router} router：{len(questions)} 題，已完成 {len(questions) - len(todo)} 題，這次要跑 {len(todo)} 題 → {routes_path}")
    if not todo:
        return

    classifier = make_router(args.router, args.base_url)
    # 跟 ModelRouterMiddleware.before_agent 送出的請求相同：state 是最後一則 HumanMessage，
    # 問題 ID 是 model_route，criteria 是各 route 的描述
    questions_spec = {"model_route": Choice(instructions=ROUTER_INSTRUCTIONS, criteria=ROUTER_CRITERIA)}
    counts = Counter()
    with open(routes_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            started = time.perf_counter()
            try:
                response = classifier.invoke(
                    {"state": HumanMessage(q["prompt"]), "questions": questions_spec}
                )
                answer = response.choices["model_route"]
                r = {
                    "id": q["id"], "ok": True, "route": answer.choice,
                    # confidence 不用服務回傳的（Winnow 的算法不一定跟文件一樣），
                    # 一律用 classifier_jevk5 的文件公式從 probabilities 算，兩個 router 才比得起來
                    "probabilities": answer.probabilities,
                    "confidence": choice_confidence(list(answer.probabilities.values())),
                    "input_tokens": response.usage.input_tokens,
                }
                counts[answer.choice] += 1
            except Exception as e:  # noqa: BLE001
                r = {"id": q["id"], "ok": False, "route": None, "error": f"{type(e).__name__}: {e}"}
                counts["failed"] += 1
            r["latency_s"] = round(time.perf_counter() - started, 3)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} 題：{dict(counts)}")


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def cmd_report(bench, args):
    questions = bench.load_questions(args)
    fast = read_jsonl(os.path.join(bench.answers_dir, f"{args.fast}.jsonl"))
    powerful = read_jsonl(os.path.join(bench.answers_dir, f"{args.powerful}.jsonl"))
    routes = read_jsonl(bench.routes_path(args.router))

    rows = []
    for q in questions:
        f_r, p_r, route = fast.get(q["id"]), powerful.get(q["id"]), routes.get(q["id"])
        if not (f_r and f_r["ok"] and p_r and p_r["ok"]):
            continue  # 兩個 model 都要有答案才能定 ground truth
        gt = "fast" if f_r["correct"] else "powerful" if p_r["correct"] else "none"
        rows.append(
            {
                "id": q["id"], "domain": q["domain"], "subdomain": q["subdomain"],
                "fast_correct": f_r["correct"], "powerful_correct": p_r["correct"],
                "ground_truth": gt,
                "route": route["route"] if route and route["ok"] else None,
                "route_probabilities": route.get("probabilities") if route else None,
                "fast_tokens": (f_r.get("input_tokens") or 0) + (f_r.get("output_tokens") or 0),
                "powerful_tokens": (p_r.get("input_tokens") or 0) + (p_r.get("output_tokens") or 0),
                "prompt": q["prompt"],
            }
        )
    if not rows:
        sys.exit("還沒有兩個 model 都答完的題目；先跑 answer --model <fast> 和 answer --model <powerful>")

    gt_path = os.path.join(bench.dir, "ground_truth.jsonl")
    jb_path = os.path.join(bench.dir, f"{bench.name}_routing.jsonl")
    with open(gt_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "prompt"}, ensure_ascii=False) + "\n")
    export_jevbench(bench, rows, args, jb_path)

    n = len(rows)
    gt = Counter(r["ground_truth"] for r in rows)
    print(f"\n========== {bench.title} routing：{n} 題（fast = {args.fast}，powerful = {args.powerful}）==========")
    if n < len(questions):
        print(f"（另有 {len(questions) - n} 題還沒被兩個 model 都答完，沒有列入）")
    print("\n[各 model 單獨作答]")
    n_fast = sum(r["fast_correct"] for r in rows)
    n_pow = sum(r["powerful_correct"] for r in rows)
    n_either = sum(r["fast_correct"] or r["powerful_correct"] for r in rows)
    print(f"  全部給 fast       答對 {n_fast:>3}/{n}  {pct(n_fast, n)}")
    print(f"  全部給 powerful   答對 {n_pow:>3}/{n}  {pct(n_pow, n)}")
    print(f"  完美 router       答對 {n_either:>3}/{n}  {pct(n_either, n)}  （任一個答對就算對）")

    print("\n[Routing ground truth：能答對的最便宜 model]")
    print(f"  fast 就夠          {gt['fast']:>3} 題  {pct(gt['fast'], n)}")
    print(f"  需要 powerful      {gt['powerful']:>3} 題  {pct(gt['powerful'], n)}")
    print(f"  兩個都答錯         {gt['none']:>3} 題  {pct(gt['none'], n)}")
    print(f"  另外：fast 對但 powerful 錯的有 {sum(r['fast_correct'] and not r['powerful_correct'] for r in rows)} 題")

    routed = [r for r in rows if r["route"]]
    if not routed:
        print(f"\n還沒有 {args.router} 的 routing 結果；先跑 python {bench.name}_routing.py route --router {args.router}")
        return
    m = len(routed)
    to_fast = [r for r in routed if r["route"] == "fast"]
    to_pow = [r for r in routed if r["route"] == "powerful"]
    n_routed_correct = sum(r["fast_correct"] for r in to_fast) + sum(r["powerful_correct"] for r in to_pow)
    base_correct = sum(r["powerful_correct"] for r in routed)
    saved_tokens = sum(r["powerful_tokens"] for r in to_fast)
    total_pow_tokens = sum(r["powerful_tokens"] for r in routed)

    print(f"\n[{args.router} router：{m} 題]")
    print(f"  選 fast      {len(to_fast):>3} 題  {pct(len(to_fast), m)}")
    print(f"  選 powerful  {len(to_pow):>3} 題  {pct(len(to_pow), m)}")
    print(f"  → 跟「全部給 powerful」相比，省下 {len(to_fast)} 次大 model 呼叫（{pct(len(to_fast), m)}）")
    print(f"    powerful model 的 token 省下 {saved_tokens:,} / {total_pow_tokens:,}（{pct(saved_tokens, total_pow_tokens)}）")
    print(f"  照 router 分配後答對 {n_routed_correct}/{m}  {pct(n_routed_correct, m)}"
          f"（全部給 powerful 是 {base_correct}/{m}，差 {n_routed_correct - base_correct:+d} 題）")

    print("\n[Router 選擇 vs ground truth]")
    print(f"  {'':<20}{'router=fast':>12}{'router=powerful':>17}")
    for label in ("fast", "powerful", "none"):
        a = sum(r["ground_truth"] == label and r["route"] == "fast" for r in routed)
        b = sum(r["ground_truth"] == label and r["route"] == "powerful" for r in routed)
        print(f"  {'ground truth=' + label:<20}{a:>12}{b:>17}")
    decisive = [r for r in routed if r["ground_truth"] != "none"]
    hit = sum(r["route"] == r["ground_truth"] for r in decisive)
    wasted = sum(r["ground_truth"] == "fast" and r["route"] == "powerful" for r in routed)
    lost = sum(r["ground_truth"] == "powerful" and r["route"] == "fast" for r in routed)
    print(f"  router 準確率（不含兩個都答錯的題目）：{hit}/{len(decisive)}  {pct(hit, len(decisive))}")
    print(f"  白用大 model：{wasted} 題（fast 就能答對，卻給了 powerful）")
    print(f"  該用沒用：    {lost} 題（只有 powerful 答得對，卻給了 fast → 這些題答錯）")

    print("\n[各分組]")
    for domain in sorted({r["domain"] for r in routed}):
        d = [r for r in routed if r["domain"] == domain]
        print(f"  {domain:<14} {len(d):>3} 題，router 選 fast {sum(r['route'] == 'fast' for r in d):>3} 題，"
              f"ground truth 需要 powerful {sum(r['ground_truth'] == 'powerful' for r in d):>3} 題")
    print(f"\n輸出：{gt_path}\n      {jb_path}")


def export_jevbench(bench, rows, args, path):
    """把 ground truth 存成 JevBench 的題目格式，給 evaluate_winnow.py / evaluate_jevk5.py 評測 router 用。"""
    sys.path.insert(0, os.path.expanduser("~/jevbench"))
    from jevbench.tasks import Task

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            task = Task.from_dict(
                {
                    "id": f"{bench.name}-{r['id']}",
                    "family": f"{bench.name}_{r['domain'].lower().replace(' ', '_')}",
                    "split": "private", "group": None,
                    "state": {"role": "user", "content": r["prompt"]},
                    "question": {"type": "choice", "instructions": ROUTER_INSTRUCTIONS, "criteria": ROUTER_CRITERIA},
                    "labels": ["fast", "powerful"],
                    # 兩個都答錯的題目沒有正確的 route → expected=None，JevBench 會把它排除在正確率之外
                    "expected": None if r["ground_truth"] == "none" else r["ground_truth"],
                    "provenance": {
                        "source": bench.source,
                        "label_basis": f"cheapest correct model: fast={args.fast}, powerful={args.powerful}",
                        "fast_correct": r["fast_correct"], "powerful_correct": r["powerful_correct"],
                        "subdomain": r["subdomain"],
                    },
                }
            )
            f.write(task.to_json() + "\n")


def main(bench, description):
    ap = argparse.ArgumentParser(description=description)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("answer", help="用指定 model 回答每一題")
    a.add_argument("--model", required=True, choices=MODELS)
    a.add_argument("--workers", type=int, default=8, help="同時送出的請求數")
    a.add_argument("--retries", type=int, default=2, help="每題失敗時重試次數")
    a.add_argument("--timeout", type=float, default=1200, help="每次呼叫 model 的逾時秒數（預設 20 分鐘）")
    a.add_argument("--limit", type=int, default=None, help="只跑前 N 題")
    a.set_defaults(func=cmd_answer)

    g = sub.add_parser("regrade", help="不重新呼叫 model，用已存的回答重新評分")
    g.add_argument("--model", required=True, choices=MODELS)
    g.add_argument("--workers", type=int, default=4, help="同時評分的題數")
    g.set_defaults(func=cmd_regrade)

    r = sub.add_parser("route", help="用 Jev router（Winnow / jevk5）為每一題選 fast / powerful")
    r.add_argument("--router", choices=ROUTERS, default="winnow", help="用哪個 classifier 當 router")
    r.add_argument("--base-url", default=None, help="router 服務網址；預設 Winnow 8091、jevk5 8090")
    r.add_argument("--limit", type=int, default=None, help="只跑前 N 題")
    r.set_defaults(func=cmd_route)

    p = sub.add_parser("report", help="產生 ground truth 並統計 router 省下多少次大 model")
    p.add_argument("--fast", default="qwen3.8-27b", choices=MODELS, help="deep1_router.py 的 fast")
    p.add_argument("--powerful", default="qwen3.8-flash", choices=MODELS, help="deep1_router.py 的 powerful")
    p.add_argument("--router", choices=ROUTERS, default="winnow", help="統計哪個 router 的結果")
    p.set_defaults(func=cmd_report)

    for parser in (a, g, r, p):
        bench.add_arguments(parser)
    args = ap.parse_args()
    args.func(bench, args)
