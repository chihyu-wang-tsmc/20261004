"""Run DecisionBench 1.0 against an imajev (Jev-compatible /v1/systemone) server without the official harness.

It mirrors the harness exactly where it matters for the score:
  * rows       : Hanno-Labs/decision-bench @ b7c8107e (the revision the leaderboard record used), split "eval", 23,900 rows
  * request    : decision_bench.models.jev_openrouter.build_jev_request  (noul / choice / score)
  * answer     : decision_bench.models.jev_openrouter.prediction_from_jev_answer
  * scoring    : argmax == gold_candidate_id; NLL against gold_probabilities; 15-bin top-label ECE
  * primary    : correct / all requested rows (rows over --max-candidates or that error count as misses)

usage:
  pip install "datasets>=4" httpx pyarrow
  python run_decisionbench_diy.py --base-url http://127.0.0.1:8765 --out runs/diy            # all 23,900 rows
  python run_decisionbench_diy.py --base-url ... --out runs/try --per-task 20                   # ~860-row quick check
  python run_decisionbench_diy.py --base-url ... --out runs/fin --tasks finqa-numerical-reasoning,route
  python run_decisionbench_diy.py --parquet eval-00000-of-00001.parquet ...                     # offline copy
Re-running with the same --out resumes: rows already in predictions.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = "Hanno-Labs/decision-bench"
REVISION = "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"
ECE_BINS = 15


# ---------- rows ----------
def load_rows(parquet: str | None, revision: str) -> list[dict]:
    from datasets import load_dataset
    if parquet:
        ds = load_dataset("parquet", data_files=parquet, split="train")
    else:
        ds = load_dataset(REPO, split="eval", revision=revision)
    rows = []
    for r in ds:
        rows.append({
            "row_id": r["row_id"],
            "task_name": r["task_name"],
            "primitive": r["primitive"],
            "instruction": r["instruction"],
            "state": json.loads(r["state_json"]),
            "candidates": json.loads(r["candidates_json"]),
            "gold_candidate_id": r["gold_candidate_id"],
            "gold_probabilities": list(r["gold_probabilities"]),
        })
    return rows


# ---------- request / answer (same as the harness) ----------
def describe(c: dict) -> str:
    return c["label"] if c.get("description") is None else f"{c['label']}: {c['description']}"


def build_request(row: dict, model: str) -> tuple[dict, list[str]]:
    cands = row["candidates"]
    if row["primitive"] == "binary_classification":
        assert len(cands) == 2, row["row_id"]
        q = {"type": "noul", "instructions": row["instruction"],
             "criteria": {"true": describe(cands[0]), "false": describe(cands[1])}}
        order = [c["id"] for c in cands]
    elif row["primitive"] == "candidate_selection":
        q = {"type": "choice", "instructions": row["instruction"],
             "criteria": {c["id"]: describe(c) for c in cands}}
        order = [c["id"] for c in cands]
    else:  # ordinal_scoring
        ordered = sorted(cands, key=lambda c: float(c.get("ordinal_value") or 0.0))
        q = {"type": "score", "instructions": row["instruction"],
             "criteria": [describe(c) for c in ordered]}
        order = [c["id"] for c in ordered]
    return {"model": model, "state": row["state"], "questions": {"decision": q}}, order


def to_probabilities(row: dict, answer: dict, order: list[str]) -> list[float]:
    cands = row["candidates"]
    if row["primitive"] == "binary_classification":
        p_true = float(answer["noul"])  # first candidate is the "true" outcome
        probs = [p_true if c["id"] == cands[0]["id"] else 1.0 - p_true for c in cands]
    else:
        raw = answer["probabilities"]
        if row["primitive"] == "candidate_selection":
            by_id = {cid: float(raw[cid]) for cid in order}
        else:
            by_id = {cid: float(raw[str(i)]) for i, cid in enumerate(order)}
        probs = [by_id[c["id"]] for c in cands]
    # same validation + renormalisation as decision_bench.schemas.DecisionPrediction
    if not probs or any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs):
        raise ValueError("probabilities must be finite values from 0 to 1")
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("probabilities must have a positive sum")
    return [p / total for p in probs]


# ---------- running ----------
def run_one(client: httpx.Client, row: dict, model: str, max_candidates: int, retries: int) -> dict:
    base = {"row_id": row["row_id"], "task": row["task_name"], "candidates": len(row["candidates"])}
    if len(row["candidates"]) > max_candidates:
        return {**base, "status": "unsupported"}
    request, order = build_request(row, model)
    started, last = time.monotonic(), None
    for attempt in range(retries + 1):
        try:
            r = client.post("/v1/systemone", json=request)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            probs = to_probabilities(row, r.json()["answers"]["decision"], order)
            ids = [c["id"] for c in row["candidates"]]
            top = max(range(len(probs)), key=probs.__getitem__)
            nll = -sum(t * math.log(max(p, 1e-15)) for t, p in zip(row["gold_probabilities"], probs))
            return {**base, "status": "ok", "probabilities": probs, "selected": ids[top],
                    "gold": row["gold_candidate_id"], "correct": ids[top] == row["gold_candidate_id"],
                    "nll": nll, "latency": time.monotonic() - started}
        except Exception as e:  # noqa: BLE001
            last = e
            if "HTTP 4" in str(e) and "HTTP 429" not in str(e):
                break  # a 4xx (e.g. input too long) will not succeed on retry
            time.sleep(min(2 ** attempt, 16))
    return {**base, "status": "error", "error": str(last)[:500]}


def ece(records: list[dict]) -> float:
    n = [0] * ECE_BINS; conf = [0.0] * ECE_BINS; acc = [0.0] * ECE_BINS
    for r in records:
        c = max(r["probabilities"]); b = min(int(c * ECE_BINS), ECE_BINS - 1)
        n[b] += 1; conf[b] += c; acc[b] += float(r["correct"])
    return sum((k / len(records)) * abs(a / k - s / k) for k, s, a in zip(n, conf, acc) if k)


def summarize(records: list[dict], requested: int) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    out = {
        "requested_rows": requested,
        "successful_rows": len(ok),
        "unsupported_rows": sum(r["status"] == "unsupported" for r in records),
        "error_rows": sum(r["status"] == "error" for r in records),
        "missing_rows": requested - len(records),
        "primary_accuracy": sum(r["correct"] for r in ok) / requested if requested else 0.0,
        "supported_accuracy": sum(r["correct"] for r in ok) / len(ok) if ok else 0.0,
        "coverage": len(ok) / requested if requested else 0.0,
        "mean_negative_log_likelihood": sum(r["nll"] for r in ok) / len(ok) if ok else None,
        "expected_calibration_error": ece(ok) if ok else None,
        "mean_latency_seconds": sum(r["latency"] for r in ok) / len(ok) if ok else None,
        "views": {},
    }
    by_task = collections.defaultdict(list)
    for r in ok:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        out["views"][f"task:{task}"] = {"rows": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="imajev-4b")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--parquet", help="local copy of data/eval-00000-of-00001.parquet")
    ap.add_argument("--tasks", help="comma-separated task_name values (e.g. route,finqa-numerical-reasoning)")
    ap.add_argument("--per-task", type=int, help="sample at most N rows per task (seed 0)")
    ap.add_argument("--max-candidates", type=int, default=255)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()

    rows = load_rows(a.parquet, a.revision)
    print(f"loaded {len(rows)} rows")
    if a.tasks:
        keep = set(a.tasks.split(","))
        rows = [r for r in rows if r["task_name"] in keep]
    if a.per_task:
        rng, grouped = random.Random(0), collections.defaultdict(list)
        for r in rows:
            grouped[r["task_name"]].append(r)
        rows = [r for g in grouped.values() for r in (rng.sample(g, a.per_task) if len(g) > a.per_task else g)]
    print(f"selected {len(rows)} rows across {len({r['task_name'] for r in rows})} tasks")

    a.out.mkdir(parents=True, exist_ok=True)
    pred_path = a.out / "predictions.jsonl"
    done: dict[str, dict] = {}
    if pred_path.exists():
        for line in pred_path.open():
            rec = json.loads(line)
            if rec["status"] != "error":  # errors are retried on resume
                done[rec["row_id"]] = rec
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run")

    lock, t0 = threading.Lock(), time.time()
    with httpx.Client(base_url=a.base_url, timeout=a.timeout) as client, pred_path.open("a") as f, \
            ThreadPoolExecutor(a.concurrency) as pool:
        futures = [pool.submit(run_one, client, r, a.model, a.max_candidates, a.retries) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                done[rec["row_id"]] = rec
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"{i}/{len(todo)}  {rate:.2f} rows/s  eta {(len(todo) - i) / rate / 3600:.1f} h", flush=True)

    wanted = {r["row_id"] for r in rows}
    summary = summarize([rec for rid, rec in done.items() if rid in wanted], len(rows))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "views"}, indent=1))


if __name__ == "__main__":
    main()
