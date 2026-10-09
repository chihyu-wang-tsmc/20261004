"""Run DecisionBench 1.0 against an imajev (Jev-compatible /v1/systemone) server without the official harness.

Reproduces imajev-4b's leaderboard record (primary accuracy 79.65, 2026-09-28, phase-3 adapter). It mirrors the
official `decision-bench run-system-one-http` runner (Hanno-Labs/decision-bench @ 47ea5a47) where it matters for the score:
  * rows       : Hanno-Labs/decision-bench @ b7c8107e (the revision the leaderboard record used), split "eval", 23,900 rows
  * request    : decision_bench.models.jev_openrouter.build_jev_request  (noul / choice / score)
  * answer     : decision_bench.models.jev_openrouter.prediction_from_jev_answer, then divide by the sum
  * eligibility: decision_bench.models.system_one_http.validate_example (candidates, rendered state, request bytes)
  * retries    : the same as system_one_http.predict (4 retries on 429/5xx and transport/parse errors; other 4xx fail)
  * scoring    : argmax == gold_candidate_id; NLL against gold_probabilities (floor 1e-15); 15-bin top-label ECE
  * primary    : correct / all requested rows (unsupported rows and errors count as misses) = the leaderboard number

The score also depends on how the server is started. imajev's phase-3 run (results/benchmarks/decisionbench/
pod_decisionbench_p3.sh) used:
  PYTHONPATH=src:scripts python scripts/playground/server.py --backend torch --model-bundle artifacts/model-qwen4b.json \
    --adapter adapters/imajev-4b --calibration adapters/imajev-4b/calibration-rot4.json --rotations 4 \
    --max-input-tokens 65536 --model-name imajev-4b --port 8765
with the phase-3 adapter (256-code readout, so up to 255 candidates) and --max-candidates 255 here.

usage:
  pip install httpx pyarrow huggingface_hub
  python run_decisionbench.py --out runs/full                                   # all 23,900 rows (hours)
  python run_decisionbench.py --out runs/try --per-task 20                      # ~800-row quick check
  python run_decisionbench.py --out runs/fin --tasks finqa-numerical-reasoning,route
  python run_decisionbench.py --parquet path/to/eval-00000-of-00001.parquet ... # a local copy (checked against b7c8107e)
Re-running with the same --out resumes: rows already in predictions.jsonl are skipped (errors are retried).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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
DATA_FILE = "data/eval-00000-of-00001.parquet"
DATA_SHA256 = "6c97d3f3b5f79ca8566d2ecdf2b0e0f78c20102898810cdd8b7439ac776893ca"  # manifest.json @ b7c8107e
ECE_BINS = 15
# system_one_http.SystemOneHTTPDecisionModel defaults
MAX_RENDERED_STATE_CHARACTERS = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 8 * 1024 * 1024
RETRY_STATUS = {429, 500, 502, 503, 504}
# imajev-4b's published record (Hanno-Labs/decision-bench-results#68), for comparison
PUBLISHED_PRIMARY_ACCURACY = 0.7965


# ---------- rows ----------
def load_rows(parquet: str | None, revision: str) -> list[dict]:
    import pyarrow.parquet as pq

    if parquet:
        path = Path(parquet)
        head = path.read_bytes()[:200]
        if head.startswith(b"version https://git-lfs"):
            raise SystemExit(
                f"{path} is a Git LFS pointer ({path.stat().st_size} bytes), not the data. "
                "Run without --parquet to download the exact revision, or `git lfs pull` in that clone "
                "(note: a clone of the current main has a newer file than the leaderboard used).")
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, DATA_FILE, repo_type="dataset", revision=revision))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if revision == REVISION and digest != DATA_SHA256:
        print(f"WARNING: {path} sha256 {digest[:12]}… is not the b7c8107e file ({DATA_SHA256[:12]}…); "
              "scores will not be comparable with the leaderboard")
    columns = ["row_id", "task_name", "primitive", "instruction", "state_json", "candidates_json",
               "gold_candidate_id", "gold_probabilities"]
    rows = []
    for r in pq.read_table(path, columns=columns).to_pylist():
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


def render_state(state) -> str:
    """system_one_http._render_state: the length the harness checks against the serving limit."""
    if isinstance(state, str):
        return state
    messages = state.get("messages") if isinstance(state, dict) else state
    if isinstance(messages, list) and all(isinstance(m, dict) and "role" in m for m in messages):
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    return json.dumps(state, indent=2)


def unsupported_reason(row: dict, request: dict, max_candidates: int) -> str | None:
    """system_one_http.validate_example: rows outside the serving contract are not sent (they count as misses)."""
    if len(row["candidates"]) > max_candidates:
        return f"more than {max_candidates} candidates"
    if len(render_state(row["state"])) > MAX_RENDERED_STATE_CHARACTERS:
        return "rendered state over the character limit"
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        return "serialized request over the byte limit"
    return None


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
    request, order = build_request(row, model)
    reason = unsupported_reason(row, request, max_candidates)
    if reason:
        return {**base, "status": "unsupported", "reason": reason}
    started, last = time.monotonic(), None
    for attempt in range(retries + 1):
        try:
            r = client.post("/v1/systemone", json=request)
            if r.is_error:
                if r.status_code not in RETRY_STATUS:  # e.g. 422 input too long: will not succeed on retry
                    return {**base, "status": "error", "error": f"HTTP {r.status_code}: {r.text[:500]}"}
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            probs = to_probabilities(row, r.json()["answers"]["decision"], order)
            ids = [c["id"] for c in row["candidates"]]
            top = max(range(len(probs)), key=probs.__getitem__)
            nll = -sum(t * math.log(max(p, 1e-15)) for t, p in zip(row["gold_probabilities"], probs))
            return {**base, "status": "ok", "probabilities": probs, "selected": ids[top],
                    "gold": row["gold_candidate_id"], "correct": ids[top] == row["gold_candidate_id"],
                    "nll": nll, "latency": time.monotonic() - started}
        except Exception as e:  # noqa: BLE001 - transport / parse errors are retried, like the harness
            last = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 16))
    return {**base, "status": "error", "error": str(last)[:500]}


def check_server(base_url: str, max_candidates: int) -> dict:
    """Read /v1/models so a server that cannot take max_candidates is caught before a multi-hour run."""
    info = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=30).json()
    print("server:", json.dumps(info))
    served = info.get("max_options")
    if served is not None and served < max_candidates:
        raise SystemExit(
            f"the server accepts at most {served} options but --max-candidates is {max_candidates}: those rows would "
            f"error instead of being unsupported. Use the phase-3 adapter (256-code readout) or --max-candidates {served}.")
    return info


def ece(records: list[dict]) -> float:
    n = [0] * ECE_BINS; conf = [0.0] * ECE_BINS; acc = [0.0] * ECE_BINS
    for r in records:
        c = max(r["probabilities"]); b = min(int(c * ECE_BINS), ECE_BINS - 1)
        n[b] += 1; conf[b] += c; acc[b] += float(r["correct"])
    return sum((k / len(records)) * abs(a / k - s / k) for k, s, a in zip(n, conf, acc) if k)


def summarize(records: list[dict], requested: int) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    primary = sum(r["correct"] for r in ok) / requested if requested else 0.0
    out = {
        "requested_rows": requested,
        "successful_rows": len(ok),
        "unsupported_rows": sum(r["status"] == "unsupported" for r in records),
        "error_rows": sum(r["status"] == "error" for r in records),
        "missing_rows": requested - len(records),
        "primary_accuracy": primary,
        "leaderboard_score": round(primary * 100, 2),  # what the leaderboard shows (primary_accuracy x 100)
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
    ap.add_argument("--parquet", help="local copy of data/eval-00000-of-00001.parquet (default: download the revision)")
    ap.add_argument("--tasks", help="comma-separated task_name values (e.g. route,finqa-numerical-reasoning)")
    ap.add_argument("--per-task", type=int, help="sample at most N rows per task (seed 0)")
    ap.add_argument("--max-candidates", type=int, default=255, help="255 for the phase-3 adapter, 254 for the previous one")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--retries", type=int, default=4, help="the harness default")
    a = ap.parse_args()

    server = check_server(a.base_url, a.max_candidates)
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
    summary["server"] = server
    summary["dataset"] = {"repo": REPO, "revision": a.revision, "parquet": a.parquet}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("views", "server")}, indent=1))
    if len(rows) == 23900:
        print(f"leaderboard score {summary['leaderboard_score']:.2f}  (imajev-4b published {PUBLISHED_PRIMARY_ACCURACY * 100:.2f})")
    else:
        print(f"subset of {len(rows)} rows: not comparable with the published full-suite score "
              f"{PUBLISHED_PRIMARY_ACCURACY * 100:.2f}; per-task accuracy is in summary.json views")


if __name__ == "__main__":
    main()
