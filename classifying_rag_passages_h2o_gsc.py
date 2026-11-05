"""classifying_rag_passages_h2o.py 的 gsc_composer_refs.csv 版：照
https://docs.typesafe.ai/cookbooks/classifying_rag_passages 的範例，用 Jev 的問法（本機 h2o-lightning，
classifier_h2o.py）逐段判斷 gsc_composer_refs.csv 裡每一題取回的 reference。

跟 classifying_rag_passages_h2o.py 的差別：
  * 資料來源換成 gsc_composer_refs.csv（欄位 query, ground_truth_chunk, reference_chunks）。
    reference_chunks 是 JSON 陣列，已經是檢索系統取回、排好順序的段落，所以不再做 embedding 檢索，
    也不需要 OPENAI_API_KEY。
  * 每一段只問兩題 Noul：is_relevant、contains_answer_evidence（拿掉 contradicts_query_premise、
    contains_prompt_injection）。route() 也只剩這兩條門檻。
  * 不產生回答（沒有 Claude 那段），改成把每一段的分數、路由存成 CSV，並跟 ground_truth_chunk 對照：
      - 每一段先用「8 字片段重疊 ≥ 60%」判斷它是不是標準答案那段（gt_match），
        方法和 gsc_composer_refs_analysis.md 的檢索命中判斷相同；
      - 預設再多問一段：標準答案本身（rank 0，id 是 r{列}-gt），當作「應該兩題都是 yes」的對照組。
        不要的話加 --no-ground-truth。
  * 快取、結果放在 classifying_rag_passages_h2o_gsc/。快取 key 含 query 和段落全文（超過 200 字換成
    sha256），所以 CSV 內容改了只會重問改到的段落。

**noul_floor 要注意**（見 classifier_h2o.py 開頭）：shim 預設把 max(p, 1-p) < 0.801 的 Noul 推到邊界，
這裡的兩個門檻（0.45、0.55）都落在被推掉的那一段，等於「原始機率有沒有過 0.5」。
要看原始機率就把 serve_config.json 的 noul_floor 改成 0、重啟 shim，再刪掉快取重跑。

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741
用法：
    python classifying_rag_passages_h2o_gsc.py                        # 讀同目錄的 gsc_composer_refs.csv
    python classifying_rag_passages_h2o_gsc.py --csv ~/work/gsc_composer_refs.csv
    python classifying_rag_passages_h2o_gsc.py --rows 0,1             # 只跑第 0、1 列
    python classifying_rag_passages_h2o_gsc.py --limit 5 --no-ground-truth
"""

import argparse
import csv
import functools
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from langchain_typesafe import Noul
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from classifier_h2o import H2OClassifier

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "gsc_composer_refs.csv"
OUT_DIR = HERE / "classifying_rag_passages_h2o_gsc"

# Every number the routing reads lives in this dict and nowhere else.
THRESHOLDS = {
    "relevant_min": 0.45,  # below this the passage is not about the query at all
    "evidence_min": 0.55,  # above this it states something usable in an answer
}

GT_SHINGLE = 8  # 判斷「這段是不是標準答案」：8 字片段
GT_OVERLAP_MIN = 0.60  # 標準答案的片段有 60% 以上出現在這段裡，就算同一段

MAX_KEY_CHARS = 200  # 同 cooksafe：超過的 key 換成 sha256


class JsonCache:
    """同 classifying_rag_passages_h2o.py：key 是函式名稱和參數，超過 200 字換成 sha256。"""

    def __init__(self, path: Path, *, indent: int | None = 2) -> None:
        self.path = path
        self.indent = indent
        self._lock = threading.Lock()
        self._data: dict = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = "|".join(
                [fn.__name__]
                + [str(a) for a in args]
                + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            )
            if len(key) > MAX_KEY_CHARS:
                key = hashlib.sha256(key.encode()).hexdigest()
            if key not in self._data:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._data[key] = result
                    ordered = {k: self._data[k] for k in sorted(self._data)}
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = self.path.with_suffix(".tmp")
                    temporary_path.write_text(
                        json.dumps(ordered, indent=self.indent, ensure_ascii=False)
                    )
                    temporary_path.replace(self.path)
            return json.loads(json.dumps(self._data[key]))

        return wrapper


client = H2OClassifier(timeout=120.0).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=3,
    exponential_jitter_params={"initial": 0.5, "max": 5.0},
)
json_cache = JsonCache(OUT_DIR / "json_cache.json")


# ## Load gsc_composer_refs.csv


def load_rows(path: Path) -> list[dict]:
    """每一列：query、ground_truth、references（JSON 陣列解開，照原本的排名順序）。"""
    csv.field_size_limit(sys.maxsize)  # 一格 reference_chunks 可以到 1 萬多字
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for index, record in enumerate(csv.DictReader(f)):
            raw = (record.get("reference_chunks") or "").strip()
            references = json.loads(raw) if raw else []
            rows.append(
                {
                    "row": index,
                    "query": record["query"],
                    "ground_truth": record["ground_truth_chunk"],
                    "references": references,
                }
            )
    return rows


# ## 這段是不是標準答案（跟 gsc_composer_refs_analysis.md 的命中判斷同一個方法）

MEDIA = re.compile(r"!\[[^\]]*\]\([^)]*\)|https?://\S+|image:/\S+")


def shingles(text: str) -> set[str]:
    flat = re.sub(r"\s+", "", MEDIA.sub("", text))
    return {flat[i : i + GT_SHINGLE] for i in range(max(0, len(flat) - GT_SHINGLE + 1))}


def gt_overlap(passage: str, ground_truth: str) -> float:
    wanted = shingles(ground_truth)
    return len(wanted & shingles(passage)) / len(wanted) if wanted else 0.0


# ## Ask two questions about each passage

PASSAGE_QUESTIONS = {
    "is_relevant": Noul(
        instructions="Does this passage address the subject of the query?",
    ),
    "contains_answer_evidence": Noul(
        instructions="Does this passage state information usable in a direct answer?",
    ),
}


def gate_document(query: str, passage_id: str, text: str) -> dict:
    return {"query": query, "passage": {"id": passage_id, "text": text}}


@json_cache
def gate(query: str, passage_id: str, text: str) -> dict:
    started = perf_counter()
    response = client.invoke(
        {
            "state": gate_document(query, passage_id, text),
            "questions": PASSAGE_QUESTIONS,
        }
    )
    answers = {key: response.answers[key].noul for key in PASSAGE_QUESTIONS}
    answers["seconds"] = round(perf_counter() - started, 2)
    answers["input_tokens"] = response.usage.input_tokens or 0
    answers["output_tokens"] = response.usage.output_tokens or 0
    return answers


# ## Route each passage in code


def route(answers: dict, thresholds: dict = THRESHOLDS) -> str:
    if answers["is_relevant"] < thresholds["relevant_min"]:
        return "exclude"
    if answers["contains_answer_evidence"] > thresholds["evidence_min"]:
        return "include"
    return "exclude"


def passages_for(row: dict, with_ground_truth: bool) -> list[dict]:
    """rank 0 是標準答案（對照組），rank 1..N 是 reference_chunks 原本的排名。"""
    passages = []
    if with_ground_truth:
        passages.append(
            {"id": f"r{row['row']}-gt", "rank": 0, "text": row["ground_truth"], "gt_overlap": 1.0}
        )
    for rank, text in enumerate(row["references"], start=1):
        passages.append(
            {
                "id": f"r{row['row']}-ref{rank:02d}",
                "rank": rank,
                "text": text,
                "gt_overlap": round(gt_overlap(text, row["ground_truth"]), 3),
            }
        )
    return passages


def gate_row(row: dict, with_ground_truth: bool, workers: int) -> list[dict]:
    passages = passages_for(row, with_ground_truth)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(
            pool.map(lambda p: gate(row["query"], p["id"], p["text"]), passages)
        )
    return [
        {
            **passage,
            "gt_match": passage["gt_overlap"] >= GT_OVERLAP_MIN,
            "answers": answer,
            "route": route(answer),
        }
        for passage, answer in zip(passages, answers)
    ]


def preview(text: str, width: int) -> str:
    flat = re.sub(r"\s+", " ", MEDIA.sub("", text)).strip()
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def show_row(row: dict, routed: list[dict]) -> None:
    print(f'\n[第 {row["row"]} 列] "{preview(row["query"], 70)}"')
    print(f"  {'id':<13}{'route':<9}{'rel':>6}{'evid':>6}{'gt':>6}  passage")
    for record in routed:
        a = record["answers"]
        mark = " *" if record["gt_match"] else "  "
        print(
            f"  {record['id']:<13}{record['route']:<9}{a['is_relevant']:>6.2f}"
            f"{a['contains_answer_evidence']:>6.2f}{record['gt_overlap']:>6.2f}{mark}"
            f"{preview(record['text'], 50)}"
        )


# ## 存結果、總結


def write_results(path: Path, rows: list[dict], routed_by_row: dict[int, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row", "passage_id", "rank", "is_ground_truth", "gt_overlap", "gt_match",
                "is_relevant", "contains_answer_evidence", "route",
                "seconds", "input_tokens", "output_tokens", "query", "passage_preview",
            ]
        )
        for row in rows:
            for r in routed_by_row[row["row"]]:
                a = r["answers"]
                writer.writerow(
                    [
                        row["row"], r["id"], r["rank"], r["rank"] == 0, r["gt_overlap"],
                        r["gt_match"], a["is_relevant"], a["contains_answer_evidence"],
                        r["route"], a["seconds"], a["input_tokens"], a["output_tokens"],
                        row["query"], preview(r["text"], 120),
                    ]
                )
    print(f"\n[results saved to {path}]")


def summarize(rows: list[dict], routed_by_row: dict[int, list[dict]]) -> None:
    refs = [r for row in rows for r in routed_by_row[row["row"]] if r["rank"] > 0]
    controls = [r for row in rows for r in routed_by_row[row["row"]] if r["rank"] == 0]
    hits = [r for r in refs if r["gt_match"]]
    others = [r for r in refs if not r["gt_match"]]
    included = [r for r in refs if r["route"] == "include"]

    def share(part: list, whole: list) -> str:
        return f"{len(part)}/{len(whole)}" + (f" ({len(part) / len(whole):.0%})" if whole else "")

    print("\n## 總結\n")
    print(f"  題數                               {len(rows)}")
    print(f"  reference 段數                     {len(refs)}")
    print(f"  判為 include 的 reference          {share(included, refs)}")
    print(f"  有取回標準答案的題數               "
          f"{share([row for row in rows if any(r['gt_match'] and r['rank'] > 0 for r in routed_by_row[row['row']])], rows)}")
    print(f"  標準答案那段被 include             {share([r for r in hits if r['route'] == 'include'], hits)}")
    print(f"  不是標準答案、卻被 include         {share([r for r in others if r['route'] == 'include'], others)}")
    if controls:
        print(f"  對照組（標準答案本身）被 include   {share([r for r in controls if r['route'] == 'include'], controls)}")
    rows_without_include = [
        row["row"] for row in rows
        if row["references"]
        and not any(r["route"] == "include" and r["rank"] > 0 for r in routed_by_row[row["row"]])
    ]
    print(f"  14 段全被 exclude 的列             {rows_without_include or '無'}")
    tokens = sum(r["answers"]["input_tokens"] + r["answers"]["output_tokens"] for r in refs + controls)
    print(f"  tokens（含快取讀回的）             {tokens:,}")


def parse_rows(text: str | None) -> set[int] | None:
    return {int(x) for x in text.split(",") if x.strip()} if text else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="gsc_composer_refs.csv 的路徑")
    parser.add_argument("--rows", help="只跑這幾列（從 0 開始），例如 0,1,5")
    parser.add_argument("--limit", type=int, help="只跑前 N 列")
    parser.add_argument("--no-ground-truth", action="store_true", help="不把標準答案當對照組一起問")
    parser.add_argument("--workers", type=int, default=4, help="同時送幾個請求（預設 4）")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "results.csv", help="結果 CSV")
    args = parser.parse_args()

    print("## Load gsc_composer_refs.csv\n")
    rows = load_rows(args.csv)
    print(f"{args.csv}：{len(rows)} 列，{sum(len(r['references']) for r in rows)} 段 reference")
    wanted = parse_rows(args.rows)
    if wanted is not None:
        rows = [r for r in rows if r["row"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"這次跑 {len(rows)} 列；每段問 {len(PASSAGE_QUESTIONS)} 題：{', '.join(PASSAGE_QUESTIONS)}")

    print("\n## Route each passage in code")
    print("（gt = 標準答案的 8 字片段出現在這段的比例；* = 判定為標準答案那段；rank 0 / -gt 是對照組）")
    routed_by_row = {}
    for row in rows:
        routed_by_row[row["row"]] = gate_row(row, not args.no_ground_truth, args.workers)
        show_row(row, routed_by_row[row["row"]])

    write_results(args.out, rows, routed_by_row)
    summarize(rows, routed_by_row)


if __name__ == "__main__":
    main()
