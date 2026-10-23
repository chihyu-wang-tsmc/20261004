"""照 https://docs.typesafe.ai/cookbooks/rerank_typesafe 的範例，
把 Jev（TypeSafeClient.system_one，TYPESAFE_MODEL = "jev-1.12"）換成本機 h2o-lightning（classifier_h2o.py）。

兩步找文件：先用 BM25（fast search）從 3,565 段法院判決（CLERC 資料集，170 列併成一個 corpus）
替 40 題 query 各挑 30 個候選，再對 40 × 30 = 1,200 對 (query, 候選) 各問一題 Noul
「這段候選是不是 query 裡被拿掉的那個引用的出處」，照 noul 由高到低重排。
官網（Jev）的結果：正確段落排第一 5% → 18%、在前 5 名 15% → 35%、在前 10 名 38% → 62%。

跟官網一樣的地方：
  * 每一段照官網的順序，印官網的小標題：Setup、Ranking the passages with fast search、
    Fast search is unlikely to rank the right passage first、Re-ranking it with TypeSafe、
    Re-ranking moves the right answer toward the top。
  * CLERC 的檔案、N_ROWS = 170、N_QUERIES = 40、TOP_K = 30、seed=0、cid()、build_slice()、
    bm25_rankings()、gold_rank()、Noul 的 instructions / criteria、state 的兩個欄位、
    score_candidate() 的回傳、max_workers=12、重排的 sorted()、兩張圖的畫法和顏色，全部一字不改。
    fast search 這一步實測和官網完全一樣：corpus 3,565 段，正確段落在前 30 名 100%、
    第 1 名 5%、前 5 名 15%、前 10 名 38%（37.5%）。
  * noul 同分時 sorted() 保留 BM25 的順序（Python 的 sort 是 stable），官網也是。

環境不同才改的：
  * client：TypeSafeClient(...).system_one(state, questions, model=...) 換成
    H2OClassifier(timeout=120.0).invoke({"state", "questions"})，網址用 H2O_ENDPOINT 改。
    SDK 預設的重試（RetryPolicy：重試 2 次、backoff 0.5~5 秒）換成 LangChain 的 with_retry。
    Noul、NoulCriteria 改從 langchain_typesafe 匯入（轉出來的 JSON 和 typesafe_sdk 的一字不差）。
  * TYPESAFE_MODEL 改成 H2O_MODEL：shim 不看 model 欄位，這裡只留在 score_candidate() 的參數裡當快取 key。
    官網把題目的 JSON 直接當 dict 送給 SDK；H2OClassifier 要 Noul 物件，所以用 Noul.model_validate_json
    轉回來（快取 key 一樣是那串 JSON）。
  * PRICE 改成 (0, 0)：本機服務不按 token 算錢（官網 jev-1.12 是 input $0.042 / 1M、output $0）。
    token 數照印；h2o 用 Qwen 的 tokenizer，數字不會和官網的 1,536,002 / 25,200 一樣。
  * cooksafe.JsonCache 換成照抄的 JsonCache（同 citation_check_h2o.py，key 格式照 cooksafe 0.2.0），
    放在 rerank_h2o/json_cache.json。寫檔改成先寫 .tmp 再換名（同 classifying_rag_passages_h2o.py）：
    快取裡有整個 corpus（約 8 MB），每問完一對就重寫一次，中途按 Ctrl-C 才不會留下寫一半的檔。
    官網附的快取沒有公開，所以第一次跑會真的下載 CLERC（串流讀到湊滿 1,000 列，幾秒）、問 h2o 1,200 次；
    之後重跑全部讀快取。
  * IPython 的 display(fig) 換成存 PNG（rerank_h2o/fast_search.png、rerank_h2o/rerank.png），
    第二張圖的圖例 "+ TypeSafe re-rank" 改成 "+ h2o re-rank"，印出來的 "TypeSafe calls" 改成 "h2o calls"。
  * 官網開頭的 playground 連結（TypeSafe 雲端的功能）換成把同一組 state 和題目寫進
    rerank_h2o/playground_questions.json。連結裡是 query 509948 和它的正確段落（Lebron v. Amtrak），
    兩段文字都和這裡的 corpus 一字不差。這段放到最後印，順便印 h2o 給這一對的 noul。
  * 最後多印一段「跟官網（Jev）對照」：fast search 和重排的命中率、token 數並排，
    再加上 noul 同分的統計（見下面）。

**noul_floor 要注意**（見 classifier_h2o.py 開頭）：shim 預設把 max(p, 1-p) < 0.801 的 Noul 推到邊界，
0.5～0.801 的全部變成 0.801、0.199～0.5 的全部變成 0.199。重排就是照 noul 排，所以落在這一段的候選
彼此同分，順序退回 BM25 原本的順序，模型在這一段分出來的高低全部丟掉。法律引用這種難題，很多候選會
落在中間，重排的效果會因此打折，不能直接和官網的 Jev 比。「跟官網（Jev）對照」那段會印有幾個候選
跟別的候選同分、正確段落同分的有幾題。要比原始機率，就把 serve_config.json 的 noul_floor 改成 0、
重啟 shim，再刪掉 rerank_h2o/json_cache.json 重跑（CLERC 會重新下載，幾秒）。

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    pip install bm25s datasets matplotlib langchain-typesafe httpx2
    python rerank_h2o.py
    H2O_ENDPOINT=http://127.0.0.1:8741 python rerank_h2o.py   # 指定服務網址
"""

import functools
import hashlib
import json
import os
import random
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
from langchain_typesafe import Noul, NoulCriteria
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from classifier_h2o import H2O_BASE_URL, H2O_MODEL, H2OClassifier

matplotlib.use("Agg")  # 官網在 notebook 裡 display(fig)；這裡存成 PNG

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "rerank_h2o"  # 快取、兩張圖、playground_questions.json
MODEL_LABEL = "h2o"

# ---- cooksafe.JsonCache（照抄 cooksafe 0.2.0 的 cache.py）----

MAX_KEY_CHARS = (
    200  # keys longer than this (e.g. whole documents) are stored as a sha256 digest
)


class JsonCache:
    """Memoize a function's JSON-serializable results to a single JSON file.

    跟 cooksafe 0.2.0 相同，只多兩件事：寫檔前先建資料夾（官網的快取放在目前目錄，這裡放在 OUT_DIR），
    以及先寫 .tmp 再換名（快取裡有約 8 MB 的 corpus，每問完一對就重寫一次，中斷時不會留下寫一半的檔）。
    """

    def __init__(self, path: str | Path, *, indent: int | None = 2) -> None:
        self.path = Path(path)
        self.indent = indent
        self._lock = threading.Lock()
        self._data: dict = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    def __call__(self, fn: Callable) -> Callable:
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
                    ordered = {key: self._data[key] for key in sorted(self._data)}
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = self.path.with_suffix(".tmp")
                    temporary_path.write_text(json.dumps(ordered, indent=self.indent))
                    temporary_path.replace(self.path)
            # deep-copy via JSON so callers can mutate results without corrupting the cache
            return json.loads(json.dumps(self._data[key]))

        return wrapper


# ### Setup

# 官網：TYPESAFE_MODEL = "jev-1.12"。shim 不看 model 欄位，這裡只當 score_candidate() 的快取 key
TYPESAFE_MODEL = H2O_MODEL
PRICE = (
    0.0,
    0.0,
)  # $ per 1M tokens (input, output)；本機服務不按 token 算錢（官網 jev-1.12 是 0.042, 0.00）
N_ROWS = 170  # CLERC rows pooled into the shared corpus
N_QUERIES = 40  # rows we evaluate
TOP_K = 30  # candidates the shortlist hands to the re-ranker, per query

# 官網：TypeSafeClient(api_key=..., base_url=os.environ.get("TYPESAFE_ENDPOINT"), timeout=120.0)，
# SDK 預設 RetryPolicy 重試 2 次、backoff 0.5~5 秒。H2OClassifier 是 LangChain Runnable，
# 用 with_retry 做同樣的事（第一次 + 重試 2 次）。
ENDPOINT = os.environ.get("H2O_ENDPOINT", H2O_BASE_URL)
client = H2OClassifier(base_url=ENDPOINT, timeout=120.0).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=3,
    exponential_jitter_params={"initial": 0.5, "max": 5.0},
)
json_cache = JsonCache(OUT_DIR / "json_cache.json")


# ### Ranking the passages with fast search

CLERC_FILE = (
    "https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/main/"
    "teva_train_dir/train_data.jsonl.gz"
)


def cid(text: str) -> str:
    """Corpus id: a content hash, so passages shared across queries dedupe."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@json_cache
def build_slice(n_rows: int, n_queries: int, seed: int) -> dict:
    """Stream CLERC rows, pool ``n_rows`` of them into a corpus, pick ``n_queries`` to evaluate."""
    from datasets import load_dataset  # heavy import, keep local

    stream = load_dataset("json", data_files=CLERC_FILE, streaming=True, split="train")
    rows = []
    for row in stream:
        if (
            row.get("positive_passages")
            and len(row.get("negative_passages") or []) == 20
        ):
            rows.append(row)
        if len(rows) >= 1000:
            break

    rng = random.Random(seed)
    picked = rng.sample(rows, n_rows)
    corpus, pool = {}, []
    for row in picked:
        gold = row["positive_passages"][0]["text"]
        corpus[cid(gold)] = gold
        for neg in row["negative_passages"]:
            corpus[cid(neg["text"])] = neg["text"]
        pool.append(
            {"qid": str(row["query_id"]), "query": row["query"], "gold": cid(gold)}
        )
    # hold out the first 20 pooled rows; evaluate on the rest
    queries = rng.sample(pool[20:], n_queries)
    # sort the corpus by id so every run — live or cache replay — iterates it identically
    return {"queries": queries, "corpus": dict(sorted(corpus.items()))}


def bm25_rankings(corpus: dict[str, str], queries: dict[str, str], k: int = 100):
    """Rank every passage in the corpus by word overlap with each query."""
    import bm25s

    cids = list(corpus)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([corpus[c] for c in cids], stopwords="en"))
    qids = list(queries)
    idxs, _ = retriever.retrieve(
        bm25s.tokenize([queries[q] for q in qids], stopwords="en"), k=min(k, len(cids))
    )
    return {q: [cids[i] for i in idxs[row]] for row, q in enumerate(qids)}


def gold_rank(ranked: list[str], gold: str) -> int | None:
    """1-based rank of the gold id, or None if it isn't in the list."""
    return ranked.index(gold) + 1 if gold in ranked else None


SURFACE, INK, INK2, MUTED = "#f8f8f2", "#34342f", "#34342f", "#7c7c77"
GRID, AXIS, BLUE, GREEN = "#d8d8cf", "#d8d8cf", "#5d76a2", "#6f9b52"


def save(fig, path: Path) -> None:
    """官網：display(fig)。這裡存成 PNG，印出路徑。

    bbox_inches="tight" 同 notebook inline 顯示的預設：第一張圖的標題比 5 吋寬，不加會被切掉。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[chart saved to {path}]")


def bar_chart(labels: list[str], shares: list[float], title: str, path: Path) -> None:
    """A small single-series bar chart of shares (0-1, shown as percentages)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    bars = ax.bar(labels, shares, width=0.55, color=[BLUE, GREEN][: len(labels)])
    ax.bar_label(
        bars,
        labels=[f"{s * 100:.0f}%" for s in shares],
        padding=4,
        color=INK,
        fontsize=11,
    )
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(f"share of {len(queries)} queries", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    plt.tight_layout()
    save(fig, path)  # 官網：display(fig)
    plt.close(fig)


# 官網在模組載入時就建好這幾個；這裡等 main() 再填（bar_chart / chart_before_after 照官網讀全域變數）
corpus: dict[str, str] = {}
queries: dict[str, str] = {}
golds: dict[str, str] = {}


# ### Re-ranking it with TypeSafe

is_cited_source = Noul(
    instructions=(
        "The query excerpt comes from a US federal court opinion and was written "
        "immediately around a citation to a precedent; the citation itself has been "
        "removed. Could the candidate passage be from that cited precedent — does it "
        "establish the specific legal proposition the query excerpt invokes at its "
        "citation point?"
    ),
    criteria=NoulCriteria(
        true=(
            "The candidate passage states or establishes the specific rule, standard, "
            "holding, or fact pattern that the query excerpt attributes to its removed "
            "citation."
        ),
        false=(
            "The candidate passage is merely on a similar topic or doctrine; it does not "
            "supply the specific proposition the query excerpt relies on."
        ),
    ),
)


@json_cache
def score_candidate(model: str, query: str, candidate: str, question_json: str) -> dict:
    """One TypeSafe call about one (query, candidate) pair: a noul, plus token usage."""
    # 官網：question = json.loads(question_json)，SDK 收 dict。H2OClassifier 要 Noul 物件，
    # 所以從同一串 JSON 轉回 Noul。model 只當快取 key：shim 不看 model 欄位，不送
    question = Noul.model_validate_json(question_json)
    # 官網：client.system_one(state=..., questions=..., model=model)
    response = client.invoke(
        {
            "state": {"query_excerpt": query, "candidate_passage": candidate},
            "questions": {"is_cited_source": question},
        }
    )
    return {
        "noul": response.answers["is_cited_source"].noul,
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }


def chart_before_after(
    runs: dict[str, dict[str, list[str]]], thresholds: list[int], path: Path
) -> None:
    """Grouped bar chart: how often the correct passage lands in the top N, for each run."""
    import numpy as np
    import matplotlib.pyplot as plt

    labels = list(runs)
    colors = [BLUE, GREEN]

    def share_in_top(rankings, k):
        return sum(
            gold_rank(rankings[q], golds[q]) in range(1, k + 1) for q in queries
        ) / len(queries)

    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    x = np.arange(len(thresholds))
    width = 0.35
    for i, (label, rankings) in enumerate(runs.items()):
        shares = [share_in_top(rankings, k) for k in thresholds]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, shares, width * 0.92, color=colors[i], label=label)
        ax.bar_label(
            bars,
            labels=[f"{s * 100:.0f}%" for s in shares],
            padding=3,
            color=INK2,
            fontsize=8.5,
        )

    ax.set_xticks(x, [f"top {k}" for k in thresholds])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(f"share of {len(queries)} queries", color=INK2, fontsize=9)
    ax.set_title(
        "How often the correct passage lands near the top",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    plt.tight_layout()
    save(fig, path)  # 官網：display(fig)
    plt.close(fig)


# ---- 官網沒有的：印出兩張圖的數字、playground 的那一對、跟官網（Jev）對照 ----

THRESHOLDS = [1, 5, 10]  # 官網第二張圖的三個門檻


def share_in_top(rankings: dict[str, list[str]], k: int) -> float:
    """正確段落在前 k 名的題目比例（同 chart_before_after 裡的那個）。"""
    return sum(
        gold_rank(rankings[q], golds[q]) in range(1, k + 1) for q in queries
    ) / len(queries)


# 官網開頭 playground 連結裡的那一對：query 509948（Freddie Mac 不是政府機關，引用 Lebron v. Amtrak 的框架），
# 候選是它的正確段落。連結裡兩段文字和這裡 corpus 的一字不差，題目也就是上面的 is_cited_source。
PLAYGROUND_QID = "509948"

# 官網（Jev）的數字：第一張圖、第二張圖、最後那一行
OFFICIAL = {
    "corpus": 3565,
    "in_top_k": 1.00,
    "fast": {1: 0.05, 5: 0.15, 10: 0.38},
    "rerank": {1: 0.18, 5: 0.35, 10: 0.62},
    "calls": 1200,
    "input_tokens": 1_536_002,
    "output_tokens": 25_200,
    "cost": 0.0645,
}


def pct(share: float) -> str:
    return f"{share * 100:.0f}%"


def compare_with_official(
    candidates: dict[str, list[str]],
    reranked: dict[str, list[str]],
    pair_scores: dict[str, dict[str, dict]],
    cost: float,
) -> None:
    calls = [pair_scores[q][c] for q in queries for c in pair_scores[q]]
    ours = {
        "corpus": f"{len(corpus):,}",
        f"fast search, in top {TOP_K}": pct(share_in_top(candidates, TOP_K)),
        **{
            f"fast search, top {k}": pct(share_in_top(candidates, k))
            for k in THRESHOLDS
        },
        **{f"+ re-rank, top {k}": pct(share_in_top(reranked, k)) for k in THRESHOLDS},
        "calls": f"{len(calls):,}",
        "input tokens": f"{sum(c['input_tokens'] for c in calls):,}",
        "output tokens": f"{sum(c['output_tokens'] for c in calls):,}",
        "cost": f"${cost:.4f}",
    }
    official = {
        "corpus": f"{OFFICIAL['corpus']:,}",
        f"fast search, in top {TOP_K}": pct(OFFICIAL["in_top_k"]),
        **{f"fast search, top {k}": pct(OFFICIAL["fast"][k]) for k in THRESHOLDS},
        **{f"+ re-rank, top {k}": pct(OFFICIAL["rerank"][k]) for k in THRESHOLDS},
        "calls": f"{OFFICIAL['calls']:,}",
        "input tokens": f"{OFFICIAL['input_tokens']:,}",
        "output tokens": f"{OFFICIAL['output_tokens']:,}",
        "cost": f"${OFFICIAL['cost']:.4f}",
    }
    print(f"  {'':<26}{'jev (official)':>16}{MODEL_LABEL + ' (this run)':>18}")
    for name, value in ours.items():
        print(f"  {name:<26}{official[name]:>16}{value:>18}")

    # 正確段落的名次，重排前後逐題比
    up = down = same = 0
    for q in queries:
        before = gold_rank(candidates[q], golds[q])
        after = gold_rank(reranked[q], golds[q])
        if before is None:
            continue
        up += after < before
        down += after > before
        same += after == before
    print(
        f"\nThe correct passage moved up in {up}, down in {down}, "
        f"and stayed put in {same} of {len(queries)} queries."
    )

    # 同分：sorted() 是 stable，同分的候選照 BM25 的順序排（見檔頭的 noul_floor）
    tied = 0
    tied_values: Counter[float] = Counter()
    gold_tied = 0
    for q in queries:
        counts = Counter(pair_scores[q][c]["noul"] for c in candidates[q])
        for c in candidates[q]:
            noul = pair_scores[q][c]["noul"]
            if counts[noul] > 1:
                tied += 1
                tied_values[noul] += 1
        gold_noul = pair_scores[q].get(golds[q], {}).get("noul")
        gold_tied += gold_noul is not None and counts[gold_noul] > 1
    print(
        f"{tied:,} of {len(calls):,} candidates share their noul with another candidate "
        f"on the same shortlist, so their order fell back to BM25's."
    )
    if tied_values:
        common = ", ".join(f"{v:.3f} x{n}" for v, n in tied_values.most_common(3))
        print(f"  most common tied values: {common}")
    print(
        f"  the correct passage was in such a tie for {gold_tied} of {len(queries)} queries"
    )


def try_it_yourself(
    qid: str,
    candidates: dict[str, list[str]],
    reranked: dict[str, list[str]],
    pair_scores: dict[str, dict[str, dict]],
) -> None:
    gold = golds[qid]
    request = {
        "state": {"query_excerpt": queries[qid], "candidate_passage": corpus[gold]},
        "questions": {
            "is_cited_source": is_cited_source.model_dump(
                mode="json", exclude_none=True
            )
        },
    }
    path = OUT_DIR / "playground_questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=1, ensure_ascii=False) + "\n")
    print(f"query {qid} + its correct passage, plus the question, written to {path}")
    print(
        f"  {MODEL_LABEL} noul {pair_scores[qid][gold]['noul']:.3f}; "
        f"rank {gold_rank(candidates[qid], gold)} by fast search, "
        f"{gold_rank(reranked[qid], gold)} after re-ranking"
    )
    print("run the same call live:")
    print(
        f"  curl -s {ENDPOINT.rstrip('/')}/v1/systemone"
        f" -H 'Content-Type: application/json' -d @{path}"
    )


def main() -> None:
    print("### Setup\n")
    print(
        f"{MODEL_LABEL} at {ENDPOINT}; N_ROWS={N_ROWS}, N_QUERIES={N_QUERIES}, TOP_K={TOP_K}"
    )

    print("\n### Ranking the passages with fast search\n")
    ds = build_slice(N_ROWS, N_QUERIES, seed=0)
    corpus.update(ds["corpus"])
    queries.update({q["qid"]: q["query"] for q in ds["queries"]})
    golds.update({q["qid"]: q["gold"] for q in ds["queries"]})

    candidates = {
        q: ranked[:TOP_K] for q, ranked in bm25_rankings(corpus, queries).items()
    }

    in_top_k = sum(golds[q] in candidates[q] for q in queries)
    at_rank_1 = sum(candidates[q][0] == golds[q] for q in queries)

    bar_chart(
        [f"In top {TOP_K}", "At rank 1"],
        [in_top_k / len(queries), at_rank_1 / len(queries)],
        f"Where the correct passage lands, {len(queries)} queries against {len(corpus):,} candidates",
        OUT_DIR / "fast_search.png",
    )

    print("\n### Fast search is unlikely to rank the right passage first\n")
    # 官網這段是文字，數字來自上面那張圖
    print(
        f"Out of {len(corpus):,} candidates, the shortlist contains the correct passage for "
        f"{pct(in_top_k / len(queries))} of the {len(queries)} queries, "
        f"but ranks it first only {pct(at_rank_1 / len(queries))} of the time."
    )

    print("\n### Re-ranking it with TypeSafe\n")
    # Each of the 40 queries has 30 candidates, so re-ranking every shortlist means 1,200 independent
    # calls — cheap enough to fire all at once with a thread pool instead of one after another.
    pair_list = [(q, c) for q in queries for c in candidates[q]]
    question_json = is_cited_source.model_dump_json(exclude_none=True)
    print(f"{len(pair_list):,} (query, candidate) pairs, 12 at a time")
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = pool.map(
            lambda p: score_candidate(
                TYPESAFE_MODEL, queries[p[0]], corpus[p[1]], question_json
            ),
            pair_list,
        )
    pair_scores = {q: {} for q in queries}
    for (q, c), result in zip(pair_list, results):
        pair_scores[q][c] = result

    reranked = {
        q: sorted(candidates[q], key=lambda c: -pair_scores[q][c]["noul"])
        for q in queries
    }

    chart_before_after(
        {"Fast search": candidates, f"+ {MODEL_LABEL} re-rank": reranked},
        THRESHOLDS,
        OUT_DIR / "rerank.png",
    )

    calls = [pair_scores[q][c] for q in queries for c in pair_scores[q]]
    input_tokens = sum(call["input_tokens"] for call in calls)
    output_tokens = sum(call["output_tokens"] for call in calls)
    cost = input_tokens / 1_000_000 * PRICE[0] + output_tokens / 1_000_000 * PRICE[1]
    print(
        f"{len(calls)} {MODEL_LABEL} calls used {input_tokens:,} input and "
        f"{output_tokens:,} output tokens, costing ${cost:.4f}."
    )

    print("\n### Re-ranking moves the right answer toward the top\n")
    # 官網這段是三行文字，數字來自上面那張圖
    for k in THRESHOLDS:
        print(
            f"  * Top {k} — {pct(share_in_top(candidates, k))} → {pct(share_in_top(reranked, k))}"
        )

    # ## Try it yourself（官網放在最上面，是 TypeSafe playground 的連結；這裡把同一組 state 和題目存成 JSON，
    # 放到最後才印得出 h2o 給這一對的 noul）
    print("\n## Try it yourself\n")
    qid = PLAYGROUND_QID
    if qid not in queries or golds[qid] not in pair_scores[qid]:
        # CLERC 或 bm25s 改版、抽到的題目不一樣時才會走到這裡
        print(f"query {qid} isn't in this run's slice or shortlist; skipped")
    else:
        try_it_yourself(qid, candidates, reranked, pair_scores)

    # 官網沒有這段
    print("\n## 跟官網（Jev）對照\n")
    compare_with_official(candidates, reranked, pair_scores, cost)


if __name__ == "__main__":
    main()
