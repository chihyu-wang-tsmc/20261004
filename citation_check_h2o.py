"""citation_check_jevk5.py 的 H2O-Lightning-4B 版：照
https://docs.typesafe.ai/cookbooks/citation_check 的範例，
把 Jev（TypeSafeClient.system_one，TYPESAFE_MODEL = "jev-1.12"）換成本機 h2o-lightning（classifier_h2o.py）。

跟 citation_check_jevk5.py 只差這幾處，其餘（RFC 7519、八筆 citation、字串比對、Choice 題目、
AUTO_ACCEPT、每一段的印法）一行都沒改，結果可以直接和 jevk5 版、官網的 Jev 對照：
  * client 問的是 H2OClassifier（預設 http://localhost:8741），網址用 H2O_ENDPOINT 改。
  * 快取和 playground_questions.json 放在 citation_check_h2o/。快取一定要分開：key 只有
    claim 和段落，跟 jevk5 共用同一個檔會把 jevk5 的答案當成 h2o 的。
  * 最後的對照表標 h2o。

官網跟 citation_check_jevk5.py 不同的地方（重建的 citations.json、照抄的 JsonCache、with_retry、
playground 連結換成 JSON）見 citation_check_jevk5.py 開頭，這裡做法相同；
citation_check/rfc7519.txt 和 citation_check/citations.json 也是同一份。

這裡只有一題 Choice，classifier_h2o.py 開頭說的 noul_floor 只夾 Noul，不影響這裡的
relation 機率和 confidence。shim 的 Choice confidence 本來就和 TypeSafe 文件的公式相同。

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    python citation_check_h2o.py
    H2O_ENDPOINT=http://127.0.0.1:8741 python citation_check_h2o.py   # 指定服務網址
"""

import functools
import hashlib
import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from langchain_typesafe import Choice
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from classifier_h2o import H2O_BASE_URL, H2OClassifier

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "citation_check"  # rfc7519.txt、citations.json，h2o 版共用
OUT_DIR = HERE / "citation_check_h2o"  # 快取和 playground_questions.json

# ---- cooksafe.JsonCache（照抄 cooksafe 0.2.0 的 cache.py）----

MAX_KEY_CHARS = (
    200  # keys longer than this (e.g. whole documents) are stored as a sha256 digest
)


class JsonCache:
    """Memoize a function's JSON-serializable results to a single JSON file.

    跟 cooksafe 0.2.0 相同，只多一行：寫檔前先建資料夾（官網的快取放在目前目錄，這裡放在 OUT_DIR）。
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
                    self.path.write_text(json.dumps(ordered, indent=self.indent))
            # deep-copy via JSON so callers can mutate results without corrupting the cache
            return json.loads(json.dumps(self._data[key]))

        return wrapper


# ---- Setup ----

# 官網的 TYPESAFE_MODEL = "jev-1.12" 拿掉：h2o-lightning 的 shim 不看 model 欄位
AUTO_ACCEPT = 0.8  # start high for more human review as you build trust in the model

# 官網：TypeSafeClient(api_key=..., base_url=os.environ.get("TYPESAFE_ENDPOINT"), timeout=120.0)
# H2OClassifier 是 LangChain Runnable，重試用 with_retry（第一次 + 重試 5 次，同 hierarchical_*_h2o.py）
ENDPOINT = os.environ.get("H2O_ENDPOINT", H2O_BASE_URL)
client = H2OClassifier(base_url=ENDPOINT, timeout=120.0).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=6,
    exponential_jitter_params={"initial": 1.0, "max": 20.0},
)
json_cache = JsonCache(OUT_DIR / "json_cache.json")


# ---- Load the source and the citations ----


def load_source() -> str:
    """RFC 7519 verbatim, minus the page headers and footers that interrupt its paragraphs."""
    lines = []
    for line in (DATA_DIR / "rfc7519.txt").read_text().splitlines():
        bare = line.lstrip("\f")
        if re.match(r"Jones, et al\.\s.*\[Page \d+\]$", bare):
            continue
        if re.match(r"RFC 7519\s+JSON Web Token \(JWT\)\s+May 2015$", bare):
            continue
        lines.append(bare)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def split_sections(source: str) -> dict[str, str]:
    """Map each numbered section ("4.1.3") to its text, split on the RFC's header lines."""
    boundary = re.compile(r"(?m)^(?:(\d+(?:\.\d+)*)\.  .+|Appendix [A-Z]\..*)$")
    marks = list(boundary.finditer(source))
    sections = {}
    for mark, nxt in zip(marks, marks[1:] + [None]):
        # an appendix header only terminates the section before it
        if mark.group(1) is None:
            continue
        sections[mark.group(1)] = source[
            mark.start() : nxt.start() if nxt else len(source)
        ].strip()
    return sections


SOURCE = load_source()
SECTIONS = split_sections(SOURCE)
CITATIONS = json.loads((DATA_DIR / "citations.json").read_text())


# ---- Find each quote in the source ----


def normalize(text: str) -> str:
    """Collapse whitespace and fold curly quotes, so a quote matches across line wraps."""
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    return re.sub(r"\s+", " ", text.translate(table)).strip()


def find_quote(sections: dict[str, str], quote: str) -> str | None:
    """The number of the section that contains the quote verbatim, or None."""
    needle = normalize(quote)
    for number in sorted(sections, key=lambda n: [int(p) for p in n.split(".")]):
        if needle in normalize(sections[number]):
            return number
    return None


def locate(sections: dict[str, str], citation: dict) -> tuple[str, str | None]:
    """Step 1 for one citation: a status, plus the section step 2 will read."""
    if citation["quote"] is None:
        return "section-only", sections[citation["section"]]
    number = find_quote(sections, citation["quote"])
    if number is None:
        return "missing", None
    return "found", sections[number]


# ---- Verify whether the source supports the claim ----

QUESTIONS = {
    "relation": Choice(
        instructions="How does the section relate to the claim?",
        criteria={
            "supports": "The section states the claim or directly implies that it is true",
            "contradicts": "The section states the opposite of the claim or implies it is false",
            "says_nothing": "The section does not address what the claim asserts, either way",
        },
    ),
}

RELATION_TO_VERDICT = {
    "supports": "verified",
    "contradicts": "contradicted",
    "says_nothing": "unsupported",
}


@json_cache
def ask(claim: str, section: str) -> dict:
    started = perf_counter()
    # 官網：client.system_one(state=..., questions=QUESTIONS, model=TYPESAFE_MODEL)
    response = client.invoke(
        {"state": {"claim": claim, "section": section}, "questions": QUESTIONS}
    )
    answer = response.answers["relation"]
    return {
        "choice": answer.choice,
        "probabilities": answer.probabilities,
        "confidence": answer.confidence,
        "seconds": round(perf_counter() - started, 2),
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }


def verdict(status: str, answer: dict | None) -> dict:
    """Fold step 1 and step 2 into one of the four labels, plus an auto-or-review flag."""
    if status == "missing":
        # confidence None: no model was called, so there is no model confidence to report
        return {"verdict": "fabricated", "confidence": None, "auto": True}
    return {
        "verdict": RELATION_TO_VERDICT[answer["choice"]],
        "confidence": answer["confidence"],
        "auto": answer["confidence"] >= AUTO_ACCEPT,
    }


def check_citation(sections: dict[str, str], citation: dict) -> dict:
    status, section = locate(sections, citation)
    answer = ask(citation["claim"], section) if section is not None else None
    return {
        "id": citation["id"],
        "status": status,
        "answer": answer,
        **verdict(status, answer),
    }


# 官網（Jev, jev-1.12, 2026-08-16）的結果：quote、relation、conf、verdict、action
OFFICIAL = {
    "epoch_seconds": ("found", "supports", 0.93, "verified", "auto"),
    "aud_reject": ("found", "supports", 0.95, "verified", "auto"),
    "sig_reporting": ("missing", "-", None, "fabricated", "auto"),
    "clock_skew": ("found", "supports", 0.99, "verified", "auto"),
    "exp_required": ("found", "contradicts", 0.99, "contradicted", "auto"),
    "pii_encryption": ("found", "says_nothing", 0.27, "unsupported", "review"),
    "iat_future": ("section-only", "says_nothing", 0.56, "unsupported", "review"),
    "duplicate_names": ("found", "supports", 0.99, "verified", "auto"),
}


def main() -> None:
    # ## Load the source and the citations
    print("## Load the source and the citations\n")
    print(
        f"{len(SOURCE):,} characters, {len(SECTIONS)} numbered sections, {len(CITATIONS)} citations"
    )
    print("\nA citation with a quote:")
    print(json.dumps(CITATIONS[1], indent=2))
    print("\nA claim-only citation:")
    print(json.dumps(next(c for c in CITATIONS if c["quote"] is None), indent=2))

    # ## Find each quote in the source
    print("\n## Find each quote in the source\n")
    for citation in CITATIONS:
        status, section = locate(SECTIONS, citation)
        where = f"section of {len(section):,} chars" if section else "not in the source"
        print(f"{citation['id']:<18}{status:<14}{where}")

    # ## Check every citation
    print("\n## Check every citation\n")
    results = {}
    print(
        f"{'citation':<18}{'quote':<14}{'relation':<14}{'conf':>6}  {'verdict':<13}{'action':>7}"
    )
    for citation in CITATIONS:
        result = check_citation(SECTIONS, citation)
        results[result["id"]] = result
        answer = result["answer"]
        relation = answer["choice"] if answer else "-"
        conf = f"{answer['confidence']:.2f}" if answer else "-"
        action = "auto" if result["auto"] else "review"
        print(
            f"{result['id']:<18}{result['status']:<14}{relation:<14}{conf:>6}"
            f"  {result['verdict']:<13}{action:>7}"
        )

    # 官網寫在表格下面的那句話：幾筆 verified / fabricated / contradicted / unsupported
    counts = {
        label: sum(r["verdict"] == label for r in results.values())
        for label in ("verified", "fabricated", "contradicted", "unsupported")
    }
    review = sum(not r["auto"] for r in results.values())
    print(
        "\n"
        + ", ".join(f"{n} {label}" for label, n in counts.items())
        + f"; {review} sent to a human"
    )

    # ## Open it in the playground（官網產生 TypeSafe playground 連結；這裡把同一組 request 存成 JSON）
    print("\n## Open it in the playground\n")
    example = next(c for c in CITATIONS if c["id"] == "exp_required")
    _, example_section = locate(SECTIONS, example)
    request = {
        "state": {"claim": example["claim"], "section": example_section},
        "questions": {
            qid: question.model_dump(mode="json", exclude_none=True)
            for qid, question in QUESTIONS.items()
        },
    }
    path = OUT_DIR / "playground_questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=1, ensure_ascii=False) + "\n")
    print(f"one citation's claim + section, plus the question, written to {path}")
    print("run the same call live:")
    print(
        f"  curl -s {ENDPOINT.rstrip('/')}/v1/systemone"
        f" -H 'Content-Type: application/json' -d @{path}"
    )

    # 官網沒有這段：八筆和官網（Jev）的結果並排
    print("\n## 跟官網（Jev）對照\n")
    same_verdict = same_action = 0
    for citation_id, result in results.items():
        _, relation, conf, label, action = OFFICIAL[citation_id]
        answer = result["answer"]
        mine_action = "auto" if result["auto"] else "review"
        same_verdict += result["verdict"] == label
        same_action += mine_action == action
        mark = "same" if result["verdict"] == label else "DIFF"
        mine_relation = answer["choice"] if answer else "-"
        mine_conf = f"{answer['confidence']:.2f}" if answer else "-"
        official_conf = f"{conf:.2f}" if conf is not None else "-"
        print(
            f"  {mark}  {citation_id:<18}h2o   {mine_relation:<14}{mine_conf:>6}"
            f"  {result['verdict']:<13}{mine_action:>7}"
        )
        print(
            f"        {'':<18}jev   {relation:<14}{official_conf:>6}"
            f"  {label:<13}{action:>7}"
        )
    print(
        f"\n{same_verdict} of {len(results)} verdicts and {same_action} of {len(results)}"
        " auto/review actions match the official Jev results"
    )


if __name__ == "__main__":
    main()
