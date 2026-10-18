"""照 https://docs.typesafe.ai/cookbooks/citation_check 的範例，
把 Jev（TypeSafeClient.system_one，TYPESAFE_MODEL = "jev-1.12"）換成本機 jevk5（classifier_jevk5.py）。

LLM 回答問題時附上 citation：一句 claim、出處的段落編號、引用的原文。先用字串比對找引用的原文
在不在 RFC 7519 裡，找不到就是 fabricated，不用問模型；找到的（或只給段落、沒引原文的）再問一題
Choice：這個段落跟 claim 的關係是 supports / contradicts / says_nothing，對應 verified /
contradicted / unsupported。confidence 低於 AUTO_ACCEPT（0.8）的交給人確認。

jevk5-serve 要開在 8090。用法：
    python citation_check_jevk5.py
    JEVK5_ENDPOINT=http://127.0.0.1:8090 python citation_check_jevk5.py   # 指定服務網址

資料（h2o 版共用同一份）：
  citation_check/rfc7519.txt     RFC 7519 原文（rfc-editor.org），官網也是把這份放在 cookbook 旁邊
  citation_check/citations.json  八筆 citation（見下面「環境不同才改的」）

跟官網一樣的地方：
  * 筆記本的每一段照順序、照原本的印法：Setup、Load the source and the citations、
    Find each quote in the source、Verify whether the source supports the claim、
    Check every citation、Open it in the playground。
  * load_source / split_sections / normalize / find_quote / locate / QUESTIONS /
    RELATION_TO_VERDICT / ask / verdict / check_citation 照抄，AUTO_ACCEPT 一樣是 0.8。
  * RFC 切出來一樣是 58,365 個字元、45 個有編號的段落；「Find each quote in the source」
    那張表（每筆的 status 和段落字數）和官網逐字相同。
  * Choice 題目（instructions 和三個選項的說明）跟官網 playground 連結解出來的原文逐字相同。

環境不同才改的：
  * 官網的 citations.json 沒有公開（CookSafe repo 是私有的），citation_check/citations.json 是重建的：
    - aud_reject、iat_future 頁面上有完整印出來，逐字照抄；
    - exp_required 的 claim 是從官網 playground 連結解出來的原文，quote 照頁面說的「逐字引用 4.1.4」；
    - 其他五筆（epoch_seconds、sig_reporting、clock_skew、pii_encryption、duplicate_names）照頁面的
      說明補：id、順序、哪四筆是對的、哪四筆故意改壞、quote 落在哪一段都對得上，
      但 claim 和 quote 的字句是這裡寫的。
    所以跟官網對照時，這五筆的差異可能來自模型，也可能來自字句。
  * client 是 JevK5Classifier，不用 API key，也不送 model（jevk5 服務兩者都不看），
    包上 with_retry（同 hierarchical_*_jevk5.py）。官網用 TYPESAFE_ENDPOINT 改網址，這裡用 JEVK5_ENDPOINT。
  * cooksafe.JsonCache 換成這裡照抄的 JsonCache（同 cooksafe 0.2.0，key 一樣是
    "ask|claim|section" 的 sha256），快取放在 citation_check_jevk5/json_cache.json。
    官網附的快取是 Jev 的答案不能用，所以第一次跑會真的問 jevk5，之後重跑直接讀快取；刪掉就重問。
  * confidence 是 classifier_jevk5.py 照 https://docs.typesafe.ai/confidence 重算的
    (n * p_max - 1) / (n - 1)，跟官網同一條公式，可以直接拿 AUTO_ACCEPT 比。
  * 最後的 playground 連結拿掉（TypeSafe 雲端的功能，同 function_calling_jevk5.py），改成把同一組
    state + questions 寫進 citation_check_jevk5/playground_questions.json。格式就是 /v1/systemone
    的 request body，可以直接用 curl 送給 jevk5。IPython 只有這段用到，所以也拿掉。
  * 「Check every citation」後面多印一行計數（官網寫在表格下面的那句話），
    最後多印一段「跟官網（Jev）對照」：八筆的 relation、confidence、verdict 並排。
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

from classifier_jevk5 import JEVK5_BASE_URL, JevK5Classifier

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "citation_check"  # rfc7519.txt、citations.json，h2o 版共用
OUT_DIR = HERE / "citation_check_jevk5"  # 快取和 playground_questions.json

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

# 官網的 TYPESAFE_MODEL = "jev-1.12" 拿掉：jevk5 服務不看 model 欄位
AUTO_ACCEPT = 0.8  # start high for more human review as you build trust in the model

# 官網：TypeSafeClient(api_key=..., base_url=os.environ.get("TYPESAFE_ENDPOINT"), timeout=120.0)
# JevK5Classifier 是 LangChain Runnable，重試用 with_retry（第一次 + 重試 5 次，同 hierarchical_*_jevk5.py）
ENDPOINT = os.environ.get("JEVK5_ENDPOINT", JEVK5_BASE_URL)
client = JevK5Classifier(base_url=ENDPOINT, timeout=120.0).with_retry(
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
            f"  {mark}  {citation_id:<18}jevk5 {mine_relation:<14}{mine_conf:>6}"
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
