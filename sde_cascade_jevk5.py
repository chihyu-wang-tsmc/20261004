"""照 https://docs.typesafe.ai/cookbooks/sde_cascade 的範例，
把 Jev（TypeSafeClient.system_one, TS_MODEL = "jev-1.12"）換成本機 jevk5（classifier_jevk5.py）。

結構化資料擷取（SDE）的兩階 cascade：先讓便宜的 gpt-5.4-mini 照 JSON Schema 擷取，再把擷取結果
每個欄位拆成幾題 Noul（true = 這個欄位有問題），一次送給 jevk5 驗證。任何一個欄位的
P(wrong) > FIRE_T（0.7）就升級給 gpt-5.5（reasoning_effort="high"）重抽，否則直接用便宜的結果。
整份紀錄的 __overall__::judge 只印出來對照，不參與升級判斷。

跟官網一樣的地方：
  * 每一段照順序、照原本的印法：Step 1 the data、Step 2 extract with the mini model、
    Step 3 verify、Step 4 the escalation gate、Step 5 escalate to the reasoning model、
    Step 6 what this looks like on 100 prompts（官網只有一張內部結果的圖、沒有程式，這裡只印圖的網址）。
  * 資料一樣是 scrapegraphai/scrapegraphai-100k 同一個 revision 的第 516 列（NYU 活動行事曆頁面）。
  * 所有常數（MINI、REASONING、FIRE_T = 0.7）、EXTRACT_SYSTEM、extract() 的 prompt 和解析、
    寫死的 mini_record、MAIN_QUESTIONS 七題、ABSENCE_*、OVERALL_JUDGE*、is_empty() / field_spec() /
    build_questions()、verify 的 state、any_flag 門檻，全部一字不改。
    送出去的 state 和 9 題，跟官網 playground 連結解出來的原文逐字相同（比對過）。

環境不同才改的：
  * ts.system_one(state=..., questions=..., model=TS_MODEL) 換成
    JevK5Classifier.invoke({"state": ..., "questions": ...})；TS_MODEL 拿掉（jevk5 服務不看 model 欄位），
    也不需要 TYPESAFE_API_KEY。TypeSafe SDK 預設的重試（RetryPolicy：重試 2 次、backoff 0.5~5 秒）
    換成 LangChain 的 with_retry。官網 timeout 30 秒；本機 9 題一起問可能比較久，放寬到 120 秒。
  * Noul / NoulCriteria 從 langchain_typesafe import（官網是 typesafe_sdk），欄位一樣。
  * cooksafe.JsonCache 換成這裡的 JsonCache（key 的算法相同）。官網附的 json_cache.json 沒有公開，
    所以第一次跑會真的問 jevk5，升級時也會真的呼叫 gpt-5.5；之後重跑全部讀快取，要重問就刪掉快取檔。
      - verify() 的快取放 sde_cascade_jevk5/json_cache.json，一定要跟 h2o 版分開：
        key 只有擷取結果、沒有模型名稱，共用會把另一顆的答案當成自己的。
      - extract() 的快取放 sde_cascade/extract_cache.json，跟 h2o 版共用：gpt-5.5 的輸入跟驗證器無關，
        共用就只付一次錢，兩版升級後拿到的也會是同一份結果。兩支不要同時跑（兩邊都會整檔重寫這個檔）。
  * 官網在模組載入時就讀資料；這裡等 main() 再讀。第一次會下載約 470 MB 的 parquet，之後讀
    HuggingFace 的本機快取。
  * Step 3 的 playground 連結（TypeSafe 雲端的功能，同 function_calling_jevk5.py）拿掉，改成把同一組
    state 和 9 題寫進 sde_cascade_jevk5/playground.json；IPython 的 display(Markdown(...)) 換成 print，
    所以不用裝 ipython 和 cooksafe。
  * 沒設 OPENAI_API_KEY 時，Step 5 印一行說明就跳過（官網這時會直接報錯）。
  * 每一段前面多印一行標題；最後多印一段「跟官網（Jev）對照」：9 題的 P(wrong)、門檻的結果、
    最後的擷取結果並排。

jevk5-serve 要開在 8090；Step 5 升級時要 OPENAI_API_KEY。用法：
    pip install openai datasets jsonschema langchain-typesafe httpx2
    python sde_cascade_jevk5.py

官網的結果（Jev, jev-1.12）：description::hallucinated 0.95、description::off_target 0.85 超過 0.7
→ ESCALATE；gpt-5.5 把捏造的 description 改回 ""（registration_open_date 本來就是 ""）。
"""

import functools
import hashlib
import json
import os
import threading
from pathlib import Path

import jsonschema
import openai
from datasets import load_dataset
from langchain_typesafe import Noul, NoulCriteria
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from openai import OpenAI

from classifier_jevk5 import JevK5Classifier

NAME = "jevk5"  # 對照表裡取代官網的 "jev"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "sde_cascade"  # extract() 的快取，h2o 版共用
OUT_DIR = HERE / "sde_cascade_jevk5"  # verify() 的快取、playground.json


# ## Setup

MINI = "gpt-5.4-mini"  # rung 0: cheap + fast
REASONING = "gpt-5.5"  # rung 1: strong, run with reasoning_effort="high"
# 官網的 TS_MODEL = "jev-1.12" 拿掉：jevk5 服務不看 model 欄位
FIRE_T = 0.7  # escalate if any per-field P(wrong) exceeds this; also the "<== FIRES" display marker

# 官網：OpenAI()。沒設 key 時先放一個假的，讀快取照樣能跑，真的要呼叫時才報 AuthenticationError
oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "cache-only"))

# 官網：TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], timeout=30.0)，SDK 預設 RetryPolicy
# 重試 2 次、backoff 0.5~5 秒。JevK5Classifier 是 LangChain Runnable，用 with_retry 做同樣的事
# （第一次 + 重試 2 次）
ts = JevK5Classifier(timeout=120.0).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=3,
    exponential_jitter_params={"initial": 0.5, "max": 5.0},
)


class JsonCache:
    """取代 cooksafe.JsonCache：函式結果依參數存進一個 JSON 檔，重跑時直接讀。

    key 跟 cooksafe 一樣：函式名稱和參數用 | 接起來，超過 200 字就換成 sha256。
    每存一筆就整檔重寫（先寫暫存檔再換名，中斷不會寫壞）；回傳的是用 JSON 複製出來的新物件，
    呼叫端改了也不會動到快取。
    """

    MAX_KEY_CHARS = 200

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    def __call__(self, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            key = "|".join(
                [function.__name__]
                + [str(a) for a in args]
                + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            )
            if len(key) > self.MAX_KEY_CHARS:
                key = hashlib.sha256(key.encode()).hexdigest()
            with self.lock:
                cached = self.data.get(key)
            if cached is None:
                cached = function(*args, **kwargs)
                with self.lock:
                    self.data[key] = cached
                    ordered = {k: self.data[k] for k in sorted(self.data)}
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(ordered, indent=2), encoding="utf-8"
                    )
                    temporary.replace(self.path)
            return json.loads(json.dumps(cached))

        return wrapper


# 官網兩個函式共用一個 json_cache.json；這裡分兩個檔（見開頭說明）
# gpt-5.5 的回答，跟 h2o 版共用
extract_cache = JsonCache(DATA_DIR / "extract_cache.json")
# jevk5 的回答，不能跟 h2o 版共用
json_cache = JsonCache(OUT_DIR / "json_cache.json")


# ## Step 1: the data

SCRAPEGRAPHAI_REVISION = "4bb9fba1dff9181c5acdb60a5a26fea62fa54fe9"

# 官網在模組載入時就讀資料；這裡等 main() 再填。field_spec() 和 verify() 照官網讀這兩個 global
row: dict = {}
schema: dict = {}


def step_1_the_data() -> tuple[str, str]:
    row.update(
        load_dataset(
            "scrapegraphai/scrapegraphai-100k",
            revision=SCRAPEGRAPHAI_REVISION,
            split="train",
        )[516]
    )
    schema.update(json.loads(row["schema"]))
    prompt = row["prompt"]
    content = row["content"]

    print(
        f"""
PROMPT
===========
{prompt}

SCHEMA
===========
{json.dumps(schema, indent=2)}

CONTENT
===========
{content}
""".strip()
    )
    return prompt, content


# ## Step 2: extract with the mini model (text mode)

EXTRACT_SYSTEM = (
    "You extract structured data from documents. Return only values supported by the text. "
    "Follow any value format specified by the schema or its field descriptions."
)


@extract_cache
def extract(
    model: str,
    prompt: str,
    schema: dict,
    content: str,
    *,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> dict:
    user = (
        f"{prompt}\n\nReturn ONLY a JSON object matching this JSON Schema:\n"
        f"{json.dumps(schema, indent=2)}\n\nDocument:\n{content}"
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        kwargs["temperature"] = temperature
    text = oai.chat.completions.create(**kwargs).choices[0].message.content
    # The prompt asks for ONLY a JSON object, so parse the reply as-is -- no regex fishing a
    # substring out of a malformed reply. If ``json.loads`` fails, treat it as an empty extraction
    # (the record-level analog of NaN): every field reads as absent, which the verifier flags and the
    # gate escalates -- the safe direction. Schema-following errors are rare here (see the overview).
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return {}


def step_2_extract_with_the_mini_model() -> dict:
    # Hard-coded canonical fabrication (see note above); a real pipeline would use extract(MINI, prompt, schema, content, temperature=0).
    mini_record = {
        "registration_open_date": "",
        "description": "Registration opens for the fall semester",
    }
    print("mini extraction:\n", json.dumps(mini_record, indent=2))

    # The record is a perfect fit for the JSON Schema -- and still wrong. Schema validation is necessary
    # but not sufficient: it catches structural errors, never semantic ones. That gap is the whole point.
    print(
        "\nschema-valid:", jsonschema.Draft202012Validator(schema).is_valid(mini_record)
    )
    return mini_record


# ## Step 3: verify with TypeSafe（這裡是 jevk5）

# metric -> (question, NoulCriteria)
MAIN_QUESTIONS = {
    "name_desc_mismatch": (
        "Does the `extracted_field` fail to match the field at `path` or the `description` in the "
        "`field_spec`? If the `description` is empty, judge against the `path` alone.",
        NoulCriteria(
            true="the `extracted_field` does not match the field name or its `description`",
            false="the `extracted_field` matches the field name and `description`",
        ),
    ),
    "type_mismatch": (
        "Does the `extracted_field` violate the `type` declared in the `field_spec`?",
        NoulCriteria(
            true="the `extracted_field` violates the declared `type`",
            false="the `extracted_field` conforms to the declared `type`",
        ),
    ),
    "unreasonable": (
        "Is the `extracted_field` one that a reasonable person would not have extracted for this "
        "`field_spec`?",
        NoulCriteria(
            true="a reasonable person would not have extracted this value",
            false="the extraction is reasonable",
        ),
    ),
    "hallucinated": (
        "Is the `extracted_field` unsupported by, or absent from, the source text?",
        NoulCriteria(
            true="the `extracted_field` is a hallucination -- not supported by, or absent "
            "from, the source text",
            false="the `extracted_field` is supported by the source text",
        ),
    ),
    "off_target": (
        "Does the source text fail to genuinely report the thing the `field_spec` describes, so the "
        "value was pulled from incidental text?",
        NoulCriteria(
            true="the source does not genuinely provide this field -- the value was pulled "
            "from incidental text",
            false="the source genuinely reports this field",
        ),
    ),
    "incomplete": (
        "Does the `extracted_field` fail to capture a value the source supports (note whether the "
        "`field_spec` is `required`)?",
        NoulCriteria(
            true="the field is wrongly empty, null, or missing a value the source supports",
            false="the field captures the value the source supports",
        ),
    ),
    "format_violation": (
        "Does the `extracted_field` violate the format or constraints implied by the `description`, "
        "the schema `type`, and the extraction instructions (e.g. date format, units, enum membership)?",
        NoulCriteria(
            true="the `extracted_field` violates the implied format or constraints",
            false="the `extracted_field` satisfies the format and constraints",
        ),
    ),
}
ABSENCE_QUESTION = (
    "The `extracted_field` is empty, null, or an empty collection. Does the source text contain the "
    "information the `field_spec` describes, making the empty result wrong?"
)
ABSENCE_CRITERIA = NoulCriteria(
    true="a value was wrongly omitted", false="returning nothing is correct"
)

# The pipeline also asks one holistic, whole-record head: "should this be escalated?"
OVERALL_JUDGE = (
    "Is this extracted record an incorrect extraction -- some value unsupported by the source or "
    "not conforming to the schema, required information missing or wrong, or some field hallucinated -- "
    "so it should be escalated to a smarter model?"
)
OVERALL_JUDGE_CRITERIA = NoulCriteria(
    true="the record is an incorrect extraction",
    false="the record is a correct extraction",
)


def is_empty(v) -> bool:
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0)


def field_spec(name: str) -> dict:
    """Minimal spec pulled from the schema (unwrapping anyOf/null for optional fields)."""
    p = schema["properties"][name]
    branches = p.get("anyOf") or []
    typ = p.get("type") or next(
        (b["type"] for b in branches if b.get("type") != "null"), "unknown"
    )
    return {
        "path": name,
        "type": typ,
        "description": p.get("description", ""),
        "required": name in schema.get("required", []),
    }


def build_questions(record: dict) -> dict[str, Noul]:
    """The verify question set: one holistic ``__overall__::judge`` head plus a per-field battery,
    keyed ``field::metric`` (mirrors build_verify_prompts)."""
    questions: dict[str, Noul] = {
        "__overall__::judge": Noul(
            instructions=OVERALL_JUDGE, criteria=OVERALL_JUDGE_CRITERIA
        ),
    }
    for name, value in record.items():
        spec = field_spec(name)
        if is_empty(value):
            questions[f"{name}::absence_wrong"] = Noul(
                instructions={
                    "field_spec": spec,
                    "extracted_field": value,
                    "main_question": ABSENCE_QUESTION,
                },
                criteria=ABSENCE_CRITERIA,
            )
            continue
        for metric, (question, criteria) in MAIN_QUESTIONS.items():
            if metric == "type_mismatch" and spec["type"] == "unknown":
                continue
            questions[f"{name}::{metric}"] = Noul(
                instructions={
                    "field_spec": spec,
                    "extracted_field": value,
                    "main_question": question,
                },
                criteria=criteria,
            )
    return questions


def verify_state(record: dict) -> dict:
    """官網寫在 verify() 裡的 state，原樣抽出來，playground.json 也用同一份。"""
    return {
        "system_message": EXTRACT_SYSTEM,
        "instruction": "Extract the structured record from this document",
        "source_text": row["content"],
        "schema": schema,
        "extraction": record,
    }


@json_cache
def verify(record: dict) -> dict[str, float]:
    """Run the whole Noul battery over a record in one TypeSafe call; return ``{field::metric: P(true)}``."""
    state = verify_state(record)
    questions = build_questions(record)
    # 官網：ts.system_one(state=state, questions=questions, model=TS_MODEL).answers
    answers = ts.invoke({"state": state, "questions": questions}).answers
    # 官網還會併上 {"playground_link": make_playground_link(state, questions)}；
    # 這裡改由 write_playground() 寫成 JSON，不放進快取
    return {qid: ans.noul for qid, ans in answers.items()}


def write_playground(record: dict) -> Path:
    """取代官網的 make_playground_link：把送出去的 state 和題目原樣寫成 JSON。"""
    path = OUT_DIR / "playground.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": verify_state(record),
        "questions": {
            qid: question.model_dump(mode="json", exclude_none=True)
            for qid, question in build_questions(record).items()
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def step_3_verify(mini_record: dict) -> dict[str, float]:
    # ### Run the whole battery over the mini extraction
    checks = verify(mini_record)
    # 官網：playground_link = checks.pop("playground_link")，再 display(Markdown(...)) 出連結
    playground_path = write_playground(mini_record)
    print(f"🔗 This verification's state and questions written to {playground_path}\n")

    print(f"{'qid':<40}{'P(wrong)':>9}")
    print("-" * 50)
    for fld, p in sorted(checks.items(), key=lambda c: -c[-1]):
        flag = "  <== FIRES" if p > FIRE_T else ""
        print(f"{fld:<40}{p:>9.2f}{flag}")
    return checks


# ## Step 4: the escalation gate


def step_4_the_escalation_gate(checks: dict[str, float]) -> bool:
    # any_flag is a per-field gate: the holistic __overall__ head is shown above but not part of it
    fired = {
        qid: p
        for qid, p in checks.items()
        if not qid.startswith("__overall__") and p > FIRE_T
    }
    escalate = bool(fired)

    print(
        f"any_flag gate (threshold {FIRE_T}): {'ESCALATE' if escalate else 'ACCEPT cheap result'}"
    )
    for qid, p in sorted(fired.items(), key=lambda c: -c[1]):
        print(f"  fired: {qid}  (P={p:.2f})")
    return escalate


# ## Step 5: escalate to the reasoning model


def step_5_escalate(
    mini_record: dict, escalate: bool, prompt: str, content: str
) -> dict | None:
    try:
        final_record = (
            extract(REASONING, prompt, schema, content, reasoning_effort="high")
            if escalate
            else mini_record
        )
    except openai.AuthenticationError:
        # 官網沒有這段：沒設 OPENAI_API_KEY 時不中斷（錯誤不會寫進快取，設了 key 再跑就會補上）
        print(
            f"(skipped: set OPENAI_API_KEY to have {REASONING} re-extract this record)"
        )
        return None

    print("mini      :", json.dumps(mini_record))
    print("reasoning :", json.dumps(final_record))
    print("\nfield-level diff (mini -> final):")
    for name in mini_record:
        if mini_record[name] != final_record.get(name):
            print(f"  {name}: {mini_record[name]!r}  ->  {final_record.get(name)!r}")
    return final_record


# ## Step 6: what this looks like on 100 prompts

# 官網這一段只有一張 TypeSafe 內部結果的圖（100 個 scrapegraphai prompt 掃門檻的 cost / quality），
# 頁面上沒有程式、也沒公開是哪 100 題和怎麼評分，所以沒辦法照做，這裡只印圖的網址
PARETO_PNG = (
    "https://mintcdn.com/ts-docs/2NirYCl-v96cw05F/cookbooks/sde_cascade/pareto_100prompts.png"
    "?fit=max&auto=format&n=2NirYCl-v96cw05F&q=85&s=ca6731507e07b501a6627616ba0b767a"
)


def step_6_what_this_looks_like_on_100_prompts() -> None:
    print("Internal TypeSafe results (a chart on the page, no code to re-run):")
    print(f"  {PARETO_PNG}")


# ## 跟官網（Jev）對照（官網沒有這段）

# 官網 Step 3 的表（Jev, jev-1.12），照官網的排序
OFFICIAL_CHECKS = {
    "description::hallucinated": 0.95,
    "description::off_target": 0.85,
    "description::unreasonable": 0.58,
    "__overall__::judge": 0.56,
    "description::incomplete": 0.16,
    "registration_open_date::absence_wrong": 0.14,
    "description::format_violation": 0.10,
    "description::name_desc_mismatch": 0.08,
    "description::type_mismatch": 0.02,
}
# 官網 Step 5 gpt-5.5 的結果（欄位順序照官網印出來的）
OFFICIAL_FINAL = {"description": "", "registration_open_date": ""}


def fired_qids(checks: dict[str, float]) -> list[str]:
    """Step 4 的 any_flag：__overall__ 以外超過 FIRE_T 的題目，照 P 由大到小。"""
    fired = [
        qid
        for qid, p in checks.items()
        if not qid.startswith("__overall__") and p > FIRE_T
    ]
    return sorted(fired, key=lambda qid: -checks[qid])


def gate_text(checks: dict[str, float]) -> str:
    fired = fired_qids(checks)
    return f"ESCALATE ({', '.join(fired)})" if fired else "ACCEPT cheap result"


def compare_with_official(checks: dict[str, float], final_record: dict | None) -> None:
    def cell(p: float | None) -> str:
        if p is None:
            return f"{'-':>7} "
        return f"{p:>7.2f}" + ("*" if p > FIRE_T else " ")

    print(f"{'qid':<40}{'jev':>7} {NAME:>7}  {NAME} - jev")
    print("-" * 66)
    for qid, jev in OFFICIAL_CHECKS.items():
        ours = checks.get(qid)
        delta = "" if ours is None else f"{ours - jev:+.2f}"
        print(f"{qid:<40}{cell(jev)}{cell(ours)}  {delta:>7}")
    for qid in checks.keys() - OFFICIAL_CHECKS.keys():  # 照理不會有：兩邊題目一樣
        print(f"{qid:<40}{cell(None)}{cell(checks[qid])}")
    print(
        f"(* = above FIRE_T {FIRE_T}; __overall__::judge is shown but not part of the gate)"
    )

    print(f"\nany_flag gate (threshold {FIRE_T}):")
    print(f"  {'jev':<6}{gate_text(OFFICIAL_CHECKS)}")
    print(f"  {NAME:<6}{gate_text(checks)}")

    print("\nfinal record:")
    print(f"  {'jev':<6}{json.dumps(OFFICIAL_FINAL)}")
    if final_record is None:
        print(f"  {NAME:<6}(skipped: no {REASONING} result, see Step 5)")
    else:
        print(f"  {NAME:<6}{json.dumps(final_record)}")


def main() -> None:
    print("## Step 1: the data\n")
    prompt, content = step_1_the_data()

    print("\n## Step 2: extract with the mini model (text mode)\n")
    mini_record = step_2_extract_with_the_mini_model()

    print(f"\n## Step 3: verify with TypeSafe ({NAME})\n")
    checks = step_3_verify(mini_record)

    print("\n## Step 4: the escalation gate\n")
    escalate = step_4_the_escalation_gate(checks)

    print("\n## Step 5: escalate to the reasoning model\n")
    final_record = step_5_escalate(mini_record, escalate, prompt, content)

    print("\n## Step 6: what this looks like on 100 prompts\n")
    step_6_what_this_looks_like_on_100_prompts()

    print(f"\n## 跟官網（Jev）對照: jev-1.12 vs {NAME}\n")
    compare_with_official(checks, final_record)


if __name__ == "__main__":
    main()
