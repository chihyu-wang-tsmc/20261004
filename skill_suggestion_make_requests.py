"""產生 skill_suggestion_jevk5.py / _h2o.py 共用的 skill_suggestion/requests.json（488 句請求）。

官網 https://docs.typesafe.ai/cookbooks/skill_suggestion 的 requests.json 放在 CookSafe repo，沒有公開。
頁面只描述了它是怎麼來的，這支照同樣的配方重做一份：

  * 488 句單輪請求：315 句剛好由一個 skill 負責（共 171 個不同的 skill），173 句沒有任何 skill 負責。
  * 有 skill 的那 315 句，官網是 Claude Sonnet 5 讀每個 skill 自己的 SKILL.md 寫的；這裡也一樣，
    把 hermes_roster.json 的 description_full 和 body（SKILL.md 開頭 1600 字）給模型，
    請它寫「需要這個 skill、而且 roster 裡其他 skill 都不對」的請求。
  * 沒有 skill 的 173 句照官網的三種：85 句日常請求、42 句沒有 skill 管的技術問題
    （例如 explain what a monad is）、46 句要求某個 roster 剛好沒有的東西
    （例如 roster 只有 X/Twitter 時的 post this to Mastodon）。寫的時候把整份 roster 給模型看，
    叫它避開。
  * 頁面上印出來的五句照原文放進去：1password 那句（POSITIVES[0]）、Trello 那句（NEGATIVES[0]），
    以及 Step 3 的三句 DEMO（Notes.app、pitch deck、Mastodon）。

哪 171 個 skill 有題目、哪 27 個只有一句，頁面沒說；這裡固定成：EXCLUDED 的 11 個不出題
（Hermes 自己內部用的、和別的 skill 重疊太多而標準答案會不明確的，以及跟繞過模型防護有關的），
剩下 171 個裡用固定 seed 抽 27 個只出一句，其餘 144 個各兩句（144 × 2 + 27 = 315）。

所以這份 requests.json 跟官網的不是同一份，skill_suggestion_*.py 跑出來的數字只能和官網的表
大致對照，不會逐位相同。

產生的結果先存在 skill_suggestion/requests_cache.json（中斷後重跑會接著做），全部湊齊才寫
skill_suggestion/requests.json。

用法：
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=...
    python skill_suggestion_make_requests.py                 # 預設 claude-sonnet-5（官網用的那一顆）
    python skill_suggestion_make_requests.py --model claude-sonnet-5-5
    python skill_suggestion_make_requests.py --check         # 只檢查現有的 requests.json
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "skill_suggestion"
ROSTER_PATH = DATA_DIR / "hermes_roster.json"
OUT_PATH = DATA_DIR / "requests.json"
CACHE_PATH = DATA_DIR / "requests_cache.json"

DEFAULT_MODEL = "claude-sonnet-5"  # 官網：written by Claude Sonnet 5
WORKERS = 8
SEED = 0

# 這 11 個 skill 不出題，所以有題目的 skill 是 182 - 11 = 171 個，和官網一樣
EXCLUDED = (
    "godmode",
    "obliteratus",
    "inspecting-hermes-desktop-dom",
    "hermes-s6-container-supervision",
    "hermes-agent-skill-authoring",
    "subagent-driven-development",
    "simplify-code",
    "spike",
    "dogfood",
    "kanban-video-orchestrator",
    "pinecone-research",
)
SINGLE_COUNT = 27  # 只出一句的 skill 數
UNCOVERED_COUNTS = {"everyday": 85, "technical": 42, "missing": 46}
BATCH = 15  # 沒有 skill 的請求一次請模型寫幾句

# 官網頁面上印出來的原文
PINNED_COVERED = [
    (
        "1password",
        "I've got a config.yaml with `{{ op://app-prod/db/password }}` placeholders in it — can "
        "you set up my project to pull the real values in at runtime instead of hardcoding them?",
    ),
    (
        "apple-notes",
        "Can you save this recipe as a new note in my 'Recipes' folder in Notes.app so it syncs"
        " to my phone? Just write it up in whatever editor pops up.",
    ),
    (
        "pptx-author",
        "Can you put together a pitch deck skeleton (cover, situation overview, comps, precedent"
        " transactions, DCF, LBO) as a .pptx, using our firm-template.pptx for branding and"
        " footnoting each valuation number back to the cell it came from in the model?",
    ),
]
PINNED_UNCOVERED = [
    ("missing", "Add these three cards to our Trello backlog."),
    ("missing", "Post this announcement to my Mastodon account."),
]

COVERED_PROMPT = """You are writing test data for an AI agent that picks which skill to load.

The agent has a roster of {roster_size} skills. Below is the full index, then the complete \
description and the opening of the SKILL.md for one skill: {name}.

Write {n} different single-turn requests that a user might send to the agent, where loading \
{name} is clearly the right call and no other skill in the index would be the right one.

Requirements:
- Base each request on what the SKILL.md says the skill actually does.
- Write as the user, in the first person, the way a real person types to an assistant: \
concrete and mid-task, with file names, data, or details where natural.
- Vary length and tone. Do not name the skill, and do not just restate its description.
- Each request must make sense on its own, with no prior conversation.
- Write in English.
{existing}
=== ROSTER INDEX ===
{index}

=== SKILL: {name} ===
description: {description}

{body}
"""

UNCOVERED_KINDS = {
    "everyday": (
        "everyday requests: ordinary things people ask an assistant in daily life (advice, "
        "writing help, planning, recommendations, general knowledge) that a capable assistant "
        "answers directly, without any skill"
    ),
    "technical": (
        "technical questions that no skill serves: conceptual questions about programming, "
        "computer science, mathematics or engineering that are answered with an explanation "
        "(for example: explain what a monad is), and that do not ask the assistant to operate "
        "any tool, service, file or account"
    ),
    "missing": (
        "requests for something specific the roster has no skill for: the user asks the "
        "assistant to act on a particular product, service, platform or device that is not "
        "covered by any skill in the index, even when a neighbouring one is (for example, "
        "with a skill for X/Twitter and none for Mastodon: post this announcement to my "
        "Mastodon account)"
    ),
}

UNCOVERED_PROMPT = """You are writing test data for an AI agent that picks which skill to load.

The agent has a roster of {roster_size} skills, listed in the index below. Write {n} different \
single-turn user requests of this kind:

{kind}

None of them may be served by any skill in the index: an agent that loads any skill for them \
has made a mistake. Read the index carefully and stay clear of everything it covers.

Requirements:
- Write as the user, in the first person, the way a real person types to an assistant.
- Vary topic, length and tone. Each request must make sense on its own.
- Write in English.
{existing}
=== ROSTER INDEX ===
{index}
"""


# 每次呼叫才填的兩個位置。SKILL.md 裡常有 { }，所以不能再用 str.format 填，改用字串取代
N_MARK, EXISTING_MARK = "<<N>>", "<<EXISTING>>"


class Requests(BaseModel):
    requests: list[str]


def roster_index(roster: list[dict]) -> str:
    return "\n".join(
        f"- {skill['name']}: {skill['description_full'][:200]}" for skill in roster
    )


def plan(roster: list[dict]) -> dict[str, int]:
    """每個 skill 要幾句（含頁面上那三句原文）。"""
    names = [s["name"] for s in roster if s["name"] not in EXCLUDED]
    assert len(names) == 171, len(names)
    pinned = {gold for gold, _ in PINNED_COVERED}
    single = set(random.Random(SEED).sample(sorted(set(names) - pinned), SINGLE_COUNT))
    return {name: 1 if name in single else 2 for name in names}


class Cache:
    """job id -> 模型寫好的請求，每完成一個 job 就存檔，中斷後重跑接著做。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key: str) -> list[str] | None:
        with self.lock:
            return self.data.get(key)

    def put(self, key: str, value: list[str]) -> None:
        with self.lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(self.path)


def ask(client: anthropic.Anthropic, model: str, prompt: str, n: int) -> list[str]:
    """請模型寫 n 句；不夠就把已有的給它看，再要剩下的（最多四輪）。"""
    got: list[str] = []
    for _ in range(4):
        need = n - len(got)
        if need <= 0:
            break
        existing = ""
        if got:
            listed = "\n".join(f"- {text}" for text in got)
            existing = f"- These already exist; do not repeat or paraphrase them:\n{listed}\n"
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            messages=[
                {
                    "role": "user",
                    "content": prompt.replace(N_MARK, str(need)).replace(EXISTING_MARK, existing),
                }
            ],
            output_format=Requests,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(f"model returned no requests (stop_reason={response.stop_reason})")
        seen = set(got)
        for text in response.parsed_output.requests:
            text = text.strip()
            if text and text not in seen:
                got.append(text)
                seen.add(text)
    if len(got) < n:
        raise RuntimeError(f"only {len(got)} of {n} requests written")
    return got[:n]


def covered_job(client, model, cache, roster, skill, n) -> None:
    key = f"covered::{skill['name']}"
    if cache.get(key) is not None or n == 0:
        return
    prompt = COVERED_PROMPT.format(
        roster_size=len(roster),
        name=skill["name"],
        description=skill["description_full"],
        body=skill["body"],
        index=roster_index(roster),
        n=N_MARK,
        existing=EXISTING_MARK,
    )
    cache.put(key, ask(client, model, prompt, n))
    print(f"[ok] {key}: {n}", flush=True)


def uncovered_job(client, model, cache, roster, kind, batch, n, previous: list[str]) -> None:
    key = f"{kind}::{batch}"
    if cache.get(key) is not None:
        return
    prompt = UNCOVERED_PROMPT.format(
        roster_size=len(roster),
        kind=UNCOVERED_KINDS[kind],
        index=roster_index(roster),
        n=N_MARK,
        existing=EXISTING_MARK,
    )
    # 同一種的前幾批也給模型看，避免每批都寫出差不多的東西
    if previous:
        listed = "\n".join(f"- {text}" for text in previous)
        prompt = prompt.replace(
            EXISTING_MARK, f"- Avoid these, which were already written:\n{listed}\n{EXISTING_MARK}"
        )
    cache.put(key, ask(client, model, prompt, n))
    print(f"[ok] {key}: {n}", flush=True)


def assemble(roster: list[dict], counts: dict[str, int], cache: Cache) -> list[dict]:
    pinned_by_gold: dict[str, list[str]] = {}
    for gold, text in PINNED_COVERED:
        pinned_by_gold.setdefault(gold, []).append(text)
    positives = []
    for name in sorted(counts):
        texts = pinned_by_gold.get(name, []) + (cache.get(f"covered::{name}") or [])
        positives += [{"text": t, "gold": name, "kind": "covered"} for t in texts[: counts[name]]]
    negatives = [{"text": t, "gold": None, "kind": k} for k, t in PINNED_UNCOVERED]
    for kind, total in UNCOVERED_COUNTS.items():
        have = sum(1 for k, _ in PINNED_UNCOVERED if k == kind)
        texts = []
        batch = 0
        while cache.get(f"{kind}::{batch}") is not None:
            texts += cache.get(f"{kind}::{batch}")
            batch += 1
        negatives += [{"text": t, "gold": None, "kind": kind} for t in texts[: total - have]]
    # 同一句不能出現兩次（量測時用句子當 key）
    seen, requests = set(), []
    for request in positives + negatives:
        if request["text"] not in seen:
            seen.add(request["text"])
            requests.append(request)
    return requests


def check(requests: list[dict], roster: list[dict]) -> bool:
    names = {s["name"] for s in roster}
    positives = [r for r in requests if r["gold"]]
    negatives = [r for r in requests if not r["gold"]]
    kinds = {k: sum(1 for r in negatives if r["kind"] == k) for k in UNCOVERED_COUNTS}
    print(
        f"{len(requests)} requests: {len(positives)} covered by a skill "
        f"({len({r['gold'] for r in positives})} distinct skills), {len(negatives)} covered by none {kinds}"
    )
    ok = (
        len(requests) == 488
        and len(positives) == 315
        and len({r["gold"] for r in positives}) == 171
        and kinds == UNCOVERED_COUNTS
        and all(r["gold"] in names for r in positives)
        and len({r["text"] for r in requests}) == len(requests)
        and positives[0]["gold"] == "1password"
        and negatives[0]["text"] == PINNED_UNCOVERED[0][1]
    )
    print("跟官網的配方一致" if ok else "跟官網的配方不一致")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--check", action="store_true", help="只檢查現有的 requests.json")
    args = parser.parse_args()

    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    if args.check:
        sys.exit(0 if check(json.loads(OUT_PATH.read_text(encoding="utf-8")), roster) else 1)

    counts = plan(roster)
    by_name = {s["name"]: s for s in roster}
    pinned = {}
    for gold, _ in PINNED_COVERED:
        pinned[gold] = pinned.get(gold, 0) + 1
    client = anthropic.Anthropic(max_retries=6)
    cache = Cache(CACHE_PATH)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(
                covered_job, client, args.model, cache, roster, by_name[name],
                max(0, n - pinned.get(name, 0)),
            )
            for name, n in counts.items()
        ]
        for job in jobs:
            job.result()

    # 沒有 skill 的請求一批一批寫，後面的批次看得到前面寫過的
    for kind, total in UNCOVERED_COUNTS.items():
        remaining = total - sum(1 for k, _ in PINNED_UNCOVERED if k == kind)
        previous = [t for k, t in PINNED_UNCOVERED if k == kind]
        batch = 0
        while remaining > 0:
            n = min(BATCH, remaining)
            uncovered_job(client, args.model, cache, roster, kind, batch, n, previous)
            previous += cache.get(f"{kind}::{batch}")
            remaining -= n
            batch += 1

    requests = assemble(roster, counts, cache)
    if not check(requests, roster):
        sys.exit("數量不對（多半是有重複的句子被拿掉了）：刪掉 requests_cache.json 裡相關的 job 再跑一次")
    OUT_PATH.write_text(json.dumps(requests, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
