"""照 https://docs.typesafe.ai/cookbooks/skill_suggestion 的範例，
把 Jev（TypeSafeClient.system_one, TYPESAFE_MODEL = "jev-1.12"）換成本機 jevk5（classifier_jevk5.py）。

每一輪 agent 回話前送兩次分類請求，決定要不要、要提示哪一個 skill：
  1. rank_wide：182 個 skill 一題 Choice 排名，加三題 Noul 問這句話需不需要 skill（平均 < 0.30 就不提示）。
  2. rerank：前三名帶完整說明和 SKILL.md 開頭再問一次 Choice，每個候選再一題「真的做得到嗎」的 Noul
     （最高的 < 0.30 就不提示）。
贏的那個名字放進 agent 系統提示的一個 <skill_relevance> 區塊。最後拿 Claude Haiku 4.5 當 agent，
488 句請求各跑三次（不提示 / jevk5 提示 / 直接給答案），比「載錯 skill」和「不該載卻載了」兩個錯誤率。

jevk5-serve 要開在 8090，量測 agent 要 ANTHROPIC_API_KEY。用法：
    pip install anthropic matplotlib langchain-typesafe httpx2
    python skill_suggestion_make_requests.py                 # 第一次要先產生 requests.json（見下面）
    python skill_suggestion_jevk5.py                         # 照官網跑一遍
    python skill_suggestion_jevk5.py "post this to my bluesky"   # 只看自己的句子會提示哪個 skill（不用 agent）

跟官網一樣的地方：
  * 每一段照順序、照原本的印法：Step 1 roster、Step 2 agent 自己跑、Step 3 排名、Step 4 重排、
    suggest() / suggestion_block()、Step 5 三組比較和圖、playground。
  * 所有常數（SHORTLIST 3、EXCERPT_CHARS 700、GATE_THRESHOLD 0.30、FITS_THRESHOLD 0.30、WORKERS 8）、
    所有題目文字（CHOICE_INSTRUCTIONS、三題 GATE_QUESTIONS、RERANK_INSTRUCTIONS、fits:: 的問法）、
    Hermes 的 IDENTITY / PREAMBLE / FOOTER、skill_view 等四個工具、suggestion_block 的字句、
    agent 模型 claude-haiku-4-5-20251001、三句 DEMO，全部逐字照抄。
  * skill_suggestion/hermes_roster.json 是從 hermes-agent 的 git 歷史重建的（skill_suggestion_build_roster.py），
    Step 1 印出來的數字（182 skills / 33 categories / 16,089 characters / 平均 54、最長 60）
    和頁面上看得到的 11 個 description 都跟官網一模一樣。

環境不同才改的：
  * client.system_one(state=..., questions=..., model=TYPESAFE_MODEL) 換成
    JevK5Classifier.invoke({"state": ..., "questions": ...})；TYPESAFE_MODEL 拿掉（jevk5 不看 model 欄位）。
    TypeSafe 的 RetryPolicy 換成 LangChain 的 with_retry（同 hierarchical_*.py）。
    182 個選項的 Choice 比較久，timeout 放寬到 120 秒。
  * Choice / Noul 從 langchain_typesafe import（官網是 typesafe_sdk），欄位一樣。
  * cooksafe.JsonCache 換成這裡的 JsonCache（同樣的 key：函式名稱加參數，太長就取 sha256）。
    分類器的快取放 skill_suggestion_jevk5/json_cache.json，一定要跟 h2o 版分開（key 沒有模型名稱）。
    agent 的回答（run_turn）放 skill_suggestion/agent_cache.json，和 h2o 版共用：
    baseline、oracle 兩組的輸入跟分類器無關，共用就只要付一次錢，兩版的這兩列也會是同一批樣本，
    差別只剩中間那一列。兩支不要同時跑（兩邊都會整檔重寫 agent_cache.json）。
  * requests.json 官網沒有公開。skill_suggestion_make_requests.py 照頁面的配方重做一份
    （315 句有 skill、171 個 skill、173 句沒有 skill = 85 日常 + 42 技術問題 + 46 roster 沒有的東西，
    頁面上的五句照原文放進去），所以 Step 2、Step 5 的數字只能和官網大致對照。
  * 中間那一組（官網叫 TypeSafe）改叫 jevk5。
  * IPython 的 display(圖) 換成存 PNG（skill_suggestion_jevk5/skill_suggestion.png），Markdown 換成 print。
  * 最後的 playground 連結拿掉（TypeSafe 雲端的功能，同 function_calling_jevk5.py），改成把同一組題目
    寫進 skill_suggestion_jevk5/playground_questions.json。
  * 最後多印一段「跟官網（Jev）對照」：Step 3、Step 4 三句 DEMO 和 Step 5 的表並排。
"""

import functools
import hashlib
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import anthropic
import matplotlib
import matplotlib.pyplot as plt
from langchain_typesafe import Choice, Noul
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from matplotlib.ticker import PercentFormatter

from classifier_jevk5 import JevK5Classifier

matplotlib.use("Agg")  # headless render

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "skill_suggestion"  # hermes_roster.json、requests.json，h2o 版共用
OUT_DIR = HERE / "skill_suggestion_jevk5"
ARM = "jevk5"  # 官網這一組叫 "TypeSafe"

# 官網的 TYPESAFE_MODEL = "jev-1.12" 拿掉：jevk5 服務不看 model 欄位
AGENT_MODEL = (
    "claude-haiku-4-5-20251001"  # the agent under test, pinned so scores are stable
)

SHORTLIST = 3  # candidates carried from the first request into the second
EXCERPT_CHARS = (
    700  # SKILL.md characters each candidate brings; the roster file stores 1600
)
GATE_THRESHOLD = (
    0.30  # mean of the three request nouls, below which nothing is suggested
)
FITS_THRESHOLD = (
    0.30  # a shortlist whose best "does this fit" noul is under this is dropped
)
WORKERS = 8  # small pool: enough to keep a live run to minutes, gentle on rate limits

assert EXCERPT_CHARS <= 1600, (
    "the shipped roster file stores 1600 body characters per skill"
)

# 官網是 TypeSafeClient(api_key=..., base_url=..., timeout=120.0)，預設帶 RetryPolicy；
# JevK5Classifier 是 LangChain Runnable，用 with_retry 做同樣的事（第一次 + 重試 5 次）
client = JevK5Classifier(timeout=120).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=6,
    exponential_jitter_params={"initial": 1.0, "max": 20.0},
)
agent = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "cache-only"))


class JsonCache:
    """取代 cooksafe.JsonCache：函式結果依參數存進一個 JSON 檔，重跑時直接讀。

    key 跟 cooksafe 一樣：函式名稱和參數用 | 接起來，超過 200 字就換成 sha256。
    ThreadPoolExecutor 會同時呼叫，讀寫都加鎖；每存一筆就整檔重寫（先寫暫存檔再換名，中斷不會寫壞）。
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
                    temporary.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
                    temporary.replace(self.path)
            # deep-copy via JSON so callers can mutate results without corrupting the cache
            return json.loads(json.dumps(cached))

        return wrapper


json_cache = JsonCache(OUT_DIR / "json_cache.json")  # 分類器的回答，不能跟 h2o 版共用
agent_cache = JsonCache(DATA_DIR / "agent_cache.json")  # agent 的回答，跟 h2o 版共用


# ## Step 1: load the roster

ROSTER = json.loads((DATA_DIR / "hermes_roster.json").read_text(encoding="utf-8"))
BY_NAME = {skill["name"]: skill for skill in ROSTER}

# Verbatim from hermes-agent agent/prompt_builder.py:build_skills_system_prompt.
PREAMBLE = (
    "## Skills (mandatory)\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant "
    "to your task, you MUST load it with skill_view(name) and follow its instructions. "
    "Err on the side of loading — it is always better to have context you don't need "
    "than to miss critical steps, pitfalls, or established workflows. "
    "Skills contain specialized knowledge — API endpoints, tool-specific commands, "
    "and proven workflows that outperform general-purpose approaches. Load the skill "
    "even if you think you could handle the task with basic tools like web_search or terminal. "
    "Skills also encode the user's preferred approach, conventions, and quality standards "
    "for tasks like code review, planning, and testing — load them even for tasks you "
    "already know how to do, because the skill defines how it should be done here.\n"
    "Whenever the user asks you to configure, set up, install, enable, disable, modify, "
    "or troubleshoot Hermes Agent itself — its CLI, config, models, providers, tools, "
    "skills, voice, gateway, plugins, or any feature — load the `hermes-agent` skill "
    "first. It has the actual commands (e.g. `hermes config set …`, `hermes tools`, "
    "`hermes setup`) so you don't have to guess or invent workarounds.\n"
    "If a skill has issues, fix it with skill_manage(action='patch').\n"
    "After difficult/iterative tasks, offer to save as a skill. "
    "If a skill you loaded was missing steps, had wrong commands, or needed "
    "pitfalls you discovered, update it before finishing.\n"
    "\n"
)
FOOTER = "\n\nOnly proceed without loading a skill if genuinely none are relevant to the task."
IDENTITY = (
    "You are Hermes, a capable AI assistant with access to tools and a library "
    "of skills. You help the user with coding, research, and everyday tasks.\n\n"
)


def render_index() -> str:
    """The body of <available_skills>: skills grouped by category, both sorted by name."""
    by_category = defaultdict(list)
    for skill in ROSTER:
        by_category[skill["category"]].append(skill)
    lines = []
    for category in sorted(by_category):
        lines.append(f"  {category}:")
        for skill in sorted(by_category[category], key=lambda s: s["name"]):
            lines.append(f"    - {skill['name']}: {skill['description']}")
    return "\n".join(lines)


CATALOG_PROMPT = (
    IDENTITY
    + PREAMBLE
    + "<available_skills>\n"
    + render_index()
    + "\n</available_skills>"
    + FOOTER
)


# ## Step 2: score the agent on its own

REQUESTS_PATH = DATA_DIR / "requests.json"
REQUESTS = (
    json.loads(REQUESTS_PATH.read_text(encoding="utf-8"))
    if REQUESTS_PATH.exists()
    else []
)
POSITIVES = [p for p in REQUESTS if p["gold"]]
NEGATIVES = [p for p in REQUESTS if not p["gold"]]

# Verbatim from hermes-agent tools/skills_tool.py:SKILL_VIEW_SCHEMA.
SKILL_VIEW_DESCRIPTION = (
    "Skills allow for loading information about specific tasks and workflows, as "
    "well as scripts and templates. Load a skill's full content or access its "
    "linked files (references, templates, scripts). First call returns SKILL.md "
    "content plus a 'linked_files' dict showing available references/templates/"
    "scripts. To access those, call again with file_path parameter."
)
TOOLS = [
    {
        "name": "skill_view",
        "description": SKILL_VIEW_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill name."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "terminal",
        "description": "Run a shell command on the user's machine and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the user's filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web and return result snippets.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


@agent_cache
def run_turn(model: str, arm: str, request: str, suggestion: str) -> dict:
    """One measured turn. ``arm`` is in the key so each arm samples independently."""
    system = [
        {"type": "text", "text": CATALOG_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    if suggestion:
        system.append({"type": "text", "text": suggestion})  # after the breakpoint
    response = agent.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        tools=TOOLS,
        messages=[{"role": "user", "content": request}],
    )
    usage = response.usage
    return {
        "loaded": [
            str(block.input.get("name", ""))
            for block in response.content
            if block.type == "tool_use" and block.name == "skill_view"
        ],
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
    }


def summarise(turns: dict[str, dict]) -> dict[str, float]:
    """Two failure rates: wrong loads on covered requests, needless ones on uncovered."""
    hits = [turns[p["text"]]["loaded"][:1] == [p["gold"]] for p in POSITIVES]
    over = [bool(turns[p["text"]]["loaded"]) for p in NEGATIVES]
    return {
        # both metrics are errors, so the two columns read the same direction
        "wrong_load": 1 - sum(hits) / len(hits),
        "needless_load": sum(over) / len(over),
    }


def run_arm(arm: str, suggestions: dict[str, str]) -> dict[str, dict]:
    """One measured turn per request, in a small pool. 488 calls."""
    texts = [request["text"] for request in REQUESTS]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        turns = pool.map(
            lambda t: run_turn(AGENT_MODEL, arm, t, suggestions.get(t, "")), texts
        )
        return dict(zip(texts, turns))


# ## Step 3: rank the whole roster

CHOICE_INSTRUCTIONS = (
    "Which of these skills, if any, is the right one to load to help with the "
    "user's latest request?"
)
GATE_QUESTIONS = {
    "acts_on_user_system": (
        "Is the assistant being asked to act on the user's files, accounts, devices, "
        "or online services, rather than only to explain or advise?"
    ),
    "would_follow_documented_procedure": (
        "Would a careful expert answering this consult a specific documented procedure "
        "or set of commands, rather than answering from general understanding?"
    ),
    "prose_suffices": (
        "Could a knowledgeable generalist fully satisfy this request in prose, with "
        "no tools, no documentation, and no access to the user's files or accounts?"
    ),
}
INVERTED = {"prose_suffices"}  # a yes here points away from needing a skill


def build_state(request: str) -> dict:
    return {"request": request, "recent_context": ""}


@json_cache
def rank_wide(request: str) -> dict:
    """Request 1: rank all 182 skills, and score the request for whether a skill applies."""
    questions = {
        "which": Choice(
            instructions=CHOICE_INSTRUCTIONS,
            criteria={skill["name"]: skill["description"] for skill in ROSTER},
        )
    }
    for key, text in GATE_QUESTIONS.items():
        questions[f"gate::{key}"] = Noul(instructions=text)
    started = perf_counter()
    # 官網：client.system_one(state=build_state(request), questions=questions, model=TYPESAFE_MODEL)
    response = client.invoke({"state": build_state(request), "questions": questions})
    ranked = sorted(
        response.answers["which"].probabilities.items(), key=lambda kv: -kv[1]
    )
    values = {
        key.removeprefix("gate::"): answer.noul
        for key, answer in response.answers.items()
        if key.startswith("gate::")
    }
    oriented = [(1.0 - v) if k in INVERTED else v for k, v in values.items()]
    return {
        "ranked": ranked[
            :12
        ],  # more than any shortlist needs, and keeps the cache small
        "gate": sum(oriented) / len(oriented),
        "values": values,
        "seconds": round(perf_counter() - started, 2),
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }


DEMO = [
    "Can you save this recipe as a new note in my 'Recipes' folder in Notes.app so it syncs"
    " to my phone? Just write it up in whatever editor pops up.",
    "Can you put together a pitch deck skeleton (cover, situation overview, comps, precedent"
    " transactions, DCF, LBO) as a .pptx, using our firm-template.pptx for branding and"
    " footnoting each valuation number back to the cell it came from in the model?",
    "Post this announcement to my Mastodon account.",
]


# ## Step 4: rerank the top three

RERANK_INSTRUCTIONS = (
    "Exactly one of these skills is the right one to load for the user's latest "
    "request. Which one? Read what each actually does, not just its name."
)


def rerank_criteria(names: tuple[str, ...], excerpt: int) -> dict[str, str]:
    return {
        name: f"{BY_NAME[name]['description_full']} — {BY_NAME[name]['body'][:excerpt]}"
        for name in names
    }


def rerank_questions(names: tuple[str, ...], excerpt: int) -> dict:
    questions = {
        "which": Choice(
            instructions=RERANK_INSTRUCTIONS, criteria=rerank_criteria(names, excerpt)
        )
    }
    for name in names:
        questions[f"fits::{name}"] = Noul(
            instructions=(
                f"Does the skill '{name}' do the specific thing the user's request asks "
                f"for? It is described as: {BY_NAME[name]['description_full']}"
            )
        )
    return questions


@json_cache
def rerank(request: str, names: tuple[str, ...], excerpt: int) -> dict:
    """Request 2: the same Choice over a shortlist, plus one absolute noul per candidate."""
    started = perf_counter()
    # 官網：client.system_one(state=..., questions=rerank_questions(...), model=TYPESAFE_MODEL)
    response = client.invoke(
        {"state": build_state(request), "questions": rerank_questions(names, excerpt)}
    )
    return {
        "winner": response.answers["which"].choice,
        "fits": {
            key.removeprefix("fits::"): answer.noul
            for key, answer in response.answers.items()
            if key.startswith("fits::")
        },
        "seconds": round(perf_counter() - started, 2),
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }


def suggest(request: str) -> tuple[str, ...]:
    """At most one skill name for a request, or () for "nothing here applies"."""
    wide = rank_wide(request)
    if wide["gate"] < GATE_THRESHOLD:
        return ()
    shortlist = tuple(name for name, _ in wide["ranked"][:SHORTLIST])
    result = rerank(request, shortlist, EXCERPT_CHARS)
    if max(result["fits"].values()) < FITS_THRESHOLD:
        return ()
    return (result["winner"],)


def suggestion_block(names: tuple[str, ...]) -> str:
    """What gets appended after the roster, in the suggestion.

    This string is a measured input rather than prose: it goes to the agent, so it is part
    of every graded turn's cache key. Editing a word here silently invalidates the shipped
    results and costs a live re-run to restore them.
    """
    body = (
        f"Relevant to the current request: {', '.join(names)}. Ignore this if it does not "
        "fit what the user actually asked for."
        if names
        else "No skill in the roster appears relevant to this request."
    )
    return f"\n\n<skill_relevance>\n{body}\n</skill_relevance>"


# ## Step 5: measure the suggestion（圖）

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE, ORANGE = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"

ARM_COLOR = {"baseline": BLUE, ARM: ORANGE, "oracle": MUTED}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.set_axisbelow(True)


def plot_scores(scores: dict[str, dict[str, float]], path: Path) -> None:
    """官網用 display(fig)；這裡存成 PNG。"""
    panels = [
        ("wrong_load", f"wrong loads\n{len(POSITIVES)} covered requests"),
        ("needless_load", f"needless loads\n{len(NEGATIVES)} uncovered requests"),
    ]
    names = list(scores)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        values = [scores[arm][metric] for arm in names]
        bars = ax.bar(
            names,
            values,
            0.58,
            color=[ARM_COLOR[arm] for arm in names],
            # the oracle is a ceiling, not a competitor: gray, and hatched so it never depends
            # on colour alone
            hatch=["", "", "///"],
            edgecolor=SURFACE,
            linewidth=1.2,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.1%}" for v in values],
            padding=3,
            color=INK2,
            fontsize=9,
        )
        ax.set_title(title, loc="left", color=INK2, fontsize=9.5)
        ax.set_ylim(0, max(max(values) * 1.28, 0.01))  # 全部是 0 時 ylim 不能是 (0, 0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("% of those requests - lower is better", color=INK2, fontsize=9)
    fig.suptitle(
        f"Hermes' {len(ROSTER)}-skill roster, {len(REQUESTS)} requests, {AGENT_MODEL}",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[chart saved to {path}]")


# 官網（Jev, jev-1.12）印出來的結果，最後並排對照用
OFFICIAL_WIDE = [
    (0.75, [(0.990, "apple-notes"), (0.010, "computer-use"), (0.000, "concept-diagrams")]),
    (0.76, [(0.700, "powerpoint"), (0.300, "pptx-author"), (0.000, "chroma")]),
    (0.78, [(0.550, "xurl"), (0.140, "computer-use"), (0.080, "openhands")]),
]
OFFICIAL_RERANK = [
    ("apple-notes", {"apple-notes": 0.60, "computer-use": 0.54, "concept-diagrams": 0.01}),
    ("pptx-author", {"powerpoint": 0.73, "pptx-author": 0.38, "chroma": 0.02}),
    ("xurl", {"xurl": 0.56, "computer-use": 0.38, "openhands": 0.05}),
]
OFFICIAL_SCORES = {
    "baseline": {"wrong_load": 0.168, "needless_load": 0.098},
    "TypeSafe": {"wrong_load": 0.073, "needless_load": 0.040},
    "oracle": {"wrong_load": 0.025, "needless_load": 0.012},
}


def suggest_own(requests: list[str]) -> None:
    """python skill_suggestion_jevk5.py "句子" ...：只跑兩次分類請求，看會提示哪個 skill。"""
    for request in requests:
        wide = rank_wide(request)
        print(f'"{request[:78]}"')
        print(f"  needs a skill {wide['gate']:.2f}   " + "  ".join(
            f"{k} {v:.2f}" for k, v in wide["values"].items()
        ))
        for name, probability in wide["ranked"][:SHORTLIST]:
            print(f"    {probability:.3f}  {name:<38}{BY_NAME[name]['description']}")
        if wide["gate"] >= GATE_THRESHOLD:
            shortlist = tuple(name for name, _ in wide["ranked"][:SHORTLIST])
            result = rerank(request, shortlist, EXCERPT_CHARS)
            print(f"  rerank -> {result['winner']}   " + "  ".join(
                f"fits {v:.2f} {k}" for k, v in result["fits"].items()
            ))
        print(suggestion_block(suggest(request)))
        print()


def main() -> None:
    if len(sys.argv) > 1:
        suggest_own(sys.argv[1:])
        return

    # ## Step 1: load the roster
    widths = [len(skill["description"]) for skill in ROSTER]
    print(f"{len(ROSTER)} skills in {len({s['category'] for s in ROSTER})} categories")
    print(f"roster prompt: {len(CATALOG_PROMPT):,} characters")
    print(
        f"index description: {sum(widths) / len(widths):.0f} characters on average, "
        f"{max(widths)} at most"
    )
    print("\none category, as the agent reads it:")
    index_lines = render_index().splitlines()
    start = index_lines.index("  apple:")
    end = next(
        i
        for i in range(start + 1, len(index_lines))
        if not index_lines[i].startswith("    ")
    )
    print("\n".join(index_lines[start:end]))

    # ## Step 2: score the agent on its own
    print("\n## Step 2: score the agent on its own\n")
    if not REQUESTS:
        sys.exit(
            f"{REQUESTS_PATH} 不存在：官網的 requests.json 沒有公開，"
            "先跑 python skill_suggestion_make_requests.py 產生一份"
        )
    print(
        f"{len(REQUESTS)} requests: {len(POSITIVES)} covered by a skill "
        f"({len({p['gold'] for p in POSITIVES})} distinct skills), {len(NEGATIVES)} covered by none"
    )
    print(f"\ncovered   [{POSITIVES[0]['gold']}]  {POSITIVES[0]['text']}")
    print(f"uncovered  {NEGATIVES[0]['text']}")

    baseline = run_arm("baseline", {})
    base_scores = summarise(baseline)
    print(
        f"\nwrong loads    {base_scores['wrong_load']:.1%}   ({len(POSITIVES)} covered requests)"
    )
    print(
        f"needless loads {base_scores['needless_load']:.1%}   ({len(NEGATIVES)} uncovered requests)"
    )

    # where the wrong loads land: a neighbour of the right skill, or somewhere unrelated?
    misses = [
        (p["gold"], baseline[p["text"]]["loaded"][0])
        for p in POSITIVES
        if baseline[p["text"]]["loaded"] and baseline[p["text"]]["loaded"][0] != p["gold"]
    ]
    same_category = sum(
        1
        for gold, got in misses
        if got in BY_NAME and BY_NAME[got]["category"] == BY_NAME[gold]["category"]
    )
    print(
        f"\nof {len(misses)} wrong first picks, {same_category} came from the right skill's own "
        f"category"
    )

    # ## Step 3: rank the whole roster
    print("\n## Step 3: rank the whole roster\n")
    for request in DEMO:
        wide = rank_wide(request)
        verdict = "suggest" if wide["gate"] >= GATE_THRESHOLD else "stay quiet"
        print(f'"{request[:78]}"')
        print(f"  needs a skill {wide['gate']:.2f} -> {verdict}   ({wide['seconds']}s)")
        for name, probability in wide["ranked"][:SHORTLIST]:
            print(f"    {probability:.3f}  {name:<38}{BY_NAME[name]['description']}")
        print()

    # ## Step 4: rerank the top three
    print("## Step 4: rerank the top three\n")
    for request in DEMO:
        wide = rank_wide(request)
        if wide["gate"] < GATE_THRESHOLD:
            print(f'"{request[:78]}"\n  scored too low, nothing suggested\n')
            continue
        shortlist = tuple(name for name, _ in wide["ranked"][:SHORTLIST])
        result = rerank(request, shortlist, EXCERPT_CHARS)
        best = max(result["fits"].values())
        verdict = result["winner"] if best >= FITS_THRESHOLD else "nothing fits"
        print(f'"{request[:78]}"')
        print(f"  was {shortlist[0]} -> {verdict}   ({result['seconds']}s)")
        for name in shortlist:
            print(f"    fits {result['fits'][name]:.2f}  {name}")
        print()

    print(suggestion_block(suggest(DEMO[1])))
    print(suggestion_block(suggest(DEMO[2])))

    # ## Step 5: measure the suggestion
    print("\n## Step 5: measure the suggestion\n")
    texts = [request["text"] for request in REQUESTS]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:  # up to 488 x 2 classifier requests
        suggested = dict(zip(texts, pool.map(suggest, texts)))

    arms = {
        "baseline": {},
        ARM: {
            request["text"]: suggestion_block(suggested[request["text"]])
            for request in REQUESTS
        },
        "oracle": {
            request["text"]: suggestion_block((request["gold"],) if request["gold"] else ())
            for request in REQUESTS
        },
    }
    scores = {
        arm: summarise(run_arm(arm, suggestions)) for arm, suggestions in arms.items()
    }

    print(f"{'run':<10}{'wrong loads':>13}{'needless loads':>16}")
    for arm, row in scores.items():
        print(f"{arm:<10}{row['wrong_load']:>13.1%}{row['needless_load']:>16.1%}")

    def fewer(metric: str) -> str:
        """The plain ratio between the two arms' error rates."""
        if not scores[ARM][metric]:  # 官網沒有這個保護：中間那組是 0% 時比值沒有意義
            return "no errors left, so no ratio"
        return f"{scores['baseline'][metric] / scores[ARM][metric]:.1f}x fewer"

    print(
        f"\nbaseline -> {ARM}:  {fewer('wrong_load')} wrong loads, "
        f"{fewer('needless_load')} needless ones"
    )

    moved = [
        (
            baseline[p["text"]]["loaded"][:1] == [p["gold"]],
            run_turn(AGENT_MODEL, ARM, p["text"], arms[ARM][p["text"]])["loaded"][:1]
            == [p["gold"]],
        )
        for p in POSITIVES
    ]
    print(
        f"of {len(POSITIVES)} covered requests: {sum(not b and a for b, a in moved)} the suggestion "
        f"fixed, {sum(b and not a for b, a in moved)} it broke"
    )

    plot_scores(scores, OUT_DIR / "skill_suggestion.png")

    # ## Open it in the playground（官網產生 TypeSafe playground 連結；這裡把同一組題目存成 JSON）
    print("\n## Open it in the playground\n")
    demo_shortlist = tuple(name for name, _ in rank_wide(DEMO[1])["ranked"][:SHORTLIST])
    questions = rerank_questions(demo_shortlist, EXCERPT_CHARS)
    path = OUT_DIR / "playground_questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "state": build_state(DEMO[1]),
                "questions": {
                    key: question.model_dump(mode="json", exclude_none=True)
                    for key, question in questions.items()
                },
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(questions)} questions for the deck request written to {path}")

    # 官網沒有這段：三句 DEMO 和最後的表，跟官網（Jev）的結果並排
    print("\n## 跟官網（Jev）對照\n")
    for request, (gate, top), (winner, fits) in zip(DEMO, OFFICIAL_WIDE, OFFICIAL_RERANK):
        wide = rank_wide(request)
        print(f'"{request[:78]}"')
        print(f"  needs a skill  {ARM} {wide['gate']:.2f}   jev {gate:.2f}")
        for (name, probability), (official_p, official_name) in zip(
            wide["ranked"][:SHORTLIST], top
        ):
            print(
                f"    {ARM} {probability:.3f}  {name:<30}jev {official_p:.3f}  {official_name}"
            )
        mine = suggest(request)
        print(f"  suggests       {ARM} {mine[0] if mine else '-':<30}jev {winner}")
        print()
    print(f"{'run':<10}{'wrong loads':>13}{'needless loads':>16}      official (jev-1.12)")
    for (arm, row), (official_arm, official) in zip(scores.items(), OFFICIAL_SCORES.items()):
        print(
            f"{arm:<10}{row['wrong_load']:>13.1%}{row['needless_load']:>16.1%}      "
            f"{official_arm:<10}{official['wrong_load']:>7.1%}{official['needless_load']:>8.1%}"
        )
    print(
        "\n（requests.json 是照官網配方重做的，不是同一份，所以 baseline 和 oracle 兩列"
        "也只能大致對照）"
    )


if __name__ == "__main__":
    main()
