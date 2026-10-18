"""重建官網 skill suggestion cookbook 的 hermes_roster.json（skill_suggestion_jevk5.py / _h2o.py 共用）。

官網 https://docs.typesafe.ai/cookbooks/skill_suggestion 的 hermes_roster.json 放在 CookSafe repo，
沒有公開。頁面只說它是 NousResearch/hermes-agent（MIT）某一個 pinned commit 的 182 個 skill，
每筆有 name、category、index 上看到的 description、完整的 description_full、SKILL.md 開頭（body，
存 1600 字）。這裡從 hermes-agent 的 git 歷史照同樣的欄位重建：

  * commit：e444d165807f（main 上 2026-07-31 最後一個 commit；官網的結果是 2026-07-31 render 的）。
  * skill 的來源：skills/、optional-skills/，加上 plugins/google_meet/SKILL.md。三個根目錄各自照
    Hermes 的 agent/prompt_builder.py:_build_snapshot_entry 算 category（路徑去掉最後兩段；
    直接放在根目錄底下的 skill，category 就是自己的資料夾名，例如 yuanbao、google_meet）。
  * description：照 agent/skill_utils.py:extract_skill_description，超過 60 字就切成 57 字加 "..."。
  * name：frontmatter 的 name（Hermes 的 skill_view 認這個），沒有才用資料夾名。

驗證（python skill_suggestion_build_roster.py --check 會再算一次）：跟官網 Step 1 印出來的完全一樣
  182 skills in 33 categories / roster prompt: 16,089 characters /
  index description: 54 characters on average, 60 at most，
  以及頁面上出現過的 11 個 description（apple 那一類四個、computer-use、concept-diagrams、
  powerpoint、pptx-author、chroma、xurl、openhands）逐字相同。
沒辦法驗證的：skill 在檔案裡的順序（這裡照 category、name 排；Step 3 的 Choice 選項照這個順序），
和 body 開頭有沒有 strip（這裡 strip 掉前後空白）。

用法：
    python skill_suggestion_build_roster.py                    # 自動 clone hermes-agent（只抓需要的檔案）
    python skill_suggestion_build_roster.py --repo ~/hermes-agent   # 用現成的 clone
    python skill_suggestion_build_roster.py --check            # 只檢查現有的 hermes_roster.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "skill_suggestion" / "hermes_roster.json"

HERMES_URL = "https://github.com/NousResearch/hermes-agent"
PINNED_COMMIT = "e444d165807f489b5c1ab8e4a612c8d09c2e67a2"
# 三個根目錄；plugins/ 只收 plugins/<name>/SKILL.md，這個 commit 只有 google_meet
ROOTS = ("skills/", "optional-skills/", "plugins/")
SKILL_PROMPT_DESC_LIMIT = 60  # agent/skill_utils.py
BODY_CHARS = 1600  # 官網：the roster file stores 1600 body characters per skill

# 官網頁面上印出來的 description，用來檢查重建的結果
PUBLISHED = {
    "apple-notes": "Manage Apple Notes via memo CLI: create, search, edit.",
    "apple-reminders": "Apple Reminders via remindctl: add, list, complete.",
    "findmy": "Track Apple devices/AirTags via FindMy.app on macOS.",
    "imessage": "Send and receive iMessages/SMS via the imsg CLI on macOS.",
    "computer-use": "Drive the user's desktop in the background — clicking, ty...",
    "concept-diagrams": "Generate flat, minimal educational SVG visuals as HTML.",
    "powerpoint": "Create, read, edit .pptx decks, slides, notes, templates.",
    "pptx-author": "Build PowerPoint decks headless with python-pptx.",
    "chroma": "Embedding database for RAG and semantic search.",
    "xurl": "X/Twitter via xurl CLI: raw post search, posting, DM, media.",
    "openhands": "Delegate coding to OpenHands CLI (model-agnostic, LiteLLM).",
}
PUBLISHED_STATS = (182, 33, 16089, 54, 60)


def git(repo: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """agent/skill_utils.py:parse_frontmatter 的同樣做法。"""
    if content.startswith("﻿"):
        content = content[1:]
    if not content.startswith("---"):
        return {}, content
    end = re.search(r"\n---\s*\n", content[3:])
    if not end:
        return {}, content
    yaml_content = content[3 : end.start() + 3]
    body = content[end.end() + 3 :]
    try:
        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
        parsed = yaml.load(yaml_content, Loader=loader)
        return (parsed if isinstance(parsed, dict) else {}), body
    except Exception:
        frontmatter = {}
        for line in yaml_content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip()
        return frontmatter, body


def skill_files(repo: Path) -> list[tuple[str, str]]:
    """pinned commit 裡所有 skill 的 (SKILL.md 路徑, blob id)。"""
    files = []
    for line in git(repo, "ls-tree", "-r", PINNED_COMMIT).splitlines():
        meta, path = line.split("\t", 1)
        if not path.endswith("/SKILL.md") or not path.startswith(ROOTS):
            continue
        # plugins/ 只收 plugins/<name>/SKILL.md（plugin 自己的說明檔放在更深的地方不算）
        if path.startswith("plugins/") and path.count("/") != 2:
            continue
        files.append((path, meta.split()[2]))
    return files


def fetch_blobs(repo: Path, blob_ids: list[str]) -> None:
    """blob:none 的 clone 缺檔案內容，一次補抓（不然 cat-file 會一個一個慢慢抓）。"""
    checked = git(repo, "cat-file", "--batch-check", stdin="\n".join(blob_ids) + "\n")
    missing = [line.split()[0] for line in checked.splitlines() if line.endswith("missing")]
    if missing:
        subprocess.run(
            ["git", "-C", str(repo), "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q",
             "origin", "--no-tags", "--filter=blob:none", "--stdin"],
            input="\n".join(missing) + "\n",
            text=True,
            check=True,
        )


def build(repo: Path) -> list[dict]:
    files = skill_files(repo)
    fetch_blobs(repo, [blob for _, blob in files])
    roster = []
    for path, blob in files:
        frontmatter, body = parse_frontmatter(git(repo, "cat-file", "blob", blob))
        # _build_snapshot_entry：相對各自根目錄的路徑
        parts = path.split("/")[1:]
        folder = parts[-2]
        category = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
        raw = frontmatter.get("description", "")
        full = str(raw).strip().strip("'\"") if raw else ""
        index = (
            full[: SKILL_PROMPT_DESC_LIMIT - 3] + "..."
            if len(full) > SKILL_PROMPT_DESC_LIMIT
            else full
        )
        roster.append(
            {
                "name": str(frontmatter.get("name", folder)),
                "category": category,
                "description": index,
                "description_full": full,
                "body": body.strip()[:BODY_CHARS],
                "path": path,
            }
        )
    roster.sort(key=lambda s: (s["category"], s["name"]))
    names = [s["name"] for s in roster]
    assert len(names) == len(set(names)), "skill 名稱重複"
    return roster


# 跟 skill_suggestion_jevk5.py 的 IDENTITY / PREAMBLE / FOOTER 相同（官網逐字抄 Hermes 的），
# 這裡另外放一份，因為 import 那兩支會直接把整個 cookbook 跑起來
IDENTITY = (
    "You are Hermes, a capable AI assistant with access to tools and a library "
    "of skills. You help the user with coding, research, and everyday tasks.\n\n"
)
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


def stats(roster: list[dict]) -> tuple[int, int, int, int, int]:
    """官網 Step 1 印的那幾個數字：skill 數、category 數、roster prompt 長度、平均 / 最長 description。"""
    by_category = defaultdict(list)
    for skill in roster:
        by_category[skill["category"]].append(skill)
    lines = []
    for category in sorted(by_category):
        lines.append(f"  {category}:")
        for skill in sorted(by_category[category], key=lambda s: s["name"]):
            lines.append(f"    - {skill['name']}: {skill['description']}")
    prompt = (
        IDENTITY + PREAMBLE + "<available_skills>\n" + "\n".join(lines)
        + "\n</available_skills>" + FOOTER
    )
    widths = [len(s["description"]) for s in roster]
    return (
        len(roster), len(by_category), len(prompt),
        round(sum(widths) / len(widths)), max(widths),
    )


def check(roster: list[dict]) -> bool:
    by_name = {s["name"]: s for s in roster}
    ok = True
    got = stats(roster)
    print(f"skills / categories / prompt chars / mean / max description: {got}")
    if got != PUBLISHED_STATS:
        print(f"  官網是 {PUBLISHED_STATS}，不一樣")
        ok = False
    for name, description in PUBLISHED.items():
        if by_name.get(name, {}).get("description") != description:
            print(f"  {name}: {by_name.get(name, {}).get('description')!r} != {description!r}")
            ok = False
    print("跟官網一致" if ok else "跟官網不一致")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, help="現成的 hermes-agent clone；不給就 clone 到暫存資料夾")
    parser.add_argument("--check", action="store_true", help="只檢查現有的 hermes_roster.json")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check(json.loads(OUT_PATH.read_text(encoding="utf-8"))) else 1)

    with tempfile.TemporaryDirectory() as temporary:
        repo = args.repo
        if repo is None:
            repo = Path(temporary) / "hermes-agent"
            subprocess.run(
                ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", HERMES_URL, str(repo)],
                check=True,
            )
        roster = build(repo.expanduser())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(roster, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(roster)} skills written to {OUT_PATH}")
    check(roster)


if __name__ == "__main__":
    main()
