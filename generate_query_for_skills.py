"""讀 anthropics/skills 的每個 skill，請 Gemini 想像 agent 會用到這個 skill 的使用情境，生成使用者對 agent 的要求。

用法：
    python generate_query_for_skills.py                      # 全部 skill，每個 30 個情境
    python generate_query_for_skills.py --only pdf docx      # 只跑指定的 skill
    python generate_query_for_skills.py --overwrite          # 已經生成過的也重跑

每個 skill 輸出一個 <out-dir>/<skill>.json，最後把全部合併成 <out-dir>/all.jsonl（一行一個情境）。
已經有輸出檔的 skill 預設跳過，中斷後重跑會接著做。

送給 Gemini 的內容：SKILL.md 全文，加上 skill 資料夾裡其他 .md 參考文件（總長度在 --max-chars 以內），
其餘檔案（程式、XSD schema、字型、圖片等）只列檔名。docx/pptx/xlsx 的 schema 和 claude-api 的各語言文件很大，
全部塞進去沒有幫助，只會變慢變貴。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from llm_models import gemini

DEFAULT_SKILLS_DIR = os.path.expanduser("~/anthropic-skills/skills")
DEFAULT_OUT_DIR = os.path.expanduser("~/jevk5/skill_queries")
DEFAULT_MODEL = "gemini-3.7-flash"


class Scenario(BaseModel):
    scenario: str = Field(description="使用情境：使用者是誰、手上有什麼、為什麼需要做這件事（1-2 句）")
    user_query: str = Field(description="使用者實際打給 agent 的訊息，原文照抄，第一人稱，像真人會打的字")
    explicitness: Literal["explicit", "implicit"] = Field(
        description="explicit：訊息直接提到這個 skill 處理的檔案格式／工具／領域；implicit：只描述目標，要 agent 自己判斷該用這個 skill"
    )


class ScenarioList(BaseModel):
    scenarios: list[Scenario]


PROMPT = """You are helping build an evaluation set for skill routing in an AI agent.

The agent has a library of "skills". A skill is a folder of instructions and resources the agent loads when a user's request matches it. The agent decides whether to load a skill mainly from its name and description, then follows SKILL.md.

Below is the full content of one skill. Read it carefully, then imagine realistic situations in which a user would ask the agent for something and the agent SHOULD load this skill.

Write exactly {n} different scenarios. For each, give the situation and the exact message the user types to the agent.

Requirements:
- Every request must genuinely need this skill; a capable agent without the skill would do a worse job.
- Cover the full range of what the skill can do, not just its headline use. Use the reference files and scripts as hints for less obvious capabilities.
- Vary the users: different jobs, industries, and levels of technical skill.
- Vary the messages: short one-liners and detailed multi-sentence requests; casual and formal; some mention file names, paths or concrete data, some don't.
- Mix explicitness: about half "explicit" (names the file format, tool or domain the skill covers) and half "implicit" (only states the goal; the agent has to realise this skill applies).
- Make the requests concrete and actionable, as if a real person is mid-task. Avoid generic phrasing like "help me with X".
- No two scenarios should be near-duplicates.
- Write user_query in {language}.
{extra}
=== SKILL: {name} ===
description: {description}

{content}
"""


def parse_frontmatter(text):
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        return yaml.safe_load(fm) or {}, body
    return {}, text


def load_skill(skill_dir, max_chars):
    """回傳 (name, description, 要送給模型的內容)。"""
    skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    meta, _ = parse_frontmatter(skill_md)
    parts = [f"--- SKILL.md ---\n{skill_md}"]
    used = len(skill_md)

    others = sorted(p for p in skill_dir.rglob("*") if p.is_file() and p.name != "SKILL.md")
    # 淺層、短的 .md 先放，比較可能是重點參考文件
    docs = sorted((p for p in others if p.suffix == ".md"), key=lambda p: (len(p.relative_to(skill_dir).parts), p.stat().st_size))
    included = set()
    for p in docs:
        text = p.read_text(encoding="utf-8", errors="replace")
        if used + len(text) > max_chars:
            continue
        parts.append(f"--- {p.relative_to(skill_dir)} ---\n{text}")
        used += len(text)
        included.add(p)

    listed = [str(p.relative_to(skill_dir)) for p in others if p not in included]
    if listed:
        parts.append("--- other files in this skill (names only) ---\n" + "\n".join(listed))
    return meta.get("name", skill_dir.name), meta.get("description", ""), "\n\n".join(parts)


def generate(llm, name, description, content, n, language):
    """呼叫模型直到湊滿 n 個情境；不足就把已有的給它看，再要剩下的。"""
    structured = llm.with_structured_output(ScenarioList)
    scenarios = []
    for _ in range(4):
        need = n - len(scenarios)
        if need <= 0:
            break
        extra = ""
        if scenarios:
            existing = "\n".join(f"- {s.user_query}" for s in scenarios)
            extra = f"- These scenarios already exist; do not repeat or paraphrase them:\n{existing}\n"
        prompt = PROMPT.format(n=need, language=language, extra=extra, name=name, description=description, content=content)
        result = structured.invoke(prompt)
        seen = {s.user_query.strip() for s in scenarios}
        scenarios += [s for s in result.scenarios if s.user_query.strip() not in seen][:need]
    if len(scenarios) < n:
        raise RuntimeError(f"只生成了 {len(scenarios)}/{n} 個情境")
    return scenarios


def process(skill_dir, args, llm):
    out_path = Path(args.out_dir) / f"{skill_dir.name}.json"
    if out_path.exists() and not args.overwrite:
        return skill_dir.name, "skip"
    name, description, content = load_skill(skill_dir, args.max_chars)
    start = time.time()
    scenarios = generate(llm, name, description, content, args.n, args.language)
    out = {
        "skill": name,
        "description": description,
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenarios": [{"id": f"{name}-{i:02d}", **s.model_dump()} for i, s in enumerate(scenarios, 1)],
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return name, f"{len(scenarios)} 個情境，{time.time() - start:.0f}s"


def merge(out_dir):
    rows = 0
    with open(Path(out_dir) / "all.jsonl", "w", encoding="utf-8") as f:
        for p in sorted(Path(out_dir).glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            for s in d["scenarios"]:
                f.write(json.dumps({"skill": d["skill"], **s}, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills-dir", default=DEFAULT_SKILLS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-n", type=int, default=30, help="每個 skill 的情境數")
    ap.add_argument("--language", default="English", help="user_query 用的語言，例如 English、Traditional Chinese (Taiwan)")
    ap.add_argument("--max-chars", type=int, default=150_000, help="每個 skill 送給模型的內容長度上限（字元）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="+", help="只跑這些 skill（資料夾名稱）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    skill_dirs = sorted(p.parent for p in Path(args.skills_dir).glob("*/SKILL.md"))
    if args.only:
        skill_dirs = [d for d in skill_dirs if d.name in args.only]
    if not skill_dirs:
        sys.exit(f"{args.skills_dir} 底下找不到 skill")
    os.makedirs(args.out_dir, exist_ok=True)

    llm = gemini(args.model, max_retries=6)
    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, d, args, llm): d.name for d in skill_dirs}
        for fut in as_completed(futures):
            try:
                name, status = fut.result()
                print(f"[ok]   {name}: {status}", flush=True)
            except Exception as e:
                failed.append(futures[fut])
                print(f"[fail] {futures[fut]}: {type(e).__name__}: {e}", flush=True)

    print(f"all.jsonl：{merge(args.out_dir)} 筆，輸出在 {args.out_dir}")
    if failed:
        sys.exit(f"失敗的 skill：{' '.join(failed)}（重跑會只跑這些）")


if __name__ == "__main__":
    main()
