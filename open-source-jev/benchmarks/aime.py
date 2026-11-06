"""AIME 2025 + 2026（MathArena/aime_2025、MathArena/aime_2026）：共 60 題，答案是 0–999 的整數。

題目檔 datasets/aime/aime_2025.jsonl、aime_2026.jsonl；problem_idx 1–15 是 AIME I、16–30 是 AIME II。
"""

import json
import re

from .base import Benchmark, last_boxed

YEARS = ("2025", "2026")

# MathArena 的 AIME 提示
PROMPT = """Please reason step by step, and put your final answer within \\boxed{{}}. The answer is an integer between 0 and 999 inclusive.

{problem}"""

# 沒有 \boxed 時的備援：最後一個 "answer is 123" / "Answer: 123"
FALLBACK = re.compile(r"(?i)answer\s*(?:is|:|：)\s*\**\s*\$?\s*(\d{1,3})\b")


def load(folder):
    questions = []
    for year in YEARS:
        with open(folder / f"aime_{year}.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = int(row["problem_idx"])
                questions.append({
                    "id": f"aime{year}-{idx:02d}",
                    "group": f"AIME {year} {'I' if idx <= 15 else 'II'}",
                    "prompt": PROMPT.format(problem=row["problem"].strip()),
                    "answer": int(row["answer"]),
                })
    return questions


def to_int(s):
    """\\text{70}、70.0、1,000、70° 這類寫法都轉成整數；不是整數就回 None。"""
    s = re.sub(r"\\(?:text|textbf|mathbf|mathrm)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,!; ]|\\quad|,|\s|\$|°|\^\{?\\circ\}?", "", s)
    m = re.fullmatch(r"0*(\d+)(?:\.0+)?", s)
    return int(m.group(1)) if m else None


def grade(text, q):
    text = text or ""
    boxed = last_boxed(text)
    if boxed is not None:
        answer = to_int(boxed)
    else:
        matches = FALLBACK.findall(text)
        answer = int(matches[-1]) if matches else None
    return answer, answer == q["answer"]


AIME = Benchmark(name="aime", dataset="AIME", title="AIME 2025/2026", load=load, grade=grade)
