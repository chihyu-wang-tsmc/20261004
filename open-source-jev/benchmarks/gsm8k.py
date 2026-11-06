"""GSM8K 的 test split（openai/gsm8k，main config）：1319 題小學程度的數學應用題，答案是整數。

題目檔 datasets/gsm8k/gsm8k_test.jsonl（每題 id、question、answer、steps；steps 是標準解法的行數，報表用來分組）。
"""

import json
import re

from .base import Benchmark, last_boxed

PROMPT = """Please reason step by step, and put your final answer within \\boxed{{}}. The answer is an integer.

{question}"""

# 沒有 \boxed 時的備援：最後一個 "answer is 18" / "Answer: $1,200"
FALLBACK = re.compile(r"(?i)answer\s*(?:is|:|：)\s*\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\b")


def steps_group(steps):
    return "steps 2-3" if steps <= 3 else "steps 4-5" if steps <= 5 else "steps 6+"


def load(folder):
    with open(folder / "gsm8k_test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"id": row["id"], "group": steps_group(row["steps"]),
             "prompt": PROMPT.format(question=row["question"]), "answer": row["answer"]} for row in rows]


def to_int(s):
    """\\text{18}、18.0、$1,200、-3 這類寫法都轉成整數；不是整數就回 None。"""
    s = re.sub(r"\\(?:text|textbf|mathbf|mathrm)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,!; ]|\\quad|,|\s|\$|\\%|%", "", s)
    m = re.fullmatch(r"(-?)0*(\d+)(?:\.0+)?", s)
    return int(m.group(1) + m.group(2)) if m else None


def grade(text, q):
    text = text or ""
    boxed = last_boxed(text)
    if boxed is not None:
        answer = to_int(boxed)
    else:
        matches = FALLBACK.findall(text)
        answer = to_int(matches[-1]) if matches else None
    return answer, answer == q["answer"]


GSM8K = Benchmark(name="gsm8k", dataset="GSM8K", title="GSM8K", load=load, grade=grade)
