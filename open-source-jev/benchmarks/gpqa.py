"""GPQA-Diamond（Idavidrein/gpqa，gated）：198 題研究所程度的科學選擇題。

題目檔 datasets/gpqa/gpqa_diamond.csv。資料集條款要求不要把題目公開到網路上（避免被爬進訓練資料），
所以 datasets/gpqa/ 不要 push 到公開的 repo。
"""

import csv
import random

from .base import Benchmark, letter_grader

# simple-evals 的 GPQA zero-shot CoT 提示
PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def load(folder):
    """選項用 Record ID 當 seed 打亂，所以每個 model、每次執行看到的順序都一樣。"""
    with open(folder / "gpqa_diamond.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    questions = []
    for row in rows:
        options = [row["Correct Answer"].strip()] + [row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
        order = list(range(4))
        random.Random(row["Record ID"]).shuffle(order)
        shuffled = dict(zip("ABCD", (options[i] for i in order)))
        questions.append({
            "id": row["Record ID"],
            "group": row["High-level domain"],
            "prompt": PROMPT.format(question=row["Question"].strip(), **shuffled),
            "answer": "ABCD"[order.index(0)],
        })
    return questions


GPQA = Benchmark(name="gpqa", dataset="GPQA", title="GPQA-Diamond", load=load,
                 grade=letter_grader(r"(?i)answer\s*[:：]\s*\**\s*\(?\$?([ABCD])\b"))
