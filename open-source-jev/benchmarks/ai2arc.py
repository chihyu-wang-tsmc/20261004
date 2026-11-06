"""AI2 ARC 的 test split（allenai/ai2_arc）：ARC-Challenge 1172 題 + ARC-Easy 2376 題，小學到高中的科學選擇題。

題目檔 datasets/ai2arc/arc_challenge_test.jsonl、arc_easy_test.jsonl（每題 id、question、options、answer；
選項數 3–5，答案一律是字母 A–E）。同一個領域有難有易，正好看 router 分不分得出來。
"""

import json

from .base import Benchmark, letter_grader

SUBSETS = {"challenge": "ARC-Challenge", "easy": "ARC-Easy"}

# simple-evals 的選擇題提示；選項數不固定，所以字母是算出來的
PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{options}"""


def load(folder):
    questions = []
    for subset, title in SUBSETS.items():
        with open(folder / f"arc_{subset}_test.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            letters = "ABCDE"[: len(row["options"])]
            questions.append({
                "id": f"{subset}-{row['id']}",
                "group": title,
                "prompt": PROMPT.format(letters=letters, question=row["question"],
                                        options="\n".join(f"{x}) {t}" for x, t in zip(letters, row["options"]))),
                "answer": row["answer"],
            })
    return questions


AI2ARC = Benchmark(name="ai2arc", dataset="AI2ARC", title="AI2 ARC", load=load,
                   grade=letter_grader(r"(?i)answer\s*[:：]\s*\**\s*\(?\$?([A-E])\b"))
