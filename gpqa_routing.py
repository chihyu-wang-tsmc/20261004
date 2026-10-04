"""用 GPQA-Diamond 做 model routing 的 ground truth，並統計 Jev router（Winnow / jevk5）省下多少次大 model。

流程和 ground truth 的定義見 routing_bench.py。在 ~/jevk5 底下、jevk5 環境執行：

    python gpqa_routing.py answer --model qwen3.8-27b       # deep1_router.py 的 fast
    python gpqa_routing.py answer --model qwen3.8-flash     # deep1_router.py 的 powerful
    python gpqa_routing.py route                            # Winnow 當 router（server 開在 8091）
    python gpqa_routing.py route --router jevk5             # jevk5 當 router（jevk5-serve 開在 8090）
    python gpqa_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash                  # 統計 Winnow
    python gpqa_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash --router jevk5   # 統計 jevk5
    python evaluate_winnow.py --tier gpqa                   # 用 JevBench 的指標評測 router
    python evaluate_jevk5.py --tier gpqa                    # 同上，評測 jevk5

題目在 gpqa/gpqa_diamond.csv（HuggingFace 的 Idavidrein/gpqa，gated，需要接受條款）。
GPQA 的資料集條款要求不要把題目公開到網路上（避免被爬進訓練資料），
所以 gpqa/ 底下的檔案不要 push 到公開的 git repo。
"""

import csv
import os
import random
import re

from routing_bench import HERE, Bench, main

CSV_PATH = os.path.join(HERE, "gpqa", "gpqa_diamond.csv")

# simple-evals 的 GPQA zero-shot CoT 提示
PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""

ANSWER_RE = re.compile(r"(?i)answer\s*[:：]\s*\**\s*\(?\$?([ABCD])\b")


def load_questions(args):
    """讀 GPQA-Diamond，每題的選項用 Record ID 當 seed 打亂，所以每個 model、每次執行看到的順序都一樣。"""
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    questions = []
    for row in rows:
        options = [row["Correct Answer"].strip()] + [
            row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)
        ]
        order = list(range(4))
        random.Random(row["Record ID"]).shuffle(order)
        shuffled = [options[i] for i in order]
        prompt = PROMPT.format(question=row["Question"].strip(), **dict(zip("ABCD", shuffled)))
        questions.append(
            {
                "id": row["Record ID"],
                "domain": row["High-level domain"],
                "subdomain": row["Subdomain"],
                "prompt": prompt,
                "correct": "ABCD"[order.index(0)],
            }
        )
    return questions


def grade(text, q):
    matches = ANSWER_RE.findall(text or "")
    letter = matches[-1].upper() if matches else None
    return letter, letter == q["correct"]


GPQA = Bench(
    name="gpqa",
    title="GPQA-Diamond",
    load_questions=load_questions,
    grade=grade,
    source="GPQA-Diamond (Idavidrein/gpqa); do not publish",
)

if __name__ == "__main__":
    main(GPQA, "用 GPQA-Diamond 做 model routing 的 ground truth")
