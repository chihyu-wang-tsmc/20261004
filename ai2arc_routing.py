"""用 AI2 ARC 做 model routing 的 ground truth，並統計 Jev router（Winnow / jevk5）省下多少次大 model。

流程和 ground truth 的定義見 routing_bench.py。在 ~/jevk5 底下、jevk5 環境執行：

    python ai2arc_routing.py answer --model qwen3.8-27b       # deep1_router.py 的 fast
    python ai2arc_routing.py answer --model qwen3.8-flash     # deep1_router.py 的 powerful
    python ai2arc_routing.py route                            # Winnow 當 router（server 開在 8091）
    python ai2arc_routing.py route --router jevk5              # jevk5 當 router（jevk5-serve 開在 8090）
    python ai2arc_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash                  # 統計 Winnow
    python ai2arc_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash --router jevk5   # 統計 jevk5
    python evaluate_winnow.py --tier ai2arc                   # 用 JevBench 的指標評測 router
    python evaluate_jevk5.py --tier ai2arc                    # 同上，評測 jevk5
    python ai2arc_routing.py regrade --model qwen3.8-27b      # 評分方式改過後，不重問 model 直接重新評分

題目在 ai2arc/arc_challenge_test.jsonl、ai2arc/arc_easy_test.jsonl（HuggingFace 的 allenai/ai2_arc，
test split：ARC-Challenge 1172 題、ARC-Easy 2376 題），先跑 python download_benchmarks.py --only ai2arc 下載。

每個子命令都可以加：
    --subset challenge / easy / all   用哪個 subset（預設 all，兩個都用）
    --n N                            每個 subset 固定 seed 隨機抽 N 題（預設 150；0 = 全部）
固定 seed 的意思是每個子命令、每次執行抽到的都是同一批題目。

Easy 和 Challenge 一起跑的用意是：同一個領域（小學到高中的科學題）裡有難易兩種，
正好可以看 router 分不分得出來——理想的 router 在 Easy 上幾乎都選 fast，Challenge 才會多選 powerful。
選項數 3–5 題都有（原始資料的 label 有 ABCD 也有 1234），download_benchmarks.py 已經一律換成 A/B/C/D/E。
"""

import json
import os
import random
import re

from routing_bench import HERE, Bench, main

DATA_DIR = os.path.join(HERE, "ai2arc")
"""題目檔所在的目錄。load_questions() 每次呼叫才組路徑，所以這個常數可以在 import 之後改掉：
deep14.py --cn 會把它指到 ai2arc_cn/（translate_cn.py 產出的中文版），同時把 Bench.name 改成
ai2arc_cn，讓 answers_dir 一起指到 ai2arc_cn/answers/。"""

SUBSETS = ("challenge", "easy")
TITLES = {"challenge": "ARC-Challenge", "easy": "ARC-Easy"}

# 抽題用的 seed，換掉就會抽到不同的題目（已經跑過的答案就對不上了，不要隨便改）
SAMPLE_SEED = 20251004

# simple-evals 的 multiple choice 提示，選項數不固定所以字母集合是算出來的
PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{options}"""

ANSWER_RE = re.compile(r"(?i)answer\s*[:：]\s*\**\s*\(?\$?([A-E])\b")


def load_questions(args):
    """讀 ARC test split；--n 用固定 seed 對每個 subset 各抽 N 題，抽完照原本的順序排。"""
    subsets = SUBSETS if args.subset == "all" else (args.subset,)
    questions = []
    for subset in subsets:
        path = os.path.join(DATA_DIR, f"arc_{subset}_test.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if args.n:
            # seed 加上 subset 名稱，兩個 subset 才不會抽到同樣的位置
            rng = random.Random(f"{SAMPLE_SEED}-{subset}")
            picked = set(rng.sample(range(len(rows)), min(args.n, len(rows))))
            rows = [row for i, row in enumerate(rows) if i in picked]
        for row in rows:
            letters = "ABCDE"[: len(row["options"])]
            questions.append(
                {
                    "id": f"{subset}-{row['id']}",
                    "domain": TITLES[subset],
                    # id 的前綴是題目出自哪個考試，例如 Mercury、MCAS、AKDE&ED
                    "subdomain": row["id"].split("_")[0],
                    "prompt": PROMPT.format(
                        letters=letters,
                        question=row["question"],
                        options="\n".join(f"{l}) {t}" for l, t in zip(letters, row["options"])),
                    ),
                    "correct": row["answer"],
                }
            )
    return questions


def grade(text, q):
    matches = ANSWER_RE.findall(text or "")
    letter = matches[-1].upper() if matches else None
    return letter, letter == q["correct"]


def add_arguments(parser):
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all", help="用哪個 ARC subset（預設 all）")
    parser.add_argument("--n", type=int, default=150, help="每個 subset 隨機抽幾題（固定 seed；0 = 全部）")


AI2ARC = Bench(
    name="ai2arc",
    title="AI2 ARC",
    load_questions=load_questions,
    grade=grade,
    source="AI2 ARC test split (allenai/ai2_arc, ARC-Challenge + ARC-Easy)",
    add_arguments=add_arguments,
)

if __name__ == "__main__":
    main(AI2ARC, "用 AI2 ARC 做 model routing 的 ground truth")
