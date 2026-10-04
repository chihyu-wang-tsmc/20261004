"""用 GSM8K 做 model routing 的 ground truth，並統計 Jev router（Winnow / jevk5）省下多少次大 model。

流程和 ground truth 的定義見 routing_bench.py。在 ~/jevk5 底下、jevk5 環境執行：

    python gsm8k_routing.py answer --model qwen3.8-27b       # deep1_router.py 的 fast
    python gsm8k_routing.py answer --model qwen3.8-flash     # deep1_router.py 的 powerful
    python gsm8k_routing.py route                            # Winnow 當 router（server 開在 8091）
    python gsm8k_routing.py route --router jevk5              # jevk5 當 router（jevk5-serve 開在 8090）
    python gsm8k_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash                  # 統計 Winnow
    python gsm8k_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash --router jevk5   # 統計 jevk5
    python evaluate_winnow.py --tier gsm8k                   # 用 JevBench 的指標評測 router
    python evaluate_jevk5.py --tier gsm8k                    # 同上，評測 jevk5
    python gsm8k_routing.py regrade --model qwen3.8-27b      # 評分方式改過後，不重問 model 直接重新評分

題目在 gsm8k/gsm8k_test.jsonl（HuggingFace 的 openai/gsm8k，main config 的 test split，1319 題），
先跑 python download_benchmarks.py --only gsm8k 下載。

每個子命令都可以加 --n（預設 200）：1319 題跑兩個 model 太貴，所以固定 seed 隨機抽 N 題，
每個子命令、每次執行抽到的都是同一批。--n 0 是全部 1319 題。
GSM8K 是小學程度的應用題，對現在的 model 幾乎都是 fast 就夠，所以這個 benchmark 主要在看
router 會不會把簡單題誤判成 powerful（白用大 model）。報表的分組是解題步數（gold solution 的行數），
步數越多的題目才比較有機會需要 powerful。
"""

import json
import os
import random
import re

from aime_routing import last_boxed  # 同樣的 \boxed{...} 解析（處理巢狀大括號）
from routing_bench import HERE, Bench, main

JSONL_PATH = os.path.join(HERE, "gsm8k", "gsm8k_test.jsonl")

# 抽題用的 seed，換掉就會抽到不同的題目（已經跑過的答案就對不上了，不要隨便改）
SAMPLE_SEED = 20251004

PROMPT = """Please reason step by step, and put your final answer within \\boxed{{}}. The answer is an integer.

{question}"""

# 沒有 \boxed 時的備援：最後一個 "answer is 18" / "Answer: $1,200"
FALLBACK_RE = re.compile(r"(?i)answer\s*(?:is|:|：)\s*\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\b")


def steps_group(steps):
    return "steps 2-3" if steps <= 3 else "steps 4-5" if steps <= 5 else "steps 6+"


def load_questions(args):
    """讀 GSM8K test split；--n 用固定 seed 抽樣，抽完照原本的順序排，所以每次執行都一樣。"""
    with open(JSONL_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n:
        picked = set(random.Random(SAMPLE_SEED).sample(range(len(rows)), min(args.n, len(rows))))
        rows = [row for i, row in enumerate(rows) if i in picked]
    return [
        {
            "id": row["id"],
            "domain": steps_group(row["steps"]),
            "subdomain": f"{row['steps']} steps",
            "prompt": PROMPT.format(question=row["question"]),
            "correct": row["answer"],
        }
        for row in rows
    ]


def to_int(s):
    # \text{18}、\mathbf{18} 之類只留裡面的內容；再去掉 LaTeX 空白、千分位逗號、$ 等，
    # 只接受整數（可以是 18.0 這種寫法，也可以是負數）
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
        matches = FALLBACK_RE.findall(text)
        answer = to_int(matches[-1]) if matches else None
    return answer, answer == q["correct"]


def add_arguments(parser):
    parser.add_argument("--n", type=int, default=200, help="隨機抽幾題（固定 seed；0 = 全部 1319 題）")


GSM8K = Bench(
    name="gsm8k",
    title="GSM8K",
    load_questions=load_questions,
    grade=grade,
    source="GSM8K test split (openai/gsm8k, main config)",
    add_arguments=add_arguments,
)

if __name__ == "__main__":
    main(GSM8K, "用 GSM8K 做 model routing 的 ground truth")
