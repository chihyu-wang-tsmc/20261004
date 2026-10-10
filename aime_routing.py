"""用 AIME 做 model routing 的 ground truth，並統計 Jev router（Winnow / jevk5）省下多少次大 model。

流程和 ground truth 的定義見 routing_bench.py。在 ~/jevk5 底下、jevk5 環境執行：

    python aime_routing.py answer --model qwen3.8-27b       # deep1_router.py 的 fast
    python aime_routing.py answer --model qwen3.8-flash     # deep1_router.py 的 powerful
    python aime_routing.py route                            # Winnow 當 router（server 開在 8091）
    python aime_routing.py route --router jevk5             # jevk5 當 router（jevk5-serve 開在 8090）
    python aime_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash                  # 統計 Winnow
    python aime_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash --router jevk5   # 統計 jevk5
    python evaluate_winnow.py --tier aime                   # 用 JevBench 的指標評測 router
    python evaluate_jevk5.py --tier aime                    # 同上，評測 jevk5

每個子命令都可以加 --year 2025 / 2026 / all（預設 all，共 60 題）。
題目在 aime/aime_2025.jsonl、aime/aime_2026.jsonl（HuggingFace 的 MathArena/aime_2025、
MathArena/aime_2026，各 30 題：problem_idx 1–15 是 AIME I，16–30 是 AIME II）。
2025 的題目比較可能已經被新 model 訓練過，要看沒被污染的結果用 --year 2026。
"""

import json
import os
import re

from routing_bench import HERE, Bench, main

DATA_DIR = os.path.join(HERE, "aime")
"""題目檔所在的目錄。load_questions() 每次呼叫才組路徑，所以這個常數可以在 import 之後改掉：
deep14.py --cn 會把它指到 aime_cn/（translate_cn.py 產出的中文版），同時把 Bench.name 改成
aime_cn，讓 answers_dir 一起指到 aime_cn/answers/。"""

YEARS = ("2025", "2026")

# MathArena 的 AIME 提示
PROMPT = """Please reason step by step, and put your final answer within \\boxed{{}}. The answer is an integer between 0 and 999 inclusive.

{problem}"""

# 沒有 \boxed 時的備援：最後一個 "answer is 123" / "Answer: 123"
FALLBACK_RE = re.compile(r"(?i)answer\s*(?:is|:|：)\s*\**\s*\$?\s*(\d{1,3})\b")


def load_questions(args):
    years = YEARS if args.year == "all" else (args.year,)
    questions = []
    for year in years:
        with open(os.path.join(DATA_DIR, f"aime_{year}.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                idx = int(row["problem_idx"])
                questions.append(
                    {
                        "id": f"aime{year}-{idx:02d}",
                        "domain": f"AIME {year} {'I' if idx <= 15 else 'II'}",
                        "subdomain": row.get("problem_type") or "",
                        "prompt": PROMPT.format(problem=row["problem"].strip()),
                        "correct": int(row["answer"]),
                    }
                )
    return questions


def last_boxed(text):
    """回傳最後一個 \\boxed{...} 裡的內容（處理巢狀大括號）。"""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def to_int(s):
    # \text{70}、\mathbf{70} 之類只留裡面的內容；再去掉 LaTeX 空白、千分位逗號、$、度數符號等，
    # 只接受純整數（可以是 12.0 這種寫法）
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
        matches = FALLBACK_RE.findall(text)
        answer = int(matches[-1]) if matches else None
    return answer, answer == q["correct"]


def add_arguments(parser):
    parser.add_argument("--year", choices=[*YEARS, "all"], default="all", help="用哪一年的 AIME（預設 all）")


AIME = Bench(
    name="aime",
    title="AIME",
    load_questions=load_questions,
    grade=grade,
    source="AIME 2025/2026 via MathArena/aime_2025, MathArena/aime_2026",
    add_arguments=add_arguments,
)

if __name__ == "__main__":
    main(AIME, "用 AIME 做 model routing 的 ground truth")
