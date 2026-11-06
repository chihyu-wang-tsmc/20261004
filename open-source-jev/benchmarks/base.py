"""一個有標準答案的 benchmark：怎麼讀題目、怎麼對答案。"""

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Benchmark:
    name: str
    """gpqa / aime / ai2arc / gsm8k：datasets/ 底下的資料夾名稱，也是 results/ 底下放作答的資料夾名稱。"""
    dataset: str
    """prompt_datasets.py 裡的資料集名稱（GPQA / AIME / AI2ARC / GSM8K），coding_report.py 的結果檔用這個名字。"""
    title: str
    load: Callable
    """datasets/<name>/ → 題目 list，每題 {"id", "group", "prompt", "answer"}（group 是報表分組用的）。"""
    grade: Callable
    """(model 的回答全文, 題目) → (抽出來的答案, 是否答對)。"""

    def questions(self, data_dir):
        return self.load(data_dir / self.name)


def last_boxed(text):
    """最後一個 \\boxed{...} 裡的內容（處理巢狀大括號）；沒有就回 None。"""
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


def letter_grader(pattern):
    """選擇題：取最後一個 'Answer: X'。"""
    regex = re.compile(pattern)

    def grade(text, q):
        matches = regex.findall(text or "")
        letter = matches[-1].upper() if matches else None
        return letter, letter == q["answer"]

    return grade
