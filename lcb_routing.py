"""用 LiveCodeBench 做 model routing 的 ground truth，並統計 Jev router（Winnow / jevk5）省下多少次大 model。

流程和 ground truth 的定義見 routing_bench.py。在 ~/jevk5 底下、jevk5 環境執行：

    python lcb_routing.py answer --model qwen3.8-27b       # deep1_router.py 的 fast
    python lcb_routing.py answer --model qwen3.8-flash     # deep1_router.py 的 powerful
    python lcb_routing.py route                            # Winnow 當 router（server 開在 8091）
    python lcb_routing.py route --router jevk5             # jevk5 當 router（jevk5-serve 開在 8090）
    python lcb_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash                  # 統計 Winnow
    python lcb_routing.py report --fast qwen3.8-27b --powerful qwen3.8-flash --router jevk5   # 統計 jevk5
    python evaluate_winnow.py --tier lcb                   # 用 JevBench 的指標評測 router
    python evaluate_jevk5.py --tier lcb                    # 同上，評測 jevk5
    python lcb_routing.py regrade --model qwen3.8-27b      # 評分方式改過後，不重問 model 直接重新評分

題目在 lcb/test6.jsonl：HuggingFace livecodebench/code_generation_lite 的 release_v6 新增題
（175 題，2025-01 ~ 2025-04 的 AtCoder / LeetCode）。每個子命令都可以加
--platform atcoder|leetcode、--difficulty easy|medium|hard 只跑其中一部分。

評分照 LiveCodeBench 官方的做法：抽出回答裡最後一個 ```python 區塊，對公開 + 私有測資逐筆執行，
全部通過才算答對（pass@1）；遇到第一筆失敗就停。
  * stdin 題（AtCoder）：逐行比對輸出，數字用 Decimal 比對（1.0 等於 1）
  * functional 題（LeetCode）：每筆測資的輸入是一行一個 JSON 參數，呼叫 Solution().<func_name>(*args)
  * 每筆測資 6 秒

model 寫的程式碼會在這台機器上執行，所以：
  * 用 `unshare -rn` 放進沒有網路的 namespace（不需要 root）
  * 不帶任何環境變數（API key 不會被讀到），在臨時資料夾裡執行，跑完就刪
  * 限制記憶體 4 GB、寫檔 64 MB
這不是完整的沙盒（還是讀得到你家目錄的檔案），只跑你信任的 model 的輸出。
"""

import base64
import io
import json
import os
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import zlib
from decimal import Decimal, InvalidOperation

from routing_bench import HERE, Bench, main

DATA_PATH = os.path.join(HERE, "lcb", "test6.jsonl")
TEST_TIMEOUT = 6  # 秒／每筆測資，跟官方相同

# LiveCodeBench 官方給 chat model 的提示（system prompt 併進同一則使用者訊息，
# 因為 router 看的是最後一則 HumanMessage）
SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests."
)
FORMAT_STARTER = (
    "### Format: You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters.\n```python\n{starter_code}\n```\n\n"
)
FORMAT_STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
    "output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
)

# 官方在執行前加在程式碼最前面的 import
IMPORT_HEADER = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\n"
    "import copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\n"
    "import operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)

# functional 題在子行程裡跑的程式：逐筆呼叫 Solution().<func_name>，結果寫到 result.json
# （不用 stdout，因為解答本身可能會 print）
FUNCTIONAL_RUNNER = r'''
import json, signal, sys
fn, per_test = sys.argv[1], float(sys.argv[2])
tests = json.load(open("tests.json"))
n = len(tests)

def report(ok, msg):
    json.dump({"ok": ok, "msg": msg}, open("result.json", "w"))
    sys.stdout.flush()
    import os; os._exit(0)

class TimeLimit(BaseException):
    pass

def on_alarm(*_):
    raise TimeLimit

signal.signal(signal.SIGALRM, on_alarm)
ns = {"__name__": "solution"}
try:
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
    cls = ns["Solution"]
except BaseException as e:
    report(False, f"compile error: {type(e).__name__}: {e}")
for i, t in enumerate(tests, 1):
    args = [json.loads(line) for line in t["input"].split("\n")]
    expected = json.loads(t["output"])
    signal.setitimer(signal.ITIMER_REAL, per_test)
    try:
        result = getattr(cls(), fn)(*args)
    except TimeLimit:
        report(False, f"test {i}/{n}: time limit")
    except BaseException as e:
        report(False, f"test {i}/{n}: runtime error: {type(e).__name__}: {str(e)[:200]}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if isinstance(result, tuple):
        result = list(result)
    if result != expected:
        report(False, f"test {i}/{n}: wrong answer")
report(True, f"passed {n}/{n}")
'''

SANDBOX = ["unshare", "-rn"] if shutil.which("unshare") else []
SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
CODE_BLOCK_RE = re.compile(r"```(?:python|py|Python3|python3)?[ \t]*\n(.*?)```", re.DOTALL)


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # 深遞迴的解法需要比較大的 stack（官方把 recursionlimit 設到 50000）
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = 1 << 30
    resource.setrlimit(resource.RLIMIT_STACK, (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))


def _run(args, cwd, timeout, stdin=None):
    return subprocess.run(
        SANDBOX + [sys.executable, "-I", *args], input=stdin, capture_output=True, text=True,
        timeout=timeout, cwd=cwd, env=SANDBOX_ENV, preexec_fn=_limits,
    )


class _StrOnlyUnpickler(pickle.Unpickler):
    """官方的私有測資是 pickle 過的字串；只允許字串，任何物件都擋掉，避免反序列化時執行程式碼。"""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def decode_tests(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        data = zlib.decompress(base64.b64decode(raw))
        return json.loads(_StrOnlyUnpickler(io.BytesIO(data)).load())


def load_questions(args):
    questions = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if args.platform and row["platform"] != args.platform:
                continue
            if args.difficulty and row["difficulty"] != args.difficulty:
                continue
            fmt = FORMAT_STARTER.format(starter_code=row["starter_code"]) if row["starter_code"] else FORMAT_STDIN
            prompt = (
                f"{SYSTEM}\n\n### Question:\n{row['question_content']}\n\n{fmt}"
                "### Answer: (use the provided format with backticks)\n\n"
            )
            questions.append(
                {
                    "id": f"{row['platform']}-{row['question_id']}",
                    "domain": f"{row['platform']} {row['difficulty']}",
                    "subdomain": row["contest_id"],
                    "prompt": prompt,
                    "correct": "pass all tests",
                    # 測資很大，評分時才解開
                    "_public": row["public_test_cases"],
                    "_private": row["private_test_cases"],
                    "_func_name": json.loads(row["metadata"] or "{}").get("func_name"),
                }
            )
    return questions


def extract_code(text):
    blocks = CODE_BLOCK_RE.findall(text or "")
    return blocks[-1] if blocks else None


def same_stdout(actual, expected):
    a_lines = [line.strip() for line in actual.strip().split("\n")]
    e_lines = [line.strip() for line in expected.strip().split("\n")]
    if a_lines == e_lines:
        return True
    if len(a_lines) != len(e_lines):
        return False
    for a, e in zip(a_lines, e_lines):
        if a == e:
            continue
        a_tok, e_tok = a.split(), e.split()
        if len(a_tok) != len(e_tok):
            return False
        try:
            if [Decimal(x) for x in a_tok] != [Decimal(x) for x in e_tok]:
                return False
        except InvalidOperation:
            return False
    return True


def grade_stdin(code, tests, workdir):
    with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as f:
        f.write(IMPORT_HEADER + code)
    n = len(tests)
    for i, t in enumerate(tests, 1):
        try:
            p = _run(["solution.py"], workdir, TEST_TIMEOUT, stdin=t["input"])
        except subprocess.TimeoutExpired:
            return f"test {i}/{n}: time limit", False
        if p.returncode != 0:
            err = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
            return f"test {i}/{n}: runtime error: {err}", False
        if not same_stdout(p.stdout, t["output"]):
            return f"test {i}/{n}: wrong answer", False
    return f"passed {n}/{n}", True


def grade_functional(code, tests, func_name, workdir):
    for name, content in (("solution.py", IMPORT_HEADER + code), ("runner.py", FUNCTIONAL_RUNNER)):
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
            f.write(content)
    with open(os.path.join(workdir, "tests.json"), "w", encoding="utf-8") as f:
        json.dump(tests, f)
    try:
        _run(["runner.py", func_name, str(TEST_TIMEOUT)], workdir, TEST_TIMEOUT * len(tests) + 30)
    except subprocess.TimeoutExpired:
        return "time limit (whole run)", False
    try:
        with open(os.path.join(workdir, "result.json"), encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "crashed (no result, e.g. out of memory or stack overflow)", False
    return result["msg"], result["ok"]


def grade(text, q):
    code = extract_code(text)
    if code is None:
        return "no code block", False
    tests = decode_tests(q["_public"]) + decode_tests(q["_private"])
    with tempfile.TemporaryDirectory(prefix="lcb_") as workdir:
        if q["_func_name"]:
            return grade_functional(code, tests, q["_func_name"], workdir)
        return grade_stdin(code, tests, workdir)


def add_arguments(parser):
    parser.add_argument("--platform", choices=["atcoder", "leetcode"], default=None, help="只跑某個平台")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None, help="只跑某個難度")


LCB = Bench(
    name="lcb",
    title="LiveCodeBench v6",
    load_questions=load_questions,
    grade=grade,
    source="LiveCodeBench code_generation_lite release_v6 increment (test6.jsonl)",
    add_arguments=add_arguments,
)

if __name__ == "__main__":
    main(LCB, "用 LiveCodeBench 做 model routing 的 ground truth")
