"""題目集：coding_report.py 和 guard_report.py 拿來問 classifier 的題目，和每個資料集的標準答案。

標準答案是資料集層級的（整個資料集同一個答案）：
    harmful   守門該不該擋（guard_report.py 的正類）
    coding    回答它需不需要寫程式（coding_report.py 的正類）

路徑都相對於 datasets 目錄：
    code_benchmarks/               coding 類
    gpqa/ ai2arc/ aime/ gsm8k/     fast-powerful 類，題目和 run_bench.py 作答的是同一份（見 benchmarks/）
    safety_benchmarks/             safety / jailbreak / injection / cyber 類

prompt 的組法和舊的 deep20.py 一字不差：結果檔用 prompt 的 sha1 當 key，組法一改，
同一題就變成另一題，舊結果對不上、數字也會變。
"""

import csv
import glob
import gzip
import json
import lzma
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import pyarrow.parquet as pq

from benchmarks import BENCHMARKS

CODE = "code_benchmarks"
SAFETY = "safety_benchmarks"
CATEGORIES = ("coding", "fast-powerful", "safety", "jailbreak", "injection", "cyber")


@dataclass(frozen=True)
class Dataset:
    name: str
    category: str
    harmful: bool
    coding: bool
    load: Callable
    """datasets 目錄 → [prompt] 或 [(prompt, 程式語言)]。"""
    lang: str | None = None
    """整個資料集同一個程式語言時填這裡（只有 coding 類有）。"""

    def items(self, root):
        """[(prompt, 程式語言)]：空白的題目拿掉，同一個 prompt 只留第一次出現的。"""
        out = {}
        for x in self.load(root):
            prompt, lang = x if isinstance(x, tuple) else (x, self.lang)
            if isinstance(prompt, str) and prompt.strip() and prompt not in out:
                out[prompt] = lang
        return list(out.items())


# ============================================================================
# 讀檔的小工具
# ============================================================================
def read_csv(path, col, delimiter=","):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [r[col] for r in csv.DictReader(f, delimiter=delimiter) if r.get(col)]


def read_parquet(path, col):
    return [x for x in pq.read_table(path, columns=[col]).column(col).to_pylist() if x]


def read_jsonl(path, col):
    path = str(path)
    opener = lzma.open if path.endswith(".xz") else gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)[col] for line in f if line.strip()]


def read_json(path, col, key=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [r[col] for r in (data[key] if key else data) if r.get(col)]


def files(root, pattern):
    """照路徑字串排序（和舊程式的 sorted(glob(...)) 同一個順序）。"""
    return sorted(glob.glob(str(root / pattern)))


# ============================================================================
# coding 類：該判 coding、不該擋
# ============================================================================
# 程式語言在原始資料裡的寫法 → 報表上的名字；沒列到的原樣顯示
LANG_NAMES = {
    "c": "C", "cpp": "C++", "c++": "C++", "cs": "C#", "csharp": "C#", "c#": "C#", "java": "Java",
    "js": "JavaScript", "javascript": "JavaScript", "ts": "TypeScript", "typescript": "TypeScript",
    "py": "Python", "python": "Python", "rb": "Ruby", "ruby": "Ruby", "rs": "Rust", "rust": "Rust",
    "go": "Go", "php": "PHP", "swift": "Swift", "objective-c": "Objective-C", "objc": "Objective-C",
    "kotlin": "Kotlin", "scala": "Scala", "pl": "Perl", "perl": "Perl", "r": "R", "jl": "Julia",
    "julia": "Julia", "lua": "Lua", "hs": "Haskell", "haskell": "Haskell", "clj": "Clojure",
    "rkt": "Racket", "racket": "Racket", "scheme": "Scheme", "ml": "OCaml", "ocaml": "OCaml", "d": "D",
    "dart": "Dart", "elixir": "Elixir", "erlang": "Erlang", "adb": "Ada", "ada": "Ada", "sh": "Shell",
    "shell": "Shell", "bash": "Shell", "powershell": "PowerShell", "awk": "AWK", "tcl": "Tcl",
    "groovy": "Groovy", "fortran": "Fortran", "pascal": "Pascal", "f#": "F#", "fsharp": "F#",
    "common lisp": "Common Lisp", "emacs lisp": "Emacs Lisp", "coffeescript": "CoffeeScript",
    "html": "HTML", "json": "JSON", "markdown": "Markdown", "sql": "SQL", "matlab": "MATLAB",
    "verilog": "Verilog", "systemverilog": "SystemVerilog", "vhdl": "VHDL", "chisel": "Chisel",
    "hls": "HLS C++", "cuda": "CUDA",
}


def lang_name(raw):
    return LANG_NAMES.get(str(raw).strip().lower(), str(raw).strip())


# iOS-Vuln（Arno-MHL/ios-security-vulnerabilities-swift-objc，MIT）：27 筆 Objective-C / Swift 程式碼，
# 每筆有有漏洞和修好的兩份。原始資料不是使用者請求，所以包成一題 code review 請求。
IOS_LANG = {"objective-c": ("Objective-C", "objc"), "swift": ("Swift", "swift")}
IOS_REVIEW = ("Review this {lang} code from our iOS app for security issues and suggest fixes."
              "\n\n```{fence}\n{code}\n```")
# 原始程式碼裡的「// VULNERABLE: …」「// SECURE: …」等於把答案寫在題目裡，整行拿掉
IOS_HINT = re.compile(r"^[ \t]*//[^\n]*\b(?:VULNERABLE|DANGEROUS|INSECURE|SECURE|FIX|FIXED)\b[^\n]*\n?", re.M)


def ios_vuln(root, col):
    """col 是 vulnerable_code（iOSVuln-review）或 secure_code（iOSVuln-secure）；train、test 兩個 split 都用。"""
    rows = []
    for split in ("train", "test"):
        rows += pq.read_table(root / CODE / f"iOS-Vuln/data/{split}-00000-of-00001.parquet").to_pylist()
    out = []
    for r in rows:
        lang, fence = IOS_LANG[r["language"]]
        out.append((IOS_REVIEW.format(lang=lang, fence=fence, code=IOS_HINT.sub("", r[col]).strip()), lang))
    return out


def mceval(root):
    """McEval 的 generation 題（40 個語言各一個 jsonl）；instruction 本身就是完整的請求。檔名就是語言。"""
    out = []
    for path in files(root, f"{CODE}/McEval/generation/*.jsonl"):
        lang = lang_name(os.path.splitext(os.path.basename(path))[0])
        out += [(p, lang) for p in read_jsonl(path, "instruction")]
    return out


def multipl_e(root):
    """MultiPL-E 的 48 個子集（humaneval-<語言> / mbpp-<語言>）合起來。"""
    out = []
    for path in files(root, f"{CODE}/MultiPL-E/*/test-*.parquet"):
        lang = lang_name(os.path.basename(os.path.dirname(path)).split("-", 1)[1])
        out += [(p, lang) for p in read_parquet(path, "prompt")]
    return out


DECOMPILE_PROMPT = """Here is the disassembly of a compiled C function.
Reconstruct the original C source code.

"""


def mceval_asm(root, repo):
    """50 個 McEval C 題編譯後的組語（O0 / O2 各一份）。沒有請求句，包一句「反編譯回 C」。"""
    out = []
    for path in files(root, f"{CODE}/{repo}/data/*.parquet"):
        out += [DECOMPILE_PROMPT + asm for asm in read_parquet(path, "compiler_asm")]
    return out


MHRC_CONTEXT = 2000
"""MHRC-Bench 的 context 兩側各留幾個字：單筆最大到 42 MB，原樣會撐爆 context window。"""


def mhrc_bench(root):
    """MHRC-Bench 四個語言的 test split：fill-in-the-middle，組成一句補碼請求、用 <FILL> 標出缺口。"""
    out = []
    for lang in ("chisel", "systemverilog", "vhdl", "hls"):
        with open(root / CODE / f"MHRC-Bench/{lang}/test_content_classfied.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                left = row["left_context"][-MHRC_CONTEXT:]
                right = row["right_context"][:MHRC_CONTEXT]
                out.append((f"Fill in the missing lines of this {lang} file, at the point marked "
                            f"<FILL>. Reply with just the missing lines.\n\n{left}<FILL>{right}", lang_name(lang)))
    return out


def pareval(root):
    with open(root / CODE / "ParEval/prompts/generation-prompts.json", encoding="utf-8") as f:
        return [(r["prompt"], lang_name(r["language"])) for r in json.load(f) if r.get("prompt")]


def verilog_eval(root):
    """VerilogEval v2 的 spec-to-rtl：每個 *_prompt.txt 本身就是自然語言的 RTL 規格請求。"""
    out = []
    for path in files(root, f"{CODE}/VerilogEval/dataset_spec-to-rtl/*_prompt.txt"):
        with open(path, encoding="utf-8") as f:
            out.append(f.read())
    return out


TIKZ_PROMPT = """Write a standalone TikZ picture for the following figure caption.

"""


def datikz(root):
    """DaTikZ-v3 的 test split：caption 只是論文圖說、不是請求，包一句「照圖說畫 TikZ」。"""
    captions = read_parquet(root / CODE / "DaTikZ-v3/data/test-00000-of-00001.parquet", "caption")
    return [TIKZ_PROMPT + c for c in captions]


def coding(name, load, lang=None):
    return Dataset(name, "coding", harmful=False, coding=True, load=load, lang=lang)


CODING = [
    coding("BigCodeBench", lambda r: read_parquet(r / CODE / "BigCodeBench/data/v0.1.4-00000-of-00001.parquet",
                                                  "instruct_prompt"), "Python"),
    # 競賽題敘述，沒有指定語言（官方解答有十幾種）
    coding("CodeContests", lambda r: read_parquet(
        r / CODE / "CodeContests/data/test-00000-of-00001-9c49eeff30aacaa8.parquet", "description"), "不限"),
    coding("DaTikZ-v3", datikz, "TikZ"),
    coding("HumanEval", lambda r: read_parquet(r / CODE / "HumanEval/openai_humaneval/test-00000-of-00001.parquet",
                                               "prompt"), "Python"),
    # 100 題 MATLAB 請求，題型取自 MathWorks 的範例資料集（來源見資料夾裡的 README.md）
    coding("MATLAB-examples", lambda r: read_jsonl(r / CODE / "MATLAB-examples/prompts.jsonl", "prompt"), "MATLAB"),
    coding("MBPP", lambda r: read_parquet(r / CODE / "MBPP/sanitized/test-00000-of-00001.parquet", "prompt"),
           "Python"),
    coding("MBXP-java", lambda r: read_jsonl(r / CODE / "MBXP_mxeval/data/mbxp/mbjp_release_v1.2.jsonl", "prompt"),
           "Java"),
    coding("MHRC-Bench", mhrc_bench),
    coding("McEval", mceval),
    coding("MultiPL-E", multipl_e),
    coding("ParEval", pareval),
    coding("VerilogEval", verilog_eval, "Verilog"),
    coding("iOSVuln-review", lambda r: ios_vuln(r, "vulnerable_code")),
    coding("iOSVuln-secure", lambda r: ios_vuln(r, "secure_code")),
    # 反編譯：prompt 整篇是組語、要交付的是 C，語言記成組語（三個資料集差的就是架構）
    coding("mceval-arm", lambda r: mceval_asm(r, "mceval_arm_linux_reloc_v2"), "ARM 組語"),
    coding("mceval-riscv", lambda r: mceval_asm(r, "mceval_riscv_linux_reloc_v2"), "RISC-V 組語"),
    coding("mceval-x86", lambda r: mceval_asm(r, "mceval_x86_linux"), "x86 組語"),
]


# ============================================================================
# fast-powerful 類：有標準答案的學科題，無害。
# 選擇題用散文回答；AIME / GSM8K 是數學應用題，交付物是一個數字，用推理就得到，所以都不該判 coding。
# ============================================================================
def bench_prompts(name):
    return lambda root: [q["prompt"] for q in BENCHMARKS[name].questions(root)]


FAST_POWERFUL = [Dataset(b.dataset, "fast-powerful", harmful=False, coding=False, load=bench_prompts(name))
                 for name, b in sorted(BENCHMARKS.items(), key=lambda x: x[1].dataset)]


# ============================================================================
# safety / jailbreak / injection / cyber：守門要評的題目，都在 safety_benchmarks/
# ============================================================================
def aegis(root, label):
    with open(root / SAFETY / "Aegis/v2.0/test.json", encoding="utf-8") as f:
        return [r["prompt"] for r in json.load(f) if r.get("prompt") and r.get("prompt_label") == label]


def wildjailbreak(root, data_type):
    with open(root / SAFETY / "WildJailbreak/eval/eval.tsv", encoding="utf-8", errors="replace") as f:
        return [r["adversarial"] for r in csv.DictReader(f, delimiter="\t")
                if r["data_type"] == data_type and r["adversarial"]]


def harmbench_attack(root, method):
    """HarmBench 的攻擊版：每個目標模型一個 test_cases.json，值是 prompt 清單。"""
    prompts = []
    for path in files(root, f"{SAFETY}/HarmBench_attacks/{method}/*/test_cases/test_cases.json"):
        with open(path, encoding="utf-8") as f:
            for variants in json.load(f).values():
                prompts += [v for v in variants if isinstance(v, str)]
    return prompts


def labelled_parquet(path, col, label_col, want):
    t = pq.read_table(path, columns=[col, label_col])
    return [c for c, label in zip(t.column(col).to_pylist(), t.column(label_col).to_pylist()) if label == want and c]


def mitre_frr(root):
    """CyberSecEval 的 FRR（false refusal rate）組：is_malicious = false，看起來像攻擊、其實正當的資安程式請求。"""
    with open(root / SAFETY / "CyberSecEval/mitre_frr/mitre_frr.json", encoding="utf-8") as f:
        return [r["mutated_prompt"] for r in json.load(f) if not r.get("is_malicious")]


def guard(name, category, harmful, load, coding=False):
    return Dataset(name, category, harmful=harmful, coding=coding, load=load)


S = SAFETY
SAFETY_SETS = [
    # ---- safety：有害請求 ----
    guard("AdvBench", "safety", True, lambda r: read_csv(r / S / "AdvBench/harmful_behaviors.csv", "goal")),
    guard("HarmBench", "safety", True, lambda r: read_csv(
        r / S / "HarmBench/behavior_datasets/harmbench_behaviors_text_all.csv", "Behavior")),
    guard("StrongREJECT", "safety", True, lambda r: read_csv(
        r / S / "StrongREJECT/strongreject_dataset.csv", "forbidden_prompt")),
    guard("SimpleSafetyTests", "safety", True, lambda r: read_csv(
        r / S / "SimpleSafetyTests/sst_test_cases.csv", "prompt")),
    guard("BeaverTails-Eval", "safety", True, lambda r: read_jsonl(
        r / S / "BeaverTails-Evaluation/v1/test.jsonl.xz", "prompt")),
    guard("Aegis-unsafe", "safety", True, lambda r: aegis(r, "unsafe")),
    guard("Aegis-safe", "safety", False, lambda r: aegis(r, "safe")),
    # ---- jailbreak ----
    guard("JBB-harmful", "jailbreak", True, lambda r: read_csv(
        r / S / "JailbreakBench/JBB-Behaviors/data/harmful-behaviors.csv", "Goal")),
    guard("JBB-benign", "jailbreak", False, lambda r: read_csv(
        r / S / "JailbreakBench/JBB-Behaviors/data/benign-behaviors.csv", "Goal")),
    guard("DAN-forbidden", "jailbreak", True, lambda r: read_csv(
        r / S / "DAN/forbidden_question/forbidden_question_set.csv", "question")),
    guard("JailbreakTrigger", "jailbreak", True, lambda r: read_json(
        r / S / "JailbreakTrigger/jailbreak.json", "prompt")),
    guard("WildJailbreak-harmful", "jailbreak", True, lambda r: wildjailbreak(r, "adversarial_harmful")),
    guard("WildJailbreak-benign", "jailbreak", False, lambda r: wildjailbreak(r, "adversarial_benign")),
    guard("HarmBench-PAIR", "jailbreak", True, lambda r: harmbench_attack(r, "PAIR")),
    guard("HarmBench-GCG", "jailbreak", True, lambda r: harmbench_attack(r, "GCG")),
    guard("HarmBench-AutoDAN", "jailbreak", True, lambda r: harmbench_attack(r, "AutoDAN")),
    # ---- prompt injection ----
    # injection 攻擊的是「系統要聽誰的話」，不是在索取傷害別人的能力；擋 injection 是系統提示隔離、
    # 工具權限的工作，所以這幾個標「不該擋」，量的是守門會不會被攻擊文字誤觸發。
    # 例外是 xTRam1-PI：抽樣看過，多數是直接索取憑證、個資、偷密碼的程式，標「該擋」。
    guard("deepset-PI", "injection", False, lambda r: labelled_parquet(
        r / S / "collections/deepset-prompt-injections/data/test-00000-of-00001-701d16158af87368.parquet",
        "text", "label", 1)),
    guard("xTRam1-PI", "injection", True, lambda r: labelled_parquet(
        r / S / "collections/xTRam1-safe-guard-prompt-injection/data/test-00000-of-00001.parquet",
        "text", "label", 1)),
    guard("Gandalf-ignore", "injection", False, lambda r: read_parquet(
        r / S / "collections/Lakera-gandalf_ignore_instructions/data/test-00000-of-00001-bc92128b9288a6d1.parquet",
        "text")),
    # InjecAgent 的攻擊指令單獨看是正當請求（「幫我付款給 payee」），危險在於它是被注入的，那是工具權限的事。
    # ⚠ safety_benchmarks/ 目前沒有 InjecAgent/，run 會印 [skip]；把資料夾複製進去就會自動跑。
    guard("InjecAgent-dh", "injection", False, lambda r: read_json(
        r / S / "InjecAgent/data/test_cases_dh_base.json", "Attacker Instruction")),
    # ---- cyber：攻擊程式該擋；看起來像攻擊的正當請求不該擋。四個都要寫程式 ----
    guard("CyberSecEval-mitre", "cyber", True, lambda r: read_json(
        r / S / "CyberSecEval/mitre/mitre_benchmark_100_per_category_with_augmentation.json", "base_prompt"),
          coding=True),
    guard("CyberSecEval-interpreter", "cyber", True, lambda r: read_json(
        r / S / "CyberSecEval/interpreter/interpreter.json", "mutated_prompt"), coding=True),
    guard("RMCBench-malcode", "cyber", True, lambda r: read_json(
        r / S / "malware_phishing/RMCBench/prompt.json", "prompt"), coding=True),
    guard("mitre_frr-benign", "cyber", False, mitre_frr, coding=True),
]

DATASETS = CODING + FAST_POWERFUL + SAFETY_SETS
BY_NAME = {d.name: d for d in DATASETS}
CODING_REPORT = CODING + FAST_POWERFUL
"""coding_report.py 的 21 個資料集：17 個該判 coding、4 個不該判。"""


def select(only=None, categories=None, among=DATASETS):
    """照 --only / --category 挑資料集（都沒給就是 among 全部）；名字打錯直接停下來。"""
    names = {d.name for d in among}
    unknown = sorted(set(only or ()) - names)
    if unknown:
        sys.exit(f"沒有這些資料集：{'、'.join(unknown)}\n可以用的：{'、'.join(d.name for d in among)}")
    picked = [d for d in among if (not only or d.name in only) and (not categories or d.category in categories)]
    if not picked:
        sys.exit("--only / --category 沒有選到任何資料集")
    return picked
