"""每一題的程式語言：deep20_coding_report.py 的 [C6] 用的對照表。

語言是從**原始資料集**取的，不是從 prompt 文字猜的。兩種來源：

    FIXED      整個資料集只有一個語言（HumanEval 全是 Python、VerilogEval 全是 Verilog…），
               直接查表就好，一個原始檔都不用讀。
    PER_ROW    一個資料集混多個語言（McEval 40 種、MultiPL-E 24 種、CyberSecEval 8 種…），
               要回頭讀原始檔，把 prompt 對回它那一列的語言。

「語言」指**這一題實際在處理的程式語言**，也就是 prompt 裡那段程式碼（或請求指名）的語言。
23 個 coding 資料集裡，只有 mceval-* 那三個反編譯資料集的「讀進來的語言」和「交付的語言」
不一樣：prompt 整篇是 ARM／RISC-V／x86 組語，要交付的是 C。它們記成組語，不是 C——
三個資料集差的就是架構，記成 C 的話三份會長得一模一樣，而且在語言彙總那張表裡會和
CyberSecEval 真正寫 C 的題目併成同一列，那個數字沒有意義。

APPS / CodeContests 的題目是競賽題敘述，整篇沒有指定語言（官方解答同時有十幾種），
所以記成「不限」，不是某個語言。

---- 為什麼 import deep20 ----

其他三份報表都標榜「只讀 result 檔、不 import deep20」。這支破例，而且是必須的：

對照表的 key 是 prompt 字串本身（result 的 row 裡只有 prompt），要對得起來，這裡組出來的
prompt 就得和 deep20.py 當初存進 result 的那個字元級相同。mceval_asm / mhrc_bench /
ios_vuln / datikz 的 prompt 都是 loader 用模板包出來的（DECOMPILE_PROMPT、IOS_REVIEW、
MHRC_CONTEXT 的截斷長度…），在這裡抄一份模板，哪天 deep20.py 改一個字就會整個資料集對不上，
而且是靜默地全部變「未知」。所以直接 import deep20、重用它的常數和 loader 輔助函式。

deep20_coding_report.py 本身仍然不 import deep20：它是在 [C6] 那一節才 import 這支，
原始資料集不在、或 import 失敗，就只跳過 [C6]，前面五節照印。

---- 用法 ----

    python deep20_coding_langs.py check          # 自我檢查：對照表的 prompt 和 loader 是否一題不差
    python deep20_coding_langs.py list           # 印每個資料集的語言分布（整份資料集，不是抽樣）

    from deep20_coding_langs import lang_of
    lang_of("McEval", prompt)                    # → "Python"／"未知"
"""

import glob
import json
import os
import sys
from collections import Counter

import deep20 as d20

UNKNOWN = "未知"
ANY_LANG = "不限"
MIXED = "多語言"

# 原始資料集的寫法 → 報表顯示的名字。沒列到的就原樣輸出（只做首字母大寫以外的處理一律不做）。
NORM = {
    "c": "C", "cpp": "C++", "c++": "C++", "cs": "C#", "csharp": "C#", "c#": "C#",
    "java": "Java", "js": "JavaScript", "javascript": "JavaScript", "ts": "TypeScript",
    "typescript": "TypeScript", "py": "Python", "python": "Python", "rb": "Ruby",
    "ruby": "Ruby", "rs": "Rust", "rust": "Rust", "go": "Go", "php": "PHP",
    "swift": "Swift", "objective-c": "Objective-C", "objc": "Objective-C",
    "kotlin": "Kotlin", "scala": "Scala", "pl": "Perl", "perl": "Perl",
    "r": "R", "jl": "Julia", "julia": "Julia", "lua": "Lua", "hs": "Haskell",
    "haskell": "Haskell", "clj": "Clojure", "rkt": "Racket", "racket": "Racket",
    "scheme": "Scheme", "ml": "OCaml", "ocaml": "OCaml", "d": "D", "dart": "Dart",
    "elixir": "Elixir", "erlang": "Erlang", "adb": "Ada", "ada": "Ada",
    "sh": "Shell", "shell": "Shell", "bash": "Shell", "powershell": "PowerShell",
    "awk": "AWK", "tcl": "Tcl", "groovy": "Groovy", "fortran": "Fortran",
    "pascal": "Pascal", "f#": "F#", "fsharp": "F#", "common lisp": "Common Lisp",
    "emacs lisp": "Emacs Lisp", "coffeescript": "CoffeeScript", "html": "HTML",
    "json": "JSON", "markdown": "Markdown", "sql": "SQL", "matlab": "MATLAB",
    "verilog": "Verilog", "systemverilog": "SystemVerilog", "vhdl": "VHDL",
    "chisel": "Chisel", "hls": "HLS C++", "tikz": "TikZ", "latex": "LaTeX",
    "arm": "ARM 組語", "riscv": "RISC-V 組語", "x86": "x86 組語",
}

# ---- 整個資料集一個語言 ----
# 理由寫在旁邊的，是那種「prompt 沒寫語言、但 benchmark 的交付物就是這個語言」的。
FIXED = {
    "HumanEval": "Python",
    "MBPP": "Python",
    "MBXP-java": "Java",
    "BigCodeBench": "Python",
    "LiveCodeBench": "Python",          # 競賽題，但 LCB 的 starter_code 和評測都是 Python
    "SecurityEval": "Python",
    "MATLAB-examples": "MATLAB",
    "VerilogEval": "Verilog",
    "Spider2-Lite": "SQL",
    "DaTikZ-v3": "TikZ",
    # 反編譯：prompt 整篇是組語，要交付的是 C。記的是組語（為什麼見檔頭）
    "mceval-arm": "ARM 組語",
    "mceval-riscv": "RISC-V 組語",
    "mceval-x86": "x86 組語",
    "APPS": ANY_LANG,                   # 競賽題敘述，沒指定語言
    "CodeContests": ANY_LANG,
}


def norm(x):
    return NORM.get(str(x).strip().lower(), str(x).strip())


# ============================================================================
# 混多語言的資料集：prompt → 語言
#
# 每一個都和 deep20.py 同名 loader 走同一份檔案、同一個欄位、同一個順序，差別只在
# 一併把語言帶出來。改 deep20.py 的 loader 時這裡要一起改，check 子命令會抓到對不上。
# ============================================================================
def _cyberseceval(sub):
    """CyberSecEval instruct / autocomplete：每一列自己帶 language 欄。"""
    rows = json.load(open(f"{d20.CYBER}/CyberSecEval/{sub}/{sub}.json", encoding="utf-8"))
    return [(r["test_case_prompt"], norm(r["language"]))
            for r in rows if r.get("test_case_prompt")]


def _ios_vuln(col):
    """iOSVuln-review / -secure：prompt 要用 deep20 的 IOS_REVIEW 模板重組一次。"""
    rows = []
    for split in ("train", "test"):
        rows += d20.pq.read_table(f"{d20.CODE}/iOS-Vuln/data/{split}-00000-of-00001.parquet").to_pylist()
    out = []
    for r in rows:
        lang, fence = d20.IOS_LANG[r["language"]]
        out.append((d20.IOS_REVIEW.format(lang=lang, fence=fence,
                                          code=d20.IOS_HINT.sub("", r[col]).strip()), norm(lang)))
    return out


def _mceval():
    """McEval：generation/ 底下一個語言一個 jsonl，檔名去掉副檔名就是語言。"""
    out = []
    for path in sorted(glob.glob(f"{d20.CODE}/McEval/generation/*.jsonl")):
        lang = norm(os.path.splitext(os.path.basename(path))[0])
        out += [(p, lang) for p in d20.jsonl_col(path, "instruction")]
    return out


def _multipl_e():
    """MultiPL-E：目錄名是 humaneval-<ext> / mbpp-<ext>，ext 就是語言。"""
    out = []
    for path in sorted(glob.glob(f"{d20.CODE}/MultiPL-E/*/test-*.parquet")):
        lang = norm(os.path.basename(os.path.dirname(path)).split("-", 1)[1])
        out += [(p, lang) for p in d20.parquet_col(path, "prompt")]
    return out


def _mhrc():
    """MHRC-Bench：四個語言各一個目錄，prompt 要照 deep20 的截斷長度重組。"""
    out = []
    for lang in ("chisel", "systemverilog", "vhdl", "hls"):
        path = f"{d20.CODE}/MHRC-Bench/{lang}/test_content_classfied.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                left = row["left_context"][-d20.MHRC_CONTEXT:]
                right = row["right_context"][:d20.MHRC_CONTEXT]
                out.append((
                    f"Fill in the missing lines of this {lang} file, at the point marked "
                    f"<FILL>. Reply with just the missing lines.\n\n{left}<FILL>{right}",
                    norm(lang)))
    return out


def _pareval():
    rows = json.load(open(f"{d20.CODE}/ParEval/prompts/generation-prompts.json", encoding="utf-8"))
    return [(r["prompt"], norm(r["language"])) for r in rows if r.get("prompt")]


PER_ROW = {
    "CyberSecEval-instruct": lambda: _cyberseceval("instruct"),
    "CyberSecEval-autocomplete": lambda: _cyberseceval("autocomplete"),
    "iOSVuln-review": lambda: _ios_vuln("vulnerable_code"),
    "iOSVuln-secure": lambda: _ios_vuln("secure_code"),
    "McEval": _mceval,
    "MultiPL-E": _multipl_e,
    "MHRC-Bench": _mhrc,
    "ParEval": _pareval,
}

_cache = {}   # dataset → {prompt: 語言}
_failed = {}  # dataset → 讀不到原始檔的原因


def table(dataset):
    """某個混多語言資料集的 {prompt: 語言}。讀不到原始檔就回空 dict（原因記在 _failed）。"""
    if dataset in _cache:
        return _cache[dataset]
    try:
        pairs = PER_ROW[dataset]()
    except (OSError, KeyError, ValueError) as e:
        _failed[dataset] = f"{type(e).__name__}: {e}"
        _cache[dataset] = {}
        return _cache[dataset]
    if not pairs:
        # McEval / MultiPL-E 是用 glob 找檔案的，根目錄不在時 glob 回空 list、不丟例外，
        # 不特別記下來的話 [C6] 只會出現一排「未知」、沒人知道是原始檔沒找到
        _failed[dataset] = "原始檔一筆都沒讀到（根目錄不在，或檔案佈局變了）"
    # 同一句 prompt 在兩個語言下都出現過（極少，通常是極短的題目）就記成「多語言」，
    # 不讓先讀到的那個語言說了算
    t = {}
    for p, lang in pairs:
        t[p] = lang if t.get(p, lang) == lang else MIXED
    _cache[dataset] = t
    return t


def lang_of(dataset, prompt):
    """一題的語言。非 coding 資料集、或原始檔讀不到，回 UNKNOWN。"""
    if dataset in FIXED:
        return FIXED[dataset]
    if dataset in PER_ROW:
        return table(dataset).get(prompt, UNKNOWN)
    return UNKNOWN


def failures():
    """哪些資料集的原始檔讀不到（要在 lang_of 叫過之後才有內容）。"""
    return dict(_failed)


# ============================================================================
# 子命令
# ============================================================================
def cmd_check():
    """對照表的 prompt 和 deep20 loader 產出的 prompt 是不是一題不差。

    對不上就表示 deep20.py 的 loader 改過、這裡沒跟著改，[C6] 會靜默地變成一堆「未知」。
    """
    by_name = {d.name: d for d in d20.DATASETS}
    bad = 0
    for name in list(FIXED) + list(PER_ROW):
        d = by_name.get(name)
        if d is None:
            print(f"  [!!] {name}：deep20.DATASETS 裡沒有這個資料集")
            bad += 1
            continue
        try:
            prompts = [p for p in d.load() if isinstance(p, str) and p.strip()]
        except Exception as e:  # noqa: BLE001 —— 原始檔不在就只是沒得比，不是錯
            print(f"  [skip] {name}：讀不到原始檔（{type(e).__name__}）")
            continue
        if name in FIXED:
            print(f"  [ok] {name}：{len(prompts)} 題，整份都是 {FIXED[name]}")
            continue
        t = table(name)
        if name in _failed:
            print(f"  [!!] {name}：對照表建不起來（{_failed[name]}）")
            bad += 1
            continue
        miss = [p for p in prompts if p not in t]
        extra = len(t) - (len(prompts) - len(miss))
        c = Counter(t[p] for p in prompts if p in t)
        flag = "ok" if not miss else "!!"
        bad += bool(miss)
        print(f"  [{flag}] {name}：loader {len(prompts)} 題，對到 {len(prompts) - len(miss)} 題"
              f"{f'，對不到 {len(miss)} 題' if miss else ''}"
              f"{f'，對照表多出 {extra} 筆' if extra > 0 else ''}"
              f"　{len(c)} 種語言")
        if miss:
            print(f"        對不到的第一題開頭：{miss[0][:90]!r}")
    print(f"\n  {'全部對得上' if not bad else f'{bad} 個資料集對不上，[C6] 會出現「未知」'}")
    return 1 if bad else 0


def cmd_list():
    """每個 coding 資料集的語言分布（整份資料集，不是 deep20 抽樣的那 80 題）。"""
    by_name = {d.name: d for d in d20.DATASETS}
    for name in [d.name for d in d20.DATASETS if d.category == "coding"]:
        if name in FIXED:
            print(f"  {name:28} {FIXED[name]}")
            continue
        if name not in PER_ROW:
            print(f"  {name:28} [!!] 不在 FIXED 也不在 PER_ROW")
            continue
        try:
            prompts = [p for p in by_name[name].load() if isinstance(p, str) and p.strip()]
        except Exception as e:  # noqa: BLE001
            print(f"  {name:28} [skip] {type(e).__name__}")
            continue
        c = Counter(lang_of(name, p) for p in prompts)
        print(f"  {name:28} {len(c)} 種、{len(prompts)} 題")
        for lang, n in c.most_common():
            print(f"  {'':28}   {lang:18} {n:5}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        raise SystemExit(cmd_check())
    elif cmd == "list":
        cmd_list()
    else:
        raise SystemExit(f"用法：python deep20_coding_langs.py [check|list]")
