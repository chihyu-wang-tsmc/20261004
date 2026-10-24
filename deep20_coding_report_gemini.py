"""`python deep20_coding_report_gemini.py report` —— deep20_gemini.py 的 coding 偵測報表。

⚠ 這支裡面沒有 classifier，也不需要換成 GeminiClassifier。

和 deep20_coding_report_h2o.py 一樣：deep20_coding_report.py 只讀 result 檔，coding 的 P(是)
是 deep20_gemini.py run 的時候、第一道那次呼叫一起問出來的，報表只是讀出來算
Precision / Recall / AUROC。deep20_coding_report.py 一行都沒動，要換的只有兩個目錄：

    g.RESULT_DIR   deep20_results/          → deep20_gemini_results/        （--results 的預設值）
    OUT_DIR        deep20_coding_report/    → deep20_coding_report_gemini/  （Excel 的輸出）

--results 的預設值是從 deep20_guard_report（import 成 g）的 RESULT_DIR 來的，所以要覆寫的是
g 那邊的，不是自己模組裡的。

---- 比 h2o 版多的一件事：讀完 result 先印兩行 ----

    classifier   每份 result 的 meta.classifier / meta.thinking_level，也就是「實際是誰問的」。
                 不是 classifier_gemini.py 現在的 GEMINI_MODEL 常數：改過常數但沒 --refresh 的話，
                 result 裡還是舊設定的答案，印常數會騙人。混了不只一種設定時會印 ⚠。
    被略過的題目  meta.gemini_blocked 的題數。這些是 Gemini 一律不讀的題目（見 deep20_gemini.py），
                 result 裡沒有它們的 row，所以不在這份報表的任何數字裡。它們幾乎都是有害請求，
                 也就是「不該判 coding」那一側，所以 FPR / 特異度的分母會少這幾題。

---- 用法 ----

所有參數原樣轉給 deep20_coding_report.main()：

    python deep20_coding_report_gemini.py report                  # 讀 deep20_gemini_results/
    python deep20_coding_report_gemini.py report --mock           # 沒有 result 時先看版面
    python deep20_coding_report_gemini.py report --threshold 0.4
    python deep20_coding_report_gemini.py report --category coding cyber
    python deep20_coding_report_gemini.py report --cn --no-excel

要同時比 jevk5、h2o、Gemini，各跑一次，輸出在不同目錄：

    python deep20_coding_report.py        report   # deep20_results/        → deep20_coding_report/
    python deep20_coding_report_h2o.py    report   # deep20_h2o_results/    → deep20_coding_report_h2o/
    python deep20_coding_report_gemini.py report   # deep20_gemini_results/ → deep20_coding_report_gemini/
"""

import os
from collections import Counter

import deep20_coding_report as coding
import deep20_guard_report as g

RESULT_DIR = os.path.join(coding.HERE, "deep20_gemini_results")
OUT_DIR = os.path.join(coding.HERE, "deep20_coding_report_gemini")

# 兩個都只在 main() 組 argparse 時才讀，所以呼叫前覆寫就有效
g.RESULT_DIR = RESULT_DIR
coding.OUT_DIR = OUT_DIR

_load_rows = g.load_rows


def load_rows(args):
    """g.load_rows 外面包一層：讀完印 result 是哪個 Gemini 設定問的、略過了幾題。

    deep20_coding_report.cmd_report 是用 g.load_rows(args) 呼叫，所以換掉 g 的名字就接得到。
    metas 只含這次 --only / --category / --cn 選到、而且有 row 的資料集，範圍和報表一致。
    """
    rows, metas = _load_rows(args)
    settings = Counter(
        f"{m.get('classifier', '（沒記，不是 deep20_gemini.py 的 result？）')}"
        f"（thinking {m.get('thinking_level', '?')}）"
        for m in metas
    )
    if len(settings) == 1:
        print(f"classifier（result 的 meta）：{next(iter(settings))}")
    else:
        print("⚠ result 混了不只一種 classifier 設定，數字會是它們的混合"
              "（換設定後要 --refresh 重跑 deep20_gemini.py）：")
        for s, n in settings.most_common():
            print(f"    {s}：{n} 個資料集")
    blocked = {m["dataset"]: len(m.get("gemini_blocked") or {}) for m in metas}
    blocked = {ds: n for ds, n in blocked.items() if n}
    if blocked:
        detail = "、".join(f"{ds} {n}" for ds, n in sorted(blocked.items()))
        print(f"Gemini 不肯讀、不在報表裡的題目：{sum(blocked.values())} 題（{detail}）")
    else:
        print("Gemini 不肯讀、不在報表裡的題目：0 題")
    return rows, metas


g.load_rows = load_rows


if __name__ == "__main__":
    print(f"result：{os.path.relpath(RESULT_DIR, coding.HERE)}/"
          f"　輸出：{os.path.relpath(OUT_DIR, coding.HERE)}/")
    coding.main()
