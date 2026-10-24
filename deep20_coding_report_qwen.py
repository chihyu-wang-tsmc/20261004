"""`python deep20_coding_report_qwen.py report` —— deep20_qwen.py 的 coding 偵測報表。

⚠ 這支裡面沒有 classifier，也不需要換成 QwenClassifier。

和 deep20_coding_report_h2o.py / deep20_coding_report_gemini.py 一樣：deep20_coding_report.py
只讀 result 檔，coding 的 P(是) 是 deep20_qwen.py run 的時候、第一道那次呼叫一起問出來的，
報表只是讀出來算 Precision / Recall / AUROC。deep20_coding_report.py 一行都沒動，要換的只有兩個目錄：

    g.RESULT_DIR   deep20_results/          → deep20_qwen_results/        （--results 的預設值）
    OUT_DIR        deep20_coding_report/    → deep20_coding_report_qwen/  （Excel 的輸出）

--results 的預設值是從 deep20_guard_report（import 成 g）的 RESULT_DIR 來的，所以要覆寫的是
g 那邊的，不是自己模組裡的。

---- 讀完 result 先印兩行（同 Gemini 版） ----

    classifier   每份 result 的 meta.classifier，也就是「實際是哪顆 Qwen 問的」。
                 不是 classifier_qwen.py 現在的 QWEN_MODEL 常數：改過常數但沒 --refresh 的話，
                 result 裡還是舊模型的答案，印常數會騙人。混了不只一顆時會印 ⚠。
    被略過的題目  meta.qwen_blocked 的題數。這些是 DashScope 內容審核擋下的題目（見 deep20_qwen.py），
                 result 裡沒有它們的 row，所以不在這份報表的任何數字裡。它們幾乎都是有害請求，
                 也就是「不該判 coding」那一側，所以 FPR / 特異度的分母會少這幾題。

---- 用法 ----

所有參數原樣轉給 deep20_coding_report.main()：

    python deep20_coding_report_qwen.py report                  # 讀 deep20_qwen_results/
    python deep20_coding_report_qwen.py report --mock           # 沒有 result 時先看版面
    python deep20_coding_report_qwen.py report --threshold 0.4
    python deep20_coding_report_qwen.py report --category coding cyber
    python deep20_coding_report_qwen.py report --cn --no-excel

要同時比幾顆 classifier，各跑一次，輸出在不同目錄：

    python deep20_coding_report.py        report   # deep20_results/        → deep20_coding_report/
    python deep20_coding_report_h2o.py    report   # deep20_h2o_results/    → deep20_coding_report_h2o/
    python deep20_coding_report_gemini.py report   # deep20_gemini_results/ → deep20_coding_report_gemini/
    python deep20_coding_report_qwen.py   report   # deep20_qwen_results/   → deep20_coding_report_qwen/
"""

import os
from collections import Counter

import deep20_coding_report as coding
import deep20_guard_report as g

RESULT_DIR = os.path.join(coding.HERE, "deep20_qwen_results")
OUT_DIR = os.path.join(coding.HERE, "deep20_coding_report_qwen")

# 兩個都只在 main() 組 argparse 時才讀，所以呼叫前覆寫就有效
g.RESULT_DIR = RESULT_DIR
coding.OUT_DIR = OUT_DIR

_load_rows = g.load_rows


def load_rows(args):
    """g.load_rows 外面包一層：讀完印 result 是哪顆 Qwen 問的、略過了幾題。

    deep20_coding_report.cmd_report 是用 g.load_rows(args) 呼叫，所以換掉 g 的名字就接得到。
    metas 只含這次 --only / --category / --cn 選到、而且有 row 的資料集，範圍和報表一致。
    """
    rows, metas = _load_rows(args)
    models = Counter(
        m.get("classifier", "（沒記，不是 deep20_qwen.py 的 result？）") for m in metas
    )
    if len(models) == 1:
        print(f"classifier（result 的 meta）：{next(iter(models))}")
    else:
        print("⚠ result 混了不只一顆 classifier，數字會是它們的混合"
              "（換模型後要 --refresh 重跑 deep20_qwen.py）：")
        for model, n in models.most_common():
            print(f"    {model}：{n} 個資料集")
    blocked = {m["dataset"]: len(m.get("qwen_blocked") or {}) for m in metas}
    blocked = {ds: n for ds, n in blocked.items() if n}
    if blocked:
        detail = "、".join(f"{ds} {n}" for ds, n in sorted(blocked.items()))
        print(f"DashScope 擋下、不在報表裡的題目：{sum(blocked.values())} 題（{detail}）")
    else:
        print("DashScope 擋下、不在報表裡的題目：0 題")
    return rows, metas


g.load_rows = load_rows


if __name__ == "__main__":
    print(f"result：{os.path.relpath(RESULT_DIR, coding.HERE)}/"
          f"　輸出：{os.path.relpath(OUT_DIR, coding.HERE)}/")
    coding.main()
