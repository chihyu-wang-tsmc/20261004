"""`python deep20_coding_report_h2o.py report` —— deep20_h2o.py 的 coding 偵測報表。

⚠ 這支裡面沒有 classifier，也不需要換成 H2OClassifier。

deep20_coding_report.py 和另外兩份報表一樣只讀 result 檔（它 docstring 最後一行寫著
「讀 result、算指標、印表格都沿用 deep20_guard_report.py 的函式，不 import deep20」）。
coding 的 P(是) 是 deep20_h2o.py run 的時候、第一道那次呼叫一起問出來的，報表只是讀出來
算 Precision / Recall / AUROC。

（[C6] 逐語言是唯一的例外：它要的「這題是什麼程式語言」不在 result 裡，由
deep20_coding_langs.py 回頭讀原始資料集。那個對照表和 classifier 無關，h2o 和 jevk5
兩邊查到的語言一樣，所以這支照樣一行都不用改；原始資料集不在就只少掉 [C6]。）

要換的只有兩個目錄：

    g.RESULT_DIR   deep20_results/          → deep20_h2o_results/        （--results 的預設值）
    OUT_DIR        deep20_coding_report/    → deep20_coding_report_h2o/  （Excel 的輸出）

這支的 --results 預設值是從 deep20_guard_report（import 成 g）的 RESULT_DIR 來的，所以
要覆寫的是 g 那邊的，不是自己模組裡的。

和 deep20_report_h2o.py / deep20_guard_report_h2o.py 不同，這份報表**不需要改名字**：
它的輸出字串裡沒有寫死「jevk5」（整份只有 docstring 第 3 行提到，不會印出來），所以
deep20_coding_report.py 一行都沒動。

---- 用法 ----

所有參數原樣轉給 deep20_coding_report.main()：

    python deep20_coding_report_h2o.py report                  # 讀 deep20_h2o_results/
    python deep20_coding_report_h2o.py report --mock           # 沒有 result 時先看版面
    python deep20_coding_report_h2o.py report --threshold 0.4
    python deep20_coding_report_h2o.py report --category coding cyber
    python deep20_coding_report_h2o.py report --cn --no-excel

要同時比 jevk5 和 h2o，兩支各跑一次，輸出在不同目錄：

    python deep20_coding_report.py     report   # deep20_results/     → deep20_coding_report/
    python deep20_coding_report_h2o.py report   # deep20_h2o_results/ → deep20_coding_report_h2o/

（也可以直接 deep20_coding_report.py report --results deep20_h2o_results --out <某處>，
這支的兩個參數都齊全，效果一樣。）
"""

import os

import deep20_coding_report as coding
import deep20_guard_report as g

from classifier_h2o import H2O_MODEL

RESULT_DIR = os.path.join(coding.HERE, "deep20_h2o_results")
OUT_DIR = os.path.join(coding.HERE, "deep20_coding_report_h2o")

# 兩個都只在 main() 組 argparse 時才讀，所以呼叫前覆寫就有效
g.RESULT_DIR = RESULT_DIR
coding.OUT_DIR = OUT_DIR


if __name__ == "__main__":
    print(f"coding 問的是：{H2O_MODEL}　result：{os.path.relpath(RESULT_DIR, coding.HERE)}/"
          f"　輸出：{os.path.relpath(OUT_DIR, coding.HERE)}/")
    coding.main()
