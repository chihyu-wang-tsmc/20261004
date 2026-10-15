"""`python deep20_guard_report_h2o.py report` —— deep20_h2o.py 跑出來的守門報表。

⚠ 這支裡面沒有 classifier，也不需要換成 H2OClassifier。

deep20_guard_report.py 只讀 result 檔，不呼叫任何模型（它刻意不 import deep20，就是為了
不要拉進 classifier_jevk5 / langchain 這些執行期相依，見它 docstring 的最後一節）。所以
「換成 H2OClassifier」在這支沒有對應的東西可以換 —— 守門是 deep20_h2o.py run 的時候問的，
報表只是把它存下來的 result 讀出來算指標。

要換的只有三個東西，都是模組全域，報表的函式都是呼叫時才查，所以覆寫得掉：

    RESULT_DIR   deep20_results/        → deep20_h2o_results/        （deep20_h2o.py 寫的地方）
    OUT_DIR      deep20_guard_report/   → deep20_guard_report_h2o/   （Excel 和圖的輸出）
    GUARD_NAME   "jevk5"                → "h2o-lightning"            （報表上 Latency 那列和
                                                                       表 C 欄名的字）

GUARD_NAME 只影響報表上的字，不影響任何數字。沒有它的話，h2o 的報表會印成
「1 次 jevk5 + 0.83 次 LLM」，看報表的人會以為守門是 jevk5 問的。

其餘所有參數都原樣轉給 deep20_guard_report.main()：report / label 子命令、--only、
--category、--cn、--isolate、--mock、--no-excel、--no-plot、--labels、--dark、
--tradeoff-* 等等，用法和 deep20_guard_report.py 完全一樣。

    python deep20_guard_report_h2o.py report                 # 讀 deep20_h2o_results/
    python deep20_guard_report_h2o.py report --mock          # 沒有 result 時先看版面
    python deep20_guard_report_h2o.py report --no-excel --no-plot
    python deep20_guard_report_h2o.py report --cn            # 中文版 result

要同時比 jevk5 和 h2o，就兩支各跑一次，輸出在不同目錄，不會互相覆蓋：

    python deep20_guard_report.py     report    # deep20_results/      → deep20_guard_report/
    python deep20_guard_report_h2o.py report    # deep20_h2o_results/  → deep20_guard_report_h2o/

（也可以直接 deep20_guard_report.py report --results deep20_h2o_results --out <某處>，
效果一樣，只是報表上的字會寫 jevk5。）
"""

import os

import deep20_guard_report as report

from classifier_h2o import H2O_MODEL

RESULT_DIR = os.path.join(report.HERE, "deep20_h2o_results")
OUT_DIR = os.path.join(report.HERE, "deep20_guard_report_h2o")

# 這三個在 deep20_guard_report 裡都只被函式在執行時讀取（RESULT_DIR / OUT_DIR 是 argparse
# 的 default，GUARD_NAME 在組表格的時候），所以在呼叫 main() 之前覆寫就有效
report.RESULT_DIR = RESULT_DIR
report.OUT_DIR = OUT_DIR
report.GUARD_NAME = "h2o-lightning"


if __name__ == "__main__":
    print(f"守門：{H2O_MODEL}　result：{os.path.relpath(RESULT_DIR, report.HERE)}/"
          f"　輸出：{os.path.relpath(OUT_DIR, report.HERE)}/")
    report.main()
