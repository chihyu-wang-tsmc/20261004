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

    # 用 measure 跑出來的 result（deep20_h2o.py run --guard-policy measure）：
    # --guard-policy measure 告訴報表 result 是怎麼跑的，--as-policy strict 讓 [M] [G1]–[G7] 和
    # dataset_guardrail*.xlsx 改成 strict（0.70）的結果；[G8] [G10] 的門檻掃描 0–1 都是確切值
    python deep20_guard_report_h2o.py report --guard-policy measure --as-policy strict \
        --dataset-out deep20_guard_report_h2o/dataset_guardrail.xlsx

要同時比 jevk5 和 h2o，就兩支各跑一次，輸出在不同目錄，不會互相覆蓋：

    python deep20_guard_report.py     report    # deep20_results/      → deep20_guard_report/
    python deep20_guard_report_h2o.py report    # deep20_h2o_results/  → deep20_guard_report_h2o/

（也可以直接 deep20_guard_report.py report --results deep20_h2o_results --out <某處>，
效果一樣，只是報表上的字會寫 jevk5。）
"""

import os
import sys

import deep20_guard_report as report

from classifier_h2o import H2O_MODEL

RESULT_DIR = os.path.join(report.HERE, "deep20_h2o_results")
OUT_DIR = os.path.join(report.HERE, "deep20_guard_report_h2o")

# 和 deep20_h2o.py 的 MEASURE_POLICY 同一組。報表重播時假設的政策必須和跑的時候一樣，
# 不然 [G8] 的數字不對（deep20_guard_report.py 有一致性檢查，對不上會印 ⚠）。
report.POLICIES["measure"] = {
    "review_threshold": 0.35,
    "action_threshold": 1.0,
    "severity_block": None,   # 跟著 action 走
}

# [G8] 表 A 的掃描格點補上 1.00（原本最大 0.95）。用量測政策跑出來的 result，
# 這一整排到 1.00 都會是確切值而不是區間。
# 只改 h2o 這一支；jevk5 版維持原樣，要一起改就把這行搬進 deep20_guard_report.py。
report.REPLAY_GRID = report.REPLAY_GRID + (1.00,)

# [G10] 的 TRADEOFF_GRID 本來就是 0.00–1.00，不用動；它超過實際門檻那段原本是估計值，
# 改用量測政策跑之後就變成確切值。

# 這三個在 deep20_guard_report 裡都只被函式在執行時讀取（RESULT_DIR / OUT_DIR 是 argparse
# 的 default，GUARD_NAME 在組表格的時候），所以在呼叫 main() 之前覆寫就有效
report.RESULT_DIR = RESULT_DIR
report.OUT_DIR = OUT_DIR
report.GUARD_NAME = "h2o-lightning"


def _take_guard_policy(argv):
    """從 argv 取出 --guard-policy <名稱>（report.main() 的 parser 不認得它，要先拿掉）。"""
    for i, a in enumerate(argv):
        if a == "--guard-policy":
            if i + 1 >= len(argv):
                sys.exit("--guard-policy 後面要接政策名稱")
            name = argv[i + 1]
            del argv[i:i + 2]
            return name
        if a.startswith("--guard-policy="):
            name = a.split("=", 1)[1]
            del argv[i]
            return name
    return None


if __name__ == "__main__":
    name = _take_guard_policy(sys.argv)
    if name is not None:
        if name not in report.POLICIES:
            sys.exit(f"沒有這個政策：{name!r}（可用：{', '.join(report.POLICIES)}）")
        report.GUARD_POLICY = name
    pol = report.POLICIES[report.GUARD_POLICY]
    print(f"守門：{H2O_MODEL}　result：{os.path.relpath(RESULT_DIR, report.HERE)}/"
          f"　輸出：{os.path.relpath(OUT_DIR, report.HERE)}/")
    print(f"假設跑的政策：{report.GUARD_POLICY}　action≥{pol['action_threshold']}"
          f"　severity_block {pol['severity_block']}"
          f"　（要和 result 的 meta.guard_policy 一致）")
    report.main()
