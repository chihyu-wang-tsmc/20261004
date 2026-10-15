"""`python deep20_report_h2o.py report` —— deep20_h2o.py 跑出來的 model_route 成本／正確率報表。

⚠ 這支裡面沒有 classifier，也不需要換成 H2OClassifier。

deep20_report.py 和 deep20_guard_report.py 一樣只讀 result 檔，不呼叫任何模型 —— 它的
line 100 寫著「這裡重寫一份而不是 import deep20，因為 deep20 會拉進 classifier_jevk5 →
httpx2 等執行期相依，報表不需要那些」。守門和 model_route 是 deep20_h2o.py run 的時候問的，
報表只是把存下來的 route_p / route 欄位讀出來算曲線。

要換的是三個模組全域：

    RESULT_DIR   deep20_results/   → deep20_h2o_results/   （deep20_h2o.py 寫的地方）
    OUT_DIR      deep20_report/    → deep20_report_h2o/    （PNG 和 Excel 的輸出）
    ROUTER_NAME  "jevk5"           → "h2o-lightning"       （報表訊息和圖例上的字）

為什麼這支需要 wrapper，而 guard_report 其實可以只靠參數：
deep20_guard_report.py 有 --results 可以指定 result 目錄，但**這支沒有**，RESULT_DIR 是在
load/scan 的函式裡直接讀模組全域的（line 235、403-408）。所以要讀 h2o 的 result，覆寫全域
是最直接的辦法（函式都是呼叫時才查，所以來得及）。

ROUTER_NAME 只影響字，不影響數字。沒有它的話，圖例會印「jevk5 model_route」，看圖的人會
以為路由是 jevk5 決定的。

---- 一件要你自己決定的事：GUARD_COST_PER_ITEM ----

deep20_report.py line 113 有 GUARD_COST_PER_ITEM = 0.0013（每題守門的美元開銷），原檔自己
標著「同樣是示意值」，PRICES_ARE_PLACEHOLDER = True 時報表會印警告。

h2o-lightning 是自架在你自己的 L4 上，沒有按次計費的帳單，所以這個數字對 h2o 的意義和對
jevk5 完全不同。我沒有自己改它 —— 填 0 會讓成本曲線變好看，但「自架」不等於「免費」
（GPU 時間、電、機器成本都在），要怎麼折算只有你知道。要改就在下面那行取消註解：

    # report.GUARD_COST_PER_ITEM = 0.0

PRICES（fast / powerful 的 token 單價）同樣是原檔的示意值，和守門是哪一顆無關，不用動。

---- 用法 ----

所有參數原樣轉給 deep20_report.main()，和 deep20_report.py 完全一樣：

    python deep20_report_h2o.py status                  # 看資料齊不齊
    python deep20_report_h2o.py report                  # 讀 deep20_h2o_results/
    python deep20_report_h2o.py report --mock           # 沒有 result 時先看版面
    python deep20_report_h2o.py report --dark --appendix-b
    python deep20_report_h2o.py prepare                 # 併 joined.jsonl（需完整環境）

要同時比 jevk5 和 h2o，兩支各跑一次，輸出在不同目錄，不會互相覆蓋：

    python deep20_report.py     report    # deep20_results/      → deep20_report/
    python deep20_report_h2o.py report    # deep20_h2o_results/  → deep20_report_h2o/
"""

import os

import deep20_report as report

from classifier_h2o import H2O_MODEL

RESULT_DIR = os.path.join(report.HERE, "deep20_h2o_results")
OUT_DIR = os.path.join(report.HERE, "deep20_report_h2o")

# 這三個在 deep20_report 裡都是函式執行時才讀（RESULT_DIR 在 load/scan、OUT_DIR 在寫檔和
# argparse default、ROUTER_NAME 在組訊息和圖例），所以在呼叫 main() 之前覆寫就有效
report.RESULT_DIR = RESULT_DIR
report.OUT_DIR = OUT_DIR
report.ROUTER_NAME = "h2o-lightning"

# 自架沒有按次計費；要把守門成本算成 0 就取消這行的註解（見上面的說明）
# report.GUARD_COST_PER_ITEM = 0.0


if __name__ == "__main__":
    print(f"守門／router：{H2O_MODEL}　result：{os.path.relpath(RESULT_DIR, report.HERE)}/"
          f"　輸出：{os.path.relpath(OUT_DIR, report.HERE)}/")
    report.main()
