"""`python deep20_guard_report_h2o_gemma4-31b.py report` —— deep20_h2o_gemma4-31b.py 跑出來的守門報表。

第一道是 H2O-Lightning-4B、第二道是 Gemma 4 31B（Gemini API 的 gemma-4-31b-it，deep20_h2o_gemma4-31b.py
run 的時候問的）。報表本身是 deep20_guard_report.py，只讀 result 檔、不呼叫任何模型，這支只換掉它的模組全域：

    RESULT_DIR      deep20_results/        → deep20_h2o_gemma4-31b_results/
    OUT_DIR         deep20_guard_report/   → deep20_guard_report_h2o_gemma4-31b/
    GUARD_NAME      "jevk5"                → "h2o-lightning-4b"   第一道
    GUARD_NAME_OUT  None                   → result 的 meta.output_classifier（gemma-4-31b-it）  第二道

GUARD_NAME / GUARD_NAME_OUT 只影響報表上的字：[M] 的「Latency／成本」那列會印成「1 次 h2o-lightning-4b
+ 0.83 次 gemma-4-31b-it + 0.83 次 LLM」，[G5] 表 C 的欄名是「守門次數/題（h2o-lightning-4b +
gemma-4-31b-it）」。數字和兩道同一顆時的算法完全一樣。

另外直接 import deep20_guard_report_h2o，所以 measure 政策、[G8] 掃到 1.00 的格點、
--guard-policy 的處理都和 h2o 版同一份。

    python deep20_guard_report_h2o_gemma4-31b.py report --as-policy strict    # 政策照 result 的 meta
    python deep20_guard_report_h2o_gemma4-31b.py report --guard-policy measure --as-policy strict
    python deep20_guard_report_h2o_gemma4-31b.py report --mock                # 沒有 result 時先看版面
    python deep20_guard_report_h2o_gemma4-31b.py label                        # 匯出人工標註用的 CSV

其餘參數都原樣轉給 deep20_guard_report.main()（--only / --category / --cn / --isolate /
--no-excel / --no-plot / --labels / --dark / --tradeoff-* …），用法和 deep20_guard_report.py 一樣。

---- 和 deep20_guard_report_h2o.py 不同的地方 ----

1. 政策可以不給，照 result 的 meta.guard_policy。
   deep20_h2o_gemma4-31b.py 跑的 result 一定記了 guard_policy / guard_thresholds，所以這支會直接讀：
       沒給 --guard-policy   照 result 的（每個資料集都要一樣，不一樣就停下來列出是哪些）
       有給 --guard-policy   要和 result 的一樣，不一樣就停下來
   報表重播時假設的政策和跑的時候不同，[G8] 的數字就不對；h2o 版只在 [G0] 印 ⚠，這支直接擋掉。

2. 第二道的設定要對、而且每個資料集都一樣，不然停下來：
       meta.output_classifier       要是 Gemma（名字裡有 gemma）。--results 指錯目錄（例如
                                    deep20_h2o_h2o31b_results/）時，報表上的字會寫 Gemma、數字卻是別顆的
       meta.output_classifier /     deep20_h2o_gemma4-31b.py 的 GEMMA_MODEL / GEMMA_THINKING 可以改，
       output_thinking_level /      改了沒 --refresh 的話，不同資料集是不同設定問的，混在一起比沒有意義
       output_json_schema
   報表上第二道的名字照 meta.output_classifier 寫，所以換成別顆 Gemma 時不用改這支。

3. [G0] 多幾行：
       - 多少題沿用 deep20_h2o_results/ 的第一道和模型回覆
       - 第二道判的回覆裡有幾題是截在 400 字的（row 的 output_truncated，見 deep20_h2o_gemma4-31b.py
         檔頭）。截過的回覆 Gemma 只看得到開頭，可能判得偏輕
       - Gemini API 不肯讀的回覆有幾題（meta.gemini_blocked）。這些題目沒有 row，連第一道都不算進
         任何指標，和 deep20_guard_report_h2o 比第一道時題數會少這幾題；被擋的通常是最有害的回覆
       - Gemma 不接受 JSON mode、改成只在提示裡要求 JSON 時（meta.output_json_schema = False）
         印一行，和 Gemini 版比時要知道兩邊的答案格式限制不同

4. dataset_guardrail*.xlsx 預設寫進 deep20_guard_report_h2o_gemma4-31b/（原本寫在 repo 根目錄，
   和 jevk5 版同一個檔名，會互相覆蓋）。--dataset-out 照樣可以指定別的地方。

---- 人工標籤 ----

labels.csv 在 deep20_guard_report_h2o_gemma4-31b/labels.csv。沿用的題目，模型回覆和 h2o 版那份一字不差
（同一個 reply），所以 deep20_guard_report_h2o/labels.csv（或 jevk5 / 31B 版的）標過的可以直接拿來用：

    python deep20_guard_report_h2o_gemma4-31b.py report --labels deep20_guard_report_h2o/labels.csv

只有整條重跑過的題目（deep20_h2o_gemma4-31b.py 印「整條跑」的那些，row 的 reused = False）回覆不一樣，
它們的 h2o 標籤不適用。全部都沿用時沒有這個問題。

---- 五支報表一起比 ----

    python deep20_guard_report.py                report   # jevk5 → LLM → jevk5    deep20_guard_report/
    python deep20_guard_report_h2o.py            report   # 4B    → LLM → 4B       deep20_guard_report_h2o/
    python deep20_guard_report_h2o_jevk5.py      report   # 4B    → LLM → jevk5    deep20_guard_report_h2o_jevk5/
    python deep20_guard_report_h2o_h2o31b.py     report   # 4B    → LLM → 31B      deep20_guard_report_h2o_h2o31b/
    python deep20_guard_report_h2o_gemma4-31b.py report   # 4B    → LLM → Gemma    deep20_guard_report_h2o_gemma4-31b/

後四支的第一道和模型回覆是同一批 result 抄過來的（全部沿用時），所以 [G2] [G3] 和 [G1] 漏斗的
輸入、呼叫模型那幾欄應該一模一樣；差別都在第二道（[G1] 的輸出欄、[G4]、[M] 的第二道欄、[G5] …）。
Gemma 版例外：Gemini API 不肯讀的那幾題整題不在，[G2] [G3] 會少這幾題（[G0] 會印是幾題）。
"""

import os
import sys
from collections import Counter

import deep20_guard_report as report
import deep20_guard_report_h2o as h2o_report  # measure 政策、REPLAY_GRID 補 1.00

from classifier_h2o import H2O_MODEL

RESULT_DIR = os.path.join(report.HERE, "deep20_h2o_gemma4-31b_results")
OUT_DIR = os.path.join(report.HERE, "deep20_guard_report_h2o_gemma4-31b")
# 沿用第一道和模型回覆的來源（deep20_h2o_gemma4-31b.py 的 SOURCE_DIR），只用在 [G0] 的字
SOURCE_DIR = h2o_report.RESULT_DIR
# 第二道的名字在讀到 result 之前（--mock）用這個；讀到之後換成 meta.output_classifier。
# 和 deep20_h2o_gemma4-31b.py 的 GEMMA_MODEL 預設一樣（那支檔名有 -，import 不到，也不該 import：
# 它一 import 就會改掉 deep20 的全域）
GEMMA_MODEL = "gemma-4-31b-it"

# 有沒有給 --guard-policy（見 __main__）；沒給就照 result 的 meta.guard_policy
POLICY_GIVEN = False

# 都只在函式執行時才讀（RESULT_DIR / OUT_DIR 是 argparse 的 default，GUARD_NAME* 在組表格的時候），
# 所以在呼叫 main() 之前覆寫就有效。deep20_guard_report_h2o 在 import 時設過一次，這裡蓋回來
report.RESULT_DIR = RESULT_DIR
report.OUT_DIR = OUT_DIR
report.GUARD_NAME = "h2o-lightning-4b"
report.GUARD_NAME_OUT = GEMMA_MODEL

# 第二道的設定：每個資料集都要一樣（見檔頭第 2 點）
_SETTINGS = ("output_classifier", "output_thinking_level", "output_json_schema")


_load_rows = report.load_rows


def load_rows(args):
    """report.load_rows 外面包一層：第二道要是 Gemma、設定一致；政策沒給 --guard-policy 就照 result。"""
    rows, metas = _load_rows(args)
    if args.mock or not metas:
        return rows, metas
    other = [f"{m['dataset']}（{m.get('output_classifier') or m.get('classifier') or '沒記'}）"
             for m in metas if "gemma" not in str(m.get("output_classifier", "")).lower()]
    if other:
        raise SystemExit(f"{'、'.join(other)} 的第二道不是 Gemma"
                         f"（不是 deep20_h2o_gemma4-31b.py 跑的？）：檢查 --results，"
                         f"或用 --only 只挑 deep20_h2o_gemma4-31b.py 跑的資料集")
    setups = Counter(tuple(m.get(k) for k in _SETTINGS) for m in metas)
    if len(setups) > 1:
        detail = "\n".join(f"  {' / '.join(map(str, s))}：{n} 個資料集" for s, n in setups.most_common())
        raise SystemExit(f"result 的第二道設定不一致（{' / '.join(_SETTINGS)}）：\n{detail}\n"
                         "deep20_h2o_gemma4-31b.py 的設定改過（GEMMA_MODEL / GEMMA_THINKING，或 JSON mode 被自動關掉）"
                         "之後要 --refresh 重跑齊，或用 --only 只挑同一組的")
    report.GUARD_NAME_OUT = metas[0]["output_classifier"]
    ran = Counter(m.get("guard_policy") for m in metas)
    if len(ran) > 1:
        detail = "、".join(f"{name or '（沒記）'} {n} 個資料集" for name, n in ran.most_common())
        raise SystemExit(f"result 的政策不一致（{detail}）：報表重播只能假設一組。\n"
                         "用同一個 --guard-policy 把這些資料集重跑齊，或用 --only 只挑同一組的")
    (name,) = ran
    if name is None:
        if POLICY_GIVEN:
            return rows, metas
        raise SystemExit("result 沒記 meta.guard_policy（不是 deep20_h2o_gemma4-31b.py 跑的？），"
                         "請加 --guard-policy 指定當時用的政策")
    if name not in report.POLICIES:
        raise SystemExit(f"result 是用 {name!r} 跑的，報表不認得這個政策（可用：{', '.join(report.POLICIES)}）")
    want = report.POLICIES[name]
    off = [m["dataset"] for m in metas if m.get("guard_thresholds") not in (None, want)]
    if off:
        raise SystemExit(f"{'、'.join(off)} 的 meta.guard_thresholds 和報表的 {name} {want} 不同，"
                         "重播出來的動作會對不上；請把 deep20_guard_report.py 的 POLICIES 和 deep20.py 同步")
    if POLICY_GIVEN and name != report.GUARD_POLICY:
        raise SystemExit(f"--guard-policy {report.GUARD_POLICY}，但 result 是用 {name} 跑的；"
                         f"拿掉 --guard-policy 就會照 result")
    report.GUARD_POLICY = name
    return rows, metas


_report_overview = report.report_overview


def report_overview(rep, rows, metas, args):
    """[G0] 多記幾行：沿用了幾題、截在 400 字的回覆、Gemini API 不肯讀的回覆、JSON mode。"""
    _report_overview(rep, rows, metas, args)
    if args.mock:
        return
    gemma = report.GUARD_NAME_OUT
    m0 = metas[0]
    src = os.path.relpath(SOURCE_DIR, report.HERE)
    reused = sum(bool(r.get("reused")) for r in rows)
    fresh = len(rows) - reused
    rep.note(f"第一道 {H2O_MODEL}、第二道 {gemma}（Gemini API，thinking {m0.get('output_thinking_level')}）："
             f"{reused}/{len(rows)} 題的第一道和模型回覆沿用 {src}/，只有第二道是 Gemma 新問的"
             + (f"；{fresh} 題整條重跑（4B → LLM → Gemma）" if fresh else ""))
    graded = [r for r in rows if r["g2"] is not None]
    cut = sum(bool(r.get("output_truncated")) for r in graded)
    if cut:
        rep.note(f"⚠ 第二道 {len(graded)} 題裡有 {cut} 題（{cut / len(graded):.1%}）判的是截在前 400 字的回覆"
                 f"（{src}/ 只存了前 400 字，h2o 版的第二道看的是完整回覆）：\n"
                 "  這些題目 Gemma 只看得到開頭，可能判得偏輕，和 deep20_guard_report_h2o 比第二道時要算進去")
    blocked = {m["dataset"]: len(m.get("gemini_blocked") or {}) for m in metas}
    n_blocked = sum(blocked.values())
    if n_blocked:
        where = "、".join(f"{ds} {n}" for ds, n in sorted(blocked.items(), key=lambda kv: -kv[1]) if n)
        rep.note(f"⚠ Gemini API 不肯讓 Gemma 讀的回覆 {n_blocked} 題（{where}；meta.gemini_blocked）：\n"
                 "  這些題目沒有 row，連第一道都不算進任何指標。被擋的通常是最有害的回覆，"
                 "略過它們會讓第二道和整個系統的擋下率稍微偏低；\n"
                 f"  和 deep20_guard_report_h2o 比第一道時，題數會少這 {n_blocked} 題")
    if m0.get("output_json_schema") is False:
        rep.note(f"{gemma} 不接受 JSON mode，答案是只在 system prompt 要求 JSON 拿到的（meta.output_json_schema = False）；"
                 "\n  Gemini 版有 responseSchema 限定格式，兩邊的答案格式限制不同")


_dataset_xlsx_out = report.dataset_xlsx_out


def dataset_xlsx_out(args):
    """dataset_guardrail*.xlsx 預設寫進 OUT_DIR，不和 jevk5 版搶 repo 根目錄的同一個檔名。"""
    path = _dataset_xlsx_out(args)
    if args.dataset_out or args.mock:
        return path
    return os.path.join(args.out, os.path.basename(path))


report.load_rows = load_rows
report.report_overview = report_overview
report.dataset_xlsx_out = dataset_xlsx_out


if __name__ == "__main__":
    name = h2o_report._take_guard_policy(sys.argv)
    if name is not None:
        if name not in report.POLICIES:
            sys.exit(f"沒有這個政策：{name!r}（可用：{', '.join(report.POLICIES)}）")
        report.GUARD_POLICY = name
        POLICY_GIVEN = True
    print(f"第一道：{H2O_MODEL}　第二道：Gemma（照 result 的 meta.output_classifier）"
          f"　result：{os.path.relpath(RESULT_DIR, report.HERE)}/（預設）"
          f"　輸出：{os.path.relpath(OUT_DIR, report.HERE)}/（預設）")
    if POLICY_GIVEN:
        pol = report.POLICIES[report.GUARD_POLICY]
        print(f"假設跑的政策：{report.GUARD_POLICY}　action≥{pol['action_threshold']}"
              f"　severity_block {pol['severity_block']}　（會和 result 的 meta.guard_policy 對）")
    else:
        print("假設跑的政策：照 result 的 meta.guard_policy")
    report.main()
