"""`python deep20_report.py report` —— model_route 的成本／正確率權衡報告（第三階段）。

deep20.py 的 run 把 jevk5 的回答存進 deep20_results/，但它只給「router 選了誰」，
沒有「選了之後答對沒有」。後者在四個 *_routing.py 的 answers/<model>.jsonl 裡。
本檔把兩邊併起來，算出整條成本–品質曲線，印成報表、畫成 PNG、寫成 Excel。

    deep12.py / deep13.py 的 report   逐題標籤的對錯（refuse / coding / route 的分類正確率）
    本檔的 report                     端到端的成本與正確率權衡（曲線、oracle、CPT、APGR）

---- 為什麼分成 prepare / report 兩段 ----

prepare 要 import <bench>_routing 才拿得到「題目 id ↔ prompt」的對照（答案檔是按 id 存的，
deep20 的 result 是按 prompt 的 sha1 存的，兩邊要靠題目本體接起來）。那條 import 會一路拉進
routing_bench → langchain_core → llm_models，和跑 deep20.py run 需要的環境相同。

report 不需要那些東西。所以：

    prepare   在跑得動 deep20.py run 的那個環境執行一次 → 產出 deep20_report/joined.jsonl
    report    只讀 joined.jsonl，純 numpy / pandas / matplotlib，在任何地方都跑得起來

joined.jsonl 一題一行，欄位就是畫圖需要的全部：

    bench        gpqa / ai2arc / aime / gsm8k
    qid          該 bench 的題目 id
    route_p      jevk5 給 powerful 的機率（deep20 的 row["route_p"]）
    route        jevk5 的 argmax 選擇（"fast" / "powerful"）
    route_conf   Choice 的 confidence，= |2·route_p − 1|（二選一時是 route_p 的函數）
    guard        輸入守門的動作 pass / review / block
    c_fast       fast model 這題答對沒有（0/1）
    c_pow        powerful model 這題答對沒有（0/1）
    tok_fast     fast 的 input + output token
    tok_pow      powerful 的 input + output token
    cost_fast    由 token × PRICES 算出的這題花費（美元）
    cost_pow     同上

---- 用法 ----

    python deep20_report.py report --mock          # 用示意資料跑完整流程（沒有真資料時先看版面）
    python deep20_report.py status                 # 檢查真資料齊不齊、缺的要跑什麼
    python deep20_report.py prepare                # 併資料（需要完整環境）
    python deep20_report.py report                 # 真資料的報表 + PNG + Excel
    python deep20_report.py report --dark          # 深色版 PNG（另存一份，不覆蓋淺色）
    python deep20_report.py report --appendix-b    # 多畫附錄 B（圖 B2 要掃 deep20_results/ 全部 50 個）
    python deep20_report.py report --with-random   # 圖 1/2 加回 Random 基準線（預設不畫）
    python deep20_report.py report --with-band     # 圖 1/2 加回 bootstrap 95% 誤差帶（預設不畫）
    python deep20_report.py report --with-band --boot 2000   # 改 bootstrap 次數
    python deep20_report.py report --out somewhere # 改輸出目錄

---- report_muli_fast：powerful 固定、比好幾個 fast ----

powerful 固定是 ROUTE_POWERFUL（gemini-3.7-flash），fast 可以一次比好幾個（deepseek、qwen、gemma…）。
router 看不到 model 身分，route_p 和升級順序對每個 fast 都一樣，所以不用重跑 deep20.py run：
先照常跑過 prepare，再讓每個 fast 對四個 bench 作答就好。

    python gpqa_routing.py answer --model qwen3.8-27b       # 每個要比的 fast、每個 bench 各跑一次
    python deep20_report.py report_muli_fast                # 自動找 answers/ 底下的 fast
    python deep20_report.py report_muli_fast --fast deepseek-flash qwen3.8-27b gemma-4-xxx
    python deep20_report.py report_muli_fast --mock         # 示意資料（三個假的 fast）
    python deep20_report.py report_muli_fast --per-fast full    # 每個 fast 另出和 report 一樣的整套
    python deep20_report.py report_muli_fast --dark         # 深色版
    （report_multi_fast 是同一個指令的別名）

只比「每個 fast 都有有效作答」的交集；缺哪個 bench 的哪些題，報表開頭的 [覆蓋] 會列出要跑的指令。
各 fast 的單價填在 FAST_PRICES，沒填的暫用 PRICES["fast"]，報表與圖上會標出來。

---- 產出 ----

    deep20_report/joined.jsonl        prepare 的產物
    deep20_report/fig1_main.png       正文圖 1：成本–品質曲線
    deep20_report/fig1_main_tau_curve.png   同上，門檻 τ 的記號標在曲線上（做法 A）
    deep20_report/fig1_main_tau_axis.png    同上，門檻 τ 當上方第二條 X 軸（做法 B）
    deep20_report/fig1_main_tau_axis_conf.png  同上，上方同時標 τ 與 route_conf（做法 C2）
    deep20_report/fig1_main_tau_axis_<bench>.png   做法 B 的四個 bench 各一張（gpqa / ai2arc / aime / gsm8k）
    deep20_report/fig1_main_tau_axis_explained.png / _<bench>_explained.png
                                      同上的說明版：圖上標 1–6，右邊逐項寫出每個數字怎麼算（需要中文字型）
    deep20_report/fig1_main_tau_axis_explained_no_down.png / _<bench>_explained_no_down.png
                                      同上，但不標往下掉的階（▼），只到 ⑤
    deep20_report/fig1_main_tau_axis[_<bench>]_explained_no_down_chart.png / .txt
                                      不標 ▼ 那版拆成只有圖、只有文字兩個檔，給投影片用
    deep20_report/fig2_per_bench.png  正文圖 2：四個 bench 分開看
    deep20_report/fig2_per_bench_tau_curve.png / _tau_axis.png   同上的兩種 τ 版本
    deep20_report/fig2_per_bench_zoom.png   同上，Y 軸每格各自從藍線附近開始（跨距各格不同）
    deep20_report/fig2_per_bench_zoom_tau_curve.png / _tau_axis.png   同上的兩種 τ 版本
    deep20_report/fig2_per_bench_norm.png   同上，Y 改成「補回 oracle 差距的比例」，四格同一把尺
    deep20_report/fig2_per_bench_norm_tau_axis.png   同上，上方標 τ
    deep20_report/fig3_routep.png     正文圖 3：route_p 按 ground truth 分層
    deep20_report/figA1_y_axis.png    附錄 A 圖 A1：為什麼 Y 軸用 end-to-end 正確率
    deep20_report/figA2_calibration.png  附錄 A 圖 A2：校準圖 + ECE
    deep20_report/figB1_guard_cost.png   附錄 B 圖 B1：含／不含守門開銷
    deep20_report/figB2_conf_by_dataset.png  附錄 B 圖 B2：50 個資料集的 route_conf
    deep20_report/multi_fast/         report_muli_fast 的產物：
        curve_tau_axis[_<bench>].png        各 fast 的藍線疊在一起，上方 τ（每個 fast 共用）
        cost[_<bench>].png                  X 換成 $ 每千題（各 fast 的 $ 不同，不能共用呼叫比例軸）
        per_bench_tau_axis.png              四個 bench 各一格
        curve_tau_axis[_<bench>]_explained.png / _explained_chart.png / _explained.txt
                                            說明版（圖上標 1–5、右邊逐項推導）與拆開的圖／文字
        tables.xlsx                         各 fast × 各 bench 的摘要、作答覆蓋、曲線、逐題資料
        <fast>/                             每個 fast 單一 fast 的 fig1_main_tau_axis 與說明版
                                            （--per-fast full 時是和 report 一樣的整套）
    deep20_report/tables.xlsx         表 1（摘要）／表 A1（各 bench）／表 A2（能力重疊，逐 dataset
                                      的 fast×powerful 2×2）／表 A3（混淆矩陣）／校準分箱／
                                      曲線原始值／方法參數／逐題資料

---- 一定要自己改的東西 ----

PRICES（下面）是**示意單價**。報表每次都會把用到的單價印出來並寫進 Excel 的「方法參數」頁，
但數字對不對只有你知道。沒填對之前，所有 $ 的數字都不要引用。
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import unicodedata
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "deep20_results")
OUT_DIR = os.path.join(HERE, "deep20_report")
JOINED = "joined.jsonl"

# deep20.py 的 DATASETS 裡，有 route_gold 的那四個（DS 名稱 → *_routing.py 的模組前綴）
BENCHES = {"GPQA": "gpqa", "AI2ARC": "ai2arc", "AIME": "aime", "GSM8K": "gsm8k"}

# deep20.py:483 的同一組常數。這裡重寫一份而不是 import deep20，因為 deep20 會拉進
# classifier_jevk5 → httpx2 等執行期相依，報表不需要那些。改 deep20 的配對時這裡要跟著改。
ROUTE_FAST = "deepseek-flash"
ROUTE_POWERFUL = "gemini-3.7-flash"

# 守門／router 是誰問的，只影響報表和圖例上的字（deep20_report_h2o.py 會覆寫）
ROUTER_NAME = "jevk5"

# ---- 單價：$ / 百萬 token。**示意值，務必換成實際報價** ----
PRICES = {
    "fast": {"in": 0.27, "out": 1.10},
    "powerful": {"in": 0.30, "out": 2.50},
}
PRICES_ARE_PLACEHOLDER = True  # 填完真實單價後改成 False，報表就不再印警告

# 每題守門的開銷（美元）。deep19/deep20 的流程是輸入端一次 jevk5、有呼叫模型時輸出端再一次。
# 同樣是示意值。
GUARD_COST_PER_ITEM = 0.0013

N_BINS = 10          # 校準圖的分箱數
GRID = np.linspace(0.0, 1.0, 201)   # 曲線統一插值到這個呼叫比例格點上（算 APGR / bootstrap 用）

# ============================================================================
# 調色盤：來自 dataviz skill 的 reference palette，淺／深兩套各自選過，不是自動翻轉
# ============================================================================
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "router": "#2a78d6",    # slot 1 blue
        "mark": "#eb6834",      # slot 2 orange（★ 現行門檻 / 建議門檻）
        "oracle": "#1baf7a",    # slot 3 aqua
        "band": "#2a78d6",
        "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "router": "#3987e5", "mark": "#d95926", "oracle": "#199e70", "band": "#3987e5",
        "cat": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
    },
}

# ============================================================================
# 字型：PNG 上的中文需要 CJK 字型，找不到就整份改用英文標籤（不要讓它變成豆腐方塊）
# ============================================================================
CJK_CANDIDATES = [
    os.path.expanduser("~/.local/share/fonts/NotoSansTC-Regular.otf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msjh.ttc",
]
ZH = True  # setup_font() 會按實際找到的字型改寫


def t(zh, en):
    """同一個標籤的中英兩版；沒有 CJK 字型時自動退回英文。"""
    return zh if ZH else en


def setup_font():
    """註冊 CJK 字型並設定 matplotlib 的全域樣式。回傳有沒有找到中文字型。"""
    global ZH
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    import matplotlib.pyplot as plt

    name = None
    for path in CJK_CANDIDATES:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    if name is None:  # 系統裡本來就裝了的話也算
        have = {f.name for f in fm.fontManager.ttflist}
        for cand in ("Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei",
                     "PingFang TC", "WenQuanYi Zen Hei", "Source Han Sans TC"):
            if cand in have:
                name = cand
                break

    ZH = name is not None
    plt.rcParams.update({
        "font.family": [name] if name else ["DejaVu Sans"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",      # 格線一律實線細髮絲，不用虛線（虛線會被讀成門檻）
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
    })
    return name


def style_axes(ax, c):
    """每張圖共用的坐標軸處理：只留左下兩邊、格線退到背景。"""
    ax.set_facecolor(c["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
    ax.grid(color=c["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=c["muted"], labelcolor=c["ink2"])


# ============================================================================
# prepare：把四個 bench 的作答 + deep20 的 result 併成 joined.jsonl
# ============================================================================
def prompt_key(dataset, prompt):
    """和 deep20.py:176 一字不差。這裡重寫是為了不 import deep20（它有執行期相依）。"""
    return f"{dataset}:{hashlib.sha1(prompt.encode('utf-8')).hexdigest()}"


def read_jsonl(path):
    """和 routing_bench.read_jsonl 一樣：同一題有多筆時以最後一筆為準。"""
    records = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["id"]] = r
    return records


def result_rows(ds_name):
    """讀 deep20_results/<DS>.json，回傳 {prompt_key: row}；檔案不在就回空的。"""
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", ds_name)
    path = os.path.join(RESULT_DIR, f"{safe}.json")
    if not os.path.exists(path):
        return {}, path
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("rows", {}), path


def item_cost(in_tok, out_tok, which):
    p = PRICES[which]
    return (in_tok or 0) / 1e6 * p["in"] + (out_tok or 0) / 1e6 * p["out"]


def cmd_prepare(args):
    """需要能 import <bench>_routing 的環境（和跑 deep20.py run 的環境相同）。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    out, skipped = [], []

    for ds_name, mod_name in BENCHES.items():
        rows, res_path = result_rows(ds_name)
        if not rows:
            skipped.append(f"{ds_name}: 沒有 {os.path.relpath(res_path, HERE)}"
                           f"（先跑 python deep20.py run --only {ds_name}）")
            continue
        try:
            module = importlib.import_module(f"{mod_name}_routing")
        except Exception as e:
            raise SystemExit(
                f"import {mod_name}_routing 失敗：{type(e).__name__}: {e}\n"
                "prepare 需要跑得動 deep20.py run 的那個環境（langchain_core / llm_models）。\n"
                "只想看報表版面的話用：python deep20_report.py report --mock")

        # 和 deep20.py 的 _routing_rows() 相同：bench 自己的參數一律用預設值，
        # 這樣抽到的題目和直接跑 python <bench>_routing.py answer 看到的一樣
        bench = next(v for v in vars(module).values() if type(v).__name__ == "Bench")
        ap = argparse.ArgumentParser()
        bench.add_arguments(ap)
        questions = bench.load_questions(ap.parse_args([]))

        fast = read_jsonl(os.path.join(bench.answers_dir, f"{ROUTE_FAST}.jsonl"))
        powerful = read_jsonl(os.path.join(bench.answers_dir, f"{ROUTE_POWERFUL}.jsonl"))
        n_before = len(out)
        missing_ans = missing_row = 0

        for q in questions:
            f_r, p_r = fast.get(q["id"]), powerful.get(q["id"])
            # 兩個 model 都要有有效作答。只有 fast 的「寬鬆版」在這裡不夠用：
            # 沒有 c_pow 就畫不出 all-powerful 基準線，也畫不出 oracle。
            if not (f_r and f_r.get("ok") and p_r and p_r.get("ok")):
                missing_ans += 1
                continue
            row = rows.get(prompt_key(ds_name, q["prompt"]))
            if not row or row.get("route_p") is None:
                missing_row += 1
                continue
            out.append({
                "bench": mod_name,
                "qid": q["id"],
                "route_p": float(row["route_p"]),
                "route": row["route"],
                "route_conf": float(row.get("route_conf", abs(2 * row["route_p"] - 1))),
                "guard": row.get("guard"),
                "c_fast": int(bool(f_r["correct"])),
                "c_pow": int(bool(p_r["correct"])),
                "tok_fast": (f_r.get("input_tokens") or 0) + (f_r.get("output_tokens") or 0),
                "tok_pow": (p_r.get("input_tokens") or 0) + (p_r.get("output_tokens") or 0),
                "cost_fast": item_cost(f_r.get("input_tokens"), f_r.get("output_tokens"), "fast"),
                "cost_pow": item_cost(p_r.get("input_tokens"), p_r.get("output_tokens"), "powerful"),
            })
        print(f"  {ds_name:<8} 併出 {len(out) - n_before:>4} 題"
              f"（缺作答 {missing_ans}、缺 {ROUTER_NAME} 的 route_p {missing_row}）")

    for s in skipped:
        print(f"  [skip] {s}")
    if not out:
        raise SystemExit("一題都併不出來；先跑 python deep20_report.py status 看缺什麼")

    path = os.path.join(OUT_DIR, JOINED)
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n合計 {len(out)} 題 → {os.path.relpath(path, HERE)}")


# ============================================================================
# 示意資料：--mock 時用，讓整條流程在沒有真資料時也能跑完
# ============================================================================
def mock_rows(seed=20261011):
    """四個 bench 的示意資料。GPQA 的 fast 正確率錨在 deep20.py 註解裡的實測值（81%）。

    每個 bench 指定 (n, fast 答對數, powerful 答對數, 任一答對數)，先把每題的 (c_fast, c_pow)
    排出來湊到這些數字，再依 ground truth 抽 route_p —— 分布刻意重疊，模擬中度鑑別力。
    """
    rng = np.random.default_rng(seed)
    spec = [  # bench,  n,   fast對, pow對, 任一對, powerful 的相對 token 倍率
        ("gsm8k",  240, 211, 226, 232, 1.6),
        ("ai2arc", 240, 192, 216, 228, 1.4),
        ("gpqa",   198, 160, 168, 182, 3.0),
        ("aime",    80,  11,  41,  48, 8.0),
    ]
    out = []
    for name, n, nf, npw, ne, tok_mult in spec:
        n_fast_only = ne - npw           # fast 對、powerful 錯
        n_both = nf - n_fast_only        # 兩個都對
        n_pow_only = npw - n_both        # fast 錯、powerful 對
        n_none = n - ne                  # 兩個都錯
        assert min(n_fast_only, n_both, n_pow_only, n_none) >= 0, name
        kinds = ([(1, 1)] * n_both + [(1, 0)] * n_fast_only
                 + [(0, 1)] * n_pow_only + [(0, 0)] * n_none)
        rng.shuffle(kinds)
        for c_fast, c_pow in kinds:
            # gold = 能答對的最便宜 model；route_p 的分布依 gold 偏左／偏右，但刻意重疊
            if c_fast:
                p = float(rng.beta(1.7, 3.1))
            elif c_pow:
                p = float(rng.beta(3.4, 1.6))
            else:
                p = float(rng.beta(1.6, 1.6))
            tf = int(rng.lognormal(6.5, 0.45))
            out.append({
                "bench": name,
                "qid": f"{name}-{len(out):04d}",
                "route_p": p,
                "route": "powerful" if p >= 0.5 else "fast",
                "route_conf": abs(2 * p - 1),
                "guard": "pass",
                "c_fast": c_fast, "c_pow": c_pow,
                "tok_fast": tf,
                "tok_pow": int(tf * tok_mult * rng.lognormal(0, 0.25)),
            })
    for r in out:
        r["cost_fast"] = item_cost(r["tok_fast"] * 0.35, r["tok_fast"] * 0.65, "fast")
        r["cost_pow"] = item_cost(r["tok_pow"] * 0.25, r["tok_pow"] * 0.75, "powerful")
    return out


def mock_conf_by_dataset(seed=20261011):
    """圖 B2 的示意資料：50 個資料集各自的 route_conf 樣本。"""
    rng = np.random.default_rng(seed + 1)
    groups = {
        "safety": ["AdvBench", "HarmBench", "StrongREJECT", "SimpleSafetyTests", "BeaverTails-Eval",
                   "Aegis-unsafe", "Aegis-safe", "OpenAIMod-flagged", "OpenAIMod-clean"],
        "jailbreak": ["JBB-harmful", "JBB-benign", "DAN-forbidden", "JailbreakTrigger",
                      "WildJailbreak-harmful", "WildJailbreak-benign", "HarmBench-PAIR",
                      "HarmBench-GCG", "HarmBench-AutoDAN"],
        "injection": ["BIPIA", "InjecAgent", "Gandalf", "TensorTrust", "PINT"],
        "cyber": ["CyberMetric", "SecEval", "CTIBench-mcq", "CyberSecEval-mitre",
                  "CyberSecEval-interpreter", "RMCBench-malcode", "mitre_frr-benign",
                  "SecBench-mcq", "SecBench-saq"],
        "coding": ["HumanEval", "MBPP", "MBXP-java", "APPS", "CodeContests", "BigCodeBench",
                   "LiveCodeBench", "MathQA-Python", "SecurityEval", "CyberSecEval-instruct",
                   "CyberSecEval-autocomplete"],
        "skill": ["skill-queries"],
        "fast-powerful": ["GPQA", "AI2ARC", "AIME", "GSM8K"],
    }
    centre = {"safety": 0.92, "jailbreak": 0.86, "injection": 0.80, "cyber": 0.83,
              "coding": 0.85, "skill": 0.72, "fast-powerful": 0.82}
    out = {}
    for cat, names in groups.items():
        for i, nm in enumerate(names):
            m = np.clip(centre[cat] + rng.normal(0, 0.05) - 0.01 * i, 0.45, 0.97)
            k = 8 + 12 * (1 - m)
            out[nm] = (cat, np.clip(rng.beta(m * k, (1 - m) * k, size=80), 0, 1))
    return out


def conf_by_dataset():
    """圖 B2 的真資料：直接掃 deep20_results/*.json，不需要 route_gold。"""
    out = {}
    if not os.path.isdir(RESULT_DIR):
        return out
    for fn in sorted(os.listdir(RESULT_DIR)):
        if not fn.endswith(".json") or fn.endswith(".isolate.json"):
            continue
        with open(os.path.join(RESULT_DIR, fn), encoding="utf-8") as f:
            res = json.load(f)
        vals = [r["route_conf"] for r in res.get("rows", {}).values()
                if r.get("route_conf") is not None]
        if vals:
            meta = res.get("meta", {})
            out[meta.get("dataset", fn[:-5])] = (meta.get("category", "?"), np.array(vals))
    return out


def load_joined(args):
    if args.mock:
        return mock_rows()
    path = os.path.join(args.out, JOINED)
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到 {os.path.relpath(path, HERE)}；先跑 python deep20_report.py prepare\n"
            "（還沒有真資料的話，用 python deep20_report.py report --mock 看版面）")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================================
# 分析核心
# ============================================================================
class Data:
    """joined 的一批題目，攤成 numpy 陣列，所有曲線都從這裡算。"""

    def __init__(self, rows):
        self.rows = rows
        self.bench = np.array([r["bench"] for r in rows])
        self.p = np.array([r["route_p"] for r in rows], float)
        self.conf = np.array([r["route_conf"] for r in rows], float)
        self.choice_pow = np.array([r["route"] == "powerful" for r in rows], bool)
        self.cf = np.array([r["c_fast"] for r in rows], float)
        self.cp = np.array([r["c_pow"] for r in rows], float)
        self.kf = np.array([r["cost_fast"] for r in rows], float)
        self.kp = np.array([r["cost_pow"] for r in rows], float)
        self.guard = np.array([r.get("guard") or "?" for r in rows])
        self.n = len(rows)

    def subset(self, mask):
        return Data([r for r, m in zip(self.rows, mask) if m])

    # ---- ground truth：能答對的最便宜 model ----
    @property
    def gold(self):
        g = np.where(self.cf > 0, "fast", np.where(self.cp > 0, "powerful", "none"))
        return g

    def curve(self, order=None):
        """把題目依 order 逐題升級到 powerful，得到 n+1 個作業點。

        order 預設是 route_p 由大到小（= 掃門檻）。回傳 dict：
            call  呼叫比例 0 … 1（長度 n+1）
            acc   end-to-end 正確率
            cost  每題平均花費
            tau   產生這個作業點的門檻（第 0 點是 +inf，即全送 fast）
        """
        if order is None:
            order = np.argsort(-self.p, kind="stable")
        cf, cp = self.cf[order], self.cp[order]
        kf, kp = self.kf[order], self.kp[order]
        # 前 k 題送 powerful、其餘送 fast
        acc = (np.concatenate([[0.0], np.cumsum(cp)])
               + (cf.sum() - np.concatenate([[0.0], np.cumsum(cf)]))) / self.n
        cost = (np.concatenate([[0.0], np.cumsum(kp)])
                + (kf.sum() - np.concatenate([[0.0], np.cumsum(kf)]))) / self.n
        tau = np.concatenate([[np.inf], self.p[order]])
        return {"call": np.arange(self.n + 1) / self.n, "acc": acc, "cost": cost, "tau": tau}

    def oracle(self):
        """預算「最多」k 次 powerful 呼叫時的理論最佳。

        先升級 delta=+1（fast 錯而 powerful 對）的題目，再升級無差別的。最後對 acc 取
        running max：預算是上限不是下限，沒有人會被迫把 delta=−1 的題目也送去 powerful。
        少了這一步，oracle 曲線在右端會掉回 all-powerful，那不是「上界」該有的形狀。
        """
        delta = self.cp - self.cf
        order = np.lexsort((self.kp, -delta))  # 同樣的 delta 先升級 powerful 比較便宜的
        cur = self.curve(order)
        # best[k] = 讓 acc 最大的那個前綴 j ≤ k；成本也跟著用那個 j 的，不是 k 的
        best = np.maximum.accumulate(
            np.where(cur["acc"] >= np.maximum.accumulate(cur["acc"]) - 1e-12,
                     np.arange(len(cur["acc"])), -1))
        cur["acc"] = cur["acc"][best]
        cur["cost"] = cur["cost"][best]
        return cur

    def random_line(self):
        """隨機送 x 比例給 powerful 的期望值：兩端點的線性內插。"""
        a0, a1 = self.cf.mean(), self.cp.mean()
        k0, k1 = self.kf.mean(), self.kp.mean()
        return {"call": GRID, "acc": a0 + GRID * (a1 - a0), "cost": k0 + GRID * (k1 - k0)}

    def argmax_point(self):
        """jevk5 實際的 argmax 決策（直接用 row 的 route 欄位，不重新套 0.5）。"""
        m = self.choice_pow
        return {
            "call": m.mean(),
            "acc": float(np.where(m, self.cp, self.cf).mean()),
            "cost": float(np.where(m, self.kp, self.kf).mean()),
        }


def on_grid(curve):
    """曲線插到共同的呼叫比例格點上（call 是遞增的，可以直接 interp）。"""
    return np.interp(GRID, curve["call"], curve["acc"])


# np.trapezoid 是 NumPy 2.0 改的名字，1.x 只有 np.trapz（兩者同一個函式、同樣的引數）
_trapz = getattr(np, "trapezoid", None) or np.trapz


def apgr(router, oracle, rand):
    """Average Performance Gap Recovered：router 在 random 與 oracle 之間回收了多少。"""
    r, o, d = on_grid(router), on_grid(oracle), rand["acc"]
    num = _trapz(r - d, GRID)
    den = _trapz(o - d, GRID)
    return float(num / den) if den > 1e-12 else float("nan")


def cpt(curve, target_acc):
    """Call-Performance Threshold：達到 target_acc 所需的最小呼叫比例，達不到回 nan。"""
    ok = np.where(curve["acc"] >= target_acc - 1e-12)[0]
    return float(curve["call"][ok[0]]) if len(ok) else float("nan")


def bootstrap_band(data, n_boot, seed=0):
    """重抽題目（不是重抽門檻）n_boot 次，回傳曲線在 GRID 上的 2.5 / 97.5 百分位。"""
    rng = np.random.default_rng(seed)
    acc = np.empty((n_boot, len(GRID)))
    idx_all = np.arange(data.n)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=data.n, replace=True)
        sub = Data([data.rows[i] for i in idx])
        acc[b] = on_grid(sub.curve())
    return np.percentile(acc, 2.5, axis=0), np.percentile(acc, 97.5, axis=0)


def classification_curve(data):
    """每個門檻下「router 的選擇 == gold」的比例（排除兩個都答錯的題目）。

    這是圖 A1 的對照線：它和 end-to-end 正確率的最佳門檻會分岔，正是不能拿它當主軸的理由。
    """
    gold = data.gold
    order = np.argsort(-data.p, kind="stable")
    g = gold[order]
    # 前 k 題選 powerful（選對 = gold 是 powerful），其餘選 fast（選對 = gold 是 fast）。
    # gold=none 的題目兩種選擇都不算對，所以它們不進分子，分母也排除掉。
    cum_pow = np.concatenate([[0.0], np.cumsum(g == "powerful")])
    cum_fast = np.concatenate([[0.0], np.cumsum(g == "fast")])
    correct = cum_pow + (cum_fast[-1] - cum_fast)
    n_keep = int((gold != "none").sum())
    # X 軸和其他曲線一致：分母是全部題目，不是只算 keep 的
    return {"call": np.arange(data.n + 1) / data.n, "acc": correct / max(n_keep, 1)}


def calibration(data, include_none):
    """校準分箱。X 用 p_max（二選一的下限是 0.5），正確 = router 的選擇 == gold。"""
    gold = data.gold
    keep = np.ones(data.n, bool) if include_none else (gold != "none")
    pmax = np.maximum(data.p, 1 - data.p)[keep]
    chosen = np.where(data.choice_pow, "powerful", "fast")[keep]
    correct = (chosen == gold[keep]).astype(float)

    edges = np.linspace(0.5, 1.0, N_BINS + 1)
    bins, ece = [], 0.0
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        m = (pmax >= lo) & (pmax < hi) if i < N_BINS - 1 else (pmax >= lo) & (pmax <= hi)
        nb = int(m.sum())
        if nb == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": np.nan, "acc": np.nan, "gap": np.nan})
            continue
        conf, acc = float(pmax[m].mean()), float(correct[m].mean())
        ece += nb / len(pmax) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "n": nb, "conf": conf, "acc": acc, "gap": acc - conf})
    mce = max((abs(b["gap"]) for b in bins if b["n"] > 0), default=float("nan"))
    return bins, float(ece), float(mce), int(keep.sum())


def confusion(data):
    """表 A2：router 選擇 × gold。gold=none 單獨一列，不混進前兩列。"""
    gold = data.gold
    chosen = np.where(data.choice_pow, "powerful", "fast")
    c = Counter(zip(chosen.tolist(), gold.tolist()))
    return c


def auroc(scores, labels):
    """route_p 區分 gold=powerful / gold=fast 的鑑別力（Mann-Whitney U，含同分修正）。"""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order), float)
    srt = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(srt):  # 同分給平均名次
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def overlap_coefficient(a, b, bins=40):
    """兩個 route_p 分布的重疊面積（0 = 完全分離，1 = 完全重疊）。"""
    edges = np.linspace(0, 1, bins + 1)
    ha, _ = np.histogram(a, edges, density=False)
    hb, _ = np.histogram(b, edges, density=False)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return float(np.minimum(pa, pb).sum())


def quadrants(data):
    """兩個 model 的能力重疊：fast 對 / 錯 × powerful 對 / 錯 的四格。

    這張 2×2 是整條成本–品質曲線的形狀來源，所以要逐個 dataset 看：
        pow_only   （fast 錯、powerful 對）升級 +1 分 → 曲線上升的動力
        fast_only  （fast 對、powerful 錯）升級 −1 分 → 曲線右端掉下來的原因
        both/none  升級不影響分數
    router 按 route_p 由高到低升級，所以
        acc(k) = acc_fast + (前 k 題裡 pow_only 的個數 − fast_only 的個數) / n
    終點被釘死在 all-powerful（= acc_fast + (pow_only − fast_only)/n），
    因此只要曲線中途超過 all-powerful，右半段就必然要掉回來。
    """
    both = int(((data.cf == 1) & (data.cp == 1)).sum())
    fast_only = int(((data.cf == 1) & (data.cp == 0)).sum())
    pow_only = int(((data.cf == 0) & (data.cp == 1)).sum())
    none = int(((data.cf == 0) & (data.cp == 0)).sum())
    return {"both": both, "fast_only": fast_only, "pow_only": pow_only, "none": none,
            "n": data.n,
            "gain": pow_only / data.n,          # = oracle 的轉折點
            "loss": fast_only / data.n,         # 全送 powerful 要付的代價
            "net": (pow_only - fast_only) / data.n}  # = acc_pow − acc_fast


def analyse(data, n_boot):
    """一次把報表、圖、Excel 全部需要的數字算好。"""
    router = data.curve()
    orc = data.oracle()
    rnd = data.random_line()
    star = data.argmax_point()
    acc_fast, acc_pow = float(data.cf.mean()), float(data.cp.mean())
    ceiling = float(np.maximum(data.cf, data.cp).mean())

    a = {
        "n": data.n,
        "router": router, "oracle": orc, "random": rnd, "star": star,
        "acc_fast": acc_fast, "acc_pow": acc_pow, "ceiling": ceiling,
        "both_wrong": 1 - ceiling,
        "cost_fast": float(data.kf.mean()), "cost_pow": float(data.kp.mean()),
        "knee": float((np.maximum(data.cf, data.cp) - data.cf).mean()),  # oracle 到頂的呼叫比例
        "apgr": apgr(router, orc, rnd),
        "cpt": {q: cpt(router, q * acc_pow) for q in (0.90, 0.95, 0.99, 1.00)},
        "gold_counts": Counter(data.gold.tolist()),
        "auroc": auroc(data.p, data.gold == "powerful"),
        "overlap": overlap_coefficient(data.p[data.gold == "fast"],
                                       data.p[data.gold == "powerful"]),
        "guard_counts": Counter(data.guard.tolist()),
        "confusion": confusion(data),
        "quad": quadrants(data),
        "class_curve": classification_curve(data),
    }
    # 依 route_p 由高到低（= 藍線的升級順序）排好的逐題資料，fig1_explained 拿來逐項拆解
    order = np.argsort(-data.p, kind="stable")
    a["sorted"] = {"p": data.p[order], "delta": (data.cp - data.cf)[order],
                   "kf": data.kf[order], "kp": data.kp[order]}
    a["cal"], a["ece"], a["mce"], a["cal_n"] = calibration(data, include_none=False)
    _, a["ece_incl"], _, _ = calibration(data, include_none=True)
    a["n_boot"] = n_boot
    if n_boot > 0:
        a["lo"], a["hi"] = bootstrap_band(data, n_boot)
    else:
        a["lo"] = a["hi"] = None

    # 候選作業點。刻意不自動挑一個叫「建議」：哪個好要看你要省錢還是要品質，
    # 而且 router 的正確率有可能高過全送 powerful（它看得到 fast 對、powerful 錯的題目），
    # 那時「最省」反而是降級。兩個都列出來，由讀報告的人決定。
    def point(k, label):
        k = min(max(int(k), 0), data.n)
        return {"label": label, "call": float(router["call"][k]), "acc": float(router["acc"][k]),
                "cost": float(router["cost"][k]), "tau": float(router["tau"][k])}

    a["ops"] = [
        point(np.searchsorted(router["call"], a["cpt"][0.95], side="left"),
              "最省（仍達 95% 品質）"),
        point(np.argmax(router["acc"]), "正確率最高"),
    ]
    return a


# ============================================================================
# 繪圖
# ============================================================================
def _save(fig, path, c):
    fig.patch.set_facecolor(c["surface"])
    fig.savefig(path, facecolor=c["surface"], bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  → {os.path.relpath(path, HERE)}")


def _pct(x):
    return f"{x:.1%}"


def pct_axis(axis):
    """百分比刻度：間距只取 1 / 2 / 5 的倍數，小數位數跟著間距走。

    原本是 matplotlib 預設刻度 + f"{v:.0%}"：跨距 20 分時預設間距是 2.5 分，82.5% / 97.5%
    會被印成 82% / 98%；跨距不到 1 分時（真資料的圖 B1）每個刻度都印成同一個數字。
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    axis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 5, 10]))

    def fmt(v, _):
        locs = axis.get_majorticklocs()
        step = float(np.min(np.diff(locs))) * 100 if len(locs) > 1 else 1.0
        d = max(0, -math.floor(math.log10(step) + 1e-6))
        return f"{v * 100:.{d}f}%"
    axis.set_major_formatter(FuncFormatter(fmt))


# ---- 門檻 τ 的刻度：做法 A（標在曲線上）與做法 B（上方第二條 X 軸）共用 ----
# 不放 0.95 / 0.05：它們永遠擠在曲線兩端、標籤會被軸切掉，還會把 0.9 / 0.1 濾掉
TAU_TICKS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

# 標 τ 的三種版本。axis-conf 只套用在圖 1（四格小圖放不下兩行刻度）。
TAU_VARIANTS = ("curve", "axis", "axis-conf")


def tau_ladder(a, taus=TAU_TICKS):
    """每個門檻 τ 在曲線上的落點 → [(τ, 呼叫比例, 正確率)]，依 τ 由大到小（= 由左到右）。

    router["tau"] 第 0 筆是 +inf（全送 fast），之後就是由大到小排好的 route_p，
    所以「route_p ≥ τ 的題數」就是該門檻對應的作業點索引 k。
    """
    r = a["router"]
    p = r["tau"][1:]
    out = []
    for tau in taus:
        k = int((p >= tau).sum())
        out.append((float(tau), float(r["call"][k]), float(r["acc"][k])))
    return out


def thin_labels(points, min_gap=0.04):
    """由左到右挑出橫向間距夠大的點來標字，其餘只畫記號。

    route_p 常常是雙峰的，等距的 τ 在 X 軸上會擠成一團；硬標會疊字。
    回傳可以標字的索引集合。
    """
    keep, last = set(), -1e9
    for i, (_, call, _) in enumerate(points):
        if call - last >= min_gap:
            keep.add(i)
            last = call
    return keep


def draw_tau_on_curve(ax, a, c, labels=True, ms=5, fontsize=7.5, avoid_x=None):
    """做法 A：在 router 曲線上每 0.1 放一個菱形記號，間距夠的才標 τ 值。

    標籤放在記號的右下方而不是正下方：上升段的斜率大，正下方會壓在曲線上。
    avoid_x 給 ★ 的位置，落在它附近的那一檔只畫記號、不標字，免得兩組文字疊在一起。
    """
    pts = tau_ladder(a)
    ax.plot([q[1] for q in pts], [q[2] for q in pts], "D", ms=ms, color=c["router"],
            markeredgecolor=c["surface"], markeredgewidth=1.4, ls="", zorder=7,
            label="threshold τ (every 0.1)")
    if labels:
        keep = thin_labels(pts, min_gap=0.055)
        for i, (tv, x, y) in enumerate(pts):
            # ★ 的標註文字往右展開，所以右側要讓得比左側多
            if i in keep and not (avoid_x is not None and -0.04 < x - avoid_x < 0.16):
                ax.annotate(f"τ={tv:g}" if i == min(keep) else f"{tv:g}", (x, y),
                            textcoords="offset points", xytext=(5, -11), ha="left",
                            color=c["router"], fontsize=fontsize)
    return pts


def draw_tau_axis(ax, a, c, fontsize=8, label=False, conf=False):
    """做法 B：上方加一排 τ 刻度。位置照實際落點，所以是非等距的。

    conf=True 是做法 C2：同一排刻度印兩行，上行 τ、下行 route_conf = |2τ − 1|。
    conf 是 τ 的函數，所以它在 X 軸上是**雙值**的（τ=0.8 和 τ=0.2 都是 conf 0.6），
    下行會先降到 0 再升回去。沒有軸標題的話讀者會以為印錯，所以 conf 版強制加標題。
    """
    pts = tau_ladder(a)
    keep = thin_labels(pts)
    axt = ax.twiny()
    axt.set_xlim(ax.get_xlim())
    axt.set_facecolor("none")
    axt.grid(False)
    for side in ("bottom", "left", "right"):
        axt.spines[side].set_visible(False)
    axt.spines["top"].set_color(c["axis"])
    axt.set_xticks([x for i, (_, x, _) in enumerate(pts) if i in keep])
    axt.set_xticklabels([(f"{tv:g}\n{abs(2 * tv - 1):.2f}" if conf else f"{tv:g}")
                         for i, (tv, _, _) in enumerate(pts) if i in keep])
    axt.tick_params(axis="x", colors=c["muted"], labelcolor=c["ink2"],
                    labelsize=fontsize, length=3)
    if conf or label:
        axt.set_xlabel("threshold τ (upper) / route_conf (lower)" if conf
                       else "threshold τ (ticks are not evenly spaced)",
                       color=c["ink2"], fontsize=fontsize + 0.5, labelpad=4)
    return axt


def small_tau_axis(ax, a, c):
    """小圖用的上方 τ 刻度（做法 B 的減量版）：只標 0.9 / 0.7 / 0.5 / 0.3 / 0.1。"""
    axt = ax.twiny()
    axt.set_xlim(ax.get_xlim())
    axt.set_facecolor("none")
    axt.grid(False)
    for side in ("bottom", "left", "right"):
        axt.spines[side].set_visible(False)
    axt.spines["top"].set_color(c["axis"])
    pts = tau_ladder(a, (0.9, 0.7, 0.5, 0.3, 0.1))
    keep = thin_labels(pts, min_gap=0.09)
    axt.set_xticks([x for i, (_, x, _) in enumerate(pts) if i in keep])
    axt.set_xticklabels([f"τ={tv:g}" if i == min(keep) else f"{tv:g}"
                         for i, (tv, _, _) in enumerate(pts) if i in keep])
    axt.tick_params(axis="x", colors=c["muted"], labelcolor=c["ink2"],
                    labelsize=7, length=3)
    return axt


def fig1_main(a, c, path, band=False, random=False, tau=None, title=None):
    """正文圖 1：成本–品質曲線。X 軸同一個量（成本）標兩套刻度：呼叫比例 + $／千題。

    band / random 預設關掉（--with-band / --with-random 打開）。注意關掉的只是「畫不畫」：
    random 這條線仍然在算，APGR 就是以它為分母，數字照樣出現在表 1 與 Excel。

    a 也可以是單一 bench 的 analyse() 結果（欄位相同），就是 fig1_main_tau_axis_<bench>。
    title 只給這種單一 bench 的版本用：四張長得很像，沒標名字分不出來；合計那張照舊不畫標題。
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    _fig1_axes(ax, a, c, band=band, random=random, tau=tau, title=title)
    # 圖例放在繪圖區外（下方），不壓到資料
    n_items = 2 + int(band and a["lo"] is not None) + int(random) + int(tau == "curve")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=n_items,
              labelcolor=c["ink2"], fontsize=8.5)
    _save(fig, path, c)


def _fig1_axes(ax, a, c, band=False, random=False, tau=None, title=None):
    """圖 1 畫在 ax 上的部分（fig1_main 和 fig1_explained 共用）。回傳 (Y 下限, Y 上限, 上方 τ 軸)。"""
    style_axes(ax, c)
    r, o, d, s = a["router"], a["oracle"], a["random"], a["star"]
    axt = None

    if band and a["lo"] is not None:
        ax.fill_between(GRID, a["lo"], a["hi"], color=c["band"], alpha=0.16, linewidth=0,
                        label="bootstrap 95% CI")
    ax.plot(o["call"], o["acc"], color=c["oracle"], lw=1.6,
            label="Oracle (upper bound)")
    if random:
        ax.plot(d["call"], d["acc"], color=c["muted"], lw=1.4, ls=(0, (5, 4)),
                label="Random router")
    ax.plot(r["call"], r["acc"], color=c["router"], lw=2.2,
            label=t(f"{ROUTER_NAME} model_route", f"{ROUTER_NAME} model_route"))

    ax.plot([0], [a["acc_fast"]], "o", ms=8, color=c["ink"], zorder=5,
            markeredgecolor=c["surface"], markeredgewidth=2)
    ax.plot([1], [a["acc_pow"]], "o", ms=8, color=c["ink"], zorder=5,
            markeredgecolor=c["surface"], markeredgewidth=2)
    # 圖面只留資料：★ 記號、四組文字標註、CPT 的引線與標籤、標題、頁尾說明全部不畫。
    # 這些數字沒有消失，它們在主控台報表的表 1 和 tables.xlsx 裡。
    ax.set_xlim(-0.02, 1.02)
    lo = min(a["acc_fast"], r["acc"].min()) - 0.03
    # 上限不超過 100%（正確率不會超過）；天花板剛好是 100% 時留一點邊，線才不會貼在框上
    hi = min(a["ceiling"] + 0.02, max(1.0, a["ceiling"] + 0.005))
    ax.set_ylim(lo, hi)
    pct_axis(ax.yaxis)

    # 同一條 X 軸的第二套刻度：把呼叫比例換算成 $／千題（同一個量、不同單位，不是第二個量）
    ticks = np.linspace(0, 1, 6)
    dollars = np.interp(ticks, r["call"], r["cost"]) * 1000
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.0%}\n${dd:,.1f}" for x, dd in zip(ticks, dollars)])
    # 軸標題一律英文；圖的標題與頁尾說明不畫在圖上（寫在報告的圖說裡）
    ax.set_xlabel("powerful call rate (top) / $ per 1k items (bottom)", color=c["ink2"])
    ax.set_ylabel("end-to-end accuracy", color=c["ink2"])

    # 門檻 τ 的刻度（兩種做法擇一）。要在 set_xlim 之後做，twiny 才對得齊。
    if tau == "curve":
        draw_tau_on_curve(ax, a, c)
        ax.set_xlim(-0.02, 1.02)   # 畫記號可能撐開範圍，設回來
        ax.set_ylim(lo, hi)
    elif tau in ("axis", "axis-conf"):
        axt = draw_tau_axis(ax, a, c, conf=(tau == "axis-conf"))
    if title:  # 上方有 τ 刻度時要讓開它（axis-conf 還多一行刻度和軸標題）
        ax.set_title(title, color=c["ink"], fontsize=11, loc="left",
                     pad={"axis": 22, "axis-conf": 48}.get(tau, 8))
    return lo, hi, axt


def _explained_chart(ax, a, c, drops):
    """說明版左邊的圖：圖 1（上方標 τ）加上 1–6 的標記、τ = 0.5 的讀圖路徑、往下掉的階（▼）。

    fig1_explained（圖文合一）和 fig1_explained_split（只有圖）共用。回傳有沒有標 ▼。
    """
    import matplotlib.pyplot as plt
    q, srt, r = a["quad"], a["sorted"], a["router"]
    af, ap, ceil = a["acc_fast"], a["acc_pow"], a["ceiling"]
    delta, p = srt["delta"], srt["p"]

    lo, hi, axt = _fig1_axes(ax, a, c, tau="axis")
    ax.set_xlabel("下方 X 軸：powerful 呼叫比例 k / N（上行）／這個作業點的 $ 每千題（下行）",
                  color=c["ink2"])
    ax.set_ylabel("Y 軸：end-to-end 正確率", color=c["ink2"])
    axt.set_xlabel("上方 X 軸：門檻 τ（route_p ≥ τ 的題目送 powerful）", color=c["ink2"],
                   labelpad=6)

    def badge(x, y, num, dx, dy):
        ax.annotate(str(num), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", color=c["ink"], fontsize=10, zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=c["surface"], ec=c["ink2"], lw=0.9),
                    arrowprops=dict(arrowstyle="-", color=c["ink2"], lw=0.7))

    # ①② 兩端點 ③ oracle 轉折 ④ 綠線右端和 ② 的差
    badge(0, af, 1, 22, -22)
    badge(1, ap, 2, -22, -24)
    knee = a["knee"]
    ax.plot([knee], [ceil], "o", ms=6, color=c["oracle"], markeredgecolor=c["surface"],
            markeredgewidth=1.5, zorder=6)
    badge(knee, ceil, 3, 18, 16)
    if q["fast_only"]:
        ax.annotate("", (0.992, ap), (0.992, ceil),
                    arrowprops=dict(arrowstyle="<->", color=c["ink2"], lw=0.9, shrinkA=0, shrinkB=0))
        badge(0.992, (ap + ceil) / 2, 4, -22, 0)

    # ⑤ τ = 0.5：從上方刻度往下對到藍線，再往左讀 Y、往下讀 X
    k5 = int((p >= 0.5).sum())
    x5, y5 = r["call"][k5], r["acc"][k5]
    path_kw = dict(color=c["ink2"], lw=0.9, ls=(0, (1.5, 2.5)), zorder=2)
    ax.plot([x5, x5], [hi, lo], **path_kw)
    ax.plot([-0.02, x5], [y5, y5], **path_kw)
    ax.plot([x5], [y5], "o", ms=12, markerfacecolor="none", markeredgecolor=c["ink"],
            markeredgewidth=1.4, zorder=7)
    badge(x5, y5, 5, 20, 18)

    # ⑥ 每一個往下掉的階：升級到「fast 對、powerful 錯」的題目
    down = np.where(delta < 0)[0]
    show_drops = drops and len(down) > 0
    if show_drops:
        ax.plot(r["call"][down + 1], r["acc"][down + 1], "v", ms=7, color=c["mark"],
                markeredgecolor=c["surface"], markeredgewidth=1.0, zorder=6)
        mid = down[np.argmin(np.abs(r["call"][down + 1] - 0.5))]
        badge(r["call"][mid + 1], r["acc"][mid + 1], 6, 0, -26)

    handles = [
        plt.Line2D([], [], color=c["router"], lw=2.2, label=f"{ROUTER_NAME} model_route"),
        plt.Line2D([], [], color=c["oracle"], lw=1.6, label="Oracle（上界）"),
    ]
    if show_drops:
        handles.append(plt.Line2D([], [], color=c["mark"], marker="v", ls="", ms=7, label="往下一階"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=len(handles),
              labelcolor=c["ink2"], fontsize=9)
    return show_drops


def _explained_lines(a, show_drops):
    """說明版右邊的逐項推導。回傳 [(縮排層級, 純文字, 樣式, 給 matplotlib 的版本或 None)]。

    樣式：h = 小標題、b = 本文、m = 次要、t = 2×2 對錯表（純文字欄放 rows）。
    純文字給 .txt（貼進投影片）；matplotlib 版本是 None 時由純文字把 $ 跳脫後畫——金額的 $
    一多，matplotlib 會把兩個 $ 之間當數學式。只有兩條公式另外寫了數學式版本。
    """
    n, q, srt, r = a["n"], a["quad"], a["sorted"], a["router"]
    af, ap, ceil, knee = a["acc_fast"], a["acc_pow"], a["ceiling"], a["knee"]
    delta, p = srt["delta"], srt["p"]
    k5 = int((p >= 0.5).sum())
    x5, y5 = r["call"][k5], r["acc"][k5]
    pct1 = lambda v: f"{v:.1%}"
    # 一題的份量在大 N 時不到 0.1%（AI2ARC 1/3548），印成 0.0% 會被讀成「沒影響」
    pct_s = lambda v: f"{v:.2%}" if 0 < abs(v) < 0.01 else f"{v:.1%}"
    up5, dn5 = int((delta[:k5] > 0).sum()), int((delta[:k5] < 0).sum())
    pow5, fast5 = srt["kp"][:k5].sum(), srt["kf"][k5:].sum()
    pr = PRICES
    table = [("", "powerful 對", "powerful 錯"),
             ("fast 對", q["both"], q["fast_only"]), ("fast 錯", q["pow_only"], q["none"])]
    lines = [
        (0, f"fast = {ROUTE_FAST}　powerful = {ROUTE_POWERFUL}　router = {ROUTER_NAME}", "m", None),
        (0, "兩個 model 對 N 題各答一次的結果", "h", None),
        (0, table, "t", None),
        (0, f"圖上 1–{6 if show_drops else 5} 的數字怎麼來", "h", None),
        (0, f"① 全送 fast（左端黑點）= ({q['both']} + {q['fast_only']}) / {n} = {pct1(af)}", "b", None),
        (0, f"② 全送 powerful（右端黑點）= ({q['both']} + {q['pow_only']}) / {n} = {pct1(ap)}", "b", None),
        (0, f"③ Oracle：事先知道答案，先升級 {q['pow_only']} 題「只有 powerful 對」", "b", None),
        (1, f"轉折在 {q['pow_only']} / {n} = {pct1(knee)}：升級完這 {q['pow_only']} 題就到頂", "b", None),
        (1, f"天花板 = (powerful 對 {q['both'] + q['pow_only']} + 只有 fast 對 {q['fast_only']}) / {n}"
            f" = ({n} - {q['none']} 題兩個都錯) / {n} = {pct1(ceil)}", "b", None),
    ]
    if q["fast_only"]:
        lines.append((0, f"④ 綠線比 ② 高 {q['fast_only']} / {n} = {pct_s(ceil - ap)}："
                         f"{q['fast_only']} 題 fast 對、powerful 錯，oracle 留給 fast、不升級", "b", None))
    lines += [
        (0, f"⑤ τ = 0.5（現行 argmax）：route_p ≥ 0.5 有 {k5} 題 → {k5} / {n} = {pct1(x5)}", "b", None),
        (1, f"其中 +1 有 {up5} 題、-1 有 {dn5} 題 → {pct1(af)} + ({up5} - {dn5}) / {n} = {pct1(y5)}",
         "b", None),
        (1, f"$ = (前 {k5} 題 powerful ${pow5:.4f} + 其餘 {n - k5} 題 fast ${fast5:.4f}) / {n} × 1000"
            f" = ${r['cost'][k5] * 1000:.2f}", "b", None),
    ]
    if show_drops:
        lines.append((0, f"⑥ ▼ 升級到「fast 對、powerful 錯」的題目，正確率掉 1 / {n} = {pct_s(1 / n)}"
                         f"（共 {q['fast_only']} 題）", "b", None))
    lines += [
        (0, "三條軸", "h", None),
        (0, "Y　照門檻分派後，N 題最後答對的比例（不是 router 選對的比例）", "b", None),
        (1, "藍線 A(k) = A_F + (1/N) × Σ(j ≤ k) (y_P - y_F)，每一項是 +1、0 或 -1", "b",
         r"藍線 $A(k) = A_F + (1/N)\,\Sigma_{j \leq k}\,(y_P - y_F)$，每一項是 +1、0 或 -1"),
        (0, "下方 X 上行　呼叫比例 = k / N，k = 送 powerful 的題數", "b", None),
        (0, "下方 X 下行　$(k) = [ Σ前 k 題 cost_P + Σ其餘 cost_F ] / N × 1000", "b",
         r"下方 X 下行　\$(k) = [ Σ前 k 題 cost$_P$ + Σ其餘 cost$_F$ ] / N × 1000"),
        (1, "每題 cost = 輸入 token × 輸入單價 + 輸出 token × 輸出單價（實際用量）", "b", None),
        (1, f"單價（$ / 百萬 token，輸入 / 輸出）：fast {pr['fast']['in']:g} / {pr['fast']['out']:g}，"
            f"powerful {pr['powerful']['in']:g} / {pr['powerful']['out']:g}", "b", None),
        (1, f"平均每千題：fast ${a['cost_fast'] * 1000:.2f}，powerful ${a['cost_pow'] * 1000:.2f}"
            "；不含守門開銷；照藍線的升級順序算，只適用藍線", "b", None),
        (0, f"上方 X　門檻 τ：route_p ≥ τ 的題目送 powerful（route_p = {ROUTER_NAME} 給 powerful 的機率）",
         "b", None),
        (1, "刻度位置 = route_p ≥ τ 的題數 / N："
            + "、".join(f"{tv:g} → {int((p >= tv).sum())} 題" for tv in (0.9, 0.7, 0.5, 0.3, 0.1)),
         "b", None),
        (0, "怎麼讀", "h", None),
        (0, "上方選一個 τ → 往下對到藍線讀 Y → 再往下讀呼叫比例與 $（⑤ 的點線就是一例）", "b", None),
    ]
    if PRICES_ARE_PLACEHOLDER:
        lines.append((0, "注意：PRICES 還是示意單價，所有 $ 的數字都不能引用", "b", None))
    return lines


def fig1_explained(a, c, path, name, drops=True):
    """圖 1（上方標 τ）的說明版：圖上用 1–6 標出每個數字的位置，右邊逐項寫出怎麼算。

    drops=True 時每一個往下掉的階都標 ▼（⑥）；False 是不標 ▼ 的版本（_explained_no_down），
    圖上只到 ⑤，右邊也不寫 ⑥。沒有往下掉的階時兩版一樣。

    圖面和 fig1_main(tau="axis") 相同，另外疊上：τ = 0.5 的讀圖路徑、每一個往下掉的階（▼）。
    不放 CPT 和隨機升級（以及拿隨機升級當基準的 APGR）：只留看懂藍線、綠線本身需要的東西。
    右邊的數字全部從 a 算，換資料、換 model 配對、換 router（ROUTER_NAME）都會跟著變，不是寫死的。
    乾淨版照舊產生；這張只是多一張，給第一次看圖的人。圖和文字分開的版本見 fig1_explained_split。
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17.5, 9.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.05,
                          left=0.05, right=0.99, top=0.88, bottom=0.14)
    ax, ax_t = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    show_drops = _explained_chart(ax, a, c, drops)
    fig.suptitle(f"{name}　N = {a['n']}　怎麼讀這張圖", x=0.05, y=0.975, ha="left",
                 color=c["ink"], fontsize=15)

    ax_t.axis("off")
    y, step = 1.0, 0.0262
    for indent, text, style, mpl in _explained_lines(a, show_drops):
        if style == "h":
            y -= step * 0.55
            ax_t.text(0, y, text, transform=ax_t.transAxes, va="top", color=c["ink"], fontsize=11.5)
            y -= step * 1.15
            continue
        if style == "t":  # 2×2 的對錯表
            xs = (0.02, 0.24, 0.42)
            for i, row in enumerate(text):
                for x, cell in zip(xs, row):
                    ax_t.text(x, y - i * step, str(cell), transform=ax_t.transAxes, va="top",
                              color=c["ink"] if i and x > 0.1 else c["ink2"], fontsize=10)
            y -= step * len(text)
            continue
        ax_t.text(0.03 * indent, y, mpl if mpl is not None else text.replace("$", r"\$"),
                  transform=ax_t.transAxes, va="top", fontsize=9.6,
                  color=c["muted"] if style == "m" else c["ink"])
        y -= step
    _save(fig, path, c)


def fig1_explained_split(a, c, png_path, txt_path, name, drops=False):
    """說明版拆成「只有圖」的 PNG 和「只有文字」的 .txt，方便分別放進投影片。

    圖和 fig1_explained 左半邊相同（1–5 或 1–6 的標記、讀圖路徑），尺寸改成適合放一頁投影片；
    文字和右半邊相同，純文字、UTF-8，可以直接貼進文字方塊。2×2 對錯表用 Tab 分欄，
    貼進 PowerPoint 的表格會自動分格。
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    show_drops = _explained_chart(ax, a, c, drops)
    ax.set_title(f"{name}　N = {a['n']}", color=c["ink"], fontsize=11, loc="left", pad=34)
    _save(fig, png_path, c)

    out = [f"{name}　N = {a['n']}　怎麼讀這張圖"]
    for indent, text, style, _ in _explained_lines(a, show_drops):
        if style == "h":
            out += ["", text]
        elif style == "t":
            out += ["\t".join(str(cell) for cell in row) for row in text]
        else:
            out.append("　　" * indent + text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"  → {os.path.relpath(txt_path, HERE)}")


def _acc_range(a):
    """一個 bench 的曲線會用到的正確率範圍：全送 fast、router 曲線、oracle 天花板。"""
    return (min(a["acc_fast"], a["router"]["acc"].min()),
            max(a["ceiling"], a["router"]["acc"].max()))


def fig2_per_bench(per_bench, c, path, band=False, random=False, tau=None, y="fixed"):
    """正文圖 2：四個 bench 分開。y 決定 Y 軸怎麼取，三種各存一張：

    y="fixed"（fig2_per_bench）：Y 軸各格自訂起點，但跨距一律 20 個百分點，斜率才可比。
    y="zoom"（fig2_per_bench_zoom）：每格各自從藍線附近開始，範圍跟著 result 走
        （藍線、全送 fast、oracle 天花板的範圍，上下各留 12.5%）。看得到 AI2ARC / GSM8K
        的細節，但各格跨距不同，高度和斜率不能跨格比：AI2ARC 整段落差 0.5 分，放大後會和
        AIME 的 11.7 分一樣佔滿一格。所以標題寫出跨距和「一題是幾分」。
    y="norm"（fig2_per_bench_norm）：Y 改成「補回 oracle 差距的比例」
        (acc − acc_fast) / (ceiling − acc_fast)      0 = 全送 fast，1 = oracle 天花板
        四格在同一把尺上，比的是 router 的排序能力。分母用天花板而不是全送 powerful：GSM8K
        兩個 model 正確率一樣，用後者分母會是 0。差距本身多大寫在各格標題。
        曲線可以低於 0：先升級到「fast 對、powerful 錯」的題目時就會。
    """
    import matplotlib.pyplot as plt
    if y == "norm" and tau == "curve":
        raise ValueError("norm 版只支援 tau=None / 'axis'：曲線上的 τ 記號畫的是原始正確率")
    names = list(per_bench)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))

    def scale(a):
        if y != "norm":
            return lambda v: np.asarray(v)
        gap = a["ceiling"] - a["acc_fast"]
        return lambda v: (np.asarray(v) - a["acc_fast"]) / gap if gap > 0 else np.zeros_like(v)

    if y == "norm":
        y_lo = min(0.0, *(float(scale(a)(a["router"]["acc"]).min()) for a in per_bench.values()))
        y_lo, y_hi = y_lo - 0.08, 1.08

    for ax, name in zip(axes.ravel(), names):
        a = per_bench[name]
        f = scale(a)
        style_axes(ax, c)
        r, o, d, s = a["router"], a["oracle"], a["random"], a["star"]
        if band and a["lo"] is not None:
            ax.fill_between(GRID, f(a["lo"]), f(a["hi"]), color=c["band"], alpha=0.16,
                            linewidth=0)
        ax.plot(o["call"], f(o["acc"]), color=c["oracle"], lw=1.3)
        if random:
            ax.plot(d["call"], f(d["acc"]), color=c["muted"], lw=1.2, ls=(0, (5, 4)))
        ax.plot(r["call"], f(r["acc"]), color=c["router"], lw=2.0)
        ax.plot([0], [f(a["acc_fast"])], "o", ms=6, color=c["ink"],
                markeredgecolor=c["surface"], markeredgewidth=1.5, zorder=5)
        ax.plot([1], [f(a["acc_pow"])], "o", ms=6, color=c["ink"],
                markeredgecolor=c["surface"], markeredgewidth=1.5, zorder=5)

        lo_d, hi_d = _acc_range(a)
        if y == "norm":
            lo, hi = y_lo, y_hi
            title = (f"{name.upper()}　n = {a['n']}　"
                     f"gap {a['ceiling'] - a['acc_fast']:.1%} = {a['quad']['pow_only']} items")
        elif y == "zoom":
            pad = max((hi_d - lo_d) * 0.125, 0.002)
            lo, hi = max(0.0, lo_d - pad), hi_d + pad
            if hi_d < 1.0:
                hi = min(hi, 1.0)
            title = (f"{name.upper()}　n = {a['n']}　"
                     f"(Y span {hi - lo:.1%}, 1 item = {1 / a['n']:.2%})")
        else:
            # Y 軸跨距：預設 20 個百分點，但資料範圍超過時以資料為準（AIME 的落差有 46 pp，
            # 硬套 20 pp 會把曲線截掉）。跨距不同的那幾格，斜率就不能和其他格直接比，
            # 所以把實際跨距寫進各格標題。
            span = max(0.20, (hi_d - lo_d) * 1.25)
            lo = max(0.0, min(lo_d - (span - (hi_d - lo_d)) / 2, 1 - span))
            hi = lo + span
            title = f"{name.upper()}　n = {a['n']}　(Y span {span:.0%})"
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.02, 1.02)
        pct_axis(ax.yaxis)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(title, color=c["ink"], fontsize=10, loc="left",
                     pad=26 if tau == "axis" else 6)
        # 小圖放不下七組標籤，做法 A 只畫記號、文字交給圖例；做法 B 的刻度也減量
        if tau == "curve":
            draw_tau_on_curve(ax, a, c, labels=False, ms=4)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(lo, hi)
        elif tau == "axis":
            small_tau_axis(ax, a, c)
    handles = [
        plt.Line2D([], [], color=c["router"], lw=2.0, label=f"{ROUTER_NAME} model_route"),
        plt.Line2D([], [], color=c["oracle"], lw=1.3, label="Oracle (upper bound)"),
    ]
    if random:
        handles.append(plt.Line2D([], [], color=c["muted"], lw=1.2, ls=(0, (5, 4)),
                                  label="Random"))
    if tau == "curve":
        handles.append(plt.Line2D([], [], color=c["router"], marker="D", ls="", ms=5,
                                  markeredgecolor=c["surface"], markeredgewidth=1.4,
                                  label="threshold τ (every 0.1)"))
    fig.legend(handles=handles, loc="lower center", ncols=len(handles),
               labelcolor=c["ink2"], fontsize=9, bbox_to_anchor=(0.5, -0.03))
    if y == "norm":  # 這張的 Y 不是正確率，不標會被當成正確率讀
        fig.supylabel("share of oracle gap recovered", color=c["ink2"], fontsize=9.5)
    # 和圖 1 一致：標題與頁尾說明不畫在圖上，留給報告的圖說
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, path, c)


def fig3_routep(data, a, c, path):
    """正文圖 3：route_p 按 ground truth 分層。它解釋圖 1 的曲線為什麼長那樣。"""
    import matplotlib.pyplot as plt
    gold = data.gold
    layers = [
        ("fast", t("gold = fast（fast 就答對）", "gold = fast"), c["cat"][0]),
        ("powerful", t("gold = powerful（需要 powerful）", "gold = powerful"), c["cat"][1]),
        ("none", t("gold = none（兩個都答錯）", "gold = none"), c["muted"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 5.0), sharex=True)
    edges = np.linspace(0, 1, 41)
    for ax, (key, label, col) in zip(axes, layers):
        style_axes(ax, c)
        v = data.p[gold == key]
        # 用「佔該列的比例」而不是題數：三列的 n 差到一個數量級（例如 574 / 116 / 68），
        # 畫原始題數的話高度不能互比，看不出分布形狀的差別
        w = np.full(len(v), 1 / max(len(v), 1))
        ax.hist(v, bins=edges, weights=w, color=col, edgecolor=c["surface"], linewidth=0.4)
        ax.axvline(0.5, color=c["muted"], lw=1.0, ls=(0, (2, 3)))
        ax.set_ylabel("")
        ax.annotate(f"{label}　n = {len(v)}（{len(v) / data.n:.1%}）",
                    (0.01, 0.80), xycoords="axes fraction", color=c["ink2"], fontsize=9)
        ax.tick_params(labelleft=False)
        ax.grid(axis="x", color=c["grid"], linewidth=0.6)
        ax.grid(axis="y", visible=False)
    axes[0].annotate(t("argmax 門檻 0.5", "argmax 0.5"), (0.5, 1.0), xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(4, -12), color=c["muted"], fontsize=8)
    axes[-1].set_xlabel("route_p = P(powerful)", color=c["ink2"])
    axes[-1].set_xlim(0, 1)
    fig.text(0.01, -0.04,
             t("每列各自正規化成「佔該列的比例」，所以三列的高度可以互比（看形狀，不是看題數）。",
               "Each row is normalised within itself, so heights are comparable across rows."),
             color=c["muted"], fontsize=8)
    fig.suptitle(t(f"圖 3　route_p 依 ground truth 分層"
                   f"（AUROC {a['auroc']:.2f}、重疊係數 {a['overlap']:.2f}）",
                   f"Fig 3  route_p by ground truth (AUROC {a['auroc']:.2f}, "
                   f"overlap {a['overlap']:.2f})"),
                 color=c["ink"], fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path, c)


def figA1_y_axis(a, c, path):
    """附錄 A 圖 A1：end-to-end 正確率 vs router 分類正確率，兩者的最佳門檻會分岔。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    style_axes(ax, c)
    r, cc = a["router"], a["class_curve"]
    ax.plot(r["call"], r["acc"], color=c["router"], lw=2.2,
            label=t("end-to-end 正確率（本報告的 Y 軸）", "end-to-end accuracy (used)"))
    ax.plot(cc["call"], cc["acc"], color=c["cat"][1], lw=1.8, ls=(0, (6, 3)),
            label=t("router 分類正確率 vs gold（僅供診斷）", "classification accuracy (diagnostic)"))

    i_e = int(np.argmax(r["acc"]))
    i_c = int(np.argmax(cc["acc"]))
    for x, y, col, lab in ((r["call"][i_e], r["acc"][i_e], c["router"],
                            t("end-to-end 最佳", "best (end-to-end)")),
                           (cc["call"][i_c], cc["acc"][i_c], c["cat"][1],
                            t("分類正確率最佳", "best (classification)"))):
        ax.plot([x, x], [ax.get_ylim()[0], y], color=col, lw=1.0, ls=(0, (2, 3)))
        ax.annotate(f"{lab} {_pct(x)}", (x, y), textcoords="offset points",
                    xytext=(6, 6), color=col, fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    pct_axis(ax.yaxis)
    ax.set_xlabel(t("powerful 呼叫比例", "powerful call rate"), color=c["ink2"])
    ax.set_ylabel(t("正確率", "accuracy"), color=c["ink2"])
    ax.set_title(t("圖 A1　為什麼 Y 軸用 end-to-end 正確率",
                   "Fig A1  Why the Y axis is end-to-end accuracy"),
                 color=c["ink"], fontsize=11.5, loc="left", pad=10)
    ax.legend(loc="lower right", labelcolor=c["ink2"], fontsize=8.5)
    _save(fig, path, c)


def figA2_calibration(a, c, path):
    """附錄 A 圖 A2：校準圖 + 每箱題數。X 用 p_max，二選一的下限是 0.5。"""
    import matplotlib.pyplot as plt
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1]})
    style_axes(ax, c)
    style_axes(axn, c)
    ax.plot([0.5, 1.0], [0.5, 1.0], color=c["muted"], lw=1.2,
            label=t("完美校準", "perfect calibration"))
    xs = [(b["lo"] + b["hi"]) / 2 for b in a["cal"]]
    big = [(b["conf"], b["acc"]) for b in a["cal"] if b["n"] >= 20]
    small = [(b["conf"], b["acc"]) for b in a["cal"] if 0 < b["n"] < 20]
    if big:
        ax.plot(*zip(*big), "o", ms=8, color=c["router"], markeredgecolor=c["surface"],
                markeredgewidth=2, label=t("分箱（n ≥ 20）", "bin (n ≥ 20)"))
    if small:
        ax.plot(*zip(*small), "o", ms=8, markerfacecolor="none",
                markeredgecolor=c["router"], markeredgewidth=1.5,
                label=t("分箱（n < 20，樣本少）", "bin (n < 20)"))
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(min(0.45, min([b["acc"] for b in a["cal"] if b["n"]], default=0.5) - 0.05), 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    pct_axis(ax.yaxis)
    ax.set_ylabel(t("該箱的實際正確率", "empirical accuracy"), color=c["ink2"])
    ax.legend(loc="upper left", labelcolor=c["ink2"], fontsize=8.5)
    ax.set_title(t(f"圖 A2　router 信心的校準　ECE = {a['ece']:.3f}　MCE = {a['mce']:.3f}",
                   f"Fig A2  Calibration  ECE = {a['ece']:.3f}  MCE = {a['mce']:.3f}"),
                 color=c["ink"], fontsize=11.5, loc="left", pad=10)
    axn.bar(xs, [b["n"] for b in a["cal"]], width=0.045, color=c["muted"],
            edgecolor=c["surface"], linewidth=1.2)
    axn.set_ylabel(t("題數", "count"), color=c["ink2"])
    axn.set_xlabel("p_max = max(route_p, 1 − route_p)", color=c["ink2"])
    fig.text(0.01, -0.02,
             t(f"{N_BINS} 等寬箱；排除 gold=none（{a['cal_n']} 題）。"
               f"計入 gold=none 時 ECE = {a['ece_incl']:.3f}。",
               f"{N_BINS} equal-width bins; gold=none excluded (n = {a['cal_n']}). "
               f"Including them: ECE = {a['ece_incl']:.3f}."),
             color=c["muted"], fontsize=8)
    fig.tight_layout()
    _save(fig, path, c)


def figB1_guard_cost(a, c, path):
    """附錄 B 圖 B1：X 軸改成絕對金額，看守門開銷把曲線往右推多少。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    style_axes(ax, c)
    r = a["router"]
    x0 = r["cost"] * 1000
    x1 = x0 + GUARD_COST_PER_ITEM * 1000
    ax.plot(x0, r["acc"], color=c["router"], lw=2.2, label=t("只計模型花費", "model cost only"))
    ax.plot(x1, r["acc"], color=c["cat"][1], lw=1.8, ls=(0, (6, 3)),
            label=t("加上守門開銷", "incl. guard overhead"))
    ax.plot([x0[0]], [r["acc"][0]], "o", ms=8, color=c["ink"],
            markeredgecolor=c["surface"], markeredgewidth=2)
    ax.plot([x1[0]], [r["acc"][0]], "o", ms=8, markerfacecolor="none",
            markeredgecolor=c["cat"][1], markeredgewidth=1.6)
    ax.annotate(t(f"全送 fast 也要付守門費\n${x1[0]:,.2f}／千題",
                  f"guard is paid even at 0%\n${x1[0]:,.2f}/1k"),
                (x1[0], r["acc"][0]), textcoords="offset points", xytext=(10, -2),
                color=c["cat"][1], fontsize=8.5)
    if not math.isnan(a["cpt"][0.95]):
        k = int(np.searchsorted(r["call"], a["cpt"][0.95]))
        k = min(k, len(x0) - 1)
        # 兩個以上的 $ 會被 matplotlib 當成 mathtext，所以要跳脫
        ax.annotate(t(f"CPT(95%)　\\${x0[k]:,.2f} → \\${x1[k]:,.2f}／千題",
                      f"CPT(95%)  \\${x0[k]:,.2f} → \\${x1[k]:,.2f}/1k"),
                    (x1[k], r["acc"][k]), textcoords="offset points", xytext=(8, -18),
                    color=c["ink2"], fontsize=8.5)
    pct_axis(ax.yaxis)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.set_xlabel(t("$ 每千題", "$ per 1,000 items"), color=c["ink2"])
    ax.set_ylabel(t("end-to-end 正確率", "end-to-end accuracy"), color=c["ink2"])
    ax.set_title(t(f"圖 B1　含／不含 router 自身的守門開銷"
                   f"（固定 ${GUARD_COST_PER_ITEM * 1000:,.2f}／千題）",
                   f"Fig B1  With / without guard overhead"),
                 color=c["ink"], fontsize=11.5, loc="left", pad=10)
    ax.legend(loc="lower right", labelcolor=c["ink2"], fontsize=8.5)
    _save(fig, path, c)


def figB2_conf(groups, c, path):
    """附錄 B 圖 B2：50 個資料集的 route_conf 盒鬚，按類別分組、組內按中位數排序。"""
    import matplotlib.pyplot as plt
    order = ("safety", "jailbreak", "injection", "cyber", "coding", "skill", "fast-powerful")
    by_cat = defaultdict(list)
    for name, (cat, vals) in groups.items():
        by_cat[cat].append((float(np.median(vals)), name, vals))
    rows = []
    for cat in order:
        for med, name, vals in sorted(by_cat.get(cat, []), reverse=True):
            rows.append((cat, name, vals, med))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, max(4.0, 0.21 * len(rows) + 1.4)))
    style_axes(ax, c)
    colours = {cat: c["cat"][i % len(c["cat"])] for i, cat in enumerate(order)}
    pos = np.arange(len(rows))[::-1]
    kw = dict(positions=pos, widths=0.62, whis=(5, 95), showfliers=False, patch_artist=True)
    try:  # matplotlib ≥ 3.11 用 orientation，舊版只有 vert
        bp = ax.boxplot([r[2] for r in rows], orientation="horizontal", **kw)
    except TypeError:
        bp = ax.boxplot([r[2] for r in rows], vert=False, **kw)
    for patch, (cat, *_rest) in zip(bp["boxes"], rows):
        patch.set_facecolor(colours[cat])
        patch.set_edgecolor(c["surface"])
        patch.set_linewidth(1.2)
        patch.set_alpha(0.9)
    for key in ("whiskers", "caps"):
        for ln in bp[key]:
            ln.set_color(c["muted"])
            ln.set_linewidth(0.9)
    for ln in bp["medians"]:
        ln.set_color(c["surface"])
        ln.set_linewidth(1.6)
    ax.set_yticks(pos)
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("route_conf = |2·route_p − 1|", color=c["ink2"])
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=colours[cat], lw=6, label=cat)
               for cat in order if by_cat.get(cat)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04 - 2.2 / len(rows)),
              ncols=4, fontsize=8, labelcolor=c["ink2"])
    ax.set_title(t("圖 B2　各資料集的 route_conf（盒＝四分位，鬚＝5／95 百分位）",
                   "Fig B2  route_conf by dataset (box = IQR, whiskers = 5/95 pct)"),
                 color=c["ink"], fontsize=11.5, loc="left", pad=10)
    fig.text(0.01, -0.01,
             t("只有 fast-powerful 四個資料集有 route_gold 可驗證信心是否名實相符（圖 A2）；"
               "其餘僅為描述性，高信心不代表判斷正確。",
               "Only the fast-powerful benches have gold labels; the rest are descriptive only."),
             color=c["muted"], fontsize=8)
    fig.tight_layout()
    _save(fig, path, c)


# ============================================================================
# Excel
# ============================================================================
def write_excel(a, per_bench, data, path, args):
    import pandas as pd

    def row_for(label, call, acc, cost):
        return {"設定": label, "end-to-end 正確率": acc, "powerful 呼叫比例": call,
                "$/千題（不含守門）": cost * 1000,
                "$/千題（含守門）": (cost + GUARD_COST_PER_ITEM) * 1000,
                "相對全送 powerful": acc / a["acc_pow"] if a["acc_pow"] else np.nan}

    t1 = pd.DataFrame([
        row_for("全送 fast", 0.0, a["acc_fast"], a["cost_fast"]),
        row_for("全送 powerful", 1.0, a["acc_pow"], a["cost_pow"]),
        row_for("現行 argmax", a["star"]["call"], a["star"]["acc"], a["star"]["cost"]),
        *[row_for(f"{o['label']}（τ ≥ {o['tau']:.3f}）", o["call"], o["acc"], o["cost"])
          for o in a["ops"]],
        row_for("Oracle（不可部署）", a["knee"], a["ceiling"], np.nan),
    ])
    t1b = pd.DataFrame([
        {"指標": "CPT(90%)", "值": a["cpt"][0.90], "說明": "達到全送 powerful 90% 品質所需的呼叫比例"},
        {"指標": "CPT(95%)", "值": a["cpt"][0.95], "說明": "同上，95%"},
        {"指標": "CPT(99%)", "值": a["cpt"][0.99], "說明": "同上，99%"},
        {"指標": "APGR", "值": a["apgr"], "說明": "在 random 與 oracle 之間回收的差距比例"},
        {"指標": "oracle 天花板", "值": a["ceiling"], "說明": "任一 model 答對的比例"},
        {"指標": "oracle 轉折點", "值": a["knee"], "說明": "oracle 到頂所需的呼叫比例"},
        {"指標": "兩個都答錯", "值": a["both_wrong"], "說明": "routing 無法補救的部分"},
        {"指標": "AUROC", "值": a["auroc"], "說明": "route_p 區分 gold=fast / powerful 的鑑別力"},
        {"指標": "重疊係數", "值": a["overlap"], "說明": "兩個 gold 的 route_p 分布重疊面積"},
        {"指標": "ECE", "值": a["ece"], "說明": f"{N_BINS} 等寬箱，排除 gold=none"},
        {"指標": "ECE（含 gold=none）", "值": a["ece_incl"], "說明": "把兩個都答錯計為選擇錯誤"},
        {"指標": "MCE", "值": a["mce"], "說明": "最糟一箱的校準誤差"},
    ])

    tA1 = pd.DataFrame([{
        "Benchmark": name.upper(), "n": b["n"],
        "fast 正確率": b["acc_fast"], "powerful 正確率": b["acc_pow"],
        "oracle 天花板": b["ceiling"], "兩個都答錯": b["both_wrong"],
        "轉折點": b["knee"], "現行 argmax 呼叫比例": b["star"]["call"],
        "現行 argmax 正確率": b["star"]["acc"], "APGR": b["apgr"],
        "CPT(95%)": b["cpt"][0.95],
    } for name, b in per_bench.items()] + [{
        "Benchmark": "合計", "n": a["n"],
        "fast 正確率": a["acc_fast"], "powerful 正確率": a["acc_pow"],
        "oracle 天花板": a["ceiling"], "兩個都答錯": a["both_wrong"],
        "轉折點": a["knee"], "現行 argmax 呼叫比例": a["star"]["call"],
        "現行 argmax 正確率": a["star"]["acc"], "APGR": a["apgr"],
        "CPT(95%)": a["cpt"][0.95],
    }])

    # 表 A2：兩個 model 的能力重疊，一列一個 dataset。曲線的形狀就是從這張表來的。
    tA2 = pd.DataFrame([{
        "Dataset": label, "n": q["n"],
        "fast 對 & powerful 對": q["both"],
        "fast 對 & powerful 錯（升級掉分）": q["fast_only"],
        "fast 錯 & powerful 對（升級賺分）": q["pow_only"],
        "兩個都錯": q["none"],
        "升級賺分比例（= oracle 轉折點）": q["gain"],
        "升級掉分比例": q["loss"],
        "淨益（= all-powerful − all-fast）": q["net"],
    } for label, q in [(k.upper(), v["quad"]) for k, v in per_bench.items()] + [("合計", a["quad"])]])

    cm = a["confusion"]
    gc = a["gold_counts"]
    tA3 = pd.DataFrame([
        {"router 的選擇": "fast", "gold=fast": cm[("fast", "fast")],
         "gold=powerful": cm[("fast", "powerful")], "gold=none": cm[("fast", "none")]},
        {"router 的選擇": "powerful", "gold=fast": cm[("powerful", "fast")],
         "gold=powerful": cm[("powerful", "powerful")], "gold=none": cm[("powerful", "none")]},
    ])
    tA3["合計"] = tA3[["gold=fast", "gold=powerful", "gold=none"]].sum(axis=1)
    n_keep = gc["fast"] + gc["powerful"]
    tA3b = pd.DataFrame([
        {"衍生指標": "分類正確率（排除 gold=none）",
         "值": (cm[("fast", "fast")] + cm[("powerful", "powerful")]) / max(n_keep, 1)},
        {"衍生指標": "不必要的升級（gold=fast 卻選 powerful）",
         "值": cm[("powerful", "fast")] / max(gc["fast"], 1)},
        {"衍生指標": "漏升級（gold=powerful 卻選 fast）",
         "值": cm[("fast", "powerful")] / max(gc["powerful"], 1)},
    ])

    tcal = pd.DataFrame([{
        "箱": f"{b['lo']:.2f}–{b['hi']:.2f}", "n": b["n"], "n 佔比": b["n"] / max(a["cal_n"], 1),
        "平均 p_max": b["conf"], "實際正確率": b["acc"], "差（acc − conf）": b["gap"],
        "加權貢獻": (b["n"] / max(a["cal_n"], 1)) * abs(b["gap"]) if b["n"] else np.nan,
    } for b in a["cal"]])

    r = a["router"]
    step = max(1, len(r["call"]) // 400)   # 曲線太長時抽樣存，Excel 開得動
    tcurve = pd.DataFrame({
        "門檻 τ": r["tau"][::step],
        "powerful 呼叫比例": r["call"][::step],
        "end-to-end 正確率": r["acc"][::step],
        "$/千題": r["cost"][::step] * 1000,
        "oracle 正確率": a["oracle"]["acc"][::step],
        "random 正確率": np.interp(r["call"][::step], GRID, a["random"]["acc"]),
        "bootstrap 下界": (np.interp(r["call"][::step], GRID, a["lo"])
                         if a["lo"] is not None else np.nan),
        "bootstrap 上界": (np.interp(r["call"][::step], GRID, a["hi"])
                         if a["hi"] is not None else np.nan),
    })

    tmeta = pd.DataFrame([
        # report_muli_fast --per-fast full 會把 args.source 設成「joined.jsonl + 哪個 answers 檔」
        {"項目": "資料來源", "值": "示意資料（--mock）" if args.mock else
         getattr(args, "source", None) or os.path.join(args.out, JOINED)},
        {"項目": "題數 N", "值": a["n"]},
        {"項目": "fast model", "值": ROUTE_FAST},
        {"項目": "powerful model", "值": ROUTE_POWERFUL},
        {"項目": "fast 單價 $/1M（in / out）", "值": f"{PRICES['fast']['in']} / {PRICES['fast']['out']}"},
        {"項目": "powerful 單價 $/1M（in / out）",
         "值": f"{PRICES['powerful']['in']} / {PRICES['powerful']['out']}"},
        {"項目": "單價是否為示意值", "值": "是（務必替換）" if PRICES_ARE_PLACEHOLDER else "否"},
        {"項目": "守門開銷 $/題", "值": GUARD_COST_PER_ITEM},
        {"項目": "bootstrap 次數", "值": args.boot},
        {"項目": "校準分箱數", "值": N_BINS},
        {"項目": "gold 定義", "值": "fast=fast 答對；powerful=fast 錯而 powerful 對；none=兩個都錯"},
        {"項目": "守門動作分布", "值": ", ".join(f"{k}={v}" for k, v in sorted(a["guard_counts"].items()))},
        {"項目": "未納入", "值": "中文版（--cn）、其他 model 配對、其他 router 問法、cascade"},
    ])

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        t1.to_excel(xw, sheet_name="表1_摘要", index=False, startrow=0)
        t1b.to_excel(xw, sheet_name="表1_摘要", index=False, startrow=len(t1) + 2)
        tA1.to_excel(xw, sheet_name="表A1_各bench", index=False)
        tA2.to_excel(xw, sheet_name="表A2_能力重疊", index=False)
        tA3.to_excel(xw, sheet_name="表A3_混淆矩陣", index=False, startrow=0)
        tA3b.to_excel(xw, sheet_name="表A3_混淆矩陣", index=False, startrow=len(tA3) + 2)
        tcal.to_excel(xw, sheet_name="圖A2_校準分箱", index=False)
        tcurve.to_excel(xw, sheet_name="曲線原始值", index=False)
        tmeta.to_excel(xw, sheet_name="方法參數", index=False)
        import pandas as _pd
        _pd.DataFrame(data.rows).to_excel(xw, sheet_name="逐題資料", index=False)
    print(f"  → {os.path.relpath(path, HERE)}")


# ============================================================================
# 主控台報表（和 deep12.py report 同樣的風格：分節印，不中斷）
# ============================================================================
def pct(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}%}"


def dw(s):
    """字串在等寬終端機裡的顯示寬度：CJK 全形字算 2 格。

    中文欄位用 f"{s:<12}" 會對不齊，因為 format 把一個中文字當一格寬，
    終端機卻畫成兩格。所有表格都改用下面的 pad() 來補空白。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def pad(s, n, right=False):
    """照顯示寬度把 s 補到 n 格；right=True 是靠右對齊。"""
    s = str(s)
    gap = max(0, n - dw(s))
    return (" " * gap + s) if right else (s + " " * gap)


def print_report(a, per_bench, args):
    n = a["n"]
    print(f"\n{'=' * 78}")
    print(f"  model_route 成本／正確率權衡報告　N = {n}"
          f"　（{'示意資料' if args.mock else '真實資料'}）")
    print(f"  fast = {ROUTE_FAST}　powerful = {ROUTE_POWERFUL}")
    print(f"{'=' * 78}")
    if PRICES_ARE_PLACEHOLDER:
        print("\n  ⚠ PRICES 還是示意單價，所有 $ 的數字都不要引用（改 deep20_report.py 的 PRICES）")

    # ---- 表 1 ----
    print("\n[表 1] 摘要")
    print("  " + pad("", 32) + pad("正確率", 9, True) + pad("呼叫比例", 11, True)
          + pad("$/千題", 11, True) + pad("相對 powerful", 16, True))
    print("  " + "-" * 79)
    for label, call, acc, cost in (
        ("全送 fast", 0.0, a["acc_fast"], a["cost_fast"]),
        ("全送 powerful", 1.0, a["acc_pow"], a["cost_pow"]),
        ("現行 argmax", a["star"]["call"], a["star"]["acc"], a["star"]["cost"]),
        *[(f"{o['label']} τ≥{o['tau']:.3f}", o["call"], o["acc"], o["cost"])
          for o in a["ops"]],
    ):
        print("  " + pad(label, 32) + pad(pct(acc), 9, True) + pad(pct(call), 11, True)
              + pad("$" + format(cost * 1000, ",.2f"), 11, True)
              + pad(pct(acc / a["acc_pow"]) if a["acc_pow"] else "-", 16, True))
    print(f"\n  CPT(90%) {pct(a['cpt'][0.90])}　CPT(95%) {pct(a['cpt'][0.95])}"
          f"　CPT(99%) {pct(a['cpt'][0.99])}　APGR {a['apgr']:.3f}")
    if a["star"]["acc"] > a["acc_pow"]:
        print(f"  註：現行 argmax 的正確率（{pct(a['star']['acc'])}）高過全送 powerful"
              f"（{pct(a['acc_pow'])}）—— router 把「fast 對而 powerful 錯」的題目留在了 fast。"
              f"\n      這種情況下「最省」那個作業點是降級，不是改進，要自己權衡。")
    print(f"  oracle 天花板 {pct(a['ceiling'])}（轉折 {pct(a['knee'])}）"
          f"　兩個都答錯 {pct(a['both_wrong'])}")
    print(f"  守門開銷 ${GUARD_COST_PER_ITEM * 1000:,.2f}/千題（未計入上方成本）")

    # ---- 表 A1 ----
    print("\n[表 A1] 各 benchmark")
    cols = [("Bench", 10, False), ("n", 6, True), ("fast", 9, True), ("powerful", 10, True),
            ("天花板", 10, True), ("兩個都錯", 11, True), ("轉折", 9, True),
            ("APGR", 8, True), ("CPT95", 9, True)]
    print("  " + "".join(pad(c, w, r) for c, w, r in cols))
    print("  " + "-" * sum(w for _, w, _ in cols))
    for label, b in [(k.upper(), v) for k, v in per_bench.items()] + [("合計", a)]:
        vals = [label, b["n"], pct(b["acc_fast"]), pct(b["acc_pow"]), pct(b["ceiling"]),
                pct(b["both_wrong"]), pct(b["knee"]), f"{b['apgr']:.2f}", pct(b["cpt"][0.95])]
        print("  " + "".join(pad(v, w, r) for v, (_, w, r) in zip(vals, cols)))

    # ---- 表 A2：逐 dataset 的 2×2 ----
    print("\n[表 A2] 兩個 model 的能力重疊（fast 對／錯 × powerful 對／錯）")
    print("  曲線的形狀就是從這張表來的：右上角那格把曲線的右端往下拉")
    for label, b in [("合計", a)] + [(k.upper(), v) for k, v in per_bench.items()]:
        q = b["quad"]
        print(f"\n  ── {label}（n = {q['n']}）")
        print("     " + pad("", 12) + pad("powerful 對", 14, True) + pad("powerful 錯", 14, True))
        print("     " + pad("fast 對", 12) + pad(q["both"], 14, True)
              + pad(q["fast_only"], 14, True)
              + ("   ← 升級這些題會掉分" if q["fast_only"] else ""))
        print("     " + pad("fast 錯", 12) + pad(q["pow_only"], 14, True)
              + pad(q["none"], 14, True))
        print(f"     升級賺一分 {q['pow_only']:>4} 題（{pct(q['gain'])}，= oracle 轉折點）"
              f"　虧一分 {q['fast_only']:>3} 題（{pct(q['loss'])}）"
              f"　淨 {q['net']:+.1%}（= all-powerful − all-fast）")
        if q["loss"] >= 0.10:
            print(f"     ⚠ 掉分題佔 {pct(q['loss'])} ≥ 10%：「全送 powerful 是品質上界」"
                  f"在這個 dataset 上不成立，主圖的敘事要跟著改")

    # ---- 表 A3：混淆矩陣 ----
    print("\n[表 A3] router 選擇 × gold")
    cm, gc = a["confusion"], a["gold_counts"]
    print("  " + pad("", 20) + pad("gold=fast", 12, True) + pad("gold=powerful", 16, True)
          + pad("gold=none", 12, True) + pad("合計", 9, True))
    for ch in ("fast", "powerful"):
        tot = sum(cm[(ch, g)] for g in ("fast", "powerful", "none"))
        print("  " + pad(f"router 選 {ch}", 20) + pad(cm[(ch, "fast")], 12, True)
              + pad(cm[(ch, "powerful")], 16, True) + pad(cm[(ch, "none")], 12, True)
              + pad(tot, 9, True))
    print("  " + pad("合計", 20) + pad(gc["fast"], 12, True) + pad(gc["powerful"], 16, True)
          + pad(gc["none"], 12, True) + pad(n, 9, True))
    n_keep = gc["fast"] + gc["powerful"]
    print(f"\n  分類正確率（排除 gold=none）"
          f"　{pct((cm[('fast', 'fast')] + cm[('powerful', 'powerful')]) / max(n_keep, 1))}")
    print(f"  不必要的升級（gold=fast 卻選 powerful）"
          f"　{cm[('powerful', 'fast')]}/{gc['fast']} = {pct(cm[('powerful', 'fast')] / max(gc['fast'], 1))}")
    print(f"  漏升級（gold=powerful 卻選 fast）"
          f"　{cm[('fast', 'powerful')]}/{gc['powerful']} = {pct(cm[('fast', 'powerful')] / max(gc['powerful'], 1))}")

    # ---- 圖 3 / 圖 A2 的數字 ----
    print("\n[圖 3] route_p 的分離度")
    print(f"  AUROC {a['auroc']:.3f}　重疊係數 {a['overlap']:.3f}"
          f"　（gold: fast {gc['fast']}／powerful {gc['powerful']}／none {gc['none']}）")

    print(f"\n[圖 A2] 校準　ECE {a['ece']:.4f}（排除 gold=none，n={a['cal_n']}）"
          f"　含入時 {a['ece_incl']:.4f}　MCE {a['mce']:.4f}")
    print("  " + pad("箱 (p_max)", 15) + pad("n", 7, True) + pad("平均 p_max", 14, True)
          + pad("實際正確率", 14, True) + pad("差", 10, True) + "  標記")
    for b in a["cal"]:
        if not b["n"]:
            continue
        tag = ("樣本少" if b["n"] < 20 else
               "過度自信" if b["gap"] < -0.10 else "信心不足" if b["gap"] > 0.10 else "")
        print("  " + pad(f"{b['lo']:.2f}–{b['hi']:.2f}", 15) + pad(b["n"], 7, True)
              + pad(f"{b['conf']:.3f}", 14, True) + pad(f"{b['acc']:.3f}", 14, True)
              + pad(f"{b['gap']:+.3f}", 10, True) + "  " + tag)

    # ---- 方法檢查 ----
    print("\n[方法檢查] 守門在這批題目上的動作分布")
    for k, v in sorted(a["guard_counts"].items()):
        flag = "  ← block 的題目不呼叫模型，會扭曲曲線，須在附錄說明處理方式" if k == "block" else ""
        print("  " + pad(k, 12) + pad(v, 7, True) + f"　{pct(v / n)}{flag}")


# ============================================================================
# status：資料齊不齊
# ============================================================================
def cmd_status(args):
    print("\n資料齊備度（畫圖需要的五個欄位：route_p / c_fast / c_pow / tok_fast / tok_pow）\n")
    print(f"  {'Benchmark':<10}{'deep20 result':>16}{'fast 作答':>12}{'powerful 作答':>15}  下一步")
    print("  " + "-" * 86)
    todo = []
    for ds_name, mod_name in BENCHES.items():
        rows, res_path = result_rows(ds_name)
        ans_dir = os.path.join(HERE, mod_name, "answers")
        f_path = os.path.join(ans_dir, f"{ROUTE_FAST}.jsonl")
        p_path = os.path.join(ans_dir, f"{ROUTE_POWERFUL}.jsonl")
        n_f, n_p = len(read_jsonl(f_path)), len(read_jsonl(p_path))
        nxt = []
        if not rows:
            nxt.append(f"python deep20.py run --only {ds_name}")
        if not n_f:
            nxt.append(f"python {mod_name}_routing.py answer --model {ROUTE_FAST}")
        if not n_p:
            nxt.append(f"python {mod_name}_routing.py answer --model {ROUTE_POWERFUL}")
        print(f"  {ds_name:<10}{(str(len(rows)) + ' 題') if rows else '✗':>16}"
              f"{(str(n_f) + ' 題') if n_f else '✗':>12}{(str(n_p) + ' 題') if n_p else '✗':>15}"
              f"  {'齊了' if not nxt else nxt[0]}")
        todo += nxt[1:]
    for x in todo:
        print(f"  {'':<53}  {x}")

    joined = os.path.join(args.out, JOINED)
    print(f"\n  joined.jsonl　{'有（' + str(sum(1 for _ in open(joined, encoding='utf-8'))) + ' 題）' if os.path.exists(joined) else '✗ 還沒跑 prepare'}")
    print(f"\n  沒有真資料也可以先看報表版面：python {os.path.basename(__file__)} report --mock\n")


# ============================================================================
# report
# ============================================================================
def cmd_report(args):
    # --boot 沒給時：有要畫誤差帶才跑 bootstrap（1000 次），否則跳過，省幾秒
    if args.boot is None:
        args.boot = 1000 if args.with_band else 0
    os.makedirs(args.out, exist_ok=True)
    rows = load_joined(args)
    data = Data(rows)
    if data.n < 10:
        raise SystemExit(f"只有 {data.n} 題，畫不出有意義的曲線")
    render_report(data, args, args.out)


def per_bench_analyses(data, n_boot):
    """每個 bench 各跑一次 analyse()，依 bench 名稱排序。"""
    # 單一 bench 題數少，bootstrap 次數照用，但題數太少時誤差帶本來就該很寬
    return {name: analyse(data.subset(data.bench == name), n_boot)
            for name in sorted(set(data.bench))}


def fig1_tau_axis_set(a, per_bench, c, out, sfx="", **kw):
    """fig1_main_tau_axis 這一家：合計 + 每個 bench 各一張，加上說明版與拆開的圖／文字。

    report 的 --tau axis 和 report_muli_fast 的每個 fast 子目錄都用這一組。
    """
    fig1_main(a, c, os.path.join(out, f"fig1_main_tau_axis{sfx}.png"), tau="axis", **kw)
    for name, b in per_bench.items():  # 四個 bench 各一張，版面和合計那張相同
        fig1_main(b, c, os.path.join(out, f"fig1_main_tau_axis_{name}{sfx}.png"),
                  tau="axis", title=f"{name.upper()}　n = {b['n']}", **kw)
    # 說明版：圖上標 1–6，右邊逐項推導。全是中文，沒有 CJK 字型就不畫（不要變豆腐）
    # 兩版：_explained 標出每一個往下掉的階（▼），_explained_no_down 不標
    if not ZH:
        print("  [skip] fig1_main_tau_axis_*_explained：找不到中文字型")
        return
    for tag, drops in (("", True), ("_no_down", False)):
        fig1_explained(a, c, os.path.join(
            out, f"fig1_main_tau_axis_explained{tag}{sfx}.png"), "合計", drops)
        for name, b in per_bench.items():
            fig1_explained(b, c, os.path.join(
                out, f"fig1_main_tau_axis_{name}_explained{tag}{sfx}.png"),
                name.upper(), drops)
    # 不標 ▼ 那版另外拆成「只有圖」+「只有文字」，給投影片用（.txt 不分深淺色）
    for key, b, label in [("", a, "合計")] + [
            (f"_{name}", b, name.upper()) for name, b in per_bench.items()]:
        base = os.path.join(out, f"fig1_main_tau_axis{key}_explained_no_down")
        fig1_explained_split(b, c, f"{base}_chart{sfx}.png", f"{base}.txt", label)


def render_report(data, args, out):
    """report 的本體：算數字、印報表、畫 PNG、寫 Excel，全部寫進 out。

    cmd_report 用它出 deep20_report/；report_muli_fast --per-fast full 用它替每個 fast 出一份。
    """
    print(f"\n分析 {data.n} 題…（bootstrap {args.boot} 次）", flush=True)
    a = analyse(data, args.boot)
    per_bench = per_bench_analyses(data, args.boot)

    print_report(a, per_bench, args)

    font = setup_font()
    print(f"\n畫圖（字型 {font or 'DejaVu Sans（沒有 CJK 字型，標籤改用英文）'}）")
    c = PALETTE["dark" if args.dark else "light"]
    sfx = "_dark" if args.dark else ""
    kw = dict(band=args.with_band, random=args.with_random)
    # 乾淨版（不標 τ）一定產出；再按 --tau 另存標了 τ 的版本，檔名加後綴，不覆蓋
    fig1_main(a, c, os.path.join(out, f"fig1_main{sfx}.png"), **kw)
    fig2_per_bench(per_bench, c, os.path.join(out, f"fig2_per_bench{sfx}.png"), **kw)
    for y in ("zoom", "norm"):
        fig2_per_bench(per_bench, c, os.path.join(out, f"fig2_per_bench_{y}{sfx}.png"),
                       y=y, **kw)
    for variant in (list(TAU_VARIANTS) if args.tau == "all" else
                    [] if args.tau == "none" else [args.tau]):
        stem = variant.replace("-", "_")
        if variant == "axis":
            fig1_tau_axis_set(a, per_bench, c, out, sfx, **kw)
        else:
            fig1_main(a, c, os.path.join(out, f"fig1_main_tau_{stem}{sfx}.png"),
                      tau=variant, **kw)
        # axis-conf 的刻度有兩行，四格小圖放不下，所以 fig2 不出這個變體
        # 曲線上的 τ 記號畫的是原始正確率，所以 norm 版只出 axis
        if variant != "axis-conf":
            for y, tag in (("fixed", ""), ("zoom", "_zoom"), ("norm", "_norm")):
                if y == "norm" and variant != "axis":
                    continue
                fig2_per_bench(per_bench, c,
                               os.path.join(out, f"fig2_per_bench{tag}_tau_{stem}{sfx}.png"),
                               tau=variant, y=y, **kw)
    fig3_routep(data, a, c, os.path.join(out, f"fig3_routep{sfx}.png"))
    figA1_y_axis(a, c, os.path.join(out, f"figA1_y_axis{sfx}.png"))
    figA2_calibration(a, c, os.path.join(out, f"figA2_calibration{sfx}.png"))
    if args.appendix_b:
        figB1_guard_cost(a, c, os.path.join(out, f"figB1_guard_cost{sfx}.png"))
        groups = mock_conf_by_dataset() if args.mock else conf_by_dataset()
        if groups:
            figB2_conf(groups, c, os.path.join(out, f"figB2_conf_by_dataset{sfx}.png"))
        else:
            print("  [skip] 圖 B2：deep20_results/ 裡沒有 route_conf")

    print("\n寫 Excel")
    write_excel(a, per_bench, data, os.path.join(out, "tables.xlsx"), args)
    print(f"\n完成 → {os.path.relpath(out, HERE)}/\n")


# ============================================================================
# report_muli_fast：powerful 固定、fast 換好幾個 model 比
# ============================================================================
# 為什麼不用重跑 deep20.py run、也不用另一份 prepare：
#     route_p 是 router 只看題目給的（deep20.py 的 ROUTE_FAST 註解：router 看不到 model 身分），
#     所以「哪幾題先升級」對每個 fast 都一樣。換 fast 只改變每一題的 c_fast / cost_fast，
#     也就是藍線每一階是 +1、0 還是 −1。那兩個欄位直接從 <bench>/answers/<fast>.jsonl 讀
#     （答案檔是按題目 id 存的，joined.jsonl 也有 qid），不需要 import <bench>_routing。
#
# 題目範圍：joined.jsonl 的 N 題（prepare 時 ROUTE_FAST 與 ROUTE_POWERFUL 都有有效作答的那些），
# 再取「每個要比的 fast 都有有效作答」的交集。不同 fast 用不同題目比，數字就不能互比。

# 每個 fast model 的單價（$ / 百萬 token，輸入 / 輸出）。**示意值，務必換成實際報價。**
# 沒列在這裡的 model 暫用 PRICES["fast"]，報表、圖上的文字和 Excel 都會標出是哪幾個。
FAST_PRICES = {
    "deepseek-flash": {"in": 0.27, "out": 1.10},
    # "qwen3.8-27b": {"in": ..., "out": ...},
    # "gemma-4-...": {"in": ..., "out": ...},
}

MULTI_FAST_DIR = "multi_fast"   # 輸出在 <out>/multi_fast/

# --mock 用的示意 fast：(名稱, 每個 bench 的正確率增減, token 倍率)。
# 第一個就是 mock_rows() 本身；其餘從它翻動一部分題目的對錯：變弱時優先翻 route_p 高（難）的題，
# 變強時優先翻 route_p 低（簡單）的題，所以 route_p 對它們仍然有鑑別力。
MOCK_FAST = (
    ("deepseek-flash", 0.00, 1.0),
    ("qwen3.8-27b", +0.03, 1.8),
    ("gemma-4", -0.07, 0.6),
)


def fast_price(model):
    """(這個 fast 的單價, 是不是沒填、暫用 PRICES["fast"])。"""
    if model in FAST_PRICES:
        return FAST_PRICES[model], False
    return PRICES["fast"], True


def _cost(in_tok, out_tok, price):
    return (in_tok or 0) / 1e6 * price["in"] + (out_tok or 0) / 1e6 * price["out"]


def safe_name(model):
    return re.sub(r"[^0-9A-Za-z._-]", "_", model)


def answers_file(bench, model):
    """<bench>_routing.py answer --model <model> 寫的檔（routing_bench.Bench.answers_dir）。"""
    return os.path.join(HERE, bench, "answers", f"{model}.jsonl")


@contextlib.contextmanager
def as_fast_model(model):
    """暫時把 ROUTE_FAST / PRICES["fast"] 換成這個 fast，單一 fast 的圖與文字就會寫它的名字和單價。"""
    global ROUTE_FAST
    saved = ROUTE_FAST, PRICES["fast"]
    ROUTE_FAST, PRICES["fast"] = model, fast_price(model)[0]
    try:
        yield
    finally:
        ROUTE_FAST, PRICES["fast"] = saved


def detect_fast_models(benches):
    """沒給 --fast 時：每個 bench 的 answers/ 底下都有作答檔的 model（不含 powerful）。

    只有部分 bench 有作答的不自動收：收進來的話交集會把其他 bench 整個丟掉。
    回傳 (要比的, 只有部分 bench 有的)；ROUTE_FAST 排第一，其餘照名稱。
    """
    found = []
    for b in benches:
        d = os.path.join(HERE, b, "answers")
        names = {fn[:-len(".jsonl")] for fn in os.listdir(d) if fn.endswith(".jsonl")} \
            if os.path.isdir(d) else set()
        found.append(names - {ROUTE_POWERFUL})
    if not found:
        return [], []
    full, anywhere = set.intersection(*found), set.union(*found)
    order = lambda m: (m != ROUTE_FAST, m)
    return sorted(full, key=order), sorted(anywhere - full, key=order)


def multi_fast_rows(base, models):
    """真資料：joined.jsonl 的每一題換上各 fast 的 c_fast / tok_fast / cost_fast。

    回傳 ({fast: rows}, 覆蓋表 {fast: {bench: (有效作答題數, 該 bench 題數)}})。
    rows 只含交集，所以每個 fast 的 rows 是同一組題目、同樣的順序。
    """
    benches = sorted({r["bench"] for r in base})
    n_bench = Counter(r["bench"] for r in base)
    answers = {m: {b: {qid: r for qid, r in read_jsonl(answers_file(b, m)).items() if r.get("ok")}
                   for b in benches} for m in models}
    coverage = {m: {b: (sum(1 for r in base if r["bench"] == b and r["qid"] in answers[m][b]),
                        n_bench[b]) for b in benches} for m in models}
    common = [r for r in base if all(r["qid"] in answers[m][r["bench"]] for m in models)]
    out = {}
    for m in models:
        price, _ = fast_price(m)
        rows = []
        for r in common:
            f = answers[m][r["bench"]][r["qid"]]
            rows.append({**r, "fast_model": m, "c_fast": int(bool(f["correct"])),
                         "tok_fast": (f.get("input_tokens") or 0) + (f.get("output_tokens") or 0),
                         "cost_fast": _cost(f.get("input_tokens"), f.get("output_tokens"), price)})
        out[m] = rows
    return out, coverage


def mock_multi_fast(base, models, seed=20261017):
    """示意資料：以 mock_rows() 為底，照 MOCK_FAST 的設定翻出其他 fast 的對錯與 token。

    亂數用 (seed, 名字) 當種子：同一個 fast 不管和誰一起比，抽到的示意資料都一樣。
    """
    spec = {name: (shift, mult) for name, shift, mult in MOCK_FAST}
    cf = np.array([r["c_fast"] for r in base])
    p = np.array([r["route_p"] for r in base])
    bench = np.array([r["bench"] for r in base])
    out = {}
    for m in models:
        h = int(hashlib.sha1(m.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng([seed, h])
        # 不在 MOCK_FAST 的名字（--mock --fast a b c）也給一組確定的設定
        shift, mult = spec.get(m, (-0.01 * (h % 9), 0.6 + (h % 11) / 10))
        price, _ = fast_price(m)
        new = cf.copy()
        if shift:
            for b in sorted(set(bench)):
                idx = np.where(bench == b)[0]
                pool = idx[cf[idx] == (1 if shift < 0 else 0)]
                k = min(len(pool), int(round(abs(shift) * len(idx))))
                if k:
                    w = (p[pool] if shift < 0 else 1 - p[pool]) + 0.05
                    new[rng.choice(pool, size=k, replace=False, p=w / w.sum())] = int(shift > 0)
        rows = []
        for r, c_new in zip(base, new):
            tok = r["tok_fast"] if mult == 1.0 else int(r["tok_fast"] * mult * rng.lognormal(0, 0.2))
            rows.append({**r, "fast_model": m, "c_fast": int(c_new), "tok_fast": tok,
                         "cost_fast": _cost(tok * 0.35, tok * 0.65, price)})
        out[m] = rows
    return out


def load_multi_fast(args):
    """回傳 (要比的 fast, {fast: rows}, 覆蓋表或 None)。"""
    base = load_joined(args)
    if args.fast:
        models = list(dict.fromkeys(args.fast))  # 去重、保留順序（顏色照這個順序配）
    elif args.mock:
        models = [m for m, _, _ in MOCK_FAST]
    else:
        models, partial = detect_fast_models(sorted({r["bench"] for r in base}))
        print(f"\n沒給 --fast，自動找到 {len(models)} 個 fast：{' / '.join(models) or '（無）'}")
        if partial:
            print(f"  只有部分 bench 有作答、沒收進來：{' / '.join(partial)}"
                  "（要比的話用 --fast 指定，交集會只剩它有答的 bench）")
    if ROUTE_POWERFUL in models:
        raise SystemExit(f"{ROUTE_POWERFUL} 是固定的 powerful，不能同時當 fast")
    if not models:
        raise SystemExit("沒有要比的 fast。先跑 python <bench>_routing.py answer --model <名稱>，"
                         "或用 --fast 指定")
    if len(models) > len(PALETTE["light"]["cat"]):
        raise SystemExit(f"一次最多比 {len(PALETTE['light']['cat'])} 個 fast（顏色只有這麼多，"
                         "不自動生成新色）；分兩批跑")
    if args.mock:
        return models, mock_multi_fast(base, models), None
    rows_by, coverage = multi_fast_rows(base, models)
    empty = [m for m in models if not any(got for got, _ in coverage[m].values())]
    if empty:  # 名字打錯或還沒跑 answer：直接講清楚，不要只說「交集 0 題」
        benches = sorted(coverage[empty[0]])
        raise SystemExit(
            f"找不到這些 fast 的有效作答：{' / '.join(empty)}\n"
            f"  名字要和 <bench>/answers/<名稱>.jsonl 的檔名一樣，例如先跑：\n"
            + "\n".join(f"    python {b}_routing.py answer --model {empty[0]}" for b in benches))
    return models, rows_by, coverage


def tau_point(a, tau=0.5):
    """門檻 τ 在藍線上的作業點：k 題升級、呼叫比例、正確率、$、其中 +1 / −1 各幾題。"""
    srt, r = a["sorted"], a["router"]
    k = int((srt["p"] >= tau).sum())
    d = srt["delta"][:k]
    return {"k": k, "call": float(r["call"][k]), "acc": float(r["acc"][k]),
            "cost": float(r["cost"][k]), "up": int((d > 0).sum()), "down": int((d < 0).sum())}


def best_point(a):
    """藍線上正確率最高的作業點（同分取呼叫比例最小的）。"""
    r = a["router"]
    k = int(np.argmax(r["acc"]))
    return {"call": float(r["call"][k]), "acc": float(r["acc"][k]), "cost": float(r["cost"][k])}


# ---- 主控台報表 ----
def print_multi_fast(mm, models, coverage, args):
    ref = mm[models[0]]["all"]
    n = ref["n"]
    t5 = tau_point(ref)
    print(f"\n{'=' * 78}")
    print(f"  model_route 多個 fast 的比較　N = {n}"
          f"　（{'示意資料' if args.mock else '真實資料'}）")
    print(f"  powerful = {ROUTE_POWERFUL}（固定）　router = {ROUTER_NAME}")
    print(f"  fast = {' / '.join(models)}")
    print(f"{'=' * 78}")
    if PRICES_ARE_PLACEHOLDER:
        print("\n  ⚠ PRICES 還是示意單價，所有 $ 的數字都不要引用（改 deep20_report.py 的 PRICES / FAST_PRICES）")
    fallback = [m for m in models if fast_price(m)[1]]
    if fallback:
        print(f"  ⚠ 這些 fast 沒填單價，暫用 PRICES['fast']（{PRICES['fast']['in']:g} / "
              f"{PRICES['fast']['out']:g}）：{' / '.join(fallback)}　→ 填進 FAST_PRICES")

    if coverage:
        benches = sorted({b for cov in coverage.values() for b in cov})
        print("\n[覆蓋] 以 joined.jsonl 為基準，各 fast 有有效作答的題數（比較只用交集）")
        print("  " + pad("fast", 22) + "".join(pad(b.upper(), 13, True) for b in benches))
        print("  " + "-" * (22 + 13 * len(benches)))
        for m in models:
            print("  " + pad(m, 22) + "".join(
                pad(f"{coverage[m][b][0]}/{coverage[m][b][1]}", 13, True) for b in benches))
        common = Counter(mm[models[0]]["data"].bench.tolist())
        print("  " + pad("交集", 22) + "".join(pad(common.get(b, 0), 13, True) for b in benches))
        dropped = [b.upper() for b in benches if not common.get(b)]
        if dropped:
            print(f"  ⚠ 交集是 0 題的 bench 這次整個不比：{' / '.join(dropped)}")
        for m in models:
            for b in benches:
                got, tot = coverage[m][b]
                if got < tot:
                    print(f"  缺 {tot - got} 題 → python {b}_routing.py answer --model {m}")

    print(f"\n  route_p 和 fast 是誰無關：τ = 0.5 時每個 fast 都升級同樣的 {t5['k']} 題"
          f"（{pct(t5['call'])}）；全送 powerful 一律 {pct(ref['acc_pow'])}"
          f"、${ref['cost_pow'] * 1000:,.2f}/千題")

    # ---- 表 M1：正確率 ----
    print("\n[表 M1] 正確率（合計）")
    cols = [("fast", 22, False), ("全送 fast", 11, True), ("τ=0.5", 9, True),
            ("+1/−1", 10, True), ("最高正確率", 12, True), ("在呼叫比例", 12, True),
            ("天花板", 9, True),
            ("轉折", 8, True), ("APGR", 7, True), ("CPT95", 8, True)]
    print("  " + "".join(pad(c_, w, r_) for c_, w, r_ in cols))
    print("  " + "-" * sum(w for _, w, _ in cols))
    for m in models:
        a = mm[m]["all"]
        tp, bp = tau_point(a), best_point(a)
        vals = [m, pct(a["acc_fast"]), pct(tp["acc"]), f"{tp['up']}/{tp['down']}", pct(bp["acc"]),
                pct(bp["call"], 0), pct(a["ceiling"]), pct(a["knee"]), f"{a['apgr']:.2f}",
                pct(a["cpt"][0.95])]
        print("  " + "".join(pad(v, w, r_) for v, (_, w, r_) in zip(vals, cols)))

    # ---- 表 M2：花費 ----
    print("\n[表 M2] $/千題（合計，只計模型花費）")
    cols = [("fast", 22, False), ("單價 in/out", 14, True), ("全送 fast", 11, True),
            ("τ=0.5", 10, True), ("最高正確率", 12, True), ("全送 powerful", 15, True)]
    print("  " + "".join(pad(c_, w, r_) for c_, w, r_ in cols))
    print("  " + "-" * sum(w for _, w, _ in cols))
    for m in models:
        a = mm[m]["all"]
        price, fb = fast_price(m)
        vals = [m, f"{price['in']:g}/{price['out']:g}" + ("*" if fb else ""),
                f"${a['cost_fast'] * 1000:,.2f}", f"${tau_point(a)['cost'] * 1000:,.2f}",
                f"${best_point(a)['cost'] * 1000:,.2f}", f"${a['cost_pow'] * 1000:,.2f}"]
        print("  " + "".join(pad(v, w, r_) for v, (_, w, r_) in zip(vals, cols)))
    if fallback:
        print("  * 沒填單價，暫用 PRICES['fast']")

    # ---- 表 M3：2×2，合計與各 bench ----
    print("\n[表 M3] 每個 fast 和 powerful 的能力重疊（升級賺分 = 只有 powerful 對、掉分 = 只有 fast 對）")
    cols = [("fast", 22, False), ("n", 6, True), ("兩個都對", 10, True), ("只有fast對", 12, True),
            ("只有powerful對", 16, True), ("兩個都錯", 10, True), ("淨益", 9, True)]
    scopes = [("合計", "all")] + [(b.upper(), b) for b in mm[models[0]]["bench"]]
    for label, key in scopes:
        print(f"\n  ── {label}")
        print("  " + "".join(pad(c_, w, r_) for c_, w, r_ in cols))
        for m in models:
            a = mm[m]["all"] if key == "all" else mm[m]["bench"][key]
            q = a["quad"]
            vals = [m, q["n"], q["both"], q["fast_only"], q["pow_only"], q["none"], f"{q['net']:+.1%}"]
            print("  " + "".join(pad(v, w, r_) for v, (_, w, r_) in zip(vals, cols)))

    # ---- 表 M4：各 bench 的正確率 ----
    print("\n[表 M4] 各 bench：全送 fast → τ=0.5 → 天花板（全送 powerful 每個 fast 都一樣）")
    cols = [("fast", 22, False), ("全送 fast", 11, True), ("τ=0.5", 9, True), ("天花板", 9, True),
            ("APGR", 7, True)]
    for b in mm[models[0]]["bench"]:
        rb = mm[models[0]]["bench"][b]
        print(f"\n  ── {b.upper()}　n = {rb['n']}　全送 powerful {pct(rb['acc_pow'])}"
              f"　τ=0.5 升級 {tau_point(rb)['k']} 題")
        print("  " + "".join(pad(c_, w, r_) for c_, w, r_ in cols))
        for m in models:
            a = mm[m]["bench"][b]
            vals = [m, pct(a["acc_fast"]), pct(tau_point(a)["acc"]), pct(a["ceiling"]),
                    f"{a['apgr']:.2f}"]
            print("  " + "".join(pad(v, w, r_) for v, (_, w, r_) in zip(vals, cols)))


# ---- 圖 ----
def fast_colors(models, c):
    """顏色跟著 --fast 給的順序固定配，不按排名：同一個 fast 在每張圖都是同一個顏色。"""
    return {m: c["cat"][i] for i, m in enumerate(models)}


def _gutter(models):
    """左端點標籤要用的空間（資料座標）：照最長的名字估。"""
    return -min(0.42, max(0.16, 0.045 + 0.0105 * max(len(m) for m in models)))


def _label_left_ends(ax, ma, models, c, lo, hi):
    """每個 fast 的左端點旁標「名字 + 全送 fast 正確率」，靠得太近時往上推開、拉一條引線。"""
    gap = (hi - lo) * 0.075
    placed = []
    for y, m in sorted((ma[m]["acc_fast"], m) for m in models):
        placed.append([max(y, placed[-1][0] + gap) if placed else y, m, y])
    over = placed[-1][0] - (hi - gap / 2)
    if over > 0:
        for p_ in placed:
            p_[0] -= over
    for y_lab, m, y in placed:
        ax.text(-0.04, y_lab, f"{m}\n{ma[m]['acc_fast']:.1%}", ha="right", va="center",
                fontsize=8, color=c["ink2"], linespacing=1.15, zorder=8)
        if abs(y_lab - y) > 1e-9:
            ax.plot([-0.035, -0.014], [y_lab, y], color=c["muted"], lw=0.7, zorder=4)


def _cmp_curve_axes(ax, ma, models, c, labels=True, tau=True):
    """多個 fast 的藍線疊在同一張圖上。ma = {fast: analyse()}，必須是同一組題目。

    粗線 = 該 fast 的 router 曲線，同色細線 = 該 fast 的 oracle，右端黑點 = 全送 powerful（共用）。
    回傳 (Y 下限, Y 上限, 上方 τ 軸或 None)。
    """
    style_axes(ax, c)
    col = fast_colors(models, c)
    ref = ma[models[0]]
    for m in models:
        a = ma[m]
        ax.plot(a["oracle"]["call"], a["oracle"]["acc"], color=col[m], lw=1.1, alpha=0.5, zorder=2)
        ax.plot(a["router"]["call"], a["router"]["acc"], color=col[m], lw=2.0, zorder=3)
        ax.plot([0], [a["acc_fast"]], "o", ms=8, color=col[m], zorder=5,
                markeredgecolor=c["surface"], markeredgewidth=2)
    ax.plot([1], [ref["acc_pow"]], "o", ms=8, color=c["ink"], zorder=6,
            markeredgecolor=c["surface"], markeredgewidth=2)

    lo = min(min(a["acc_fast"], a["router"]["acc"].min()) for a in ma.values()) - 0.03
    ceil = max(a["ceiling"] for a in ma.values())
    # 上方留白跟著跨距放大（AIME 跨 50 分以上時，③ 的標記才不會頂到上方的 τ 刻度）
    hi = min(ceil + max(0.02, 0.07 * (ceil - lo)), max(1.0, ceil + 0.005))
    ax.set_xlim(_gutter(models) if labels else -0.02, 1.02)
    ax.set_ylim(lo, hi)
    pct_axis(ax.yaxis)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("powerful call rate k / N (same for every fast model)", color=c["ink2"])
    ax.set_ylabel("end-to-end accuracy", color=c["ink2"])
    if labels:
        _label_left_ends(ax, ma, models, c, lo, hi)
    axt = draw_tau_axis(ax, ref, c) if tau else None
    return lo, hi, axt


def _cmp_legend(target, models, c, oracle=True, extra=(), **kw):
    """圖例：第一列各 fast（照配色順序），第二列 oracle / 全送 powerful 這些共用的說明。

    matplotlib 的圖例是一欄一欄往下填的，所以把兩列交錯排好再交給它。
    fast 超過 4 個時一列放不下，改成三欄照順序排。
    """
    import matplotlib.pyplot as plt
    col = fast_colors(models, c)
    fast_h = [plt.Line2D([], [], color=col[m], lw=2.0, label=f"fast = {m}") for m in models]
    ref_h = []
    if oracle:
        ref_h.append(plt.Line2D([], [], color=c["muted"], lw=1.1, alpha=0.7,
                                label=t("同色細線 = 該 fast 的 oracle", "thin = that fast's oracle")))
    ref_h.append(plt.Line2D([], [], color=c["ink"], marker="o", ls="", ms=7,
                            label=t(f"全送 powerful（{ROUTE_POWERFUL}）",
                                    f"all powerful ({ROUTE_POWERFUL})")))
    ref_h += list(extra)
    if len(models) <= 4 and len(ref_h) <= len(models):
        ncols = len(models)
        handles = [h for i in range(ncols) for h in (fast_h[i:i + 1] + ref_h[i:i + 1])]
    else:
        ncols = 3
        handles = fast_h + ref_h
    return target.legend(handles=handles, ncols=ncols, labelcolor=c["ink2"], **kw)


def fig_multi_curve(ma, models, c, path, title=None):
    """多 fast 圖 1：X = 呼叫比例（每個 fast 共用），上方 τ，Y = end-to-end 正確率。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    _cmp_curve_axes(ax, ma, models, c)
    if title:
        ax.set_title(title, color=c["ink"], fontsize=11, loc="left", pad=22)
    _cmp_legend(ax, models, c, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)
    _save(fig, path, c)


def fig_multi_cost(ma, models, c, path, title=None):
    """多 fast 圖 2：X 換成 $ 每千題。每個 fast 的 token 用量和單價不同，$ 不能共用一條呼叫比例軸。

    每條線從「全送該 fast」走到「全送 powerful」（共用的黑點），○ 是 τ = 0.5 的作業點。
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    style_axes(ax, c)
    col = fast_colors(models, c)
    ref = ma[models[0]]
    label_bg = dict(boxstyle="square,pad=0.15", fc=c["surface"], ec="none", alpha=0.85)
    for m in models:
        a = ma[m]
        r = a["router"]
        x = r["cost"] * 1000
        ax.plot(x, r["acc"], color=col[m], lw=2.0, zorder=3)
        ax.plot([x[0]], [r["acc"][0]], "o", ms=8, color=col[m], zorder=5,
                markeredgecolor=c["surface"], markeredgewidth=2)
        tp = tau_point(a)
        ax.plot([tp["cost"] * 1000], [tp["acc"]], "o", ms=9, markerfacecolor=c["surface"],
                markeredgecolor=col[m], markeredgewidth=1.8, zorder=6)
        ax.annotate(f"{m}\n${x[0]:,.2f}", (x[0], r["acc"][0]), textcoords="offset points",
                    xytext=(0, -10), ha="center", va="top", fontsize=8, color=c["ink2"],
                    linespacing=1.15, zorder=8, bbox=label_bg)
    xp = ref["cost_pow"] * 1000
    ax.plot([xp], [ref["acc_pow"]], "o", ms=8, color=c["ink"], zorder=7,
            markeredgecolor=c["surface"], markeredgewidth=2)
    ax.annotate(t(f"全送 powerful\n${xp:,.2f}", f"all powerful\n${xp:,.2f}"), (xp, ref["acc_pow"]),
                textcoords="offset points", xytext=(0, -10), ha="center", va="top",
                fontsize=8, color=c["ink2"], linespacing=1.15, zorder=8, bbox=label_bg)
    pct_axis(ax.yaxis)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.2f}")
    ax.margins(x=0.08, y=0.12)
    ax.set_xlabel("$ per 1k items (model cost only, excl. guard)", color=c["ink2"])
    ax.set_ylabel("end-to-end accuracy", color=c["ink2"])
    if title:
        ax.set_title(title, color=c["ink"], fontsize=11, loc="left", pad=8)
    tau_h = plt.Line2D([], [], marker="o", ls="", ms=8, markerfacecolor=c["surface"],
                       markeredgecolor=c["ink2"], markeredgewidth=1.6,
                       label=t("τ = 0.5 的作業點", "τ = 0.5 operating point"))
    _cmp_legend(ax, models, c, oracle=False, extra=[tau_h],  # 這張沒畫 oracle
                loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    _save(fig, path, c)


def fig_multi_per_bench(mb, models, c, path):
    """多 fast 圖 3：每個 bench 一格，格內疊所有 fast。Y 軸各格自己放大（跨距寫在標題）。"""
    import matplotlib.pyplot as plt
    names = list(mb)
    ncols = 2 if len(names) > 1 else 1
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.4, 3.4 * nrows + 0.4), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names):
        ma = mb[name]
        ref = ma[models[0]]
        _cmp_curve_axes(ax, ma, models, c, labels=False, tau=False)
        lo_d = min(min(a["acc_fast"], a["router"]["acc"].min()) for a in ma.values())
        hi_d = max(max(a["ceiling"], a["router"]["acc"].max()) for a in ma.values())
        pad_ = max((hi_d - lo_d) * 0.125, 0.002)
        lo, hi = max(0.0, lo_d - pad_), hi_d + pad_
        if hi_d < 1.0:
            hi = min(hi, 1.0)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{name.upper()}　n = {ref['n']}　(Y span {hi - lo:.1%}, 1 item = {1 / ref['n']:.2%})",
                     color=c["ink"], fontsize=10, loc="left", pad=26)
        small_tau_axis(ax, ref, c)
    _cmp_legend(fig, models, c, loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, path, c)


# ---- 說明版 ----
def _multi_explained_chart(ax, ma, models, c):
    """說明版左邊的圖：多 fast 疊圖 + ①–⑤ 的標記 + τ = 0.5 的讀圖路徑。"""
    col = fast_colors(models, c)
    ref = ma[models[0]]
    lo, hi, axt = _cmp_curve_axes(ax, ma, models, c)
    ax.set_xlabel("下方 X 軸：powerful 呼叫比例 k / N（每個 fast 都一樣）", color=c["ink2"])
    ax.set_ylabel("Y 軸：end-to-end 正確率", color=c["ink2"])
    axt.set_xlabel("上方 X 軸：門檻 τ（route_p ≥ τ 的題目送 powerful，和 fast 是誰無關）",
                   color=c["ink2"], labelpad=6)

    def badge(x, y, num, dx, dy):
        ax.annotate(str(num), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", color=c["ink"], fontsize=10, zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=c["surface"], ec=c["ink2"], lw=0.9),
                    arrowprops=dict(arrowstyle="-", color=c["ink2"], lw=0.7))

    # ① 左端點（指最低那個，其他的旁邊有名字）② 共用的右端黑點
    low = min(models, key=lambda m: ma[m]["acc_fast"])
    badge(0, ma[low]["acc_fast"], 1, 24, -22)
    badge(1, ref["acc_pow"], 2, -22, -24)
    # ③ 每條 oracle 的轉折點：橫軸位置 = 只有 powerful 對 / N
    for m in models:
        ax.plot([ma[m]["knee"]], [ma[m]["ceiling"]], "o", ms=6, color=col[m],
                markeredgecolor=c["surface"], markeredgewidth=1.5, zorder=6)
    top = max(models, key=lambda m: ma[m]["ceiling"])
    badge(ma[top]["knee"], ma[top]["ceiling"], 3, 18, 16)
    # ④ 天花板比 ② 高的那一段 = 只有 fast 對 / N：右端每個 fast 一支同色雙箭頭，從 ② 量到自己的天花板。
    # 天花板的高度不只看「只有 powerful 對」，這一段就是各色天花板不一樣高的原因。
    ap = ref["acc_pow"]
    xs4 = []
    for i, m in enumerate(models):
        if ma[m]["quad"]["fast_only"]:
            x4 = 0.992 - 0.016 * i
            xs4.append((x4, m))
            ax.annotate("", (x4, ap), (x4, ma[m]["ceiling"]),
                        arrowprops=dict(arrowstyle="<->", color=col[m], lw=1.1,
                                        shrinkA=0, shrinkB=0), zorder=6)
    if xs4:
        x4, m4 = min(xs4)  # 最左邊那支旁邊放 ④
        badge(x4, (ap + ma[m4]["ceiling"]) / 2, 4, -22, 0)
    # ⑤ τ = 0.5：從上方刻度往下，和每條線各交一點
    tp = tau_point(ref)
    x5 = tp["call"]
    ax.plot([x5, x5], [hi, lo], color=c["ink2"], lw=0.9, ls=(0, (1.5, 2.5)), zorder=2)
    ys = []
    for m in models:
        y5 = tau_point(ma[m])["acc"]
        ys.append(y5)
        ax.plot([x5], [y5], "o", ms=10, markerfacecolor="none", markeredgecolor=col[m],
                markeredgewidth=1.6, zorder=7)
    badge(x5, max(ys), 5, 20, 18)

    _cmp_legend(ax, models, c, loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=9)


def _multi_explained_lines(ma, models):
    """說明版右邊的逐項推導：[(縮排, 內容, 樣式)]。樣式 h / b / m 同 _explained_lines；
    t 是表格，內容是 (欄位置, 列…)，每列第一格是 fast 名字（畫的時候前面加該 fast 的色塊）。
    """
    ref = ma[models[0]]
    n, tp0 = ref["n"], tau_point(ref)
    pct1 = lambda v: f"{v:.1%}"
    n_pow = ref["quad"]["both"] + ref["quad"]["pow_only"]
    t1 = ((0.035, 0.30, 0.45, 0.61, 0.82),
          ("fast", "兩個都對", "只有 fast 對", "只有 powerful 對", "兩個都錯"),
          *[(m, ma[m]["quad"]["both"], ma[m]["quad"]["fast_only"], ma[m]["quad"]["pow_only"],
             ma[m]["quad"]["none"]) for m in models])
    # ③ 轉折和 ④ 天花板各自怎麼算：轉折只看「只有 powerful 對」，天花板還要加上「只有 fast 對」
    t_orc = ((0.035, 0.27, 0.58),
             ("fast", "③ 轉折 = 只有 powerful 對 / N", "④ 天花板 = (powerful 對 + 只有 fast 對) / N"),
             *[(m, f"{ma[m]['quad']['pow_only']} / {n} = {pct1(ma[m]['knee'])}",
                f"({n_pow} + {ma[m]['quad']['fast_only']}) / {n} = {pct1(ma[m]['ceiling'])}")
               for m in models])
    rows2 = []
    for m in models:
        a, tp = ma[m], tau_point(ma[m])
        rows2.append((m, pct1(a["acc_fast"]), f"{tp['up']} / {tp['down']}", pct1(tp["acc"]),
                      f"${tp['cost'] * 1000:,.2f}"))
    t2 = ((0.035, 0.30, 0.47, 0.64, 0.81),
          ("fast", "① 全送 fast", "⑤ +1 / -1", "⑤ 正確率", "⑤ $/千題"),
          *rows2)
    lines = [
        (0, f"powerful = {ROUTE_POWERFUL}（固定）　router = {ROUTER_NAME}　fast 比 {len(models)} 個", "m"),
        (0, f"N = {n} 題：每個 fast 和 powerful 都有有效作答的交集", "m"),
        (0, "每個 fast 和 powerful 對 N 題各答一次的結果", "h"),
        (0, t1, "t"),
        (0, "圖上 1–5 的數字怎麼來", "h"),
        (0, "① 各色左端點 = 全送該 fast =（兩個都對 + 只有 fast 對）/ N", "b"),
        (0, f"② 右端黑點 = 全送 powerful =（兩個都對 + 只有 powerful 對）/ N = {n_pow} / {n}"
            f" = {pct1(ref['acc_pow'])}", "b"),
        (1, "這兩格加起來就是 powerful 答對的題數，和 fast 是誰無關，所以只有一個黑點", "b"),
        (0, "③ 同色細線 = 該 fast 的 oracle：事先知道答案，先升級「只有 powerful 對」的題目", "b"),
        (1, "轉折的橫軸位置 = 只有 powerful 對 / N：升級完這些題就到頂，再升級也不會更高", "b"),
        (0, "④ 天花板的高度 = (powerful 對 + 只有 fast 對) / N = (N - 兩個都錯) / N", "b"),
        (1, "「只有 fast 對」= powerful 錯、fast 對的題目，oracle 把它們留給 fast", "b"),
        (1, "→ 天花板比 ② 高「只有 fast 對」/ N（右端的同色雙箭頭）", "b"),
        (1, "powerful 一樣，但每個 fast 能補回 powerful 的錯題數不同 → 各色天花板不一樣高", "b"),
        (0, t_orc, "t"),
        (0, f"⑤ τ = 0.5（現行 argmax）：route_p ≥ 0.5 有 {tp0['k']} 題 → {tp0['k']} / {n}"
            f" = {pct1(tp0['call'])}，每個 fast 都升級這 {tp0['k']} 題", "b"),
        (1, "正確率 = ① + (+1 題數 - -1 題數) / N；$ = (這幾題 powerful + 其餘 fast) / N × 1000", "b"),
        (0, t2, "t"),
        (0, "為什麼每個 fast 共用同一條 X 軸", "h"),
        (0, f"route_p 是 {ROUTER_NAME} 只看題目給的，它看不到 fast 是哪個 model", "b"),
        (1, "→ 升級順序、上方 τ 刻度、下方呼叫比例，對每個 fast 都一樣", "b"),
        (1, "換 fast 只改變每一階是 +1、0 還是 -1（y_P - y_F），各色線因此分開", "b"),
        (1, "$ 不能共用：每個 fast 的 token 用量和單價不同 → X 換成 $ 的那張是 cost.png", "b"),
        (0, "單價（$ / 百萬 token，輸入 / 輸出）與全送時的 $ 每千題", "h"),
    ]
    for m in models:
        price, fb = fast_price(m)
        lines.append((0, f"fast {m}　{price['in']:g} / {price['out']:g}　${ma[m]['cost_fast'] * 1000:,.2f}"
                         + ("（沒填，暫用 PRICES['fast']）" if fb else ""), "b"))
    lines += [
        (0, f"powerful {ROUTE_POWERFUL}　{PRICES['powerful']['in']:g} / {PRICES['powerful']['out']:g}"
            f"　${ref['cost_pow'] * 1000:,.2f}；不含守門開銷", "b"),
        (0, "怎麼讀", "h"),
        (0, "上方選一個 τ → 往下對到各色線 → 同一個 τ 下線越高，那個 fast 越好（⑤ 的點線就是一例）", "b"),
    ]
    if PRICES_ARE_PLACEHOLDER:
        lines.append((0, "注意：PRICES 還是示意單價，所有 $ 的數字都不能引用", "b"))
    return lines


def _draw_multi_lines(ax_t, lines, models, c):
    """把 _multi_explained_lines 畫在右半邊。行數隨 fast 個數變多，行距跟著縮，不會超出版面。"""
    col = fast_colors(models, c)
    units = sum(1.7 if s == "h" else (len(x) - 1) if s == "t" else 1 for _, x, s in lines)
    step = min(0.0262, 0.98 / units)
    y = 1.0
    for indent, x, style in lines:
        if style == "h":
            y -= step * 0.55
            ax_t.text(0, y, x.replace("$", r"\$"), transform=ax_t.transAxes, va="top",
                      color=c["ink"], fontsize=11.5)
            y -= step * 1.15
            continue
        if style == "t":
            xs, *rows = x
            for i, row in enumerate(rows):
                if i and row[0] in col:  # 色塊：表格的 fast 名字和圖上的線對得起來
                    ax_t.plot([0.012], [y - i * step - step * 0.38], "s", ms=7, color=col[row[0]],
                              transform=ax_t.transAxes, clip_on=False)
                for xx, cell in zip(xs, row):
                    ax_t.text(xx, y - i * step, str(cell).replace("$", r"\$"),
                              transform=ax_t.transAxes, va="top", fontsize=9.6 if i else 9,
                              color=c["ink"] if i else c["ink2"])
            y -= step * len(rows)
            continue
        ax_t.text(0.03 * indent, y, x.replace("$", r"\$"), transform=ax_t.transAxes, va="top",
                  fontsize=9.6, color=c["muted"] if style == "m" else c["ink"])
        y -= step


def fig_multi_explained(ma, models, c, path, name):
    """多 fast 的說明版：左邊疊圖標 ①–⑤，右邊逐項寫出每個 fast 的數字怎麼算。"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17.5, 9.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.05,
                          left=0.05, right=0.99, top=0.88, bottom=0.16)
    ax, ax_t = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _multi_explained_chart(ax, ma, models, c)
    fig.suptitle(f"{name}　N = {ma[models[0]]['n']}　多個 fast 怎麼讀這張圖", x=0.05, y=0.975,
                 ha="left", color=c["ink"], fontsize=15)
    ax_t.axis("off")
    _draw_multi_lines(ax_t, _multi_explained_lines(ma, models), models, c)
    _save(fig, path, c)


def fig_multi_explained_split(ma, models, c, png_path, txt_path, name):
    """說明版拆成「只有圖」的 PNG 和「只有文字」的 .txt（表格用 Tab 分欄），給投影片用。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    _multi_explained_chart(ax, ma, models, c)
    ax.set_title(f"{name}　N = {ma[models[0]]['n']}", color=c["ink"], fontsize=11, loc="left", pad=34)
    _save(fig, png_path, c)

    out = [f"{name}　N = {ma[models[0]]['n']}　多個 fast 怎麼讀這張圖"]
    for indent, x, style in _multi_explained_lines(ma, models):
        if style == "h":
            out += ["", x]
        elif style == "t":
            out += ["\t".join(str(cell) for cell in row) for row in x[1:]]
        else:
            out.append("　　" * indent + x)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"  → {os.path.relpath(txt_path, HERE)}")


# ---- Excel ----
def write_excel_multi(mm, models, coverage, path, args):
    import pandas as pd
    ref = mm[models[0]]
    scopes = [("合計", None)] + [(b.upper(), b) for b in ref["bench"]]

    summary = []
    for label, key in scopes:
        for m in models:
            a = mm[m]["all"] if key is None else mm[m]["bench"][key]
            tp, bp, q = tau_point(a), best_point(a), a["quad"]
            summary.append({
                "範圍": label, "fast": m, "n": a["n"],
                "全送 fast 正確率": a["acc_fast"], "全送 powerful 正確率": a["acc_pow"],
                "oracle 天花板": a["ceiling"], "oracle 轉折點": a["knee"],
                "τ=0.5 呼叫比例": tp["call"], "τ=0.5 正確率": tp["acc"], "τ=0.5 $/千題": tp["cost"] * 1000,
                "τ=0.5 +1 題數": tp["up"], "τ=0.5 -1 題數": tp["down"],
                "現行 argmax 正確率": a["star"]["acc"],
                "最高正確率": bp["acc"], "最高正確率的呼叫比例": bp["call"],
                "最高正確率的 $/千題": bp["cost"] * 1000,
                "APGR": a["apgr"], "CPT(95%)": a["cpt"][0.95], "AUROC": a["auroc"],
                "全送 fast $/千題": a["cost_fast"] * 1000, "全送 powerful $/千題": a["cost_pow"] * 1000,
                "兩個都對": q["both"], "只有 fast 對": q["fast_only"],
                "只有 powerful 對": q["pow_only"], "兩個都錯": q["none"], "淨益": q["net"],
            })

    curves = []
    for m in models:
        a = mm[m]["all"]
        r = a["router"]
        step = max(1, len(r["call"]) // 400)
        for i in range(0, len(r["call"]), step):
            curves.append({"fast": m, "門檻 τ": r["tau"][i], "powerful 呼叫比例": r["call"][i],
                           "end-to-end 正確率": r["acc"][i], "$/千題": r["cost"][i] * 1000,
                           "oracle 正確率": a["oracle"]["acc"][i]})

    meta = [
        {"項目": "資料來源", "值": "示意資料（--mock）" if args.mock else
         f"{os.path.join(args.out, JOINED)} + <bench>/answers/<fast>.jsonl"},
        {"項目": "題數 N（交集）", "值": ref["all"]["n"]},
        {"項目": "powerful model（固定）", "值": ROUTE_POWERFUL},
        {"項目": "router", "值": ROUTER_NAME},
        {"項目": "fast models（顏色順序）", "值": " / ".join(models)},
        {"項目": "powerful 單價 $/1M（in / out）",
         "值": f"{PRICES['powerful']['in']} / {PRICES['powerful']['out']}"},
    ]
    for m in models:
        price, fb = fast_price(m)
        meta.append({"項目": f"{m} 單價 $/1M（in / out）",
                     "值": f"{price['in']} / {price['out']}" + ("（沒填，暫用 PRICES['fast']）" if fb else "")})
    meta += [
        {"項目": "單價是否為示意值", "值": "是（務必替換）" if PRICES_ARE_PLACEHOLDER else "否"},
        {"項目": "升級順序", "值": "route_p 由大到小；router 看不到 model 身分，所以每個 fast 都一樣"},
        {"項目": "gold 定義", "值": "每個 fast 各自算：fast 答對 → fast；fast 錯而 powerful 對 → powerful；都錯 → none"},
    ]

    base_cols = ["bench", "qid", "route_p", "route", "route_conf", "guard", "c_pow", "tok_pow", "cost_pow"]
    items = pd.DataFrame([{k: r.get(k) for k in base_cols} for r in ref["data"].rows])
    for m in models:
        rows = mm[m]["data"].rows
        items[f"c_fast[{m}]"] = [r["c_fast"] for r in rows]
        items[f"tok_fast[{m}]"] = [r["tok_fast"] for r in rows]
        items[f"cost_fast[{m}]"] = [r["cost_fast"] for r in rows]

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name="摘要_fast比較", index=False)
        if coverage:
            benches = sorted({b for cov in coverage.values() for b in cov})
            pd.DataFrame([{"fast": m, **{f"{b.upper()} 有效作答": coverage[m][b][0] for b in benches},
                           **{f"{b.upper()} 題數": coverage[m][b][1] for b in benches}}
                          for m in models]).to_excel(xw, sheet_name="作答覆蓋", index=False)
        pd.DataFrame(curves).to_excel(xw, sheet_name="曲線原始值", index=False)
        pd.DataFrame(meta).to_excel(xw, sheet_name="方法參數", index=False)
        items.to_excel(xw, sheet_name="逐題資料", index=False)
    print(f"  → {os.path.relpath(path, HERE)}")


# ---- 子命令 ----
def cmd_report_multi_fast(args):
    out = os.path.join(args.out, MULTI_FAST_DIR)
    os.makedirs(out, exist_ok=True)
    models, rows_by, coverage = load_multi_fast(args)
    n = len(rows_by[models[0]])
    if n < 10:
        raise SystemExit(f"每個 fast 都有作答的交集只有 {n} 題，畫不出有意義的曲線"
                         "（看上面的 [覆蓋] 缺什麼，或少比幾個 fast）")

    print(f"\n分析 {n} 題 × {len(models)} 個 fast…", flush=True)
    mm = {}
    for m in models:
        data = Data(rows_by[m])
        mm[m] = {"data": data, "all": analyse(data, 0), "bench": per_bench_analyses(data, 0)}
    print_multi_fast(mm, models, coverage, args)

    font = setup_font()
    print(f"\n畫圖（字型 {font or 'DejaVu Sans（沒有 CJK 字型，標籤改用英文）'}）→ "
          f"{os.path.relpath(out, HERE)}/")
    c = PALETTE["dark" if args.dark else "light"]
    sfx = "_dark" if args.dark else ""
    overall = {m: mm[m]["all"] for m in models}
    by_bench = {b: {m: mm[m]["bench"][b] for m in models} for b in mm[models[0]]["bench"]}

    fig_multi_curve(overall, models, c, os.path.join(out, f"curve_tau_axis{sfx}.png"))
    fig_multi_cost(overall, models, c, os.path.join(out, f"cost{sfx}.png"))
    fig_multi_per_bench(by_bench, models, c, os.path.join(out, f"per_bench_tau_axis{sfx}.png"))
    for b, ma in by_bench.items():
        fig_multi_curve(ma, models, c, os.path.join(out, f"curve_tau_axis_{b}{sfx}.png"),
                        title=f"{b.upper()}　n = {ma[models[0]]['n']}")
        fig_multi_cost(ma, models, c, os.path.join(out, f"cost_{b}{sfx}.png"),
                       title=f"{b.upper()}　n = {ma[models[0]]['n']}")
    if ZH:
        for key, ma, label in [("", overall, "合計")] + [
                (f"_{b}", ma, b.upper()) for b, ma in by_bench.items()]:
            base = os.path.join(out, f"curve_tau_axis{key}_explained")
            fig_multi_explained(ma, models, c, f"{base}{sfx}.png", label)
            fig_multi_explained_split(ma, models, c, f"{base}_chart{sfx}.png", f"{base}.txt", label)
    else:
        print("  [skip] curve_tau_axis_*_explained：找不到中文字型")

    print("\n寫 Excel")
    write_excel_multi(mm, models, coverage, os.path.join(out, "tables.xlsx"), args)

    # 每個 fast 各一份單一 fast 的圖（和 report 的同名檔一樣），放在 multi_fast/<fast>/
    if args.per_fast != "none":
        for m in models:
            sub = os.path.join(out, safe_name(m))
            os.makedirs(sub, exist_ok=True)
            print(f"\n[{m}] 單一 fast 的圖 → {os.path.relpath(sub, HERE)}/")
            with as_fast_model(m):
                if args.per_fast == "explained":
                    fig1_tau_axis_set(mm[m]["all"], mm[m]["bench"], c, sub, sfx)
                else:  # full：和 report 完全同一套（主控台報表、全部的圖、tables.xlsx）
                    sub_args = argparse.Namespace(**{
                        **vars(args), "out": sub, "tau": "all", "with_band": False,
                        "with_random": False, "appendix_b": False, "boot": 0,
                        "source": None if args.mock else
                        f"{os.path.join(args.out, JOINED)} + <bench>/answers/{m}.jsonl"})
                    render_report(mm[m]["data"], sub_args, sub)
    print(f"\n完成 → {os.path.relpath(out, HERE)}/\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=OUT_DIR, help=f"輸出目錄（預設 {os.path.relpath(OUT_DIR, HERE)}/）")

    p_prep = sub.add_parser("prepare", help="把 bench 作答 + deep20 result 併成 joined.jsonl（需完整環境）")
    common(p_prep)
    p_prep.set_defaults(func=cmd_prepare)

    p_rep = sub.add_parser("report", help="算報表、畫 PNG、寫 Excel")
    common(p_rep)
    p_rep.add_argument("--mock", action="store_true", help="用示意資料跑（沒有真資料時先看版面）")
    p_rep.add_argument("--dark", action="store_true", help="輸出深色版 PNG（檔名加 _dark，不覆蓋淺色）")
    p_rep.add_argument("--appendix-b", action="store_true", help="多畫附錄 B 的圖 B1 / B2")
    p_rep.add_argument("--tau", choices=("all",) + TAU_VARIANTS + ("none",), default="all",
                       help="圖 1 / 圖 2 另存標了門檻 τ 的版本："
                            "curve = 記號標在曲線上、axis = 上方第二條 X 軸、"
                            "axis-conf = 上方同時標 τ 與 route_conf（只有圖 1）、"
                            "all = 全部都出（預設）、none = 不出")
    p_rep.add_argument("--with-band", action="store_true",
                       help="圖 1 / 圖 2 畫出 bootstrap 95%% 誤差帶（預設不畫）")
    p_rep.add_argument("--with-random", action="store_true",
                       help="圖 1 / 圖 2 畫出 Random 基準線（預設不畫；APGR 仍以它為分母）")
    p_rep.add_argument("--boot", type=int, default=None,
                       help="bootstrap 次數（預設：有 --with-band 時 1000，否則 0）")
    p_rep.set_defaults(func=cmd_report)

    p_mf = sub.add_parser("report_muli_fast", aliases=["report_multi_fast"],
                          help=f"powerful 固定（{ROUTE_POWERFUL}）、比好幾個 fast：疊圖、說明版、Excel")
    common(p_mf)
    p_mf.add_argument("--fast", nargs="+", metavar="MODEL",
                      help="要比的 fast（= <bench>/answers/<MODEL>.jsonl 的檔名），依序配色；"
                           "不給就自動找每個 bench 都有作答檔的 model")
    p_mf.add_argument("--mock", action="store_true", help="用示意資料跑（沒有真資料時先看版面）")
    p_mf.add_argument("--dark", action="store_true", help="輸出深色版 PNG（檔名加 _dark，不覆蓋淺色）")
    p_mf.add_argument("--per-fast", choices=("explained", "full", "none"), default="explained",
                      help="每個 fast 另外出一份單一 fast 的圖到 multi_fast/<fast>/："
                           "explained = fig1_main_tau_axis 與說明版（預設）、"
                           "full = 和 report 完全一樣的整套、none = 不出")
    p_mf.set_defaults(func=cmd_report_multi_fast)

    p_st = sub.add_parser("status", help="看資料齊不齊、缺的要跑什麼")
    common(p_st)
    p_st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
