"""各支程式共用的小工具：路徑與 .env、結果檔、跑資料集的迴圈、主控台表格與 Excel。"""

import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from metrics import wilson

REPO_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_DIR / "results"

# .env 裡的網址和 token（見 .env.example）；已經設在環境變數裡的優先
load_dotenv(REPO_DIR / ".env", override=False)

CHUNK = 40
"""每問完幾題存一次檔：中斷時最多白跑這麼多題。"""
SAMPLE_SEED = 1
"""-n 抽樣的 seed（再加上資料集名稱）；換掉就會抽到不同的題目。"""
TEXT_CHARS = 400
"""結果檔只存 prompt 和模型回覆的前幾個字。"""


# ============================================================================
# 路徑
# ============================================================================
def add_data_dir_arg(parser):
    parser.add_argument("--data-dir", help="datasets 目錄（預設 ../datasets，或環境變數 JEV_DATASETS_DIR）")


def data_dir(path=None):
    """題目所在的 datasets 目錄：--data-dir > 環境變數 JEV_DATASETS_DIR > ../datasets。"""
    path = Path(path or os.environ.get("JEV_DATASETS_DIR") or REPO_DIR.parent / "datasets")
    if not path.is_dir():
        sys.exit(f"找不到 datasets 目錄 {path}；用 --data-dir 或環境變數 JEV_DATASETS_DIR 指定")
    return path


def safe_name(name):
    """可以當檔名的字串。"""
    return re.sub(r"[^0-9A-Za-z._-]", "_", name)


def rel(path):
    """印出來給人看的路徑：在 repo 裡就印相對路徑。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(path)


# ============================================================================
# 結果檔
# ============================================================================
def prompt_key(dataset, prompt):
    """一題的 key：資料集名稱 + prompt 的 sha1（和舊的 deep20 相同）。"""
    return f"{dataset}:{hashlib.sha1(prompt.encode('utf-8')).hexdigest()}"


def read_json(path):
    """讀 JSON；檔案不在回 None，壞掉的印警告後也回 None（當成還沒跑）。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] {rel(path)} 讀不起來（{type(e).__name__}），當成還沒跑")
        return None


def write_json(path, obj):
    """先寫暫存檔再改名：寫到一半被中斷時，原本的檔案還是完整的。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path):
    """讀一題一行的 jsonl，回傳 {id: 紀錄}；同一題有多筆時以最後一筆為準（重跑會追加新紀錄）。"""
    records = {}
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records[record["id"]] = record
    return records


def pick(items, n, name):
    """-n 抽樣：n 是 None 就全部；否則固定 seed 洗牌後取前 n 題（照原本的順序排回來）。

    洗牌順序只和資料集名稱有關，所以 -n 加大時前面抽到的題目不變，已經問過的都會沿用。
    """
    if n is None or n >= len(items):
        return items
    order = list(range(len(items)))
    random.Random(f"{SAMPLE_SEED}-{name}").shuffle(order)
    return [items[i] for i in sorted(order[:n])]


def run_datasets(datasets, out_dir, root, ask, meta, n=None, refresh=False):
    """把每個資料集的題目交給 ask()，結果存成 out_dir/<資料集>.json。

    ask(prompts) 回傳一樣長的 list：每一格是那一題的結果 dict，失敗的那一題是 Exception
    （不存，下次重跑同一個指令會再問）。meta(d) 是要記進結果檔的資料集層級資訊。
    結果檔的格式：
        meta     資料集名稱、類別、標準答案、誰問的…
        rows     {key: 一題的結果}，問過的題目都在這裡；重跑時只問還不在裡面的
        sample   這次選到的題目的 key；報表只看這些
    """
    out_dir = Path(out_dir)
    total_new = total_failed = 0
    for d in datasets:
        try:
            items = pick(d.items(root), n, d.name)
        except (OSError, KeyError, ValueError) as e:
            print(f"[skip] {d.name}：讀不到題目（{type(e).__name__}: {e}）")
            continue
        if not items:
            print(f"[skip] {d.name}：一題都沒讀到（檔案佈局變了？）")
            continue
        path = out_dir / f"{safe_name(d.name)}.json"
        old = None if refresh else read_json(path)
        rows = old["rows"] if old else {}
        sample = [prompt_key(d.name, p) for p, _ in items]  # items() 已經把重複的 prompt 拿掉了
        todo = [(k, p, lang) for k, (p, lang) in zip(sample, items) if k not in rows]
        result = {"meta": {**meta(d), "n": n}, "rows": rows, "sample": sample}
        if todo:
            print(f"{d.name}：{len(todo)} 題要問（結果檔裡已經有 {len(sample) - len(todo)} 題）", flush=True)
        failed = 0
        # --refresh 時，新答案進來之前不寫檔：一開始就失敗的話，舊的結果檔還在
        keep_old_file = refresh
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            results = ask([p for _, p, _ in chunk])
            errors = [r for r in results if isinstance(r, Exception)]
            # 一整批（至少 5 題）全部失敗，幾乎一定是服務沒起來或設定錯了，繼續跑只是一直失敗；
            # 只剩一兩題又都失敗，比較可能是那幾題本身的問題，記下來繼續跑下一個資料集
            if len(errors) == len(chunk) >= 5:
                if not keep_old_file:
                    save_result(path, result)
                sys.exit(f"  這一批 {len(chunk)} 題全部失敗，停下來：{type(errors[0]).__name__}: {errors[0]}\n"
                         "  服務有沒有起來、.env 的網址和 token 對不對？已經問完的都存好了，"
                         "修好後重跑同一個指令就會接著問")
            for (k, p, lang), r in zip(chunk, results):
                if not isinstance(r, Exception):
                    rows[k] = {"prompt": p[:TEXT_CHARS], **({"lang": lang} if lang else {}), **r}
                    keep_old_file = False
            failed += len(errors)
            if not keep_old_file:
                save_result(path, result)
            print(f"  {d.name} {min(i + CHUNK, len(todo))}/{len(todo)}"
                  + (f"（失敗 {len(errors)}：{type(errors[0]).__name__}: {str(errors[0])[:120]}）" if errors else ""),
                  flush=True)
        if not keep_old_file:
            save_result(path, result)
        total_new += len(todo) - failed
        total_failed += failed
        print(f"  {d.name}：{len(sample)} 題 → {rel(path)}"
              + (f"　⚠ {failed} 題失敗，重跑同一個指令會再問" if failed else ""), flush=True)
    print(f"\n這次新問 {total_new} 題" + (f"，失敗 {total_failed} 題（重跑同一個指令會再問）" if total_failed else ""))


def save_result(path, result):
    result["meta"]["updated"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, result)


def load_results(result_dir, names=None):
    """讀 result_dir 底下的結果檔，回傳 [(meta, [row, ...])]；rows 只含 sample 裡、而且已經問到的題目。

    每個 row 會多 key 和 dataset 兩欄。names 給了就只讀這些資料集。
    """
    result_dir = Path(result_dir)
    out = []
    if not result_dir.is_dir():
        return out
    for path in sorted(result_dir.glob("*.json")):
        res = read_json(path)
        if not res or "meta" not in res:
            continue
        meta = res["meta"]
        if names and meta["dataset"] not in names:
            continue
        rows = [{**res["rows"][k], "key": k, "dataset": meta["dataset"]}
                for k in res["sample"] if k in res["rows"]]
        if rows:
            out.append((meta, rows))
    return out


def print_status(datasets, result_dir):
    """每個資料集的結果檔有幾題、什麼時候更新的。"""
    result_dir = Path(result_dir)
    print(f"\n{pad('資料集', 26)}{pad('類別', 15)}{pad('題數', 8, True)}{pad('已問', 8, True)}  更新時間")
    have = 0
    for d in datasets:
        res = read_json(result_dir / f"{safe_name(d.name)}.json")
        if not res:
            print(f"{pad(d.name, 26)}{pad(d.category, 15)}{pad('-', 8, True)}{pad('-', 8, True)}  （還沒跑）")
            continue
        have += 1
        done = sum(k in res["rows"] for k in set(res["sample"]))
        total = len(set(res["sample"]))
        print(f"{pad(d.name, 26)}{pad(d.category, 15)}{pad(total, 8, True)}{pad(done, 8, True)}"
              f"  {res['meta'].get('updated', '')}" + ("　⚠ 還有題目沒問到，重跑 run" if done < total else ""))
    print(f"\n{have}/{len(datasets)} 個資料集有結果（{rel(result_dir)}/）")


# ============================================================================
# 主控台表格與 Excel
# ============================================================================
class Cell:
    """表格的一格：主控台印 text，Excel 寫 value（數字留數字，才能排序、算）。"""

    __slots__ = ("value", "text")

    def __init__(self, value, text):
        self.value, self.text = value, text

    def __str__(self):
        return self.text


NA = Cell(None, "—")


def P(x, nd=1):
    """百分比。"""
    return NA if x is None else Cell(x, f"{x:.{nd}%}")


def F(x, nd=3):
    """小數。"""
    return NA if x is None else Cell(x, f"{x:.{nd}f}")


def KP(k, n):
    """「題數（比例）」。"""
    return Cell(k, f"{k}（{k / n:.1%}）" if n else str(k))


def CI(k, n):
    """k / n 的 Wilson 95% 信賴區間。"""
    lo, hi = wilson(k, n)
    return NA if lo is None else Cell(f"{lo:.4f}–{hi:.4f}", f"{lo:.1%}–{hi:.1%}")


def dw(s):
    """字串在終端機裡的寬度：中文字算兩格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def pad(s, n, right=False):
    """照顯示寬度補空白到 n 格；right=True 靠右。"""
    s = str(s)
    gap = max(0, n - dw(s))
    return " " * gap + s if right else s + " " * gap


class Report:
    """表格同時印到主控台、也收起來寫成 Excel（一個 sheet 放一節的表格）。"""

    def __init__(self):
        self.sheets = defaultdict(list)  # sheet 名 → [(標題, 欄名, 列)]

    @staticmethod
    def section(title):
        print(f"\n{title}")

    @staticmethod
    def note(text):
        for line in text.splitlines():
            print(f"  {line}")

    def table(self, sheet, title, header, rows, align=None):
        """align 是每一欄的對齊（"<" 靠左、">" 靠右）；不給就第一欄靠左、其餘靠右。"""
        align = align or "<" + ">" * (len(header) - 1)
        widths = [max(dw(r[i]) for r in [header, *rows]) for i in range(len(header))]
        if title:
            print(f"\n  {title}")

        def line(cells):
            return ("    " + "  ".join(pad(c, w, a == ">") for c, w, a in zip(cells, widths, align))).rstrip()

        print(line(header))
        print(("    " + "  ".join("-" * w for w in widths)).rstrip())
        for r in rows:
            print(line(r))
        self.sheets[sheet].append((title, header, rows))

    def write_excel(self, path, data_sheets=()):
        """每一節一個 sheet；data_sheets 是 [(sheet 名, 欄名, 列)]，放在最後（例如逐題資料）。"""
        from openpyxl import Workbook
        from openpyxl.styles import Font

        wb = Workbook()
        wb.remove(wb.active)
        bold = Font(bold=True)
        for sheet, tables in self.sheets.items():
            ws = wb.create_sheet(sheet[:31])
            r = 1
            for title, header, rows in tables:
                if title:
                    ws.cell(r, 1, title).font = bold
                    r += 1
                for j, h in enumerate(header, 1):
                    ws.cell(r, j, str(h)).font = bold
                for row in rows:
                    r += 1
                    for j, c in enumerate(row, 1):
                        ws.cell(r, j, c.value if isinstance(c, Cell) else c)
                r += 3
        for sheet, header, rows in data_sheets:
            ws = wb.create_sheet(sheet[:31])
            ws.append(list(header))
            for c in ws[1]:
                c.font = bold
            for row in rows:
                ws.append([json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in row])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
        print(f"  → {rel(path)}")
