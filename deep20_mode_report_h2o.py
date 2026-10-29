"""`python deep20_mode_report_h2o.py run` / `report` —— h2o-lightning 判 mode（chat / agent）準不準。

deep20.py 第一道多問了一題 mode（deep20.TRIAGE_QUESTIONS["mode"]）：使用者的 query 該走
chat mode（一則文字回覆就答完）還是 agent mode（要開檔、產出檔案、跑程式、用工具）。
這份報表量 H2O-Lightning-4B 這一題判得對不對。

---- 標準答案：資料夾層級 ----

    agent   f"{DATA_HOME}/skill-queries/" 底下所有檔案的 query
            （github.com/chihyu-wang-tsmc/skill-queries 整個 repo：19 個 skill × 30 題）
    chat    chat/ 底下所有檔案的 query

正類 = agent。「偵測率」就是 Recall：該走 agent 的 query 裡判成 agent 的比例；
「誤判率」FPR：該走 chat 的 query 裡被判成 agent 的比例。

和 deep20_coding_report.py 的 coding_gold 一樣是整個資料夾一個答案：skill-queries 裡個別一則
文字回覆就答得完的 query（例如 academy-guide 問「Artifacts 怎麼用？」）也照算 agent，
[M3] 逐檔案和 [M4] 判錯的例子看得出是不是這種情況。

chat/ 在哪裡：--chat-dir 沒給時依序找下面三個，用第一個存在的

    f"{DATA_HOME}/chat"
    f"{DATA_HOME}/skill-queries/chat"     放在 skill-queries 裡面也可以：agent 那邊會跳過這個資料夾
    <本檔所在目錄>/chat

---- 為什麼不直接讀 deep20_h2o_results/ ----

deep20_h2o.py run 也會問 mode，但 skill-queries 在 deep20 裡只從 all.jsonl 抽 80 題，chat/ 更不在
DATASETS 裡。所以這支自己問：只問 mode 這一題，不呼叫 LLM、不跑守門。題目是從 deep20 import 的
同一個物件，state 也和 deep20.ask() 第 1 步一樣是 HumanMessage(query)，答案和整條跑時一致。

---- 讀檔規則 ----

    *.json    skill-queries 的格式 {"scenarios": [{"user_query": ...}, ...]}；也接受最上層直接是
              list，或 {"queries" / "items" / "data" / "examples" / "rows": [...]}。元素可以是字串，
              或帶 user_query / query / prompt / question / text 其中一個欄位的 dict
    *.jsonl   每行一個上面那種元素
    子資料夾也會讀；隱藏資料夾（.git 之類）不讀。

    同一組（agent / chat）裡重複的 query 只問一次，算在第一個出現的檔案（*.json 先於 *.jsonl）。
    skill-queries 的 all.jsonl 就是各 skill 檔的合併，整份都重複，所以會印一行 [skip] 略過。
    同一則 query 同時出現在 agent 和 chat：標準答案互相矛盾，兩邊都拿掉，並印出來。

---- result ----

    deep20_mode_h2o_results/<agent|chat>__<檔名>.json      每個來源檔一份

已經問過的 query 會沿用（key 是 query 的 sha1），-n 加大時只問新的。mode 那題的措辭一改
（meta.question_sha 不同），那份 result 就整份重問 —— 不同措辭的答案不能混在一起算。
--refresh 一律重問。

---- 報表各節 ----

    [M0] 概況           檔案數、題數、誰問的、各份 result 是不是同一個措辭問的
    [M1] 整體           門檻 0.5 的混淆矩陣與 Precision / Recall / F1 / FPR / Accuracy，
                        每個檔案一樣重的 macro 版，以及門檻無關的 AUROC / AUPRC / TPR@FPR / ECE
    [M2] 門檻掃描       P(agent) ≥ τ 判 agent，τ = 0.1–0.9，另外標出正確率最高、F1 最高的門檻
    [M3] 逐檔案         每個 skill 檔／chat 檔的 agent 率、判對率（含 95% 信賴區間）、平均 P(agent)
    [M4] 判錯的例子     判對率不到 80% 的檔案各列 3 題（全部判錯的題目在 Excel）
    [M5] 門檻 trade-off X = 誤判率 FPR、Y = 偵測率 Recall 的曲線，點上標門檻 τ，另存一張 PNG

deep20.py 存的 mode 是 argmax；兩個選項時就是 P(agent) ≥ 0.5（剛好 0.5 時是 chat，實際上碰不到），
所以門檻 0.5 的數字就是 deep20 實際的判法。

---- 用法 ----

run 要先把 h2o 服務起起來（見 classifier_h2o.py 開頭）：

    python deep20_mode_report_h2o.py run -n 5             # 先小規模試：每個檔案只問前 5 題
    python deep20_mode_report_h2o.py run                  # 全部問（已經問過的沿用）
    python deep20_mode_report_h2o.py run --only pptx docx # 只問這幾個檔案（檔名，可省略 .json；--only chat 是整組）
    python deep20_mode_report_h2o.py run --chat-dir ~/somewhere/chat
    python deep20_mode_report_h2o.py report               # 印報表、寫 Excel、畫曲線
    python deep20_mode_report_h2o.py report --threshold 0.4
    python deep20_mode_report_h2o.py report --mock        # 還沒 run 時先看版面

run 會 import deep20 拿題目，而 import deep20 時 llm_models 就會讀 DASHSCOPE_API_KEY；這支不呼叫
LLM，所以沒設時自己補一個佔位值。report 只讀 result：不 import deep20、不需要 h2o 服務。

---- 產出 ----

    deep20_mode_report_h2o/tables.xlsx   每一節一個工作表，另有「判錯的題目」與「逐題資料」
                                         （--mock 時是 tables_mock.xlsx）
    deep20_mode_report_h2o/roc.png       [M5] 的門檻曲線（--mock / --dark 時檔名加 _mock / _dark）

門檻掃描、ROC 的點、指標和表格都沿用 deep20_coding_report.py / deep20_guard_report.py 的函式。
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime

import deep20_coding_report as coding
import deep20_guard_report as g
from deep20_guard_report import CI, F, P, Cell, Report

HERE = os.path.dirname(os.path.abspath(__file__))
# 和 deep20.py 的 DATA_HOME 同一個；這裡重寫一份，report 才不用 import deep20
DATA_HOME = os.path.join(os.path.expanduser("~"), "chihyu-wang-tsmc-datasets")
AGENT_DIR = os.path.join(DATA_HOME, "skill-queries")
CHAT_DIRS = (os.path.join(DATA_HOME, "chat"), os.path.join(AGENT_DIR, "chat"), os.path.join(HERE, "chat"))
RESULT_DIR = os.path.join(HERE, "deep20_mode_h2o_results")
OUT_DIR = os.path.join(HERE, "deep20_mode_report_h2o")

SCHEMA = 1
CHUNK = 40             # 每問完一批就存檔（同 deep20.CHUNK）：中斷時已問過的留得下來
MODE_THRESHOLD = 0.5   # deep20 取 argmax，兩個選項時等於 P(agent) ≥ 0.5
GROUPS = ("agent", "chat")
GOLD_ZH = {True: "agent", False: "chat"}
QUERY_KEYS = ("user_query", "query", "prompt", "question", "text")
LIST_KEYS = ("scenarios", "queries", "items", "data", "examples", "rows")


# ============================================================================
# 讀來源檔
# ============================================================================
def qkey(query):
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def _first_existing(*candidates):
    """第一個存在的資料夾；都不存在時回傳第一個，讓錯誤訊息指向預設位置。"""
    return next((c for c in candidates if os.path.isdir(c)), candidates[0])


def _query_of(item):
    """一個元素 → (query, id)；認不得的回 (None, None)。"""
    if isinstance(item, str):
        return (item, None) if item.strip() else (None, None)
    if isinstance(item, dict):
        for k in QUERY_KEYS:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v, item.get("id")
    return None, None


def read_queries(path):
    """一個檔案 → [(id, query)]，照檔案裡的順序。格式見檔頭「讀檔規則」。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            items = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, dict):
                items = next((data[k] for k in LIST_KEYS if isinstance(data.get(k), list)), None)
                if items is None:
                    raise ValueError(f"找不到 query 清單（要有 {' / '.join(LIST_KEYS)} 其中一個 list）")
            elif isinstance(data, list):
                items = data
            else:
                raise ValueError("最上層要是 list 或 dict")
    out = []
    for i, item in enumerate(items, 1):
        query, qid = _query_of(item)
        if query is not None:
            out.append((str(qid) if qid else f"{stem}-{i:03d}", query))
    if items and not out:
        raise ValueError(f"沒有一個元素帶 query（認得的欄位：{' / '.join(QUERY_KEYS)}）")
    return out


def _files(root, skip=None):
    """root 底下（含子資料夾）的 *.json 再 *.jsonl，各自照路徑排序；skip 資料夾裡的不算。

    glob 的 ** 本來就不進隱藏資料夾，所以 .git 裡的東西不會被讀到。
    """
    out = []
    for ext in ("json", "jsonl"):
        for p in sorted(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)):
            real = os.path.realpath(p)
            if skip and os.path.commonpath([real, skip]) == skip:
                continue
            out.append(p)
    return out


def collect(agent_dir, chat_dir, only=None):
    """兩個資料夾 → [{"group", "file", "items": [(id, query)]}]（已去重），並印出略過了什麼。"""
    chat_real = os.path.realpath(chat_dir)
    seen = {name: {} for name in GROUPS}  # query 的 key → 第一次出現的檔案
    sources = []
    for group, root, skip in (("agent", agent_dir, chat_real), ("chat", chat_dir, None)):
        if not os.path.isdir(root):
            raise SystemExit(f"{group} 的資料夾不存在：{root}"
                             + ("（用 --chat-dir 指定 chat/ 在哪裡）" if group == "chat" else
                                "（用 --agent-dir 指定 skill-queries/ 在哪裡）"))
        files = _files(root, skip)
        if not files:
            raise SystemExit(f"{root} 底下沒有 *.json / *.jsonl")
        for path in files:
            rel = os.path.relpath(path, root)
            try:
                items = read_queries(path)
            except (ValueError, OSError, json.JSONDecodeError) as e:
                print(f"[skip] {group}/{rel}：{type(e).__name__}: {e}")
                continue
            uniq = []
            for qid, query in items:
                k = qkey(query)
                if k not in seen[group]:
                    seen[group][k] = rel
                    uniq.append((qid, query))
            if not uniq:
                print(f"[skip] {group}/{rel}：{len(items)} 題全部和前面的檔案重複")
                continue
            if len(uniq) < len(items):
                print(f"  {group}/{rel}：{len(items) - len(uniq)} 題和前面的檔案重複，只問一次")
            sources.append({"group": group, "file": rel, "items": uniq})

    both = set(seen["agent"]) & set(seen["chat"])
    if both:
        print(f"[warn] {len(both)} 則 query 同時在 agent 和 chat 裡（標準答案互相矛盾），兩邊都拿掉：")
        for k in sorted(both)[:5]:
            print(f"    agent/{seen['agent'][k]}  ↔  chat/{seen['chat'][k]}")
        for s in sources:
            s["items"] = [(qid, q) for qid, q in s["items"] if qkey(q) not in both]
        sources = [s for s in sources if s["items"]]

    if only:
        sources = [s for s in sources if matches(s["group"], s["file"], only)]
        if not sources:
            raise SystemExit(f"--only 沒有對到任何檔案：{' '.join(only)}")
    return sources


def matches(group, rel, only):
    """--only 的名字對得到這個檔案嗎？認得 pptx、pptx.json、agent/pptx.json，或整組的 agent / chat。"""
    stem = os.path.splitext(rel)[0]
    return bool({group, rel, stem, f"{group}/{rel}", f"{group}/{stem}"} & set(only))


# ============================================================================
# result 檔
# ============================================================================
def result_path(group, rel):
    stem = rel[:-5] if rel.endswith(".json") else rel  # pptx.json → pptx；all.jsonl 保留副檔名
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", stem.replace(os.sep, "__"))
    return os.path.join(RESULT_DIR, f"{group}__{safe}.json")


def load_result(path):
    """讀一份 result。不在、壞掉、格式版本不符都回 None（當成還沒問）。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] {os.path.relpath(path, HERE)} 讀不起來（{type(e).__name__}: {e}），當成還沒問")
        return None
    if res.get("schema") != SCHEMA:
        print(f"[warn] {os.path.relpath(path, HERE)} 是舊格式（schema {res.get('schema')} != {SCHEMA}），重問")
        return None
    return res


def save_result(path, res):
    """先寫暫存檔再 rename：寫到一半被中斷時，原本那份還是完整的。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def question_dict(question):
    """Choice 物件 → 純 dict：存進 meta，報表才看得到是用哪個措辭問的，也拿來算 sha。"""
    return {"instructions": question.instructions, "criteria": dict(question.criteria)}


def question_sha(qd):
    raw = json.dumps(qd, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# run：只問 h2o 一題 mode
# ============================================================================
def run_source(s, ask, meta, args):
    """問完一個來源檔、存成一份 result。回傳這次新問了幾題。"""
    label = f"{s['group']}/{s['file']}"
    path = result_path(s["group"], s["file"])
    old = None if args.refresh else load_result(path)
    if old and old["meta"].get("question_sha") != meta["question_sha"]:
        print(f"  {label}：mode 的措辭改了（{old['meta'].get('question_sha')} → {meta['question_sha']}），整份重問")
        old = None
    rows = old["rows"] if old else {}
    items = s["items"][:args.n] if args.n else s["items"]
    todo = [(qid, q) for qid, q in items if qkey(q) not in rows]

    def build():
        return {
            "schema": SCHEMA,
            "meta": {**meta, "group": s["group"], "file": s["file"], "gold": s["group"],
                     "n_in_file": len(s["items"]), "n": args.n,
                     "updated": datetime.now().isoformat(timespec="seconds")},
            "rows": rows,
            "sample": [qkey(q) for _, q in items],  # 這次選中的題目；報表只看這些
        }

    if todo:
        print(f"問 h2o {label}：{len(todo)} 題（result 裡已有 {len(items) - len(todo)} 題）", flush=True)
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        for (qid, query), ans in zip(chunk, ask([q for _, q in chunk])):
            rows[qkey(query)] = {"id": qid, "query": query, **ans}
        save_result(path, build())
    save_result(path, build())
    got = Counter(rows[qkey(q)]["mode"] for _, q in items)
    print(f"  {label}：{len(items)} 題，agent {got['agent']}／chat {got['chat']}"
          f"（預期 {s['group']}；新問 {len(todo)}）→ {os.path.relpath(path, HERE)}", flush=True)
    return len(todo)


def cmd_run(args):
    # import deep20 時 llm_models.qwen() 就會讀 DASHSCOPE_API_KEY；這支只問 h2o，不呼叫 LLM
    os.environ.setdefault("DASHSCOPE_API_KEY", "unused-by-mode-report")
    import deep20
    from langchain_core.messages import HumanMessage

    from classifier_h2o import H2O_MODEL, H2OClassifier, choice_confidence

    chat_dir = args.chat_dir or _first_existing(*CHAT_DIRS)
    print(f"agent（標準答案）：{args.agent_dir}\nchat （標準答案）：{chat_dir}")
    sources = collect(args.agent_dir, chat_dir, args.only)
    n_q = Counter()
    for s in sources:
        n_q[s["group"]] += len(s["items"][:args.n] if args.n else s["items"])
    print(f"{len(sources)} 個檔案：agent {sum(s['group'] == 'agent' for s in sources)} 個 {n_q['agent']} 題、"
          f"chat {sum(s['group'] == 'chat' for s in sources)} 個 {n_q['chat']} 題"
          f"{f'（每個檔案只取前 {args.n} 題）' if args.n else ''}")

    question = deep20.TRIAGE_QUESTIONS["mode"]
    qd = question_dict(question)
    meta = {"classifier": H2O_MODEL, "question_sha": question_sha(qd), "question": qd}
    classifier = H2OClassifier(timeout=600)

    def ask(queries):
        # state 和 deep20.ask() 第 1 步一樣是 HumanMessage，答案才和整條跑時一致
        responses = classifier.batch(
            [{"state": HumanMessage(q), "questions": {"mode": question}} for q in queries],
            config={"max_concurrency": 8},
        )
        out = []
        for r in responses:
            a = r.choices["mode"]
            out.append({"mode": a.choice, "mode_p": a.probabilities["agent"],
                        "mode_conf": choice_confidence(list(a.probabilities.values()))})
        return out

    done = total_new = 0
    for s in sources:
        try:
            total_new += run_source(s, ask, meta, args)
            done += 1
        except Exception as e:  # noqa: BLE001 —— 一個檔案出錯不該讓其他檔案跟著停
            print(f"[skip] {s['group']}/{s['file']}：{type(e).__name__}: {e}")
    print(f"\n完成 {done}/{len(sources)} 個檔案，這次新問 {total_new} 題"
          f"　result 在 {os.path.relpath(RESULT_DIR, HERE)}/（報表：python {os.path.basename(__file__)} report）")


# ============================================================================
# report
# ============================================================================
def load_rows(results_dir, only=None):
    """讀 results_dir 裡每一份 result 的 sample → (逐題 rows, 每份的 meta)。"""
    rows, metas = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        res = load_result(path)
        if res is None:
            continue
        m = res["meta"]
        if only and not matches(m["group"], m["file"], only):
            continue
        metas.append(m)
        for k in res["sample"]:
            r = res["rows"].get(k)
            if r is None:
                continue
            rows.append({**r, "key": k, "group": m["group"], "file": m["file"],
                         "src": f"{m['group']}/{os.path.splitext(m['file'])[0]}",
                         "gold": m["group"] == "agent"})
    return rows, metas


def _scores(rows):
    return [r["mode_p"] for r in rows], [r["gold"] for r in rows]


def report_overview(rep, rows, metas, args):
    n_files = Counter(m["group"] for m in metas)
    n_rows = Counter(r["group"] for r in rows)
    print(f"\n{'=' * 78}")
    print(f"  deep20 mode（chat / agent）判斷報表　{'示意資料' if args.mock else '真實資料'}")
    print(f"  agent {n_files['agent']} 個檔案 {n_rows['agent']} 題／chat {n_files['chat']} 個檔案 {n_rows['chat']} 題"
          f"　正類 = agent，P(agent) ≥ {args.threshold} 就判成 agent")
    who = sorted({str(m.get("classifier")) for m in metas})
    print(f"  classifier：{'、'.join(who)}")
    print(f"{'=' * 78}")
    rep.section("[M0] 讀數字之前")
    if args.threshold != MODE_THRESHOLD:
        rep.note(f"⚠ 門檻用的是 {args.threshold}；deep20.py 存的 mode 是 argmax，等於門檻 {MODE_THRESHOLD}")
    if len(who) > 1:
        rep.note(f"⚠ 這些 result 不是同一顆 classifier 問的：{'、'.join(who)}")
    shas = defaultdict(list)
    for m in metas:
        shas[m.get("question_sha")].append(f"{m['group']}/{m['file']}")
    if len(shas) > 1:
        rep.note("⚠ 這些 result 是用不同措辭的 mode 問的，數字不能放在一起比；再跑一次 run 會讓它們一致：")
        for sha, files in shas.items():
            rep.note(f"    {sha}：{len(files)} 個檔案（{'、'.join(files[:4])}{'…' if len(files) > 4 else ''}）")
    elif shas and not args.mock:
        rep.note(f"mode 的措辭：question_sha {next(iter(shas))}（每份 result 的 meta.question 存著全文）")
    if not n_rows["agent"] or not n_rows["chat"]:
        rep.note(f"⚠ 只有一類題目（agent {n_rows['agent']}／chat {n_rows['chat']}）：FPR 或 Recall 算不出來，"
                 "[M2]、[M5] 會跳過")
    rep.note("標準答案是資料夾層級的：skill-queries/ 底下全算 agent、chat/ 底下全算 chat，"
             "個別例外的題目也照算（[M4] 看得出來）")


def report_overall(rep, rows, thr):
    rep.section(f"[M1] 整體　正類 = agent；P(agent) ≥ {thr} 就判成 agent")
    scores, gold = _scores(rows)
    tp, fn, fp, tn, m = coding.at(scores, gold, thr)
    rep.table("M1_整體", "混淆矩陣", ["", "判成 agent", "判成 chat"],
              [[f"該走 agent（{tp + fn}）", f"TP = {tp}", f"FN = {fn}"],
               [f"該走 chat（{fp + tn}）", f"FP = {fp}", f"TN = {tn}"]])

    # macro：每個檔案一樣重。agent 檔的平均 agent 率 = macro Recall，
    # chat 檔的平均 agent 率 = macro FPR；(Recall + 1 − FPR) / 2 = balanced accuracy
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)
    pos = [sum(r["mode_p"] >= thr for r in s) / len(s) for s in by_src.values() if s[0]["gold"]]
    neg = [sum(r["mode_p"] >= thr for r in s) / len(s) for s in by_src.values() if not s[0]["gold"]]
    mac_rec = sum(pos) / len(pos) if pos else None
    mac_fpr = sum(neg) / len(neg) if neg else None
    mac_acc = None if mac_rec is None or mac_fpr is None else (mac_rec + 1 - mac_fpr) / 2
    rep.table("M1_整體", "指標", ["", "Precision", "Recall（偵測率）", "FPR（誤判率）", "F1", "Accuracy"], [
        ["整體（每題一樣重）", F(m["precision"]), P(m["recall"]), P(m["fpr"]), F(m["f1"]), P(m["accuracy"])],
        [f"macro（每個檔案一樣重；agent {len(pos)}／chat {len(neg)} 個）",
         Cell(None, "—"), P(mac_rec), P(mac_fpr), Cell(None, "—"), P(mac_acc)],
    ])
    rep.note(f"Recall {tp}/{tp + fn}：該走 agent 的 query 抓到幾成　FPR {fp}/{fp + tn}：該走 chat 的被誤判成 agent 幾成\n"
             "macro 的 Accuracy 是 (macro Recall + 1 − macro FPR) / 2，也就是 balanced accuracy："
             "兩類題數差很多時看這個比較公平")

    t1, th1 = g.tpr_at_fpr(scores, gold, 0.01)
    t5, th5 = g.tpr_at_fpr(scores, gold, 0.05)

    def thr_cell(t):
        return F(t if t is not None and t != float("inf") else None, 2)

    rep.table("M1_整體", "排序指標（掃過所有門檻）",
              ["AUROC", "AUPRC", "AUPRC 隨機基線", "TPR@FPR1%", "門檻", "TPR@FPR5%", "門檻", "ECE"],
              [[F(g.auroc(scores, gold)), F(g.auprc(scores, gold)),
                F(sum(gold) / len(gold) if gold else None),
                P(t1), thr_cell(th1), P(t5), thr_cell(th5), F(g.ece(scores, gold), 4)]],
              [">"] * 8)


def report_sweep(rep, rows, thr):
    scores, gold = _scores(rows)
    if all(gold) or not any(gold):
        return
    rep.section("[M2] 門檻掃描　P(agent) ≥ τ 判成 agent；τ 越小越容易判 agent")
    (t_acc, _), (t_f1, _) = coding.best_thresholds(scores, gold)
    notes = defaultdict(list)
    notes[thr].append("← 目前" if thr == MODE_THRESHOLD else "← 這次用的")
    notes[t_acc].append("正確率最高")
    notes[t_f1].append("F1 最高")
    grid = sorted(set(notes) | {x for x in coding.SWEEP if all(abs(x - t) > 1e-9 for t in notes)})
    body = []
    for t in grid:
        tp, fn, fp, tn, m = coding.at(scores, gold, t)
        body.append([f"{t:.3f}", tp, fn, fp, tn, F(m["precision"]), P(m["recall"]), P(m["fpr"]),
                     F(m["f1"]), P(m["accuracy"]), "、".join(notes.get(t, []))])
    rep.table("M2_門檻掃描", "", ["門檻", "TP", "FN", "FP", "TN", "Precision", "Recall", "FPR", "F1",
                                "Accuracy", ""], body, ["<"] + [">"] * 9 + ["<"])
    rep.note("正確率最高、F1 最高是在這批題目上挑的，換一批題目不一定還是最好；"
             "要改 deep20 的判法得把 mode 從 argmax 改成比門檻")


def rate_row(name, sub, thr):
    n = len(sub)
    k = sum(r["mode_p"] >= thr for r in sub)
    gold = sub[0]["gold"]
    good = k if gold else n - k
    flag = "ok" if good / n >= 0.8 else "!!" if good / n < 0.6 else "~"
    return [name, n, GOLD_ZH[gold], P(k / n), P(good / n), CI(good, n),
            F(sum(r["mode_p"] for r in sub) / n), flag]


def report_files(rep, rows, thr):
    head = ["", "題數", "預期", "agent 率", "判對率", "判對率 95% CI", "平均 P(agent)", ""]
    aligns = ["<", ">", "<", ">", ">", ">", ">", "<"]
    rep.section(f"[M3] 逐檔案　agent 率 = P(agent) ≥ {thr} 的比例；判對率 = agent 檔看 agent 率、chat 檔看 1 − agent 率"
                "（ok：≥ 80%　~：60–80%　!!：< 60%）")
    body = [rate_row(f"全部 {GOLD_ZH[gold]}", sub, thr)
            for gold in (True, False) if (sub := [r for r in rows if r["gold"] is gold])]
    rep.table("M3_逐檔案", "兩組彙總", ["組", *head[1:]], body, aligns)

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)
    names = sorted(by_src, key=lambda s: (not by_src[s][0]["gold"], s))
    rep.table("M3_逐檔案", f"逐檔案（{len(names)} 個）", ["檔案", *head[1:]],
              [rate_row(s, by_src[s], thr) for s in names], aligns)
    return by_src


def report_examples(rep, by_src, thr):
    wrong = [r for sub in by_src.values() for r in sub if (r["mode_p"] >= thr) != r["gold"]]
    poor = sorted(s for s, sub in by_src.items()
                  if sum((r["mode_p"] >= thr) == r["gold"] for r in sub) / len(sub) < 0.8)
    rep.section(f"[M4] 判錯的例子（判對率 < 80% 的檔案各 3 題，錯得最離譜的先列；全部 {len(wrong)} 題判錯的在 Excel）")
    if not poor:
        rep.note("每個檔案的判對率都 ≥ 80%")
    body = []
    for s in poor:
        sub = sorted((r for r in wrong if r["src"] == s), key=lambda r: abs(r["mode_p"] - thr), reverse=True)
        for r in sub[:3]:
            text = " ".join(str(r.get("query", "")).split())
            body.append([s, GOLD_ZH[r["gold"]], F(r["mode_p"]), text[:80] + ("…" if len(text) > 80 else "")])
    if body:
        rep.table("M4_判錯例子", "", ["檔案", "預期", "P(agent)", "query 開頭"], body, ["<", "<", ">", "<"])
    rep.sheets["判錯的題目"].append(("全部判錯的題目", ["檔案", "預期", "P(agent)", "id", "query"], [
        [r["src"], GOLD_ZH[r["gold"]], F(r["mode_p"]), r.get("id", ""), r.get("query", "")]
        for r in sorted(wrong, key=lambda r: (r["src"], -abs(r["mode_p"] - thr)))]))


def report_roc(rep, rows, thr):
    rep.section(f"[M5] 門檻 trade-off　X = 誤判率 FPR、Y = 偵測率 Recall；目前的門檻是 {thr}")
    scores, gold = _scores(rows)
    n_pos, n_neg = sum(gold), len(gold) - sum(gold)
    if not n_pos or not n_neg:
        rep.note(f"只有一類題目（agent {n_pos}／chat {n_neg}），畫不出曲線")
        return
    body = []
    for t in sorted(set(coding.ROC_MARKS) | {thr}, reverse=True):
        tp, fn, fp, tn, m = coding.at(scores, gold, t)
        body.append([f"{t:.2f}", P(m["fpr"]), P(m["recall"]), tp, fn, fp, tn, F(m["precision"]), F(m["f1"]),
                     "← 目前" if abs(t - thr) < 1e-9 else ""])
    rep.table("M5_門檻曲線", "曲線上的幾個門檻（τ 由大到小；越小越容易判成 agent）",
              ["門檻 τ", "誤判率 FPR（X）", "偵測率 Recall（Y）", "TP", "FN", "FP", "TN", "Precision", "F1", ""],
              body, ["<"] + [">"] * 8 + ["<"])
    rep.note(f"AUROC {F(g.auroc(scores, gold))}：曲線下的面積，門檻無關，0.5 是亂猜（圖上的對角線）\n"
             "兩個軸都是在同一類題目裡算的比例，所以不受 agent／chat 題數比例影響\n"
             "整條曲線都是確切值：每題的 P(agent) 都存在 result 裡，換門檻只是重算，不用重問")


def fig_roc(rows, path, thr, dark=False):
    """[M5] 的圖：Recall（Y）vs FPR（X），點上標門檻 τ。畫得出來回傳 True。"""
    import matplotlib.pyplot as plt

    import deep20_report as d20r  # 調色盤、字型、坐標軸樣式和其他幾份報表的圖一致

    scores, gold = _scores(rows)
    pts = coding.roc_curve(scores, gold)
    if not pts:
        return False
    d20r.setup_font()
    c = d20r.PALETTE["dark" if dark else "light"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    d20r.style_axes(ax, c)
    auc = g.auroc(scores, gold)
    ax.plot([0, 1], [0, 1], color=c["muted"], lw=1, ls="--", zorder=2,
            label=d20r.t("亂猜（AUROC 0.5）", "Random (AUROC 0.5)"))
    ax.plot([x for x, _ in pts], [y for _, y in pts], color=c["router"], lw=2, zorder=3,
            label=d20r.t(f"mode 判斷（AUROC {auc:.3f}）", f"Chat / agent mode (AUROC {auc:.3f})"))
    for t in coding.ROC_MARKS:
        _, _, _, _, m = coding.at(scores, gold, t)
        ax.plot(m["fpr"], m["recall"], "o", ms=5, color=c["router"], mew=0, zorder=5)
        ax.annotate(f"{t:.1f}", (m["fpr"], m["recall"]), textcoords="offset points",
                    xytext=(7, -10), color=c["ink2"], fontsize=8)
    _, _, _, _, cur = coding.at(scores, gold, thr)
    ax.plot(cur["fpr"], cur["recall"], "*", ms=15, color=c["mark"], mew=0, zorder=6,
            label=d20r.t(f"目前的門檻 τ={thr:g}", f"Current threshold τ={thr:g}"))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    d20r.pct_axis(ax.xaxis)
    d20r.pct_axis(ax.yaxis)
    ax.set_xlabel(d20r.t("誤判率 FPR：該走 chat 卻判成 agent", "FPR: chat queries routed to agent"),
                  color=c["ink2"])
    ax.set_ylabel(d20r.t("偵測率 Recall：該走 agent 而判成 agent", "Recall: agent queries routed to agent"),
                  color=c["ink2"])
    ax.legend(loc="lower right", labelcolor=c["ink2"], fontsize=8.5)
    d20r._save(fig, path, c)
    return True


# ---- 示意資料（--mock）：還沒 run 時先看版面 ----
MOCK = [  # 檔案, 預期 agent?, P(agent) 的中心
    ("pptx.json", True, 0.92), ("docx.json", True, 0.9), ("xlsx.json", True, 0.93), ("pdf.json", True, 0.9),
    ("webapp-testing.json", True, 0.88), ("slack-gif-creator.json", True, 0.85),
    ("web-artifacts-builder.json", True, 0.85), ("canvas-design.json", True, 0.8),
    ("algorithmic-art.json", True, 0.8), ("mcp-builder.json", True, 0.75), ("skill-creator.json", True, 0.7),
    ("frontend-design.json", True, 0.7), ("theme-factory.json", True, 0.72), ("doc-coauthoring.json", True, 0.6),
    ("claude-api.json", True, 0.5), ("brand-guidelines.json", True, 0.45), ("internal-comms.json", True, 0.4),
    ("academy-guide.json", True, 0.3), ("discernment-nudge.json", True, 0.3),
    ("general.json", False, 0.1), ("math.json", False, 0.08), ("coding.json", False, 0.3),
    ("writing.json", False, 0.2), ("advice.json", False, 0.15),
]


def mock_rows(seed=20261029):
    rng = random.Random(seed)
    rows, metas = [], []
    for name, gold, mu in MOCK:
        group = "agent" if gold else "chat"
        metas.append({"group": group, "file": name, "classifier": "（示意）", "question_sha": "mock"})
        for i in range(30):
            p = rng.betavariate(max(mu, 0.02) * 4, max(1 - mu, 0.02) * 4)
            rows.append({"key": f"{name}:{i:03d}", "group": group, "file": name,
                         "src": f"{group}/{os.path.splitext(name)[0]}", "gold": gold,
                         "id": f"{os.path.splitext(name)[0]}-{i + 1:02d}", "query": f"（示意）{name} 第 {i + 1} 題",
                         "mode": "agent" if p >= 0.5 else "chat", "mode_p": p, "mode_conf": abs(2 * p - 1)})
    return rows, metas


def cmd_report(args):
    if args.mock:
        rows, metas = mock_rows()
    else:
        rows, metas = load_rows(args.results, args.only)
        if not rows:
            raise SystemExit(f"{os.path.relpath(args.results, HERE)}/ 裡沒有 result："
                             f"先跑 python {os.path.basename(__file__)} run（或用 --mock 看版面）")
    rep = Report()
    report_overview(rep, rows, metas, args)
    report_overall(rep, rows, args.threshold)
    report_sweep(rep, rows, args.threshold)
    by_src = report_files(rep, rows, args.threshold)
    report_examples(rep, by_src, args.threshold)
    report_roc(rep, rows, args.threshold)
    if not args.no_plot:
        os.makedirs(args.out, exist_ok=True)
        print("\n畫 [M5] 的門檻曲線")
        try:
            name = "roc" + ("_mock" if args.mock else "") + ("_dark" if args.dark else "") + ".png"
            if not fig_roc(rows, os.path.join(args.out, name), args.threshold, dark=args.dark):
                print("  [skip] 只有一類題目，畫不出曲線")
        except ImportError as e:
            print(f"  [skip] 畫不了圖（{e}）；需要 numpy + matplotlib")
    if not args.no_excel:
        os.makedirs(args.out, exist_ok=True)
        print("\n寫 Excel")
        for r in rows:
            r["mode_pred"] = "agent" if r["mode_p"] >= args.threshold else "chat"
            r["gold_zh"] = GOLD_ZH[r["gold"]]
        rep.write_excel(os.path.join(args.out, "tables_mock.xlsx" if args.mock else "tables.xlsx"), rows,
                        keep=("src", "gold_zh", "mode_pred", "mode", "mode_p", "mode_conf", "id", "key", "query"))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="只問 h2o 一題 mode，每個來源檔存一份 result")
    p.add_argument("--agent-dir", default=AGENT_DIR,
                   help=f"標準答案是 agent 的資料夾（預設 {AGENT_DIR}）")
    p.add_argument("--chat-dir", help="標準答案是 chat 的資料夾（預設依序找：" + "、".join(CHAT_DIRS) + "）")
    p.add_argument("--only", nargs="+", help="只問這些檔案（pptx、pptx.json、agent/pptx.json 都可以；agent / chat 是整組）")
    p.add_argument("-n", type=int, help="每個檔案只問前 n 題（先小規模試；之後加大會沿用已問過的）")
    p.add_argument("--refresh", action="store_true", help="丟掉既有的 result，全部重問")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="印報表、寫 Excel、畫門檻曲線")
    p.add_argument("--results", default=RESULT_DIR,
                   help=f"result 目錄（預設 {os.path.relpath(RESULT_DIR, HERE)}/）")
    p.add_argument("--out", default=OUT_DIR, help=f"輸出目錄（預設 {os.path.relpath(OUT_DIR, HERE)}/）")
    p.add_argument("--only", nargs="+", help="只看這些檔案（寫法同 run --only）")
    p.add_argument("--threshold", type=float, default=MODE_THRESHOLD,
                   help=f"P(agent) ≥ 它就判成 agent（預設 {MODE_THRESHOLD}，等於 deep20 的 argmax）")
    p.add_argument("--mock", action="store_true", help="用示意資料（還沒 run 時先看版面）")
    p.add_argument("--no-excel", action="store_true", help="只印主控台，不寫 Excel")
    p.add_argument("--no-plot", action="store_true", help="不畫 [M5] 的門檻曲線")
    p.add_argument("--dark", action="store_true", help="[M5] 的圖用深色底（檔名加 _dark）")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
