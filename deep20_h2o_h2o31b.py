"""`python deep20_h2o_h2o31b.py run` —— deep20.py 的整條流程，第一道問 H2O-Lightning-4B、第二道問 H2O-Lightning-31B。

    1. 輸入守門   H2O-Lightning-4B：model_route + coding + INPUT_BATTERY（和 deep20_h2o.py 一樣）
    2. 呼叫模型   和 deep20.py 一樣（model_route 選 fast / powerful）
    3. 輸出守門   H2O-Lightning-31B：拿模型的回覆問 OUTPUT_BATTERY（classifier_h2o_31b.py）

CLI 和 deep20_h2o.py 一樣（run / status / --only / --category / -n / --full / --refresh /
--isolate / --cn / --guard-policy），另外多三個：

    --no-reuse          不沿用 deep20_h2o.py 的結果，每一題都整條跑
    --h2o4b-url URL     第一道 4B 的 shim（預設 http://localhost:8741）
    --h2o31b-url URL    第二道 31B 的 shim（預設環境變數 H2O_LIGHTNING_31B_BASE_URL，沒設就是 8741）

    python deep20_h2o_h2o31b.py run --guard-policy measure        # 先沿用 deep20_h2o.py 跑過的第 1、2 步
    python deep20_h2o_h2o31b.py run --guard-policy measure --only AdvBench -n 20
    python deep20_h2o_h2o31b.py status
    python deep20_h2o_h2o31b.py run --guard-policy measure --refresh              # 第 3 步重問 31B
    python deep20_h2o_h2o31b.py run --guard-policy measure --refresh --no-reuse   # 三步全部重跑

-n / --full / --cn / --isolate 要和當初跑 deep20_h2o.py 時一樣：抽樣的 seed 只看資料集名稱，所以
同樣的 -n 抽到同一批題目；--cn / --isolate 決定讀哪個檔。不一樣的話，多出來的題目會整條跑。

---- 沿用 deep20_h2o.py 已經跑完的第 1、2 步 ----

第 1 步問的是同一顆 4B、同樣的題目、同樣的政策，結果就是 deep20_h2o_results/ 裡那一份；
第 2 步的模型回覆也存在那裡。所以每一題先去 deep20_h2o_results/<資料集>.json 找：

    找得到   抄第 1 步的欄位（in_* / noul / score / probs / coding / route / guard …）和第 2 步的
             called_model / model / reply，只問 31B 第 3 步。不需要 4B 服務，也不呼叫 LLM。
    找不到   照 deep20.py 整條跑：4B → LLM → 31B

deep20_h2o.py 所有資料集都跑完時，就只會問 31B。4B 不用開，GPU 可以整張給 31B。

整個資料集都不沿用（整條跑，會印一行 [reuse] 說明原因）的情況：
    - deep20_h2o_results/ 裡沒有這個資料集的 result（檔名規則同 deep20：--cn / --isolate 要一致）
    - result 讀不起來，或 schema 不同
    - meta.classifier 不是 4B
    - meta.guard_policy / meta.guard_thresholds 和這次的不同。政策不同，第 1 步的 guard
      （pass / review / block）就不同，哪些題目有呼叫模型、review 有沒有加保守提示也跟著不同，
      抄過來就錯了。舊版 deep20_h2o.py（還沒有 measure 的時候）沒記政策，也一律不沿用。

本檔自己的 result（deep20_h2o_h2o31b_results/）照 deep20 的規則續跑：已經在裡面的題目整個跳過，
所以中斷後重跑、或 -n 加大，都只補做差額。--refresh 只丟掉本檔的 result，4B 那份照樣沿用；
要連 LLM 一起重跑就再加 --no-reuse。

---- ⚠ 沿用的回覆只有前 400 字 ----

deep20.ask() 存 row["reply"] 時只留前 400 字（[:400]），完整的回覆沒有存下來。所以沿用的題目，
31B 第 3 步判的是「前 400 字」，而 deep20_h2o.py 當時 4B 判的是完整的回覆：

    回覆不到 400 字   就是完整的回覆，沒有差別
    回覆被截過        31B 只看得到開頭。拒答通常開頭就看得出來，但「先鋪陳、後面才給有害細節」
                      的回覆可能判得比較輕

這些題目在 row 裡標 output_truncated = True，meta.output_truncated 記數量，每個資料集跑完也會印。
要第 3 步看完整回覆，只能 --no-reuse 重新呼叫 LLM（回覆會和 4B 那次不完全一樣）。

---- 4B 和 31B 的 shim 預設都在 8741 ----

兩顆的 classifier 預設網址一樣，而且 4B 的 shim 不檢查請求的 model 欄位，問錯服務不一定會報錯。所以本檔
每一批都檢查回應的 model（shim 回的 served model 名稱）：第一道一定要是 4B、第二道一定要是 31B，
不是就整個停下來（前面跑完的批次已經存了）。要整條跑的題目，在呼叫 LLM 之前會先問 31B 一題確認，
免得 LLM 都呼叫完了才發現第二道問錯服務。

    全部沿用     只開 31B（serve.sh 或 run_default.sh，shim 在 8741），用預設網址就好
    要整條跑     兩個都要開、port 要不同，例如 4B 的 shim 改開 8742：
                     python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8001 --port 8742
                     python deep20_h2o_h2o31b.py run --guard-policy measure --h2o4b-url http://localhost:8742
                 （兩個 vLLM 的 port、--gpu-memory-utilization 也要錯開）

---- 怎麼換掉的 ----

和 deep20_h2o_jevk5.py 同一套手法（換掉 deep20 模組看到的名字），而且直接 import deep20_h2o，所以
measure 政策、--guard-policy 的處理、4B 的 result 目錄都只有一份：

    import deep20_h2o                              先套上 h2o 版（measure 政策、build_result 包一層）
    deep20.JevK5Classifier = H2OH2O31BGuards       cmd_run 的 JevK5Classifier(timeout=600)
    deep20.RESULT_DIR      = deep20_h2o_h2o31b_results/
    deep20.run_dataset     = 包一層：記下現在跑哪個資料集，跑完印沿用了幾題
    deep20.ask             = 包一層：能沿用的只問第 3 步，其餘交給原本的 deep20.ask
    deep20.build_result    = 再包一層：meta 記兩顆 classifier 和沿用的題數

deep20.ask() 三步共用一個 classifier。H2OH2O31BGuards.batch 看這批問的是不是 OUTPUT_BATTERY
決定問誰：第 3 步一定是 {"questions": OUTPUT_BATTERY}（就是那個物件），第 1 步是
triage_questions() 新組的 dict（--isolate 時是單題的 dict），用 is 分得乾淨。deep20.ask 一行沒改。

---- result ----

    deep20_results/              jevk5 → LLM → jevk5    （deep20.py）
    deep20_h2o_results/          4B    → LLM → 4B       （deep20_h2o.py）
    deep20_h2o_jevk5_results/    4B    → LLM → jevk5    （deep20_h2o_jevk5.py）
    deep20_h2o_h2o31b_results/   4B    → LLM → 31B      （本檔）

row 多兩個欄位（和 deep20_h2o_jevk5.py 一樣）：
    reused             True = 第 1、2 步是從 deep20_h2o_results/ 抄來的，本檔只問了 31B
    output_truncated   True = 第 3 步判的是被截在 400 字的回覆（只有 reused 的題目會是 True）
meta 多記 input_classifier / output_classifier / reused_from / reused / output_truncated；
guard_policy / guard_thresholds 由 deep20_h2o.build_result 記。

---- 要先起來的東西 ----

    31B                http://localhost:8741    第 3 步，一定要（見 classifier_h2o_31b.py）
    4B                 http://localhost:8741    只有找不到可沿用的題目時才會問，要和 31B 不同 port（見上面）
    DASHSCOPE_API_KEY  LLM                      同上只有整條跑時才呼叫，但 import deep20 時就會讀，一定要設
"""

import json
import os
import sys
from collections import Counter

import deep20
import deep20_h2o  # 先套上 h2o 版：measure 政策、build_result 記 guard_policy

from classifier_h2o import H2O_BASE_URL, H2O_MODEL, H2OClassifier
from classifier_h2o_31b import H2O_31B_MODEL, H2O31BClassifier

RESULT_DIR = os.path.join(deep20.HERE, "deep20_h2o_h2o31b_results")
# 沿用第 1、2 步的來源：deep20_h2o.py 寫的地方
SOURCE_DIR = deep20_h2o.RESULT_DIR

# --no-reuse 關掉（見 __main__）
REUSE = True

# 兩顆的 shim 網址（--h2o4b-url / --h2o31b-url，見 __main__）。31B 是 None 就照 H2O31BClassifier
# 的預設：環境變數 H2O_LIGHTNING_31B_BASE_URL，沒設就是 http://localhost:8741
H2O_4B_URL = H2O_BASE_URL
H2O_31B_URL = None

# deep20.ask() 存 row["reply"] 時只留前 400 字；長度到這裡的回覆就當成被截過
REPLY_KEPT = 400

# 沿用時不抄的欄位：
#   第 3 步的結果   out_* 加上下面這三個（output_guard / output_hazard 不是 out_ 開頭，要另外列），
#                   改成 31B 重問的
#   run_dataset 加的  prompt / coding_gold / route_gold，以這次 DATASETS 的值為準
#   本檔加的          reused / output_truncated，重新算
_NOT_COPIED = {"output_guard", "output_hazard", "blocked_at",
               "prompt", "coding_gold", "route_gold",
               "reused", "output_truncated"}


def _model_tail(name):
    """"h2oai/h2o-lightning-31b" → "h2o-lightning-31b"。shim 回的可能是全名、短名或本機路徑。"""
    return str(name).rstrip("/").rsplit("/", 1)[-1].lower()


class H2OH2O31BGuards:
    """cmd_run 的 JevK5Classifier(timeout=600) 換成這個：第一道 4B、第二道 31B。

    deep20.ask() 三步共用一個 classifier，batch() 照「問的是不是 OUTPUT_BATTERY」分給兩顆，
    並檢查回答的真的是那一顆（見檔頭「4B 和 31B 的 shim 預設都在 8741」）。
    另外帶著現在這個資料集的沿用狀態（run_dataset 設、ask 用），deep20 的 run_dataset → ask
    本來就會把 classifier 傳下去，不用另開全域。
    """

    def __init__(self, timeout):
        self.input = H2OClassifier(base_url=H2O_4B_URL, timeout=timeout)
        self.output = H2O31BClassifier(
            timeout=timeout, **({"base_url": H2O_31B_URL} if H2O_31B_URL else {}))
        self.output_checked = False  # 31B 回答過一次、model 對了就是 True
        self.dataset = None   # 現在跑的資料集名稱（prompt_key 要用）
        self.reuse = None     # 這個資料集可沿用的 4B rows（key → row）；None = 還沒載入
        self.counts = Counter()

    def batch(self, inputs, config=None):
        is_out = [inp["questions"] is deep20.OUTPUT_BATTERY for inp in inputs]
        if all(is_out):
            side, clf = "第二道", self.output
        elif not any(is_out):
            side, clf = "第一道", self.input
        else:
            raise ValueError("同一批混了第一道和第二道的問題，分不出要問 4B 還是 31B")
        responses = clf.batch(inputs, config=config)
        for r in responses:
            if _model_tail(r.model) != _model_tail(clf.model):
                # SystemExit 不會被 cmd_run 的 except Exception 吃掉：服務開錯了，後面每個資料集都會一樣錯
                raise SystemExit(
                    f"{side}應該是 {clf.model}，但 {clf.base_url} 回答的是 {r.model}。\n"
                    f"4B 和 31B 的 shim 預設都在 8741：全部沿用時只要開 31B；要整條跑時兩個要開在不同 port，"
                    f"用 --h2o4b-url / --h2o31b-url 指定（見檔頭）。前面跑完的批次已經存了。")
        if clf is self.output:
            self.output_checked = True
        return responses

    def check_output(self):
        """整條跑之前先問 31B 一題，確認第二道的服務是對的，免得 LLM 呼叫完才在第 3 步停下來。"""
        if not self.output_checked:
            self.batch([{"state": "Hello.", "questions": deep20.OUTPUT_BATTERY}])


def load_reusable(name, isolate):
    """deep20_h2o_results/ 裡這個資料集可沿用的 rows（key → row）。不能沿用就印原因、回 {}。"""
    path = os.path.join(SOURCE_DIR, os.path.basename(deep20.result_path(name, isolate)))
    where = os.path.relpath(path, deep20.HERE)
    if not os.path.exists(path):
        print(f"[reuse] {name}：沒有 {where}，整條跑（4B → LLM → 31B）")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[reuse] {name}：{where} 讀不起來（{type(e).__name__}: {e}），整條跑")
        return {}
    if res.get("schema") != deep20.SCHEMA:
        print(f"[reuse] {name}：{where} 是 schema {res.get('schema')}，不是 {deep20.SCHEMA}，整條跑")
        return {}
    meta = res.get("meta", {})
    if meta.get("classifier") != H2O_MODEL:
        print(f"[reuse] {name}：{where} 的第 1 步是 {meta.get('classifier')} 問的，不是 {H2O_MODEL}，整條跑")
        return {}
    want = deep20.POLICIES[deep20.GUARD_POLICY]
    if meta.get("guard_policy") != deep20.GUARD_POLICY or meta.get("guard_thresholds") != want:
        got = (f"{meta['guard_policy']} {meta.get('guard_thresholds')}" if meta.get("guard_policy")
               else "（沒記，舊版 deep20_h2o.py 跑的）")
        print(f"[reuse] {name}：{where} 的政策是 {got}，這次是 {deep20.GUARD_POLICY} {want}，"
              f"第 1 步的 guard 會不一樣，不能沿用 → 整條跑")
        return {}
    rows = res["rows"]
    # 同 deep20.load_result：JSON 的 key 一定是字串，severity 的等級還原成 int
    for row in rows.values():
        row["probs"] = {int(k): v for k, v in row["probs"].items()}
    return rows


def regrade(guards, src_rows):
    """沿用 4B 的第 1、2 步，只重問第 3 步（31B）。和 deep20.ask() 第 3 步、收尾同一套規則。"""
    rows = [{k: v for k, v in r.items() if not k.startswith("out_") and k not in _NOT_COPIED}
            for r in src_rows]
    graded = [i for i, r in enumerate(rows) if r["called_model"] and r["reply"].strip()]
    if graded:
        outs = guards.batch(
            [{"state": rows[i]["reply"], "questions": deep20.OUTPUT_BATTERY} for i in graded],
            config={"max_concurrency": 8},
        )
        for i, r in zip(graded, outs):
            rows[i].update(deep20.unpack_out(r))
    graded = set(graded)
    for i, row in enumerate(rows):
        row["reused"] = True
        row["output_truncated"] = i in graded and len(row["reply"]) >= REPLY_KEPT
        if row["guard"] == "block":
            row["blocked_at"] = "input"
        elif row.get("output_guard") == "block":
            row["blocked_at"] = "output"
        else:
            row["blocked_at"] = None
    return rows


_ask = deep20.ask


def ask(guards, prompts, isolate):
    """deep20.ask 外面包一層：deep20_h2o_results/ 已經有的題目跳過第 1、2 步，只問第 3 步。

    沒有的題目交給原本的 deep20.ask 整條跑（guards.batch 會把第 1 步分給 4B、第 3 步分給 31B）。
    回傳的 rows 順序和 prompts 一樣，run_dataset 是用 zip 接回去的。
    """
    if guards.reuse is None:
        # 第一次有題目要問時才載入：本檔 result 已經跑完的資料集不會走到這裡，也就不會印 [reuse]
        guards.reuse = load_reusable(guards.dataset, isolate) if REUSE else {}
    keys = [deep20.prompt_key(guards.dataset, p) for p in prompts]
    reuse = [i for i, k in enumerate(keys) if k in guards.reuse]
    fresh = [i for i, k in enumerate(keys) if k not in guards.reuse]
    rows = [None] * len(prompts)
    if fresh:
        guards.check_output()
        for i, row in zip(fresh, _ask(guards, [prompts[i] for i in fresh], isolate)):
            rows[i] = {**row, "reused": False, "output_truncated": False}
    if reuse:
        for i, row in zip(reuse, regrade(guards, [guards.reuse[keys[i]] for i in reuse])):
            rows[i] = row
    guards.counts.update(reused=len(reuse), fresh=len(fresh),
                         truncated=sum(rows[i]["output_truncated"] for i in reuse))
    return rows


_run_dataset = deep20.run_dataset


def run_dataset(d, args, guards):
    """deep20.run_dataset 外面包一層：記下現在跑哪個資料集，跑完印沿用了幾題。"""
    guards.dataset = d.name
    guards.reuse = None
    guards.counts = Counter()
    try:
        return _run_dataset(d, args, guards)
    finally:
        c = guards.counts
        if c["reused"] or c["fresh"]:
            print(f"  {d.name}：{c['reused']} 題沿用 4B 的第 1、2 步、只問 31B；"
                  f"{c['fresh']} 題整條跑（4B → LLM → 31B）", flush=True)
        if c["truncated"]:
            print(f"  ⚠ {d.name}：其中 {c['truncated']} 題的回覆在 {os.path.relpath(SOURCE_DIR, deep20.HERE)}/"
                  f" 只存了前 {REPLY_KEPT} 字，31B 判的是截過的回覆", flush=True)
        guards.reuse = None


def build_result(d, args, rows, prompts):
    """deep20_h2o.build_result（已記 guard_policy / guard_thresholds）外面再包一層。"""
    res = deep20_h2o.build_result(d, args, rows, prompts)
    meta = res["meta"]
    meta["classifier"] = f"{H2O_MODEL} (input) + {H2O_31B_MODEL} (output)"
    meta["input_classifier"] = H2O_MODEL
    meta["output_classifier"] = H2O_31B_MODEL
    meta["reused_from"] = os.path.relpath(SOURCE_DIR, deep20.HERE)
    meta["reused"] = sum(1 for r in rows.values() if r.get("reused"))
    meta["output_truncated"] = sum(1 for r in rows.values() if r.get("output_truncated"))
    return res


# cmd_run 裡的 JevK5Classifier(timeout=600) 會變成 H2OH2O31BGuards(timeout=600)
deep20.JevK5Classifier = H2OH2O31BGuards
deep20.RESULT_DIR = RESULT_DIR
deep20.run_dataset = run_dataset
deep20.ask = ask
deep20.build_result = build_result


def _take_flag(argv, flag):
    """從 argv 拿掉 flag（deep20.main() 的 parser 不認得它），回傳有沒有給。"""
    found = flag in argv
    while flag in argv:
        argv.remove(flag)
    return found


def _take_option(argv, flag):
    """從 argv 拿掉 `flag 值` 或 `flag=值`，回傳值（沒給就是 None）。"""
    for i, a in enumerate(argv):
        if a == flag:
            if i + 1 >= len(argv):
                sys.exit(f"{flag} 後面要接網址")
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if a.startswith(flag + "="):
            del argv[i]
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    name = deep20_h2o._take_guard_policy(sys.argv)
    if _take_flag(sys.argv, "--no-reuse"):
        REUSE = False
    H2O_4B_URL = _take_option(sys.argv, "--h2o4b-url") or H2O_4B_URL
    H2O_31B_URL = _take_option(sys.argv, "--h2o31b-url") or H2O_31B_URL
    if name is not None:
        if name not in deep20.POLICIES:
            sys.exit(f"沒有這個政策：{name!r}（可用：{', '.join(deep20.POLICIES)}）")
        deep20.GUARD_POLICY = name
    pol = deep20.POLICIES[deep20.GUARD_POLICY]
    url_31b = H2O31BClassifier(**({"base_url": H2O_31B_URL} if H2O_31B_URL else {})).base_url
    print(f"第一道：{H2O_MODEL}（{H2O_4B_URL}）　第二道：{H2O_31B_MODEL}（{url_31b}）"
          f"　result：{os.path.relpath(RESULT_DIR, deep20.HERE)}/")
    print(f"政策：{deep20.GUARD_POLICY}　action≥{pol['action_threshold']}"
          f"　review≥{pol['review_threshold']}　severity_block {pol['severity_block']}")
    if REUSE:
        print(f"沿用：{os.path.relpath(SOURCE_DIR, deep20.HERE)}/ 裡同政策的第 1、2 步，只補問 31B"
              f"（回覆只存了前 {REPLY_KEPT} 字；不要沿用就加 --no-reuse）")
    else:
        print("沿用：關閉（--no-reuse），每一題都整條跑")
    if H2O_4B_URL.rstrip("/") == url_31b.rstrip("/"):
        print("注意：4B 和 31B 是同一個網址，只有沿用的題目跑得了；要整條跑的題目會在第一道停下來"
              "（要整條跑就用 --h2o4b-url 指到另一個 port 的 4B）")
    if deep20.GUARD_POLICY == "measure":
        print("⚠ 量測政策：第一道幾乎不擋，整條跑的題目會把有害的 prompt 真的送進模型")
    deep20.main()
