"""`python deep20_h2o_gemma4-31b.py run` —— deep20.py 的整條流程，第一道問 H2O-Lightning-4B、第二道問 Gemma 4 31B。

    1. 輸入守門   H2O-Lightning-4B：model_route + coding + INPUT_BATTERY（和 deep20_h2o.py 一樣）
    2. 呼叫模型   和 deep20.py 一樣（model_route 選 fast / powerful）
    3. 輸出守門   gemma-4-31b-it（Google Gemini API）：拿模型的回覆問 OUTPUT_BATTERY

第二道用 classifier_gemini.py 的 GeminiClassifier，只把模型換成 gemma-4-31b-it（Gemini API 有提供
Gemma 4，見 https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api），所以題目文字、verbalized
機率的格式、重試規則都和 deep20_gemini.py 一樣，數字可以並排比。

CLI 和 deep20_h2o.py 一樣（run / status / --only / --category / -n / --full / --refresh /
--isolate / --cn / --guard-policy），多一個 --no-reuse：

    python deep20_h2o_gemma4-31b.py run --guard-policy measure        # 先沿用 deep20_h2o.py 跑過的第 1、2 步
    python deep20_h2o_gemma4-31b.py run --guard-policy measure --only AdvBench -n 20
    python deep20_h2o_gemma4-31b.py status
    python deep20_h2o_gemma4-31b.py run --guard-policy measure --refresh              # 第 3 步重問 Gemma
    python deep20_h2o_gemma4-31b.py run --guard-policy measure --refresh --no-reuse   # 三步全部重跑

-n / --full / --cn / --isolate 要和當初跑 deep20_h2o.py 時一樣：抽樣的 seed 只看資料集名稱，所以
同樣的 -n 抽到同一批題目；--cn / --isolate 決定讀哪個檔。不一樣的話，多出來的題目會整條跑。

---- 沿用 deep20_h2o.py 已經跑完的第 1、2 步 ----

第 1 步問的是同一顆 4B、同樣的題目、同樣的政策，結果就是 deep20_h2o_results/ 裡那一份；
第 2 步的模型回覆也存在那裡。所以每一題先去 deep20_h2o_results/<資料集>.json 找：

    找得到   抄第 1 步的欄位（in_* / noul / score / probs / coding / route / guard …）和第 2 步的
             called_model / model / reply，只問 Gemma 第 3 步。不需要 4B 服務，也不呼叫 LLM。
    找不到   照 deep20.py 整條跑：4B → LLM → Gemma

整個資料集都不沿用（整條跑，會印一行 [reuse] 說明原因）的情況：
    - deep20_h2o_results/ 裡沒有這個資料集的 result（檔名規則同 deep20：--cn / --isolate 要一致）
    - result 讀不起來，或 schema 不同
    - meta.classifier 不是 4B
    - meta.guard_policy / meta.guard_thresholds 和這次的不同。政策不同，第 1 步的 guard
      （pass / review / block）就不同，哪些題目有呼叫模型、review 有沒有加保守提示也跟著不同，
      抄過來就錯了。舊版 deep20_h2o.py（還沒有 measure 的時候）沒記政策，也一律不沿用。

本檔自己的 result（deep20_h2o_gemma4-31b_results/）照 deep20 的規則續跑：已經在裡面的題目整個跳過，
所以中斷後重跑、或 -n 加大，都只補做差額。--refresh 只丟掉本檔的 result，4B 那份照樣沿用；
要連 LLM 一起重跑就再加 --no-reuse。換了 GEMMA_MODEL / GEMMA_THINKING 之後要 --refresh，
不然新舊兩種設定的答案會混在一起（meta.output_classifier / output_thinking_level 記著是哪個設定）。

---- ⚠ 沿用的回覆只有前 400 字 ----

deep20.ask() 存 row["reply"] 時只留前 400 字（[:400]），完整的回覆沒有存下來。所以沿用的題目，
Gemma 第 3 步判的是「前 400 字」，而 deep20_h2o.py 當時 4B 判的是完整的回覆：

    回覆不到 400 字   就是完整的回覆，沒有差別
    回覆被截過        Gemma 只看得到開頭。拒答通常開頭就看得出來，但「先鋪陳、後面才給有害細節」
                      的回覆可能判得比較輕

這些題目在 row 裡標 output_truncated = True，meta.output_truncated 記數量，每個資料集跑完也會印。
要第 3 步看完整回覆，只能 --no-reuse 重新呼叫 LLM（回覆會和 4B 那次不完全一樣）。

---- Gemma 的設定 ----

    GEMMA_MODEL      gemma-4-31b-it
    GEMMA_THINKING   minimal。Gemma 4 的思考只有開（high）和關（minimal）兩種，classifier_gemini.py
                     預設的 low 它不認。只是要一個 JSON，關掉比較快也比較便宜（同 classifier_qwen.py）。

    JSON mode        classifier_gemini.py 會送 responseSchema 限定答案的格式。Gemini API 上的 Gemma
                     支不支援沒有文件寫（Gemma 3 不支援），所以開跑前那一題如果回「不支援 JSON mode」，
                     就自動改成只在 system prompt 要求 JSON（response_schema=False），印一行說明，
                     meta.output_json_schema 記成 False。答案還是用同一套規則檢查，格式不對會重試。

開始跑之前會先問 Gemma 一題（"Hello."）。key 錯、模型名稱錯、Gemma 不支援某個參數這類設定錯誤，
每一題都會一樣錯，所以這裡直接停下來、印出 Gemini API 的錯誤訊息，不讓每個資料集各 [skip] 一次。

成本：Gemini API 一次請求只回答一個問題，第 3 步是 3 個問題，所以沿用的題目一題 3 次 Gemma 呼叫。
預設 62 個資料集 × 80 題 ≈ 4960 題，大約 1.5 萬次。429（超過配額）會自動退避重試，只是會變慢。

---- Gemini API 不肯讀的回覆 ----

classifier_gemini.py 已經把能關的安全過濾全部關掉，但還是有少數內容 Gemini API 一律不讀
（GeminiBlockedError），一批 40 則回覆裡有一則被擋，整批就失敗。做法和 deep20_gemini.py 一樣：

    1. 整批失敗在被擋 → 這批改成一則一則問
    2. 還是被擋的那題不寫進 result 的 rows，改記在 meta.gemini_blocked
       （{prompt key: {"prompt": 前 200 字, "reason": 哪一題被擋、原因}}），並印一行 [blocked]

報表會略過 sample 裡沒有 row 的題目，所以這幾題連第一道也不算進指標（和 4B 版的報表比第一道時，
題數會少這幾題）。被擋的回覆通常是最有害的那些，看 meta.gemini_blocked 有幾題就知道影響多大。
重跑時這幾題會再問一次（它們不在 rows 裡），通常一樣被擋，代價只是每題幾次請求。

整條跑的題目如果在第 3 步被擋，那一批（最多 40 題）的模型呼叫都已經發生過，一題一題重問時
整批會再呼叫一次模型（同 deep20_gemini.py）。全部沿用時不會有這個問題。

---- 怎麼換掉的 ----

和 deep20_h2o_jevk5.py / deep20_h2o_h2o31b.py 同一套手法（換掉 deep20 模組看到的名字），而且直接
import deep20_h2o，所以 measure 政策、--guard-policy 的處理、4B 的 result 目錄都只有一份：

    import deep20_h2o                              先套上 h2o 版（measure 政策、build_result 包一層）
    deep20.JevK5Classifier = H2OGemmaGuards        cmd_run 的 JevK5Classifier(timeout=600)
    deep20.RESULT_DIR      = deep20_h2o_gemma4-31b_results/
    deep20.run_dataset     = 包一層：記下現在跑哪個資料集，跑完印沿用了幾題
    deep20.ask             = 包一層：能沿用的只問第 3 步，其餘交給原本的 deep20.ask；處理被擋的回覆
    deep20.build_result    = 再包一層：拿掉被擋的 row，meta 記兩顆 classifier 和沿用的題數

deep20.ask() 三步共用一個 classifier。H2OGemmaGuards.batch 看這批問的是不是 OUTPUT_BATTERY
決定問誰：第 3 步一定是 {"questions": OUTPUT_BATTERY}（就是那個物件），第 1 步是
triage_questions() 新組的 dict（--isolate 時是單題的 dict），用 is 分得乾淨。deep20.ask 一行沒改。

第 1 步的回應會檢查 model 是不是 4B：h2o-lightning-31b 的 shim 預設也在 8741，整條跑時如果那裡
開的是 31B，4B 的問題會被 31B 回答而且不報錯，所以不是 4B 就整個停下來。

---- result ----

    deep20_results/                 jevk5 → LLM → jevk5    （deep20.py）
    deep20_h2o_results/             4B    → LLM → 4B       （deep20_h2o.py）
    deep20_h2o_jevk5_results/       4B    → LLM → jevk5    （deep20_h2o_jevk5.py）
    deep20_h2o_h2o31b_results/      4B    → LLM → 31B      （deep20_h2o_h2o31b.py）
    deep20_h2o_gemma4-31b_results/  4B    → LLM → Gemma    （本檔）

row 多兩個欄位（和 deep20_h2o_jevk5.py 一樣）：
    reused             True = 第 1、2 步是從 deep20_h2o_results/ 抄來的，本檔只問了 Gemma
    output_truncated   True = 第 3 步判的是被截在 400 字的回覆（只有 reused 的題目會是 True）
meta 多記 input_classifier / output_classifier / output_thinking_level / output_json_schema /
reused_from / reused / output_truncated / gemini_blocked；guard_policy / guard_thresholds 由 deep20_h2o.build_result 記。

---- 要先設好的 ----

    GEMINI_API_KEY     第 3 步（和 deep20_gemini.py、llm_models.py 的 gemini() 用同一個 key）
    4B                 http://localhost:8741    只有找不到可沿用的題目時才會問（見 classifier_h2o.py）
    DASHSCOPE_API_KEY  LLM                      同上只有整條跑時才呼叫，但 import deep20 時就會讀，一定要設
"""

import json
import os
import sys
from collections import Counter

import deep20
import deep20_h2o  # 先套上 h2o 版：measure 政策、build_result 記 guard_policy
from langchain_typesafe.client import TypeSafeAPIConnectionError, TypeSafeAPIError

from classifier_gemini import GeminiBlockedError, GeminiClassifier, _api_message
from classifier_h2o import H2O_MODEL, H2OClassifier

GEMMA_MODEL = "gemma-4-31b-it"
# Gemma 4 只有 "minimal"（關）和 "high"（開），見檔頭
GEMMA_THINKING = "minimal"
# 送不送 responseSchema；Gemma 回「不支援 JSON mode」時 check_output 會改成 False（見檔頭）
GEMMA_JSON_SCHEMA = True

RESULT_DIR = os.path.join(deep20.HERE, "deep20_h2o_gemma4-31b_results")
# 沿用第 1、2 步的來源：deep20_h2o.py 寫的地方
SOURCE_DIR = deep20_h2o.RESULT_DIR

# --no-reuse 關掉（見 __main__）
REUSE = True

# deep20.ask() 存 row["reply"] 時只留前 400 字；長度到這裡的回覆就當成被截過
REPLY_KEPT = 400

# row 裡用這個欄位標記「Gemini API 不肯讀」，build_result 存檔前把這些 row 拿掉、改記在 meta
# （和 deep20_gemini.py 同一個名字）
BLOCKED = "gemini_blocked"

# 沿用時不抄的欄位：
#   第 3 步的結果   out_* 加上下面這三個（output_guard / output_hazard 不是 out_ 開頭，要另外列），
#                   改成 Gemma 重問的
#   run_dataset 加的  prompt / coding_gold / route_gold，以這次 DATASETS 的值為準
#   本檔加的          reused / output_truncated，重新算
_NOT_COPIED = {"output_guard", "output_hazard", "blocked_at",
               "prompt", "coding_gold", "route_gold",
               "reused", "output_truncated"}


def _model_tail(name):
    """"h2oai/h2o-lightning-4b" → "h2o-lightning-4b"。shim 回的可能是全名、短名或本機路徑。"""
    return str(name).rstrip("/").rsplit("/", 1)[-1].lower()


class H2OGemmaGuards:
    """cmd_run 的 JevK5Classifier(timeout=600) 換成這個：第一道 4B、第二道 Gemma。

    deep20.ask() 三步共用一個 classifier，batch() 照「問的是不是 OUTPUT_BATTERY」分給兩顆。
    另外帶著現在這個資料集的沿用狀態（run_dataset 設、ask 用），deep20 的 run_dataset → ask
    本來就會把 classifier 傳下去，不用另開全域。
    """

    def __init__(self, timeout):
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("第二道要 GEMINI_API_KEY（和 deep20_gemini.py 同一個 key），先設好再跑")
        self.timeout = timeout
        self.input = H2OClassifier(timeout=timeout)
        self.output = self._gemma()
        self.output_checked = False  # 開跑前問過 Gemma 一題、設定沒問題就是 True
        self.dataset = None   # 現在跑的資料集名稱（prompt_key 要用）
        self.reuse = None     # 這個資料集可沿用的 4B rows（key → row）；None = 還沒載入
        self.counts = Counter()

    def batch(self, inputs, config=None):
        is_out = [inp["questions"] is deep20.OUTPUT_BATTERY for inp in inputs]
        if all(is_out):
            return self.output.batch(inputs, config=config)
        if any(is_out):
            raise ValueError("同一批混了第一道和第二道的問題，分不出要問 4B 還是 Gemma")
        responses = self.input.batch(inputs, config=config)
        for r in responses:
            if _model_tail(r.model) != _model_tail(H2O_MODEL):
                # SystemExit 不會被 cmd_run 的 except Exception 吃掉：服務開錯了，後面每個資料集都會一樣錯
                raise SystemExit(
                    f"第一道應該是 {H2O_MODEL}，但 {self.input.base_url} 回答的是 {r.model}。\n"
                    "h2o-lightning-31b 的 shim 預設也在 8741：要整條跑的題目需要 4B，把 8741 換回 4B 再跑"
                    "（前面跑完的批次已經存了）。")
        return responses

    def _gemma(self):
        return GeminiClassifier(model=GEMMA_MODEL, thinking_level=GEMMA_THINKING,
                                response_schema=GEMMA_JSON_SCHEMA, timeout=self.timeout)

    def check_output(self):
        """第一次要問 Gemma 之前先問一題，設定錯了（key、模型名稱、不支援的參數）就整個停下來。

        回「不支援 JSON mode」時改成不送 responseSchema 再試一次（見檔頭「Gemma 的設定」）。
        """
        global GEMMA_JSON_SCHEMA
        if self.output_checked:
            return
        try:
            self.output.batch([{"state": "Hello.", "questions": deep20.OUTPUT_BATTERY}])
        except GeminiBlockedError:
            pass  # "Hello." 被擋不是設定的問題；真的題目照常問
        except TypeSafeAPIError as e:  # 400 / 401 / 403 / 404 或重試用完
            detail = _api_message(getattr(e, "body", None)) or str(e)
            if (GEMMA_JSON_SCHEMA and getattr(e, "status_code", None) == 400
                    and any(w in detail.lower() for w in ("json", "schema", "mime"))):
                print(f"第二道：{GEMMA_MODEL} 不接受 JSON mode（{detail}），"
                      "改成只在 system prompt 要求 JSON（meta.output_json_schema = False）", flush=True)
                GEMMA_JSON_SCHEMA = False
                self.output = self._gemma()
                return self.check_output()
            raise SystemExit(f"第二道 {GEMMA_MODEL}（thinking {GEMMA_THINKING}）的請求失敗：{detail}\n"
                             "檢查 GEMINI_API_KEY、GEMMA_MODEL、GEMMA_THINKING（見檔頭「Gemma 的設定」）")
        except TypeSafeAPIConnectionError as e:
            raise SystemExit(f"連不上 Gemini API（{type(e).__name__}: {e}）")
        self.output_checked = True


def _blocked_row(prompt, error):
    print(f"  [blocked] Gemini API 不肯讓 Gemma 讀這題的回覆（{error.question}：{error.reason}）："
          f"{prompt[:60]!r}", flush=True)
    return {BLOCKED: f"output {error.question}: {error.reason}"}


def load_reusable(name, isolate):
    """deep20_h2o_results/ 裡這個資料集可沿用的 rows（key → row）。不能沿用就印原因、回 {}。"""
    path = os.path.join(SOURCE_DIR, os.path.basename(deep20.result_path(name, isolate)))
    where = os.path.relpath(path, deep20.HERE)
    if not os.path.exists(path):
        print(f"[reuse] {name}：沒有 {where}，整條跑（4B → LLM → Gemma）")
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


def grade_replies(guards, replies):
    """一串回覆問 Gemma 第 3 步，回傳每則的回應；Gemini API 不肯讀的那則換成它的 GeminiBlockedError。"""
    def one_batch(texts):
        return guards.batch([{"state": t, "questions": deep20.OUTPUT_BATTERY} for t in texts],
                            config={"max_concurrency": 8})
    try:
        return one_batch(replies)
    except GeminiBlockedError as error:
        if len(replies) == 1:
            return [error]
    print(f"  [blocked] 這批 {len(replies)} 則回覆裡有 Gemini API 不肯讀的，改成一則一則問", flush=True)
    out = []
    for text in replies:
        try:
            out += one_batch([text])
        except GeminiBlockedError as error:
            out.append(error)
    return out


def regrade(guards, src_rows):
    """沿用 4B 的第 1、2 步，只重問第 3 步（Gemma）。和 deep20.ask() 第 3 步、收尾同一套規則。"""
    rows = [{k: v for k, v in r.items() if not k.startswith("out_") and k not in _NOT_COPIED}
            for r in src_rows]
    graded = [i for i, r in enumerate(rows) if r["called_model"] and r["reply"].strip()]
    blocked = set()
    if graded:
        for i, r in zip(graded, grade_replies(guards, [rows[i]["reply"] for i in graded])):
            if isinstance(r, GeminiBlockedError):
                rows[i] = _blocked_row(src_rows[i].get("prompt", ""), r)
                blocked.add(i)
            else:
                rows[i].update(deep20.unpack_out(r))
    graded = set(graded) - blocked
    for i, row in enumerate(rows):
        if i in blocked:
            continue
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


def ask_fresh(guards, prompts, isolate):
    """原本的 deep20.ask 整條跑；第 3 步有回覆被 Gemini API 擋下時改成一題一題跑（同 deep20_gemini.py）。"""
    try:
        return _ask(guards, prompts, isolate)
    except GeminiBlockedError as error:
        if len(prompts) == 1:
            return [_blocked_row(prompts[0], error)]
    print(f"  [blocked] 這批 {len(prompts)} 題裡有 Gemini API 不肯讀的回覆，改成一題一題跑"
          "（這批的模型會再呼叫一次）", flush=True)
    rows = []
    for p in prompts:
        try:
            rows += _ask(guards, [p], isolate)
        except GeminiBlockedError as error:
            rows.append(_blocked_row(p, error))
    return rows


def ask(guards, prompts, isolate):
    """deep20.ask 外面包一層：deep20_h2o_results/ 已經有的題目跳過第 1、2 步，只問第 3 步。

    沒有的題目交給原本的 deep20.ask 整條跑（guards.batch 會把第 1 步分給 4B、第 3 步分給 Gemma）。
    回傳的 rows 順序和 prompts 一樣，run_dataset 是用 zip 接回去的。
    """
    guards.check_output()
    if guards.reuse is None:
        # 第一次有題目要問時才載入：本檔 result 已經跑完的資料集不會走到這裡，也就不會印 [reuse]
        guards.reuse = load_reusable(guards.dataset, isolate) if REUSE else {}
    keys = [deep20.prompt_key(guards.dataset, p) for p in prompts]
    reuse = [i for i, k in enumerate(keys) if k in guards.reuse]
    fresh = [i for i, k in enumerate(keys) if k not in guards.reuse]
    rows = [None] * len(prompts)
    if fresh:
        for i, row in zip(fresh, ask_fresh(guards, [prompts[i] for i in fresh], isolate)):
            rows[i] = row if BLOCKED in row else {**row, "reused": False, "output_truncated": False}
    if reuse:
        for i, row in zip(reuse, regrade(guards, [guards.reuse[keys[i]] for i in reuse])):
            rows[i] = row
    guards.counts.update(reused=sum(BLOCKED not in rows[i] for i in reuse),
                         fresh=sum(BLOCKED not in rows[i] for i in fresh),
                         blocked=sum(BLOCKED in r for r in rows),
                         truncated=sum(bool(rows[i].get("output_truncated")) for i in reuse))
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
        if c["reused"] or c["fresh"] or c["blocked"]:
            print(f"  {d.name}：{c['reused']} 題沿用 4B 的第 1、2 步、只問 Gemma；"
                  f"{c['fresh']} 題整條跑（4B → LLM → Gemma）"
                  + (f"；{c['blocked']} 題的回覆 Gemini API 不肯讀，記在 meta.gemini_blocked" if c["blocked"] else ""),
                  flush=True)
        if c["truncated"]:
            print(f"  ⚠ {d.name}：其中 {c['truncated']} 題的回覆在 {os.path.relpath(SOURCE_DIR, deep20.HERE)}/"
                  f" 只存了前 {REPLY_KEPT} 字，Gemma 判的是截過的回覆", flush=True)
        guards.reuse = None


def build_result(d, args, rows, prompts):
    """deep20_h2o.build_result（已記 guard_policy / guard_thresholds）外面再包一層：拿掉被擋的 row。"""
    blocked = {k: r for k, r in rows.items() if BLOCKED in r}
    kept = {k: r for k, r in rows.items() if k not in blocked}
    res = deep20_h2o.build_result(d, args, kept, prompts)
    meta = res["meta"]
    meta["classifier"] = f"{H2O_MODEL} (input) + {GEMMA_MODEL} (output)"
    meta["input_classifier"] = H2O_MODEL
    meta["output_classifier"] = GEMMA_MODEL
    meta["output_thinking_level"] = GEMMA_THINKING
    meta["output_json_schema"] = GEMMA_JSON_SCHEMA
    meta["reused_from"] = os.path.relpath(SOURCE_DIR, deep20.HERE)
    meta["reused"] = sum(1 for r in kept.values() if r.get("reused"))
    meta["output_truncated"] = sum(1 for r in kept.values() if r.get("output_truncated"))
    meta["gemini_blocked"] = {
        k: {"prompt": r.get("prompt", "")[:200], "reason": r[BLOCKED]}
        for k, r in blocked.items()
    }
    return res


# cmd_run 裡的 JevK5Classifier(timeout=600) 會變成 H2OGemmaGuards(timeout=600)
deep20.JevK5Classifier = H2OGemmaGuards
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


if __name__ == "__main__":
    name = deep20_h2o._take_guard_policy(sys.argv)
    if _take_flag(sys.argv, "--no-reuse"):
        REUSE = False
    if name is not None:
        if name not in deep20.POLICIES:
            sys.exit(f"沒有這個政策：{name!r}（可用：{', '.join(deep20.POLICIES)}）")
        deep20.GUARD_POLICY = name
    pol = deep20.POLICIES[deep20.GUARD_POLICY]
    print(f"第一道：{H2O_MODEL}　第二道：{GEMMA_MODEL}（Gemini API，thinking {GEMMA_THINKING}）"
          f"　result：{os.path.relpath(RESULT_DIR, deep20.HERE)}/")
    print(f"政策：{deep20.GUARD_POLICY}　action≥{pol['action_threshold']}"
          f"　review≥{pol['review_threshold']}　severity_block {pol['severity_block']}")
    if REUSE:
        print(f"沿用：{os.path.relpath(SOURCE_DIR, deep20.HERE)}/ 裡同政策的第 1、2 步，只補問 Gemma"
              f"（回覆只存了前 {REPLY_KEPT} 字；不要沿用就加 --no-reuse）")
    else:
        print("沿用：關閉（--no-reuse），每一題都整條跑")
    if deep20.GUARD_POLICY == "measure":
        print("⚠ 量測政策：第一道幾乎不擋，整條跑的題目會把有害的 prompt 真的送進模型")
    deep20.main()
