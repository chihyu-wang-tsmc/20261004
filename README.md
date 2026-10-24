# 20261004

jevk5（TypeSafe System-One 分類器）評估工作，2026-10-02 ~ 10-04 三天內改動的程式碼。

只收程式碼。快取（`deep11_cache.jsonl`，6.5 MB）和 benchmark 衍生資料（`ai2arc/` `gpqa/`
`gsm8k/` `aime/`）沒有放進來，要重現請自己跑第一階段。

## 這三天在做什麼

jevk5 對每個使用者請求同時回答四個問題，這批程式在評估那四個答案值不值得信：

| 問題 | 型別 | production 的用法 |
| --- | --- | --- |
| `refuse` | Noul | P(yes) ≥ 0.70 就擋下請求 |
| `severity` | Score | 期望值 ≥ 2.5 時把 refuse 門檻放寬到 0.40 |
| `coding` | Noul | P(yes) ≥ 0.5 就掛上 coding 的系統提示 |
| `model_route` | Choice | argmax 決定送 fast 還是 powerful model |

三天的主軸是**把門檻從「寫死的常數」變成「可以掃描的變數」**，再進一步拆成兩階段、加上校準分析。

## 檔案

### 評估主線

| 檔案 | 說明 |
| --- | --- |
| `deep9.py` | production 的 agent 本體。四個問題的定義、四個門檻常數（`REFUSE_THRESHOLD` 0.70、`SEVERE_SCORE` 2.5、`SEVERE_REFUSE_THRESHOLD` 0.40、`CODING_THRESHOLD` 0.5）、`should_refuse()` 的整合邏輯都在這裡。 |
| `deep9_calibration.py` | deep9 的校準相關工具。 |
| `deep10.py` | 50 個 benchmark 資料集的定義（`DATASETS`），涵蓋七類：safety / jailbreak / injection / cyber / coding / skill / fast-powerful。每個資料集帶 `expect_refuse`、`expect_coding`、`expect_route`（推定標籤）或 `route_gold`（逐題實測標籤）。用**固定門檻**看每個資料集達不達標。 |
| `deep11.py` | 把門檻當變數掃掉，所以分得出「門檻選錯」和「訊號本身不夠好」的差別。報表 [1]~[10]：ROC AUC、對齊誤擋率後的擋下率、逐資料集對照、coding、model_route 推定與實測。收集和算報表在同一次執行裡跑完。 |
| `deep12.py` | **本次新建。** deep11 的兩階段版，外加 confidence 與校準。詳見下節。 |
| `gen_gold_probs.py` | 產生標準答案機率。 |

### benchmark 資料與評分

| 檔案 | 說明 |
| --- | --- |
| `download_benchmarks.py` | 下載 benchmark 原始資料。 |
| `ai2arc_routing.py` / `gsm8k_routing.py` | 讓 fast 和 powerful 兩個 model 實際作答再評分，產出 model_route 的**實測**標準答案（「能答對的最便宜 model」：fast 對→fast、fast 錯 powerful 對→powerful、兩個都錯→不列入）。 |
| `evaluate_jevbench.py` / `evaluate_jevk5.py` / `evaluate_winnow.py` | 各自的評分腳本。 |

### cookbook 移植

| 檔案 | 說明 |
| --- | --- |
| `llm_guardrails.py` | TypeSafe [Guardrails for LLMs](https://docs.typesafe.ai/cookbooks/llm_guardrails) cookbook 改用 `JevK5Classifier`：輸入／輸出各一組危害 Noul 加 severity Score，`route()` 依 strict / permissive 門檻決定 pass / review / block / support。 |
| `llm_guardrails_prompts.txt` / `llm_guardrails_replies.txt` | 上面用的 10 則使用者訊息和 5 則模型回覆。 |
| `deep9_llm_guardrails.py` | deep9 的 guardrails 版：`model_route` / `coding` 照 deep9，拒絕改用 `llm_guardrails.py` 的危害 battery 和 `route()`（pass / review / block / support），輸入、輸出各檢查一次。 |

## deep12.py 做了什麼

### 1. 拆成兩階段

deep11 是收集和報表同一次跑完、報表吃記憶體裡的 rows；`deep11_cache.jsonl` 只是省下次重問的快取。
deep12 拆成兩個指令，中間用「每個資料集一份 result 檔」接起來：

```bash
python deep12.py run --only AdvBench HarmBench   # 第一階段：各自問完 jevk5，各自存成 result
python deep12.py run --category coding safety    # 也可以按七個類別挑
python deep12.py report                          # 第二階段：撈 result 算報表 [1]~[11]
python deep12.py status                          # 哪些資料集已經有 result
```

- result 檔自帶標籤（harmful / coding_gold / route_expect / route_gold / category），第二階段
  **完全不讀 benchmark 原始檔**，原始資料不在也算得出來。
- rows 存成池子、sample 另外記：`-n 80` 跑過再跑 `-n 150` 只補問差額。
- 每 40 題寫一次檔，中斷可續；單一資料集失敗不影響其他 49 個。
- 預設每個資料集抽 80 題（seed 帶資料集名稱，所以抽到哪些題和一起跑了誰無關），`--full` 才全跑。

### 2. 報表可降級

某一節的資料還不夠（例如只跑了 MBPP，整批都是無害題，誤擋率沒有分母）時，那一節印一行原因跳過、
其他節照印，不會整份中止，並建議補跑哪個資料集。「跑一個資料集看一個」是正常用法。

### 3. 每個答案都帶 confidence

照 <https://docs.typesafe.ai/confidence> 的公式計算並存進 result：

```python
choice_confidence(ps) = (max(ps) - 1/n) / (1 - 1/n)
score_confidence(ps)  = max(0, 1 - Σ pᵢ|i-m| / MAD_unif)
noul_confidence(p)    = choice_confidence([p, 1-p]) = |2p - 1|   # 文件沒有 Noul 的公式，套 n=2
```

**實測發現**：本機 jevk5 服務回傳的 `confidence` 等於 `p_max`，不是文件上的公式。四題實測全部
吻合到小數點後四位，分佈平坦時差距很大（毒品成癮小說那題 severity：服務 0.8097、文件公式 0.5287）。
另外 `NoulAnswer` 根本沒有 `confidence` 欄位，只有 `ChoiceAnswer` 和 `ScoreAnswer` 有。
所以兩種都存：`*_conf` 是文件公式，`*_conf_api` 是服務回傳的。

### 4. [11] 校準（calibration）

把 confidence 分 10 個 bin，每個 bin 對一次答案，看「說 X 把握」的那群題目是不是真的對了 X 的比例。

**對照線不是對角線。** 文件的 confidence 是正規化邊際不是機率：n=2 時 `confidence = 2·p_max - 1`，
一個完美校準的模型（說 0.8 就真的 80% 對）在這裡只會回報 0.6，畫對角線會把它誤判成「信心不足」。
所以期望答對率要換算回 `p_max = conf × (1 - 1/n) + 1/n` 再比。

預測一律用 argmax（≥0.5）而非 production 門檻：confidence 的定義就是相對 argmax 的，
「門檻該設多少」是 [3] [7] [8] 在回答的另一個問題。

實測（3904 題，50 個資料集）：

| 問題 | 題數 | 整體說 | 實際 | ECE | Brier |
| --- | --- | --- | --- | --- | --- |
| refuse | 3904 | 89.1% | 88.3% | 0.0265 | 0.0905 |
| coding | 3904 | 93.5% | 95.4% | 0.0205 | 0.0391 |
| model_route 實測 | 293 | 79.2% | 76.1% | 0.0582 | 0.1595 |
| model_route 推定 | 3603 | 77.9% | 56.1% | 0.2187 | 0.3046 |

refuse 和 coding 校準良好。「推定」那組的 ECE 大**不代表模型沒校準** —— 那組的「答對」是
「符合推定標籤」，而推定標籤有一半是政策（coding 類一律 powerful），和 criteria 字面本來就會分岔。
四組裡只有 `model_route 實測` 是逐題實測的真答案。

severity 沒有進校準分析：沒有任何資料集標了每題的嚴重度等級，沒有答案就對不了答。

### [3] 掃門檻和 [11] 校準是互補，不是接力

[3] 掃出來的 TPR/FPR **不需要機率校準就成立**，它純粹靠排序。把 `noul` 做單調變換（排序完全不變）
重跑可以驗證：

| 分數 | AUC | [3] 門檻 | 擋下率 | 誤擋率 | ECE |
| --- | --- | --- | --- | --- | --- |
| 原始 noul | 0.9308 | 0.694 | 72.1% | 3.9% | 0.0265 |
| noul\*\*0.3 | 0.9308 | 0.896 | 72.1% | 3.9% | 0.1002 |
| noul\*\*3 | 0.9308 | 0.334 | 72.1% | 3.9% | 0.0740 |

AUC 和操作點完全相同，只有門檻的「數值」變了，但 ECE 差 4 倍。[3] 看不出這三個分數有任何差別。

校準保護的是**直接把數字當機率解讀的決策**，例如「分數 ≥ 0.8 就自動處理，我接受 20% 錯誤率」：

| 分數版本 | ECE | 那條規則的實際錯誤率 |
| --- | --- | --- |
| 原始 noul | 0.0265 | 6.7%　符合 |
| noul\*\*0.3 | 0.1002 | 15.2%　符合 |
| noul\*\*0.1 | 0.4801 | 39.5%　**超標** |
| noul\*\*0.03 | 0.5993 | 65.7%　**超標** |

四列的 AUC、擋下率、誤擋率完全相同。門檻要用 [3] 實測出來的，不要用「0.8 聽起來像八成」推出來的。

### 相依（不在這三天的改動範圍內，但 import 需要，一併收錄）

| 檔案 | 說明 |
| --- | --- |
| `classifier_jevk5.py` | 連本機 jevk5 服務（`http://localhost:8090/v1/systemone`）的 LangChain Runnable，介面和 `TypeSafeClassifier` 相同。 |
| `classifier_winnow.py` | Winnow 分類器的對應實作，`evaluate_winnow.py` 用。 |
| `llm_models.py` | 共用的模型建立函式（qwen / deepseek / kimi / glm / gemini）。API key 一律讀環境變數，呼叫到才讀。 |
| `routing_bench.py` | 四個 routing 腳本共用的骨架（`Bench`、`main`）。 |
| `aime_routing.py` / `gpqa_routing.py` | 另外兩個 fast-powerful bench 的 routing 腳本，和已收錄的 ai2arc / gsm8k 成一套。 |
| `deep7.py` | `deep10.py` 從這裡 import。 |

## 執行環境

```bash
pip install langchain langchain-core langchain-openai langchain-google-genai \
            langchain-typesafe deepagents httpx2 pyarrow langsmith
```

jevk5 服務要先跑起來（預設 `http://localhost:8090`）。

要用到外部 model 的部分（`routing_bench.py`、`gen_gold_probs.py`、`deep7.py`）才需要設對應的
環境變數，只跑 jevk5 評估的話都不用：

```
DASHSCOPE_API_KEY   DEEPSEEK_API_KEY   MOONSHOT_API_KEY   ZAI_API_KEY   GEMINI_API_KEY
WINNOW_BASE_URL     WINNOW_API_KEY
```

**benchmark 原始資料沒有收錄**（第三方資料集，體積大）。用 `download_benchmarks.py` 取得，
路徑見 `deep10.py` 開頭的常數。`deep12.py report` 不需要原始資料，但 `run` 需要。
