# open-source-jev

用開源的 [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b)（報表上叫「Jev (開源)」）當 classifier，評三件事：

1. **coding 偵測**：判「這題要不要寫程式」判得準不準，和 Gemini 3.7 Flash 比
2. **routing**：model_route 決定每題送 fast 還是 powerful model，看省多少錢、掉多少分
3. **兩道守門**：輸入守門 → LLM → 輸出守門，有害請求擋下多少、正常請求誤擋多少

## 資料夾

`open-source-jev/` 和 `datasets/` 放在同一層（題目預設從 `../datasets` 讀）。

```
open-source-jev/
├── coding_report.py      1. coding 偵測          run / status / report / compare
├── run_bench.py          2. 讓 LLM 回答 GPQA / AIME / AI2 ARC / GSM8K
├── routing_report.py     2. routing 報表          status / report / multi-fast
├── guard_report.py       3. 兩道守門              run / status / report / label
├── prompt_datasets.py    每個資料集的題目從哪讀，和標準答案（該不該擋、該不該判 coding）
├── benchmarks/           GPQA / AIME / AI2 ARC / GSM8K：讀題目、對答案
├── classifiers.py        問 classifier 的題目（coding、model_route、守門的危害和嚴重度）
├── classifier_h2o.py     H2O-Lightning-4B（本機服務）
├── classifier_gemini.py  Gemini 當 classifier（請模型用 JSON 寫出機率）
├── models.py             作答的 LLM（連線設定在 .env）和單價
├── common.py  metrics.py  plotting.py    共用的小工具
└── results/              結果和報表都在這裡（自己跑出來的，不進 repo）

datasets/
├── code_benchmarks/      coding 類（HumanEval、MBPP、McEval…）
├── safety_benchmarks/    safety / jailbreak / injection / cyber 類（AdvBench、HarmBench、JailbreakBench…）
└── gpqa/  ai2arc/  aime/  gsm8k/
```

## 安裝

Python 3.10 以上。

```bash
pip install -r requirements.txt
cp .env.example .env      # 填 gateway 的網址和 token
```

**H2O-Lightning-4B**（要 GPU，L4 / L40S 都可以）：照 [model card](https://huggingface.co/h2oai/h2o-lightning-4b) 的 Run it
起 vLLM 和作者的 shim，起好之後的網址是 `http://127.0.0.1:8741/v1/systemone`（不是這個網址就改 `.env` 的 `H2O_BASE_URL`）。

```bash
hf download h2oai/h2o-lightning-4b --local-dir h2o-lightning-4b
bash h2o-lightning-4b/serve.sh     # 暖機完會印 ready
python classifier_h2o.py           # 問一題，確認服務通了
```

**LLM**：gemini-3.5-flash、gemma-4-31b、qwen3.5-0.8b 負責作答，gemini-3.7-flash 當 coding 的 classifier。
都走 OpenAI 相容的 API（`/v1/chat/completions`），
公司 AI gateway、本機 vLLM 都可以。每個 model 在 `.env` 填 `<字首>_BASE_URL`、`<字首>_TOKEN`，
字首和單價在 `models.py`。

```bash
python classifier_gemini.py        # 用 gemini-3.7-flash 問一題，確認 gateway 通了
```

## 1. coding 偵測

21 個資料集：17 個寫程式的（該判 coding），加上 GPQA / AI2ARC / AIME / GSM8K（不該判）。

```bash
python coding_report.py run --classifier h2o-lightning-4b -n 5    # 先每個資料集抽 5 題試跑
python coding_report.py run --classifier h2o-lightning-4b         # 全部題目（試跑問過的會沿用）
python coding_report.py report --classifier h2o-lightning-4b      # 印報表、寫 Excel、畫 ROC

python coding_report.py run --classifier gemini-3.7-flash
python coding_report.py report --classifier gemini-3.7-flash

python coding_report.py compare                                   # 兩個 classifier 的逐資料集 coding 率並排
python coding_report.py status --classifier h2o-lightning-4b      # 哪些資料集已經有結果
```

報表：逐資料集的判對率與 coding 率、整體的 Precision / Recall / FPR / AUROC、門檻掃描、判錯的例子、
逐程式語言、ROC 曲線。輸出在 `results/coding/<classifier>/`（`report.xlsx`、`roc.png`）。

## 2. routing（成本／正確率）

要兩樣東西：

- **route_p**：h2o 給每題「該送 powerful」的機率。第 1 節用 h2o 跑 coding 時同一個請求就會一起問，所以跑過第 1 節就有了
  （只要 routing 的話：`python coding_report.py run --classifier h2o-lightning-4b --only GPQA AI2ARC AIME GSM8K`）。
- **作答**：fast 和 powerful 都把四個 benchmark 答一遍。

```bash
python run_bench.py all answer --model gemini-3.5-flash    # powerful
python run_bench.py all answer --model gemma-4-31b         # fast
python run_bench.py all summary                            # 各 model 的正確率、token、花費

python routing_report.py status                            # 缺什麼、要跑什麼
python routing_report.py report                            # fast = gemma-4-31b、powerful = gemini-3.5-flash

# 比好幾個 fast（route_p 和 fast 是誰無關，換 fast 只要讓它作答）
python run_bench.py all answer --model qwen3.5-0.8b
python routing_report.py multi-fast --fast qwen3.5-0.8b gemma-4-31b --bench aime ai2arc
```

`answer` 中斷後重跑同一個指令會接著答；`--limit 20` 只答前 20 題，試跑用。
報表：門檻 τ 從 1 掃到 0 的「powerful 呼叫比例 → 正確率」曲線，和全送 fast、全送 powerful、oracle 比，
加上 APGR、CPT、$ 每千題。輸出在 `results/routing/<fast>/`、`results/routing/multi-fast/`。

## 3. 兩道守門

每一題走三步：第一道問使用者的 prompt → 交給 model_route 選的 LLM 回答 → 第二道問 LLM 的回覆。

```bash
python guard_report.py run --only GPQA AIME GSM8K AI2ARC
python guard_report.py run --category coding
python guard_report.py run --category safety jailbreak injection cyber
python guard_report.py status

python guard_report.py report                     # 照 strict 的門檻（block ≥ 0.70）
python guard_report.py report --policy permissive # 照 permissive（block ≥ 0.85）
```

- **run 的時候第一道不擋任何題目**，每題都送進 LLM、每個回覆都問第二道；擋不擋是 report 才照 `--policy` 的門檻算的，
  所以換門檻不用重跑。代價是**有害的 prompt 真的會送進 LLM**，而且每一題都是一次 LLM 呼叫。先用 `-n 20` 小規模試。
- LLM 預設 fast = gemma-4-31b、powerful = gemini-3.5-flash（和 routing 同一組），可用 `--fast` / `--powerful` 換。
- 「回覆是否有害」預設用啟發式（有害的 prompt、回覆不是拒答就算有害）。要人工標：
  `python guard_report.py label` 匯出 CSV，在 `reply_harmful` 欄填 1 / 0，再跑 report。

輸出在 `results/guard/<classifier>/report/`：`tables.xlsx`（每一節一個工作表）、`tradeoff*.png`（擋下率 vs 誤擋率）、
`dataset_guardrail.xlsx`（逐資料集的八欄：題數、第一道擋、第二道擋、模型自拒、ASR、FPR、空回覆、正常回答）、
`dataset_guardrail_3table.xlsx`（同樣的數字照三張表排）。要把八欄加到舊的 `dataset.xlsx` 後面：`--template dataset.xlsx`。

## 結果檔

```
results/
├── coding/<classifier>/<資料集>.json     coding_report.py run（h2o 的另外有 route_p）
├── coding/<classifier>/report.xlsx      coding_report.py report
├── <gpqa|ai2arc|aime|gsm8k>/<model>.jsonl   run_bench.py answer，一行一題
├── routing/<fast>/  routing/multi-fast/  routing_report.py
└── guard/<classifier>/<資料集>.json      guard_report.py run；report/ 底下是報表
```

每個 `run` 每問完 40 題存一次檔，中斷後重跑同一個指令會接著問；失敗的題目不存，下次重跑會再問。
`-n` 抽樣的 seed 固定，加大 `-n` 時已經問過的會沿用；`--refresh` 丟掉舊答案重問。

## 和舊指令的對照

| 舊的 | 新的 |
| --- | --- |
| `deep20_coding_report_h2o.py report` | `coding_report.py run` → `coding_report.py report` |
| `deep20_coding_report_gemini.py report` | `coding_report.py run --classifier gemini-3.7-flash` → `report --classifier gemini-3.7-flash` |
| `deep20_report_h2o.py status / report` | `routing_report.py status / report` |
| `deep20_report_h2o.py prepare` | 不用了，report 直接讀 |
| `deep20_report_h2o.py report_muli_fast --fast A B --bench aime ai2arc` | `routing_report.py multi-fast --fast A B --bench aime ai2arc` |
| `<bench>_routing.py answer --model M` | `run_bench.py <bench> answer --model M` |
| `deep20_h2o.py run --full --only GPQA --guard-policy measure` | `guard_report.py run --only GPQA`（預設就是全部題目、不擋） |
| `deep20_h2o.py run --full --category safety --guard-policy measure` | `guard_report.py run --category safety` |
| `deep20_guard_report_h2o.py report --guard-policy measure --as-policy strict --dataset-out …` | `guard_report.py report`（預設 strict，dataset_guardrail.xlsx 寫在 report/ 底下） |

prompt 的組法、問 classifier 的題目文字、守門政策的算法都和舊程式一字不差，數字可以直接比。

## 常改的地方

- **加 LLM**：`models.py` 的 `MODELS` 加一行（`.env` 字首、模型名稱、單價），`.env` 填網址和 token
- **加資料集**：`prompt_datasets.py` 加一行（名稱、類別、該不該擋、該不該判 coding、怎麼讀）
- **門檻**：coding 是 `coding_report.py` 的 `THRESHOLD`（0.5）；守門是 `guard_report.py` 的 `POLICIES`
- **單價**：`models.py`

## 注意

- GPQA 的資料集條款要求不要公開題目，`datasets/gpqa/` 不要放進公開的 repo。
- `safety_benchmarks/` 裡沒有 InjecAgent，`InjecAgent-dh` 會印 `[skip]`；把 `InjecAgent/` 複製進去就會自動跑。
- `results/` 裡有有害的 prompt 和 LLM 的回覆，`.gitignore` 已經排除，不要 commit。
