# MATLAB-examples

給 `deep20.py` 用的 MATLAB 寫程式請求，共 100 題（英文），檔案是 `prompts.jsonl`。

題目的主題取自 MathWorks 官方範例資料集（`patients.mat`、`fisheriris.mat`、`outages.csv`⋯⋯）。
題目裡只用到這些資料集的檔名和變數名稱，所以**不需要下載任何 `.mat` 檔**。

## 來源與限制

- 題目由 Claude Code 依題型表撰寫，**不是真實使用者的請求**。報表上的數字代表「這批題目」上的表現，
  不等於真實使用者的分布。
- 用到的變數名稱都對照過 MathWorks 文件；查不到名稱的（`penny.mat` 的矩陣、`Data_Airline` 裡
  `DataTable` 的欄位）題目裡就不寫出名稱。
- 還沒經過人工審閱。用之前請至少看過「附程式碼」那 28 題，確認程式碼裡的錯誤和描述的症狀對得上。
- 只有英文版。中文版（`--cn`）要另外產生，**題數和順序必須和英文版完全一樣**，同一個抽樣 seed
  才會抽到同一批題目。

## 欄位

| 欄位 | 說明 |
|---|---|
| `id` | `matlab-ex-001` ～ `matlab-ex-100` |
| `group` | 資料集組別 `D1`～`D11`（見下表） |
| `dataset` | 題目用到的檔案，例如 `patients.mat`；`D11` 是 `none` |
| `task` / `task_name` | 題型代號與名稱，例如 `T11` / `debug` |
| `phrasing` | 問法：`explicit`（題目裡有 "MATLAB"）、`implicit`（沒有 "MATLAB"，只有檔名或函式名稱）、`code`（附程式碼） |
| `prompt` | 送給 jevk5 和模型的文字 |

## 題數分配

| 組別 | 資料集 | 題數 |
|---|---|---|
| D1 | `patients.mat` | 10 |
| D2 | `fisheriris.mat` | 10 |
| D3 | `carsmall.mat` / `carbig.mat` | 10 |
| D4 | `census.mat` | 8 |
| D5 | `outages.csv` | 10 |
| D6 | `Data_Airline`（Econometrics Toolbox） | 10 |
| D7 | `humanactivity.mat` | 10 |
| D8 | `wind.mat` / `penny.mat` | 9 |
| D9 | `ngc6543a.jpg` / `peppers.png` | 9 |
| D10 | `handel.mat` / `chirp.mat` / `gong.mat` | 9 |
| D11 | 不用資料集（數值計算） | 5 |

| 題型 | 題數 | 題型 | 題數 | 題型 | 題數 |
|---|---|---|---|---|---|
| T1 載入與摘要 | 6 | T6 迴歸與擬合 | 6 | T11 除錯 | 9 |
| T2 清理與型別轉換 | 4 | T7 機器學習 | 7 | T12 向量化改寫 | 7 |
| T3 分組統計 | 5 | T8 時間序列 | 5 | T13 語言轉換 | 6 |
| T4 繪圖 | 12 | T9 訊號與影像 | 10 | T14 解釋程式 | 6 |
| T5 統計檢定 | 3 | T10 寫成函式 | 9 | T15 數值計算 | 5 |

問法：`code` 28 題（T11–T14）、`explicit` 36 題、`implicit` 36 題。

## 怎麼接到 deep20.py（尚未接）

1. 把這個資料夾複製到 `~/code_benchmarks/MATLAB-examples/`。放在 `CODE` 根目錄底下，`--cn` 才會自動改讀
   `~/code_benchmarks_cn/MATLAB-examples/`。
2. 在 `deep20.py` 的 `DATASETS` coding 區加一個 `DS`：名稱 `MATLAB-examples`、類別 `coding`、用 `jsonl_col`
   讀 `prompt` 欄位、`expect_refuse=False`、`expect_coding=True`、`expect_route="powerful"`。
3. 試跑：`python deep20.py run --only MATLAB-examples -n 10`，再用預設的 `-n 80` 正式跑。
4. 看報表：`python deep20_coding_report.py report --only MATLAB-examples`。要分開比較三種問法的判對率，
   得把 result 的逐題資料對回這個檔案的 `phrasing` 欄位。result 的 key 是
   `MATLAB-examples:` 加上完整 prompt 的 sha1（`deep20.prompt_key()`），用它對，不要用 result 裡的
   `prompt` 欄位（只存前 400 字，有幾題比 400 字長）。
