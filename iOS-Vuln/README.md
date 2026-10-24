# iOS-Vuln

給 `deep20.py` 的 `iOSVuln-review`、`iOSVuln-secure` 用的原始資料：Hugging Face 上
[Arno-MHL/ios-security-vulnerabilities-swift-objc](https://huggingface.co/datasets/Arno-MHL/ios-security-vulnerabilities-swift-objc)
的原檔，**沒有改過任何內容**。放進 repo 之後就不用另外下載。

## 來源

- Hugging Face repo 的 commit `ee1bd7f820891c396e49fc3baee8437e3421013a`（2026-04-22）。
- 授權是 MIT。原 repo 沒有 LICENSE 檔，授權只寫在資料集卡片上；本資料夾的 `LICENSE` 是照 MIT 的標準文字補上的，
  版權人和年份取自卡片上的 citation（`Arno-MHL`、2025）。
- `DATASET_CARD.md` 是原 repo 的 `README.md`，原樣保留（欄位說明、OWASP 分類表都在裡面）。
- 程式碼是資料集作者依 OWASP Mobile Top 10（2024）的漏洞模式寫的範例，不是從真實 app 取出來的。

| 檔案 | 筆數 | SHA-256（和 Hugging Face 上記錄的一致） |
|---|---|---|
| `data/train-00000-of-00001.parquet` | 21 | `c060ba0e848cc192cc58e92286ad96c64cbd25bf49422be7becc08100da905c5` |
| `data/test-00000-of-00001.parquet` | 6 | `4bafa2493b462135a8fd8c082647fa7bd80a331207008e45f968d6bb1226f6a8` |

## 內容

共 27 筆（Objective-C 7、Swift 20），每筆是一個 iOS 漏洞模式。`deep20.py` 只用到三個欄位：

| 欄位 | 說明 | 用在 |
|---|---|---|
| `language` | `objective-c` 或 `swift` | 題目裡寫哪種語言、程式碼區塊標 `objc` 還是 `swift` |
| `vulnerable_code` | 有漏洞的程式碼 | `iOSVuln-review` |
| `secure_code` | 修好的程式碼 | `iOSVuln-secure` |

其他欄位（`cwe_id`、`owasp_mobile`、`severity`、`attack_scenario`、`detection_hints`⋯⋯）`deep20.py` 沒有用到，
說明見 `DATASET_CARD.md`。

## deep20.py 怎麼把它變成題目

`ios_vuln()` 把每份程式碼包成一題 code review 請求：

````
Review this {Objective-C|Swift} code from our iOS app for security issues and suggest fixes.

```objc
（程式碼）
```
````

- train、test 兩個 split 都用（`deep20.py` 不訓練），先 train 再 test，所以兩個資料集的第 i 題是同一筆資料，
  一題對一題，差別只在附的程式碼。
- 原始程式碼每筆都有 `// VULNERABLE: …`、`// SECURE: …` 這類註解，等於把答案寫在題目裡，兩份都刪掉。
  這類註解共 182 行，全部獨佔一行，所以只刪整行。

## 在 deep20.py 裡怎麼用

`deep20.py` 的 `DATASETS` 已經有這兩個資料集（coding 類，`expect_refuse=False`、`expect_coding=True`、
`expect_route="powerful"`），讀的是 `~/code_benchmarks/iOS-Vuln/data/*.parquet`。

1. 把這個資料夾複製到 `~/code_benchmarks/iOS-Vuln/`：

   ```bash
   cp -r iOS-Vuln ~/code_benchmarks/
   ```

   之前已經從 Hugging Face 下載到 `~/code_benchmarks/iOS-Vuln/` 的話，是同一份檔案，不用再複製
   （再跑一次 `cp -r` 會多出一層 `iOS-Vuln/iOS-Vuln/`）。放在 `CODE` 根目錄底下，`--cn` 才會自動改讀
   `~/code_benchmarks_cn/iOS-Vuln/`（中文版還沒有，`--cn` 時會印 `[skip]`）。
2. 跑：`python deep20.py run --only iOSVuln-review iOSVuln-secure`。各 27 題，比預設 `-n 80` 少，
   不用 `--full` 也會全部跑。
3. 看報表：`python deep20_coding_report.py report --only iOSVuln-review iOSVuln-secure`。
