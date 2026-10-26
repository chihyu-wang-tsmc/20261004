# agent_trace_observability/ 的資料

`agent_trace_observability_h2o.py` 和五個 `agent_trace_observability_h2o_<n>_*.py` 用的資料。

| 檔案 | 來源 |
| --- | --- |
| `cases.json` | https://evals.typesafe.ai/agent_trace_observability-cases.js?v=4a3821a2（2026-10-25 下載） |
| `case5_h2o_calls_preview.md` | 第 5 題（Duplicate shifts prompt cache purge）3 次呼叫 h2o 的完整 request body，用 `agent_trace_observability_h2o.py` 的程式從 `cases.json` 產生；h2o 的輸出是 2026-10-25 實跑時 console 印的值（四捨五入到兩位）。完整的原始 response 要用 `--dump` 跑 |

頁面是 https://evals.typesafe.ai/agent_trace_observability，`cases.js` 是頁面底下「View examples」的資料。

處理：`cases.js` 的內容是 `__VIEWER_DATA__({...});`，只把括號裡的 JSON 取出來，用
`json.dump(..., ensure_ascii=False, indent=1)` 重新排版，內容一個字都沒改（讀回來和原檔解析出來的
物件相同）。原檔的 sha256 是 `4a3821a24366dc98…`，開頭 8 碼就是網址上的 `v=4a3821a2`。

## 內容

整個評測有 117 題（`eval.n_cases`），官網只公開這 5 題範例：

| `eval.examples` 的名稱 | 官網標籤 | case_id |
| --- | --- | --- |
| Email update already on file | Opus differs from the other two | `trace_185afd508b6907c6075de2931f14946c` |
| Frozen schedule board cache purge | Sol differs from the other two | `trace_ce6a5fb975fce60de84457e97ab6ed35` |
| Stale database grant cleanup request | TypeSafe differs from the other two | `trace_ebba36efa92c88a555ad4be8eabf149d` |
| Referral booking with cost discrepancy | All three miss the reference | `trace_e2ab80ce644dc6a4d4ea02cfcffcccee` |
| Duplicate shifts prompt cache purge | All three agree | `trace_9222e1c90cfe97d34a6893f13889ff1d` |

- `eval.cases[case_id].models`：官網三個模型（`opus` = anthropic:claude-opus-5、`sol` = openai:gpt-5.6-sol、
  `typesafe` = typesafe:v13_snowy_elephant，頁面圖上叫 Jev）每個 node 用哪份文件、問哪幾題、答案，
  以及 workflow 算出來的決定（`decisions.balanced`：actions、keys、outputs、每一條門檻的 tests）。
- `eval.cases[case_id].references` / `reference_answers`：reference 的決定和每題答案
  （openai:gpt-6-astra、anthropic:claude-fable-5-1，都是 high effort；第 1 題只有 astra）。
- `eval.documents`：送給分類器的 state，共 17 份，`nodes[].doc` 用 index 引用。
- `eval.questions`：題目，共 16 題，`nodes[].questions` 用 index 引用。
- `icons` / `meaning`：頁面的圖示，程式沒用到。

## 已知的缺口

- 第 2 題（Frozen schedule board cache purge）`ShiftLoom.run_standard_diagnostics` 的結果在官網資料裡
  就被截斷了（`…[534 more characters]`，`documents[3]` 和 `documents[5]`）。官網跑的時候分類器看到的
  是不是完整的不知道，這裡原樣送給 h2o。
- 只有官網三個模型實際問過的 node 才有文件。h2o 走到官網沒問過的 node 時（例如第 2 題走到
  expectation gap 要問 clarity），`agent_trace_observability_h2o.py` 照同一個 agent 的其他文件的格式
  自己組，`--check` 會驗證組出來的跟官網現有的文件一字不差。
