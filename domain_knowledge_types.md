# 領域專家知識的七種類型與管理迴圈

兩張投影片（手機拍攝）的整理稿：

- 第一張定義「領域專家知識的七種類型」。
- 第二張（Private Domain Knowledge Management Loop）把每種類型對應到儲存媒介、更新／取出機制與專職 Agent。

兩張的順序原本不同，這裡統一用第一張的順序。修正與改寫的地方列在最後一節。

## 總覽

| # | 知識類型 | 回答的問題 | 存在哪 | 怎麼更新 | 怎麼取出 | 專職 Agent |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Domain Fact（領域事實） | 這是什麼？ | 向量庫＋SQL / NoSQL | 專家輸入、ETL | 向量檢索＋SQL | Retrieval / RAG Agent |
| 2 | Topology / Relational Logic（結構關聯） | 彼此是什麼關係？ | Graph Database（Ontology） | Graph Builder 流水線 | 圖查詢、GraphRAG | Knowledge Graph Agent |
| 3 | Conditional Logic（情境條件） | 什麼時候才能這樣做？ | JSON / Python Rule Schema | 動態配置中心 | 規則引擎判定 | Guardrail / Gateway Agent |
| 4 | Causality Logic（因果邏輯） | 為什麼會這樣？ | 因果 DAG（JSON / YAML）＋貝氏網路 | 專家輸入＋機器學習 | 撈出因果子圖（Sub-DAG） | Diagnostic Task |
| 5 | Procedure Logic（程序邏輯） | 該怎麼做？ | 程式碼＋工作流定義檔 | CI/CD＋Code Review | 走到該節點就自動載入 | Working Task |
| 6 | Judgement Logic（評判邏輯） | 哪個方案比較好？ | Evaluation Prompt＋評分 Rubric | 線上 A/B 測試 | 動態渲染 Prompt | Router / Reviewer |
| 7 | Meta-Knowledge（後設／反思） | 沒碰過的問題要怎麼想？ | 軌跡記憶庫＋Self-Reflection Prompt | 任務結束後自動寫回 | 撈相似的成功軌跡當 few-shot | Supervisor Agent |

## 一、七種知識類型（第一張）

| 知識類型 | 解決的問題 | 舉例 |
| --- | --- | --- |
| Domain Fact（領域事實） | **這是什麼？** 經驗事實，包含名詞、數據、定義。 | 零件故障排除案例；RD Split Story |
| Topology / Relational Logic（結構關聯） | **這個和那個在空間或層級上是什麼關係？** 把事實拆解成具業務意義的細部構成。 | 零件組裝圖紙；硬體構成的 Topology 關係圖；RD Split 的 film stack 結構與數據關係 |
| Conditional Logic（情境條件） | **什麼時候才能這樣做？** 對領域事實的條件補充：Business Rules、硬性條件。 | Process 操作限制；觸發 alert 的條件；可以發 run card 的條件；實驗 boundary |
| Causality Logic（因果邏輯） | **為什麼會這樣？** 物理法則、因果關係。 | chamber 運作的物理原理；查案流程背後的因果邏輯 |
| Procedure Logic（程序邏輯） | **該怎麼做？** 演算法、SOP、步驟。SOP 是企業剛性資產，不能隨意發揮。 | 組裝或維修的操作說明書 |
| Judgement Logic（評判邏輯） | **遇到問題，哪個方案比較好？** 決策、權衡、價值評估、軟性條件。 | 決策依據與期望結果；事後品管時的檢驗標準 |
| Meta-Knowledge（後設／啟發） | **沒碰過這個問題時，要怎麼思考？** | 工程師遇到未知故障時的直覺與推理框架 |

## 二、知識管理迴圈（第二張）

原表六欄、每格字很多，拆成兩張表。

### 2-1 存在哪、怎麼更新、怎麼取出

| 知識類型 | 儲存與技術媒介 | 更新 | 取出 |
| --- | --- | --- | --- |
| Domain Fact | 案例文本存向量庫（Embedding）；動態紀錄存 SQL / NoSQL | 文本由專家輸入；ETL 流水線；每個案例附 metadata | 向量檢索＋SQL。重點在知識怎麼組織才找得到相關的：Tag & link、分層內容（layer content） |
| Topology / Relational | Graph Database，用節點與邊建立知識本體（Ontology） | 圖譜建構流水線（Graph Builder） | 用圖查詢語言做路徑搜尋；GraphRAG 局部社群檢索 |
| Conditional | JSON Rule Schema 或 Python Rule Schema | 動態配置中心（後台管理系統的配置介面） | 規則引擎做高效判定，並能追蹤原因 |
| Causality | 靜態：JSON / YAML 存節點與因果邊（DAG）；動態：貝氏推論網路 | 靜態由專家輸入；動態由機器學習從案例事實中沉澱 | 依目前的證據代碼，從配置庫撈出對應的因果子圖（Sub-DAG）載入記憶體 |
| Procedure | 程式碼硬編碼＋工作流引擎定義檔（軟：`skill.md`；硬：workflow、LangChain） | 走 CI/CD。SOP 是企業剛性資產，更新必須經過嚴格的 Code Review 與測試 | 確定性觸發：任務進入該節點時，自動載入對應的 Python 函式或工作流定義 |
| Judgement | Prompt Repository：附評分 Rubric 與權重的 Evaluation Prompt | 線上 A/B 測試優化評判 Prompt 或調整評分權重，類似 MCTS 與 RL 的互動機制 | 動態渲染 Prompt：需要決策時載入評判標準，把待評估的方案填進 Prompt |
| Meta-Knowledge | 軌跡記憶庫：過往任務的執行軌跡＋高階 Self-Reflection Prompts | 強化學習／記憶寫回：任務結束後自動把這次的 Trace 總結成一條經驗與反思，寫回記憶庫 | 向量檢索撈出過去相似的「成功執行軌跡」，當 few-shot 給 LLM |

### 2-2 執行時怎麼運作（做中學）、誰負責

| 知識類型 | 多層次記憶體的動態流轉與運作 | 專職 Agent |
| --- | --- | --- |
| Domain Fact | 暫存案例與數據並萃取相關事實；負責把使用者的問題轉成查詢語句，取回乾淨的事實文本來回答 | Retrieval Agent / RAG Agent |
| Topology / Relational | 用三元組（triple）表達複雜的領域實體關係，例如 Lot 生產履歷、多站點機台排程：<br>`(Lot_A)-[:PROCESSED_BY]->(Machine_B)-[:HAS_COMPONENT]->(Component_C)-[:SUPPLIED_BY]->(Vendor_D)` | Knowledge Graph Agent |
| Conditional | 用 JSON-Rules-Engine 執行；把 `reject_code`、`retrieved_facts` 與組好的 `reason_template` 交給 LLM，用自然語言解釋原因 | Guardrail Agent / Gateway Agent |
| Causality | 事件觸發時，用 Python 圖論庫（如 NetworkX）載入 JSON，轉成 Adjacency List 或矩陣，和當下撈到的 Fact 做狀態綁定推論；或啟動動態貝氏推論 | Diagnostic Task |
| Procedure | 執行器載入程序記憶。不需要太多創造力，核心是「嚴格照以下步驟做」；負責呼叫 API 工具完成發信、產報表等重複性工作 | Working Task（通常用較受控的模型） |
| Judgement | 載入評分 Prompt，在 Blackboard State 上替各個行動策略權衡打分；負責意圖分流（Routing）與品質把關（Reranking）；可用 LLM 或 MCTS | Router / Reviewer（推理要符合領域認知） |
| Meta-Knowledge | 載入 Self-Reflection Prompts，監控全域的 Blackboard State；發現其他 Agent 卡住或陷入死循環時介入，擴大搜尋範圍並修正策略 | Supervisor Agent（Orchestration）：大腦與決策官，用推理能力最強的模型 |

## 三、整理時改動的地方

| 原文 | 改成 | 原因 |
| --- | --- | --- |
| MTCS | MCTS | Monte Carlo Tree Search 的誤植 |
| 零件故障排固案例 | 零件故障排除案例 | 錯字 |
| 可以發 Run car 的條件 | 可以發 run card 的條件 | 推測是 fab 的 run card |
| 劇情文本、劇本事實、每一集的 meta | 案例文本、案例事實、每個案例的 metadata | 講者用影集當比喻，換成實際場景的說法 |
| 三元子句 | 三元組（triple） | 圖資料庫的慣用說法 |
| Topology Relational / Logic | Topology / Relational Logic | 與第一張名稱統一 |
| Meta-Knowledge（反思） | 總覽寫「後設／反思」 | 第一張叫「後設／啟發」、第二張叫「反思」，兩者合併 |
| Evaluation Prompts with Scoring Rubrics（weight MTCS） | 附評分 Rubric 與權重的 Evaluation Prompt；MCTS 移到更新與運作欄 | 原文括號意思不明確，依上下文拆開 |
