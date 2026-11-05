# PDMP 照片文字整理

> 格式：CSV，欄位 `query, ground_truth_chunk, reference_chunks`
> 標記 `[?]` 表示照片中被遮住或看不清楚的字（已依上下文補上）

## 照片 1–5

### query
同一張PDMP中為何有成功的也有失敗的

### ground_truth_chunk
- 問題描述：為何同一張 PDMP 訂單有部份 Runcard Group 未成功開單?
- PDMP 訂單內的 Runcard Group 部份沒有長出 Runcard 單號
  ![PDMP 訂單內的 Runcard Group 部份沒有長出 Runcard 單號](image:/api/v1/fetchImage/NveDwwCrte/TURYRrWCSK/2)
- 查閱 PDMP 訂單歷史會看到 PDMP 訂單執行 Fail，出現「over retry limit (10) times, stop retry. Please consider whether to delete PDMP order.」錯誤
- ECP DB SELECT * FROM PDMP.ELM_ST_RC_MAST a WHERE a.SIGNOFF_FORMNO='PDMP1802025481348'
  ![查閱 PDMP 訂單歷史會看到 PDMP 訂單執行 Fail，出現「over retry limit (10) times, stop retry. Please consider whether to delete PDMP order.」錯誤](image:/api/v1/fetchImage/NveVB5joa4/TURYRrWCSK/2)
- 可能原因：RegisterLotActionBatch 這隻程式會一直掃狀態是 EFFECTIVE (簽核完成) 的 PDMP 訂單，檢查單子內的 Wafer 是否已到站，而當 Wafer 正在 Xfer (跨 Phase 傳送) 時，程式就會撈不到資料，誤判 Wafer 不存在，此檢查若經過 Retry 上限的 10 次，便會將該 Runcard Group 的開單狀態改成 Fail，不再開單

### reference_chunks（共 5 段）

**Chunk 1**
- 解決步驟：請嘗試 [Retigger PDMP](https://tkms.digwork.tw.ent.tsmc.com/pages/H4JaALf6pM#7FI76CtytzOVyVaeoZR2K) 訂單
- 參考資料：
- [https://it-case-center.itsd.tsmc.com/case/Case-20251215-00945](https://it-case-center.itsd.tsmc.com/case/Case-20251215-00945 "https://it-case-center.itsd.tsmc.com/case/Case-20251215-00945")

***

**Chunk 2**
- 問題描述： PDMP 執行 fail，Please solve fail reason then retrigger PDMP or re-issue PDMP. Fail reason: Split Merge book lot action fail due to [[Constraint:CheckPlanPartId [CheckPlanPartId,TMUX14C-TCEAZU+BKM.001] is different from
  ![PDMP 執行 fail，[PDMP1802025373163] fail, Please solve fail reason then retrigger PDMP or re-issue PDMP. Fail reason: Split Merge book lot action fail due to [[Constraint:CheckPlanPartId [CheckPlanPartId,TMUX14C-TCEAZU+BKM.001] is different from ](image:/api/v1/fetchImage/JFSkv9QVri/TURYRrWCSK/2)
- 可能原因：Lot 到分批 / 實驗站點的 Part 全碼與先前開單時的 Part 全碼不符 (中途有 New Part 行為)，導致不會執行分批和開立 Runcard 動作
- 解決步驟：砍單重發
- 參考文件：
- [https://it-case-center.itsd.tsmc.com/case/Case-20250927-00617](https://it-case-center.itsd.tsmc.com/case/Case-20250927-00617 "https://it-case-center.itsd.tsmc.com/case/Case-20250927-00617")

***

**Chunk 3**
- FAIL：PDMP 執行 fail，例如：Lot 有未 plan wafer、Lot action 未註冊成功...等
- 若 Abort 或 Reject 後要再重新開單，可從原訂單繼續編輯或 drop 重新開一張
- PDMP 訂單狀態演進
  ![PDMP 訂單狀態演進](image:/api/v1/fetchImage/JsgWcvsRiR/TURYUuJEen/2)

***

**Chunk 4**
- Case Background
  以參考案例 [Case-20250213-02232](https://it-case-center.itsd.tsmc.com/case/Case-20250213-02232 "https://it-case-center.itsd.tsmc.com/case/Case-20250213-02232") 做說明
  user報案的這2張 PDMP 單，是綁定到 P150202449413 的另外一個 comment (STR - No. 2)
  ![PDMP 單，是綁定到 P150202449413 的另外一個 comment](image:/api/v1/fetchImage/https:/TURNj6mN4n/2)
  打開 STR - No. 2的 Condition Table，確實會看到有長出這2筆生效的PDMP單在 3.1 PDMP Lot Experiment Split Table
  ![2筆生效的PDMP單在 3.1 PDMP Lot Experiment Split Table](image:/api/v1/fetchImage/https:/TURNj6mN4n/2)
  那因為PDMP單不是綁定到STR - No. 1，所以在STR - No. 1的table看到這兩批Lot，還是會停在Experimenting狀態

**Chunk 5**
- 問題描述： PDMP 開單時，出現「PDMP WAT iWin check result: Filename validate fail.」錯誤
  ![PDMP 開單時，出現「PDMP WAT iWin check result: Filename validate fail.」錯誤](image:/api/v1/fetchImage/MLHJQzpgwx/TURYATCFTn/2)
- 可能原因：填入的 Recipe 資料有特殊字元
  ![填入的資料有特殊字元，請手動輸入 recipe (防止複製貼上特殊字元)](image:/api/v1/fetchImage/MLHTY61Cbt/TURYATCFTn/2)
- 解決步驟：請手動輸入 Recipe (防止複製貼上特殊字元)
- 參考資料：
- [https://it-case-center.itsd.tsmc.com/case/Case-20251202-02550](https://it-case-center.itsd.tsmc.com/case/Case-20251202-02550 "https://it-case-center.itsd.tsmc.com/case/Case-20251202-02550")
- （照片 5 到此截斷，接續下一張）
