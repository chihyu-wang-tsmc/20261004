# PDMP 照片文字整理

> 格式：CSV，欄位 `query, ground_truth_chunk, reference_chunks`
> 標記 `[?]` 表示照片中被遮住或看不清楚的字（已依上下文補上）

## 第 1 筆（照片 1–）

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

### reference_chunks

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

***

**Chunk 6**（照片 6）
- 問題描述： PDMP 訂單有勾選 Foup Exchange，但為何沒有換 Foup，導致實驗站點發生 Muti Lot In One Foup 情況?
- 可能原因：若欲執行 Runcard 的子批已在 PDMP 訂單定義的 Split 站點前就被分出，則 PDMP 便不會在幫忙註冊 LotAction 訂單去做拆批 & 換 Foup
- 解決步驟：若要於 PDMP 訂單定義的 Split 站點前就將欲直行 Runcard 的子批拆出，請記得一併處理 Foup 更換動作
- 參考資料：
- [https://it-case-center.itsd.tsmc.com/case/Case-20260216-00444](https://it-case-center.itsd.tsmc.com/case/Case-20260216-00444 "https://it-case-center.itsd.tsmc.com/case/Case-20260216-00444")

***

**Chunk 7**（照片 6–7）
那因為PDMP單不是綁定到STR - No. 1，所以在STR - No. 1的table看到這兩批Lot，還是會停在Experimenting狀態
![兩批 Lot 停在 Experimenting 狀態因為 PDMP 單不是綁定到 STR - No. 1](image:/api/v1/fetchImage/https:/TURNj6mN4n/2)
- Root Cause
  在開 PDMP 單之前，user 有先在 iChange Booking Lot，接者在 comment No.1的 3.2 iChange Lot Experiment Split Table 建立此 Lot 的設定，但後續發 PDMP 單不是綁定到同一個 comment，就會發生 comment No.1 殘留狀態不一致的 Lot 設定。
- Solution
  Step 1: PDMP單是綁到哪個 comment 可以到 Assign PDMP 頁面查詢
  ![在 Assign PDMP 頁面查詢 PDMP 單是綁到哪個 comment](image:/api/v1/fetchImage/https:/TURNj6mN4n/2)
  Step 2: 如果 user不需要了，可以直接按 `🗑️垃圾桶` 刪除

**Chunk 8**（照片 7–8）
- 問題描述：同一隻 recipe 有分別開 PDMP Main Route 和 PDMP Runcard， 為何兩者最後帶出來的 chamber 不同
  ![PDMP Mian Route 有設定 Bypass Tool/Chamber Inhibition](image:/api/v1/fetchImage/QTMVdXf9ST/TUR6cjJqDv/2)
  ![PDMP Runcard 有設定 Bypass Tool Inhibition](image:/api/v1/fetchImage/QTMZahUeyt/TUR6cjJqDv/2)
- 解決步驟：Runcard 的 Bypass Tool Inhibition 與Main Route 的 Bypass Tool/Chamber Inhibition 邏輯是不同的
- PDMP Main Route Bypass 禁用方式如下 :
- 若要Bypass Chamber Inhibition 需要在 Assign Tool/Chamber欄位指定 Chamber 且 Bypass Tool/Chamber Inhibition選擇 Y
- 若要Bypass Tool Inhibition 需要在 Assign Tool/Chamber欄位指定 Tool 且 Bypass Tool/Chamber Inhibition選擇 Y
- PDMP Runcard Bypass 禁用的方式如下：
- Bypass Tool Inhibition 選擇 Y, 會同時bypass tool 和 chamber 的inhibition
- 參考案例
- [https://it-case-center.itsd.tsmc.com/case/Case-20260103-00587-2](https://it-case-center.itsd.tsmc.com/case/Case-20260103-00587-2 "https://it-case-center.itsd.tsmc.com/case/Case-20260103-00587-2")

***

**Chunk 9**（照片 8–9）
Issue:在 eRD 使用 PDMP skip to opeNo 跳超過一站時，會有訊息跳出說有 duplicated的錯誤訊息
"xxx is duplicated at opeNos xxx. Please separate into two runcard for xxx"
![](image:/api/v1/fetchImage/oP2ieAHHkV/TURJdBdkM8/2)
Root cause/Action:
opeNo到 skipOpeNo 之間不能有同一支 recipe上在不同站，系統會不知道他是要上在哪一站的recipe. 請 User 將其中一站下掉在發就可以了。
例如:OpeNo: 2600.700, skipOpeNo:2600.850, Recipe：RD-SYTINBO163-{c123}, 但User 在 Recipe Library 上將此 Recipe 定義 在 2600.700 和 2600.800,
因此系統顯示 duplicated at opeNos (2600.700,2600.800). 請 User 去 recipe library 將 2600.800 recipe 移除再重新到 PDMP 發即可
![](image:/api/v1/fetchImage/oP2mxoCiyK/TURJdBdkM8/2)
參考:[https://it-case-center.titot.f15.tsmc.com/case/Case-20231002-00171](https://it-case-center.titot.f15.tsmc.com/case/Case-20231002-00171[?])

**Chunk 10**（照片 9–10）
- Case Background
  user 覺得 STR Cond Table 有資料錯誤
- Root Cause
  STR CondTable 的資料來源是 PDMP
- Solution
  PDMP 與STR CondTable的關係請參考[Cond Table功能說明](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#7BDg00dS2y0aK-gLuM-ib)。解決方式可以參考 Case-3: [Common] FCCB 找不到 PDMP 單
  Step 1: 如果 Order Status不符合規則，請 user調整
  Step 2: 若單子的時間沒有問題的話，系統會有 job 將此 PDMP 單抓入系統，有可能只是錯過班車。[系統邏輯參考 (Check Pdmp 功能說明)](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#o3fgOaWOLBv8yAv2nbpEl)
  Step 3: 若單子透過 iChange 開的，並請 user 提供綁定的畫面，檢查 Cond Table 的 [Step 3-1. (From PDMP) Check Lot Experiment Split Table](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#[?]3Mv_mQaDnw9eOFc3aP)，是否有出現 user 所發出…
- （照片 10 到此截斷，接續下一張）
