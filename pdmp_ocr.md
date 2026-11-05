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

**Chunk 10**（照片 9–11）
- Case Background
  user 覺得 STR Cond Table 有資料錯誤
- Root Cause
  STR CondTable 的資料來源是 PDMP
- Solution
  PDMP 與STR CondTable的關係請參考[Cond Table功能說明](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#7BDg00dS2y0aK-gLuM-ib)。解決方式可以參考 Case-3: [Common] FCCB 找不到 PDMP 單
  Step 1: 如果 Order Status不符合規則，請 user調整
  Step 2: 若單子的時間沒有問題的話，系統會有 job 將此 PDMP 單抓入系統，有可能只是錯過班車。[系統邏輯參考 (Check Pdmp 功能說明)](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#o3fgOaWOLBv8yAv2nbpEl)
  Step 3: 若單子透過 iChange 開的，並請 user 提供綁定的畫面，檢查 Cond Table 的 [Step 3-1. (From PDMP) Check Lot Experiment Split Table](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#lz-3Mv_mQaDnw9eQFc3aP)，是否有出現 user 所發出的 PDMP order 包含的 lot 和實驗站點和 wafer。如果有，那就是已經透過 iChange 綁定，那張 PDMP order 就不會出現在 Unassigned PDMP Order 的 table 中。
  Step 4: 比較 user 提供的 iChange 畫面的 Comment ID，和在行控 Comment Table 指令列出的所有 comment 的 ID 是否一致。如果不一致: 請 user 找 QM 把不在行控的 comment 做 recomment，這樣才能註冊到行控，todo列出的 STR comments 會新增一筆，從新增的 comment 按 Cond Table 就可看到 PDMP order 的 lot 和實驗站點

**Chunk 11**（照片 11–12）
- PDMP 訂單上半部分的 SplitOpNo / MergeOpNo 為註冊 Wafer Base 的 LotAction 訂單來進行
- 若 Lot 身上的所有的 Wafer 都用來開 Runcard (單一 R/C Group)，這種情況是不會在 SplitOpNo / MergeOpNo 註冊 Lot Action 訂單
- 當 Wafer 在 Split 站點完成拆批後，若於抵達實驗站點前又被 Merge，則當抵達實驗站點時，PDMP 系統亦會註冊 LotAction 訂單將 Wafer 再次拆出並註冊 Runcard，而其 Merge 站點亦是 PDMP 訂單上所設置的 Merge 站點
- 行為上會是 PDMP 系統認到 STHL Hold -> 註冊 LotAction Split/Merge 訂單 (註冊成功後 Lot 身上會有 LPHL Hold) -> 解 STHL Hold -> 再重新設上 STHL Hold -> 等 LotAction 完成 Split 動作 -> PDMP 系統認到 STHL Hold -> 註冊 Runcard & 設置 WFRC Hold
- Runcard Route 中的拆批行為則依然是交由 RCCS 進行
  ![PDMP 的 Wafer Base 開單邏輯說明](image:/api/v1/fetchImage/JmdpQvwYuX/TURYATCFTn/2)

***

**Chunk 12**（照片 12–13）
4. PDMP 單是透過 iChange 開的，但是有 comment 不在行控裡面
- Solution
  Step 1: 如果 Order Status不符合規則，請 user調整
  Step 2: 若單子的時間沒有問題的話，系統會有 job 將此 PDMP 單抓入系統，有可能只是錯過班車。[系統邏輯參考 (Check Pdmp 功能說明)](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#o3fgOaWOLBv8yAv2nbpEl)
  Step 3: 若單子透過 iChange 開的，並請 user 提供綁定的畫面，檢查 Cond Table 的 [Step 3-1. (From PDMP) Check Lot Experiment Split Table](https://tkms.digwork.tw.ent.tsmc.com/pages/sY7GwxXugj#lz-3Mv_mQaDnw9eQFc3aP)，是否有出現 user 所發出的 PDMP order 包含的 lot 和實驗站點和 wafer。如果有，那就是已經透過 iChange 綁定，那張 PDMP order 就不會出現在 Unassigned PDMP Order 的 table 中。
  Step 4: 比較 user 提供的 iChange 畫面的 Comment ID，和在行控 Comment Table 指令列出的所有 comment 的 ID 是否一致。如果不一致: 請 user 找 QM 把不在行控的 comment 做 recomment，這樣才能註冊到行控，todo列出的 STR comments 會新增一筆，從新增的 comment 按 Cond Table 就可看到 PDMP order 的 lot 和實驗站點

**Chunk 13**（照片 13–14）
##### Description: PDMP 單種有另外對於 user 是否有 Siview 權限作卡關
##### Root Cause: 主要原因是無權限
Action:
![](image:/api/v1/fetchImage/pwc4jRJDPV/TURYj9ojhX/2)
##### PDMP 單種的特殊邏輯:
My Worklist 有卡 PDMP 單的代簽組織成員需要有 Siview Runcard 權限才能在「Secondary」看得到
(但 Query Request 目前沒有卡，User 還是可以查詢得到)
如遇此問題，請 USER 去申請 Siview 權限或找 Sponsor 把他加進 A4 2.0 Role (RUNCARD_SIGNOFF)
※PDMP Site Sponsor:
F12: cfwuk 吳政峰
F14AB/F18AB: hyliaoo 廖宣雅
F15A/F15B: wyjhuang 莊婉瑜
SOIC, APOD, AZ (新廠區): hyliaoo 廖宣雅
相關 CASE:
[https://it-case-center.itsd.tsmc.com/case/Case-20250914-00607](https://it-case-center.itsd.tsmc.com/case/Case-20250914-00607 "https://it-case-center.itsd.tsmc.com/case/Case-20250914-00607")

（第 1 筆結束）

---

## 第 2 筆（照片 14–）

### query
NT2046077.0Y Fail Reason: (CreateRC)MFGWFR_MANAGERUNCARD-02090001:Create Runcard error.Set Future Hold (WFRC) fail -- USER已自行處理

### ground_truth_chunk
- 問題描述：Fab PDMP 設置 WFRC Hold 失敗，STHL Hold Memo 出現「Fail Reason: (CreateRC)MFGWFR_MANAGERUNCARD-02090001:Create Runcard error.Set Future Hold (WFRC) fail.」錯誤
- 可能原因：該批 Lot 在該站已有 WFRC Hold 存在，並正在執行 Runcard
  ![](image:/api/v1/fetchImage/LiTEeg13Gg/TUR4TigKJp/2)
- 解決步驟：
- 若欲執行 Fab PDMP 訂單，請砍掉 WFRC Hold Memo 上記錄的 Runcard 單
- 若不執行 Fab PDMP 訂單，請到 ECP Portal 上的 My Experiment，執行 [Fab PDMP 砍單動作](https://tkms.digwork.tw.ent.tsmc.com/pages/H4EqPa6oPH#_Yb3noeV-nHJEx_7yOP-p)
- 參考資料：
- [https://it-case-center.itsd.tsmc.com/case/Case-20251028-01152](https://it-case-center.itsd.tsmc.com/case/Case-20251028-01152 "https://it-case-center.itsd.tsmc.com/case/Case-20251028-01152")

***

### reference_chunks

**Chunk 1**（照片 15–16）
- 問題描述： FABPDMP 開單時，出現「Exception: MFGWFR_MANAGERUNCARD-02090001:Create Runcard error.LotID Route: xxx Operation: xxx RunCard(xxx) is already existed」錯誤
  ![FABPDMP 開單時，出現「Exception: MFGWFR_MANAGERUNCARD-02090001:Create Runcard error.LotID Route: xxx Operation: xxx RunCard(xxx) is already existed」錯誤](image:/api/v1/fetchImage/JmCpmq14RR/TURYATCFTn/2)
- 可能原因： Lot 身上還有 Runcard 未執行完成 (該未執行完成的 Runcard 亦有可能還處在待簽核階段)
- 解決步驟：請確認 Runcard 完成 / 取消後再開單 (待簽核狀態的 Runcard 亦須先完成簽退)
  ![請確認 Runcard 完成 / 取消後再開單 (待簽核狀態的 Runcard 亦須先完成簽退)](image:/api/v1/fetchImage/JFzdspzdJB/TURYATCFTn/2)
- 參考資料 :
- [https://it-case-center.itsd.tsmc.com/case/Case-20250914-00470](https://it-case-center.itsd.tsmc.com/case/Case-20250914-00470 "https://it-case-center.itsd.tsmc.com/case/Case-20250914-00470")

**Chunk 2**（照片 17）
- 問題描述：PDMP Create Runcard 錯誤，「MFGWFR_MANAGERUNCARD-02090001:Create Runcard error.e-R/C is rejected by MFG /Fail reason: RUNCARD(wfr) book lot action fail due to [CYHSUZZB MFG bypass signoff rule is Reject」
- 可能原因：此 PDMP 訂單欲開立的 RunCard 符合了 MFG 單位所設置的「自動拒簽規則」
- 解決步驟：您可以從 Siview -> 特殊晶片功能 -> eRuncard 功能選單 -> eSplit Runcard -> 資料維護 -> 製造部自動簽核條件 (尋找條件為 RuleFunction=Reject 且 Rule Status=Active 的 Rule) 來找到拒簽規則，並依此來調整您的 RunCard Condition 以避免拒簽,查找 Rule Function 為 Reject 且 Rule Status 為 Active 的項目，並根據這些規則進行調整。

***

**Chunk 3**（照片 17–18）
- 問題描述： 解 WFRC Hold 出現「不能解除 Runcard 母批的暫置 (等到 Runcard complete 時會自動解除)」錯誤
  ![解 WFRC Hold 出現「不能解除 Runcard 母批的暫置 (等到 Runcard complete 時會自動解除)」錯誤](image:/api/v1/fetchImage/MLFWhRZgNP/TURVjw7Ugr/2)
- 可能原因：未做 Runcard 完成
- 解決步驟：請參照「[Release WFRC Hold 操作步驟](https://tkms.digwork.tw.ent.tsmc.com/pages/G2PycHeKvr#cQByzmru5Z267xMwU1cxG)」小節操作，透過 Runcard 完成來結束 Runcard，WFRC Hold 便會自動解除
- 參考資料 :
- [https://it-case-center.itsd.tsmc.com/case/Case-20251202-02573](https://it-case-center.itsd.tsmc.com/case/Case-20251202-02573 "https://it-case-center.itsd.tsmc.com/case/Case-20251202-02573")

***

**Chunk 4**（照片 18–19）
![CSRCC504: 0011320E: More than one [WFRC] Hold records. 有超過一個以上的 Hold 存在, 請工程師先處理其他多餘的 Hold, 只留下正確且唯一的 WFRC Hold, 並在做一次 RC Complete.](image:/api/v1/fetchImage/G2vENQd9Ka/TURVCmPLvn/2)
- Step1. 使用前請先確認
- 所有子母批都已回到 Main Route 原本開 Runcard 的站點 (可參考「[Runcard 取消操作步驟](https://tkms.digwork.tw.ent.tsmc.com/pages/GY3gX1P3Z4#iLsz-vsEpsxulnl0cID_g)」小節，將 Lot 先帶回 Main Route )
- 母批身上要有 WFRC Hold
- Lot 身上只能留這些 Hold：RRAM, YRCP, WFRC, CCWM, MGHL, FHLD
- 以上 6 個 Hold，系統會自動解，不用手動解。尤其 FHLD 可以擋住 ABO auto merge，請勿解掉 (FHLD 跟 Future Hold 是不同意思，FHLD 是真的有一個 Hold 叫 FHLD)
- 以上 6 個以外的 Hold 請先手動解掉
- 如果要跳站
- 不可有 ENHL Future Hold 在跳站站點 (生門做完後，系統會自動在跳站站點再 hold 上 ENH)
- 在 Runcard 站點到跳站站點之間不可有 Future Hold 存在
- Foup 不能已 Load 到機台 (有的話請先收貨)

**Chunk 5**（照片 19–）
- Step0. 您必須知道的事 (請閱讀完以下注意事項後，在開始 Release WFRC Hold 操作步驟)
- 可以手動解 WFRC Hold 的情境：原則上當 Lot 已進去過 Runcard Route 就不允許手動解 WFRC，但以下情況可請 PE/PID/RD 手動解除 WFRC Hold
- Runcard 已退件
- 繼承到母批的 WFRC，…
- （照片 19 到此截斷，接續下一張）
