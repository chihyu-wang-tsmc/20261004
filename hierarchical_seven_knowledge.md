# 用七大知識串成一棵 hierarchical classification（設計草稿）

參考 [TypeSafe 的 hierarchical classification cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification)
和本 repo 的 `hierarchical_cpc_jevk5_path.py`，想法是把七大知識（Fact、Relational、Causality、Conditional、
Procedure、Judgement、Meta）放進**同一個** beam search 架構，而不是做成七個分開的分類器。
七大知識的定義見 `domain_knowledge_types.md`。

## 一、cookbook 的架構在做什麼

- 分類樹的每個內部節點問一題 `Choice`：「哪個直接子類別最符合這份文件？」選項用 `c0`、`c1`… 當 key，
  回來的是一組機率。
- `state` 只放文件。官方版不給模型看路徑；`hierarchical_cpc_jevk5_path.py` 把走過的路徑放進 instructions。
- 路徑分數是每次決策機率的幾何平均；只有一個子節點時機率算 1.0，不算一次決策。
- beam search 保留 K=3 條路徑，走到葉子的路徑留在 beam 裡繼續跟還沒走完的比。
- 沒有 abstain，一定會給一個葉子。`separation_ratio`（第一名分數 ÷ 第二名分數）有算，但沒拿來做任何判斷。
- 結果：greedy 4 題對 2 題，beam 4 題全對。CPC 那題 greedy 掉進了「E99Z99/00 其他」這個 catch-all 葉子。

## 二、核心想法：七種知識在樹裡扮演不同的角色

七種知識不是七層一樣的 Choice，而是分別對應到 beam search 的不同零件：

| 知識 | 在樹裡的角色 | 對應 cookbook 的哪裡 | jevk5 題型 | 要不要叫模型 |
| --- | --- | --- | --- | --- |
| Fact | 樹的前幾層：「這是什麼」；也是 `state` 的內容 | `Hierarchy.document`、靜態分類樹 | Choice | 要 |
| Relational | 決定下一層的選項：從 graph 撈「組成、上下游」當子節點；走過的路徑本身就是一條關係鏈 | `subtree()`、路徑版的 instructions | Choice | 要 |
| Causality | 邊的意思變成「可能是因為」（或「會導致」），從症狀走到根因 | 新的一段樹：因果 Sub-DAG 攤平成樹 | Choice | 要 |
| Conditional | 剪枝：問模型之前先用規則刪掉不成立、不允許的子節點；軟條件改用 Noul 問 | `choose()` 之前多一個 `prune()` | 規則引擎（硬）／Noul（軟） | 硬條件不用 |
| Procedure | 葉子：走到底對應一份 SOP，查表就好 | 只有一個子節點時機率算 1.0 的處理方式 | 查表 | 不用 |
| Judgement | 排序：beam 留下的候選方案用 rubric 打分 | 幾何平均分數、最後的 `sorted(beam)` | Score | 要 |
| Meta | 包在外面的迴圈：分數太接近、掉進 catch-all 時加寬 beam、補證據、交給人；結束後寫回 | `separation_ratio`、`BEAM_WIDTH`、`MAX_DEPTH` | 不是題目 | 視情況 |

### 串法：上一段的葉子是下一段的根

每一種知識是一「段」，每段是自己的一棵樹，可以有好幾層。某一段走到葉子，就用那個葉子去對應的知識庫撈下一段的根：

```
state = 輸入文件 + 撈到的 Fact
  │
  ├─ Fact 段         Choice   這是什麼？            選項：分類樹（像 CPC、Shopify）
  │    └ 葉子 ──────────────▶ 拿去 Graph DB 撈
  ├─ Relational 段   Choice   出在結構的哪裡？       選項：這個東西的組成、上下游
  │    └ 葉子 ──────────────▶ 拿去因果庫撈
  ├─ Causality 段    Choice   最可能的原因？         選項：因果 DAG 上指向它的原因
  │    └ 葉子 ──────────────▶ 拿去規則庫、SOP 庫撈
  ├─ Conditional     規則剪枝＋Noul                  不成立的分支直接刪，不進 beam
  ├─ Procedure 段    查表     對應哪份 SOP？         不叫模型，機率 1.0
  └─ Judgement       Score    哪個方案最好？         beam 結束後重排
Meta：包在整個 beam search 外面
```

整棵樹不用事先建好：每次展開時依「現在在哪一段」去對應的知識庫撈子節點。beam search 本身幾乎不用改，
只要把 `subtree(tree, path)` 換成依段分派的 `expand(path)`。

### 為什麼要串成同一棵，而不是六個分開的分類器

- **分開問**：每段只拿上一段的第一名往下傳，前面錯了後面全錯。這就是 greedy 的問題（cookbook 裡 greedy 只對 2/4）。
- **串成同一棵**：beam 跨段保留 K 條路徑。Fact 段選錯的那條，到了 Causality 段找不到像樣的原因，機率會散掉、
  分數下降，被另一條路徑取代。後面的段等於在幫前面的段驗算。
- **代價**：路徑變長、呼叫次數變多。靠 Conditional 的剪枝和 Procedure 的查表（都不叫模型）省回來。

### 設計時要決定的幾件事

1. **只有機率可以乘進路徑分數。** Fact、Relational、Causality 的 Choice 和軟條件的 Noul 都是機率，算進幾何平均。
   硬規則剪掉的分支直接刪；Procedure 查表是 1.0、不算決策次數；Judgement 的 Score 是期望值、不是機率，
   不能乘進去，放到最後當排序依據。
2. **Judgement 比的是「動作」，不是「路徑」。** beam 最後留下 K 條診斷（例如兩個可能的根因），每個候選動作
   在每條診斷下各打一次分，再用路徑分數加權。這樣選得出「不管是哪個根因都還算好」的動作，例如先補證據。
   jevk5 一次呼叫可以問好幾題（deep9 一次問四題），K 條路徑 × 幾個 rubric 維度可以一起送。
3. **Conditional 有兩種放法。** 硬規則掛在每次展開前剪枝；如果條件的結果本身就是要輸出的答案
   （例如 run card 能不能放行），就做成一個兩選一的節點，用 Noul 問。Noul 用在「程式要 if 分支」的地方、
   Score 用在「要跟門檻比的光譜」，跟 deep9 的原則一樣。
4. **帶路徑的風險在 Causality 段最大。** `hierarchical_cpc_jevk5_path.py` 的 docstring 提過：題目先說
   「已經分到 X」，模型在錯的分支裡也可能很有把握，beam 就剪不掉錯的路徑。串起來之後路徑一定要帶
   （不知道走到哪個零件，原因選項根本沒意義），所以要靠 Meta 盯著：分數一路都很高、但跟其他路徑
   分不開的時候，拿掉路徑再問一次這一段。
5. **Meta 的門檻要用資料掃出來。** 觸發條件例如 top/second 太接近、平均機率太低、葉子是「其他」這類 catch-all；
   具體數字跟 deep11 一樣，用有標準答案的題目掃，不要寫死。

### 程式要改哪裡（示意，不是可以直接跑的程式）

以 `hierarchical_cpc_jevk5_path.py` 為底，路徑的每一步改成 `(段, 節點)`：

```python
STAGES = ("fact", "relational", "causality", "procedure")


def expand(case, path):
    """取代 subtree()：回傳下一步的選項。某一段沒有子節點了就換下一段，用這一段的葉子撈下一段的根。"""
    stage, anchor, segment = locate(path)            # 目前在哪一段、上一段的葉子、這一段走過的節點
    if stage == "fact":
        labels = subtree(FACT_TREE, segment)         # 跟 cookbook 一樣的靜態樹
    elif stage == "relational":
        labels = graph.children(anchor, segment)     # Graph DB：組成、上下游
    elif stage == "causality":
        labels = causal_dag.causes(anchor, segment)  # 因果 Sub-DAG
    else:
        labels = sop.lookup(anchor)                  # Procedure：查表
    # Conditional（硬規則）：問模型之前先剪
    return tuple(label for label in labels if rules.allows(case, path + (label,)))


def choose(case, path, labels):
    if len(labels) == 1:
        return {labels[0]: 1.0}                      # 同 cookbook：沒得選，不算一次決策
    if is_gate(path):                                # Conditional（軟條件）：兩個子節點是「成立／不成立」
        p = ask_noul(case, gate_question(path))
        return {labels[0]: p, labels[1]: 1 - p}
    ...                                              # 其餘照原本的 child_question(labels, path)


def judge(case, beam):
    """Judgement：每個候選動作在 K 條診斷下各打分，用路徑分數加權。"""
    weights = normalize([candidate["score"] for candidate in beam])
    actions = {action_of(candidate) for candidate in beam}
    utility = {
        action: sum(w * rubric_score(case, c, action) for w, c in zip(weights, beam))
        for action in actions
    }
    return max(utility, key=utility.get)


def solve(case):
    """Meta：包在 beam_search 外面。"""
    prior = memory.similar_successes(case)           # 過去相似案例的成功路徑，當 few-shot
    for width in (3, 6):
        result = beam_search(case, width, prior)
        if healthy(result):                          # 分數夠開、不是 catch-all、平均機率夠高
            break
    else:
        return escalate(case, result)                # 交給人，不硬給答案
    action = judge(case, result["beam"])
    memory.write(case, result, action)               # 結案後寫回，下次撈得到
    return action
```

## 三、十個場景

先看總表，每個場景的細節在後面。段的縮寫：F = Fact、R = Relational、Ca = Causality、Co = Conditional、
P = Procedure、J = Judgement；Meta 都包在外面，不列在順序裡。

| # | 場景 | 段的順序 | Causality 的方向 | Conditional 的作用 | Meta 什麼時候出手 |
| --- | --- | --- | --- | --- | --- |
| 1 | 半導體 fab：機台異常到 run card 放行 | F → R → Ca → Co → P → J | 往回找根因 | 本身就是答案（能不能放行） | 兩個根因分不開 |
| 2 | 電商退貨與客訴 | F → Co → R → Ca → P → J | 往回找原因 | 提早剪枝（不可退的直接結束） | 掉進「其他」原因 |
| 3 | 保險理賠 | F → R → Ca → Co → P → J | 往回找近因 | 由原因決定保不保 | 近因分不開 |
| 4 | 合約審閱 | F → R → Ca → Co → P → J | 往前推後果 | 公司紅線 | 沒見過的條款類型 |
| 5 | 線上服務事故（on-call） | F → R → Ca → Co → P → J | 往回找根因 | 剪掉不准做的處置 | 不像任何已知模式 |
| 6 | 學生學習診斷 | F → R → Ca → Co → P → J | 往回找迷思 | 剪掉還沒教過的方法 | 補救後還是一直錯 |
| 7 | 航空班機異常調度 | F → R → Ca → Co → P → J | 往前推連鎖延誤 | 法規與安全，剪掉不合法的方案 | 多個機場互相衝突 |
| 8 | 1999 市民陳情分派 | F → R → Co → Ca → P → J | 往回找原因 | 決定受理單位 | 單位互相退件 |
| 9 | 內容審核／AI 護欄 | Co → F → R → Ca → P → J | 判斷意圖 | 零容忍的閘門 | 沒看過的黑話 |
| 10 | 家庭烘焙失敗診斷 | F → R → Ca → Co → P → J | 往回找主因和次因 | 使用者自己的限制 | 樹裡沒有的材料 |

### 1. 半導體 fab：機台異常到 run card 放行

**輸入**：ETCH-07 chamber B 連續 3 個 lot 的 CD 偏大約 2 nm；前一天剛做完 PM；RF reflected power 偶發 alarm。
工程師想知道能不能發 run card 跑實驗 lot。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 這是哪一類異常？ | 異常分類樹：製程偏移／機台 alarm／量測異常／物料 | 製程偏移 > CD 偏大 |
| 2 | Relational | 出在哪個子系統？ | Graph：ETCH-07 → chamber B → RF／gas／ESC／pump | RF > matching network |
| 3 | Causality | 最可能的原因？ | 因果 DAG：電容老化／RF cable 接觸不良／PM 後 recipe 參數沒回復 | PM 後參數沒回復 |
| 4 | Conditional | 能不能發 run card？ | 硬規則：同 chamber 有 hold lot 就剪掉「可發」；軟條件（Noul）：實驗條件在 boundary 內嗎？ | 可發 |
| 5 | Procedure | 對應哪份 SOP？ | 查表：參數回復 SOP、monitor wafer SOP、換電容 SOP | 3 份候選 |
| 6 | Judgement | 哪個處置最好？ | rubric：良率風險、停機時間、成本 | 先跑 monitor wafer 再放行 |

- **Meta**：「電容老化」和「參數沒回復」分數太接近時不放行，先跑 monitor wafer 補證據，把結果加進 state 再問一次。
  結案後把這條路徑寫回軌跡庫。
- **特點**：Conditional 的結果本身就是答案，所以做成 Noul 節點，不只是剪枝。Judgement 選「先跑 monitor wafer」，
  是因為不管根因是哪一個，這個動作都不會錯，這就是「比動作、不比路徑」的好處。

### 2. 電商退貨與客訴

**輸入**：買了一件羊毛大衣，收到第 5 天發現袖口脫線，想退貨。這筆訂單是跨境賣家出貨。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 是什麼商品？ | Shopify 分類樹（repo 已有 `hierarchical_shopify`） | Apparel & Accessories > Clothing > Outerwear > Coats & Jackets |
| 2 | Conditional | 能不能退？ | 硬規則：7 天鑑賞期、貼身衣物不可退、客製商品不可退 | 可退（第 5 天、不是貼身衣物） |
| 3 | Relational | 責任在哪一段？ | Graph：訂單 → 賣家 → 倉庫 → 物流商 | 跨境賣家 > 海外倉 |
| 4 | Causality | 為什麼要退？ | 原因樹：瑕疵（製造／運送損壞）／尺寸不合／與描述不符／不想要了 | 瑕疵 > 製造（脫線） |
| 5 | Procedure | 走哪個流程？ | 查表：製造瑕疵＋跨境賣家 → 退款並向賣家求償 | 退款 SOP |
| 6 | Judgement | 給顧客什麼方案？ | rubric：顧客終身價值、退貨成本、再發生的風險 | 免寄回直接退款（跨境運費比商品還貴） |

- **Meta**：顧客的描述跟所有原因都對不上（例如「穿了會癢」），掉進「其他」葉子，就轉真人客服；
  同類案例累積夠多，就在原因樹加新節點。
- **特點**：Conditional 放第 2 段。跟原因無關的硬規則越早剪越省，被擋下的單後面四段都不用問。

### 3. 保險理賠

**輸入**：颱風過後一樓淹水，地板和冰箱泡壞。保戶投保住宅火險，加保颱風洪水附加險。理賠人員發現牆角有長期水漬。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 哪一類理賠？ | 險種樹：住宅火險 > 附加險 > 颱風洪水 | 颱風洪水附加險 |
| 2 | Relational | 損失對到保單的哪一項？ | Graph：保單 → 主約／附約 → 建築物／動產 → 保額 | 附約 > 建築物（地板）＋動產（冰箱） |
| 3 | Causality | 損失的近因是什麼？ | 因果 DAG：颱風淹水／管線老化漏水／地下水回滲 | 地板、冰箱：颱風淹水；牆角水漬：管線老化 |
| 4 | Conditional | 保不保？ | 硬規則：除外責任（老化、漸進性損失）、等待期、保額上限 | 地板、冰箱保；牆角水漬不保 |
| 5 | Procedure | 走哪個理賠流程？ | 查表：金額低於門檻 → 書面審核；高於門檻 → 公證人勘查 | 公證人勘查 |
| 6 | Judgement | 怎麼處理？ | rubric：賠付正確性、處理天數、客訴風險、詐保風險 | 先賠沒有爭議的冰箱，地板等勘查 |

- **Meta**：近因在「淹水」和「漏水」之間分不開時，不自動拒賠，交給資深理賠人員；被推翻的判斷寫回，下次當 few-shot。
- **特點**：保險的「近因原則」讓 Causality 直接決定 Conditional 的結果，所以 Conditional 一定要放在 Causality
  後面，剛好跟電商退貨相反。一件理賠有好幾個損失項目（地板、冰箱、牆角），所以從 Relational 段開始，
  每個項目各走一條路徑，最後在 Judgement 合起來。

### 4. 合約審閱

**輸入**：供應商合約第 12 條寫「乙方對甲方之一切損害負賠償責任，不設上限，包括間接損失與商譽損失」。我方是乙方。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 這是哪種條款？ | 條款分類樹：責任限制／保密／智財／終止／付款 | 責任限制 > 損害賠償範圍 |
| 2 | Relational | 牽動合約的哪些地方？ | 條款引用圖：本條 → 定義條款 → 附件 SLA → 保險條款 | 定義條款＋保險條款 |
| 3 | Causality | 這樣寫會造成什麼後果？ | 後果樹：無上限賠償 → 現金流風險；保險不承保間接損失 → 要自己付 | 保險不涵蓋，要自己付 |
| 4 | Conditional | 踩到哪些紅線？ | 公司 playbook：賠償上限不得超過合約金額、不承擔間接損失；準據法 | 兩條都踩到 |
| 5 | Procedure | 用哪份修改範本？ | 查表：redline 範本（上限＝12 個月合約金額、排除間接損失） | 對應範本 |
| 6 | Judgement | 接受、協商還是拒簽？ | rubric：交易價值、風險金額、對方的議價空間 | 協商：提出上限條款 |

- **Meta**：條款類型對不上任何分類（例如新型的 AI 資料使用條款）時不套範本，直接交法務。
- **特點**：Causality 是往前推「後果」，不是往回找「原因」。子節點是可能發生的後果，Judgement 比的是後果有多嚴重。

### 5. 線上服務事故（on-call）

**輸入**：checkout API 的 p99 延遲從 300 ms 升到 4 s，錯誤率 8%。40 分鐘前有一次部署；現在是大促前一天，正在 change freeze。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 哪一類 alert？ | 延遲／錯誤率／資源飽和／資料不一致 | 延遲＋錯誤率 |
| 2 | Relational | 卡在哪個服務、哪個檔案？ | 服務相依圖：checkout-api → payment-svc → postgres／redis；再往下接 codebase 樹（repo 已有 `hierarchical_codebase`） | payment-svc > 連線池設定檔 |
| 3 | Causality | 最可能的原因？ | 最近部署／設定變更／流量尖峰／下游變慢 | 最近部署（連線池改小） |
| 4 | Conditional | 哪些處置不准做？ | change freeze：剪掉「上 hotfix」；error budget 用完：剪掉「擴大灰度」 | 剩 rollback、關 feature flag、擴容 |
| 5 | Procedure | 對應哪份 runbook？ | 查表 | 3 份 runbook |
| 6 | Judgement | 選哪個處置？ | rubric：恢復時間、影響範圍、資料風險 | rollback |

- **Meta**：幾個原因的分數都很低（不像任何已知模式）時，page 第二線，並加寬 beam，把第三方金流這類外部原因也納入。
- **特點**：Conditional 剪掉的是 Judgement 的候選動作（hotfix 被 freeze 擋掉），不是診斷的分支。

### 6. 學生學習診斷

**輸入**：國二學生解 x² − 5x + 6 = 0，寫 x = 2 或 x = −3。前兩題同類型也錯。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 哪個單元、哪種題型？ | 課綱分類樹：數學 > 代數 > 一元二次方程式 > 因式分解解法 | 因式分解解法 |
| 2 | Relational | 卡在哪個先備知識？ | 先備知識圖（往回走）：因式分解 ← 十字交乘 ← 整數乘法與正負號 | 十字交乘 > 正負號 |
| 3 | Causality | 是哪個錯誤迷思？ | 迷思樹：正負號配錯／移項沒變號／只看常數項 | 十字交乘時正負號配錯 |
| 4 | Conditional | 哪些教法不適用？ | 硬規則：國二還沒教公式解 → 剪掉「用公式解驗算」 | 剩圖形法、代數拆解 |
| 5 | Procedure | 補救教學怎麼做？ | 查表：概念說明 → 2 題例題 → 3 題練習 → 1 題檢測 | 對應補救單元 |
| 6 | Judgement | 下一題出什麼？ | rubric：難度、挫折風險、能不能測到這個迷思 | 先出常數項為正的簡單題 |

- **Meta**：同一個迷思補救後還是連錯 3 次，就換整套策略（從代數改用面積圖），並記下「這個學生對圖像教法反應比較好」。
- **特點**：Relational 段沿著先備知識往回走，越走越基礎。Meta 的「換策略」在教育場景最常用到。

### 7. 航空班機異常調度

**輸入**：颱風讓機場下午 2 點到 6 點關閉。某架飛機原本要接著飛 3 段，機組今天已經執勤 9 小時。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 哪一類異常？ | 天候／機械／機組／航管／旅客事件 | 天候 > 機場關閉 |
| 2 | Relational | 牽動哪些資源？ | 網路圖：航班 → 機體輪轉 → 機組排班 → 轉機旅客 | 後續 3 段航班＋機組＋轉機旅客 |
| 3 | Causality | 會連鎖影響到哪裡？ | 延誤傳播：晚 4 小時 → 第 2 段誤點 → 第 3 段撞上目的地宵禁 | 第 3 段撞宵禁 |
| 4 | Conditional | 哪些方案不合法？ | 硬規則：機組執勤時數上限、目的地宵禁、起降時間帶 | 剪掉「原機組飛完 3 段」 |
| 5 | Procedure | 候選方案各自的 SOP | 查表：換機組／取消第 3 段／換機體，各自的改票與補償規則 | 3 個候選 |
| 6 | Judgement | 選哪個方案？ | rubric：受影響旅客數、成本、恢復時間、補償金額 | 換機組，第 3 段延後 |

- **Meta**：多個機場同時受影響，各航班各自最好的方案互相衝突（搶同一組備用機組）時，交給營運控制中心整體重排。
- **特點**：Procedure 產出的是好幾個候選方案，真正的決策在 Judgement。Conditional 管的是安全和法規，只能用硬規則，不能交給模型判斷。

### 8. 1999 市民陳情分派

**輸入**：「我家門口的路面又凹了一個大洞，上個月才補過，機車騎過去很危險。」附上地址。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 哪一類陳情？ | 陳情分類樹：道路／噪音／違建／環保／號誌 | 道路 > 路面坑洞 |
| 2 | Relational | 這段路的結構？ | Graph：地址 → 行政區 → 道路等級 → 路面下的管線與最近的挖掘紀錄 | 市區道路，半年前自來水管線施工 |
| 3 | Conditional | 誰受理？ | 硬規則：依道路等級決定主管機關 | 工務單位受理 |
| 4 | Causality | 為什麼一直壞？ | 管線回填不實／重車輾壓／排水不良 | 回填不實 |
| 5 | Procedure | 派工流程 | 查表：有安全疑慮 → 先臨時補平；挖掘保固期內 → 通知挖掘單位限期重做 | 先補平，再通知自來水單位 |
| 6 | Judgement | 排第幾優先？ | rubric：安全風險、重複陳情次數、影響人數 | 高優先（機車道、第 2 次陳情） |

- **Meta**：兩個單位互相退件（「不是我們管的」）時，自動升級給上層協調；把這次的權責判定寫回，下次同一段路直接分對。
- **特點**：「誰受理」跟原因無關，所以 Conditional 在前；「誰出錢修」要看原因，交給 Causality 之後的 Procedure 查表。

### 9. 內容審核／AI 護欄

**輸入**：一則貼文附上某人的照片和住址，寫「這個人就是詐騙集團，大家去他家問清楚」。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Conditional | 有沒有碰到零容忍的規則？ | 硬規則（雜湊比對、黑名單）：兒少性剝削、恐攻宣傳 → 直接下架，後面都不問 | 沒碰到，往下 |
| 2 | Fact | 哪一類違規？ | 政策分類樹：仇恨／騷擾／詐騙／自傷／不實資訊 | 騷擾 > 公開個資（肉搜） |
| 3 | Relational | 針對誰？ | Graph：發文者 → 被指涉的人（一般人／公眾人物）→ 社群規模 | 一般人，大型社群 |
| 4 | Causality | 意圖是什麼？ | 意圖樹（先剝掉包裝，同 deep9 的 refuse）：提醒他人／煽動行動／新聞報導 | 煽動行動（「去他家」） |
| 5 | Procedure | 處置流程 | 查表：公開個資 → 遮蔽個資、下架、通知當事人 | 對應流程 |
| 6 | Judgement | 處置要多重？ | Score：嚴重度（同 deep9 的 severity）× 影響範圍 | 下架＋停權 |

- **Meta**：出現沒看過的黑話或新手法，意圖段的分數都偏低時送人工審；審完的標註寫回，累積夠多就加進政策樹。
- **特點**：Conditional 放第一段，零容忍規則不需要理解內容。這跟 deep9 直接對應：refuse 的門檻（P ≥ 0.70 就擋）
  是 Conditional，severity 的 Score 是 Judgement 的刻度。

### 10. 家庭烘焙失敗診斷

**輸入**：照食譜做吐司，結果沒長高、組織很密。用的是速發酵母，水是熱水壺剛燒開放一下就用。冬天，家裡大約 16 度。

| 段 | 知識 | 問的問題 | 選項從哪來（例子） | 這次走到 |
| --- | --- | --- | --- | --- |
| 1 | Fact | 哪一種麵包或點心？ | 品項樹：麵包 > 吐司 > 直接法 | 直接法吐司 |
| 2 | Relational | 問題在配方還是流程的哪一步？ | 配方結構（麵粉、水、酵母、鹽）＋流程（攪拌 → 一發 → 整形 → 二發 → 烘烤） | 一發 |
| 3 | Causality | 為什麼沒長高？ | 因果樹：酵母失活（水太燙）／發酵不足（室溫低）／筋性不足（揉不夠） | 主因酵母失活，次因室溫低 |
| 4 | Conditional | 使用者有什麼限制？ | 沒有溫度計、沒有發酵箱、不想用改良劑 → 剪掉需要這些的修法 | 剩「手測水溫」「烤箱放熱水當發酵箱」 |
| 5 | Procedure | 修正後的步驟 | 查表：手測水溫 → 一發延長到 90 分鐘 → 手指測試判斷發酵完成 | 修正版食譜 |
| 6 | Judgement | 先改哪一個？ | rubric：成功率、多花的時間、對口感的影響 | 先改水溫（成本最低、影響最大） |

- **Meta**：用了樹裡沒有的材料（例如無麩質粉）時，撈過去相似的成功案例當參考，並明說這部分沒有把握。
- **特點**：Causality 常有主因和次因，beam 同時保留兩條路徑，剛好能說出「主要是水太燙，室溫低也有影響」。

## 四、從十個場景看出來的規律

1. **Conditional 放哪裡，看條件需不需要先知道原因。** 不需要的（鑑賞期、零容忍、受理單位）放前面，
   一剪就省掉整棵子樹；需要的（保險近因、run card 放行）放在 Causality 後面。
2. **Causality 有兩個方向。** 往回找根因的（fab、事故、烘焙），子節點是「原因」；往前推後果的（合約、航空），
   子節點是「後果」，Judgement 比的是後果有多嚴重。
3. **Procedure 幾乎都不用叫模型。** 它是查表，真正的決策在 Judgement，而且 Judgement 要把 beam 留下的
   幾條診斷都考慮進去，不能只看第一名。
4. **Meta 的觸發條件和動作每個場景都差不多。** 觸發條件：分數太接近、掉進「其他」、同一條路重複失敗。
   動作：加寬 beam、補證據再問、交給人、寫回軌跡庫。

## 五、要驗證的話

- **每一段都要有標準答案。** 只看最後的葉子不夠：Fact 對但 Causality 錯，跟一開始就走錯，是兩種不同的問題。
- **對照組**：(a) 串成一棵、帶路徑；(b) 串成一棵、不帶路徑；(c) 每段分開各問一次。每組都比 greedy 和 beam。
  帶路徑的錨定風險要特別看 Causality 段：錯的路徑分數有沒有被墊高。
- **從現有資料起步**：電商退貨的 Fact 段可以直接用 `hierarchical_shopify`，線上事故的 Relational 段可以接
  `hierarchical_codebase`，這兩個場景最省事。
