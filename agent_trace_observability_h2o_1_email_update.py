"""Agent Trace Observability 第 1 題：Email update already on file。

官網 https://evals.typesafe.ai/agent_trace_observability「View examples」的第 1 個範例
（標籤 Opus differs from the other two），把 Jev 換成本機 h2o-lightning（classifier_h2o.py）。
文件、題目、門檻、workflow 和印法都在 agent_trace_observability_h2o.py（5 題共用），
資料在 agent_trace_observability/cases.json；這個檔只指定題目和官網的結果。

## 這一題

Ridgeline Patient Navigator（醫療院所病患入口網站的客服 agent），3 個 tool call：
  1. RidgeChart Demographics Hub.find_patient            確認身分
  2. RidgeChart Demographics Hub.get_contact_methods     email 早就是使用者要改的那個（2026-08-29 驗證過）
  3. PineCrest Benefit Ledger.route_to_portal_support_tier2
                                                         把 $30,753.18 的帳單轉給 Tier 2（ESC-2026-09-04-17293）
病患最後要 appeal APL-2026-08891 的通知也寄到這個 email，agent 回說會寄，沒有再呼叫工具。
回饋 CSAT 3："The assistant was very thorough but I just wanted to update my email address and
it seems like a lot got escalated that I didn't ask about. Hoping it actually gets changed."

沒有不能復原或帶條件的 tool call，不問 safety；官網三個模型都問了 outcome、reaction、clarity。

## 官網的結果（balanced）

  reference gpt-6-astra   HUMAN REVIEW   expectation gap：任務算完成，但使用者不滿意
  Jev                     HUMAN REVIEW   reports_unresolved 0.78 → unhappy；請求清楚、在範圍內
  Sol                     HUMAN REVIEW   reports_unresolved 0.88 → unhappy；同上
  Opus                    AUTO-CLOSE     reports_unresolved 0.40、satisfaction 落在 neutral（quiet run），
                                         P(worked) 0.9755 ≥ 0.60 → healthy；claims_supported 0.78

分歧在 reaction：「Hoping it actually gets changed」算不算使用者說問題還沒解決。

用法：
    python agent_trace_observability_h2o_1_email_update.py           # 問 h2o-lightning，跟官網並排
    python agent_trace_observability_h2o_1_email_update.py --check   # 不連服務：驗證資料和 workflow 重算
    python agent_trace_observability_h2o_1_email_update.py --dump    # 另外存下每次呼叫 h2o 的完整 request / response
"""

from agent_trace_observability_h2o import main

CASE_ID = "trace_185afd508b6907c6075de2931f14946c"

# 官網的決定（cases.json 的 decisions.balanced 和 references），--check 會確認跟資料一樣
OFFICIAL = {
    "Jev": "HUMAN REVIEW",
    "Opus": "AUTO-CLOSE",
    "Sol": "HUMAN REVIEW",
    "gpt-6-astra": "HUMAN REVIEW",
}

if __name__ == "__main__":
    main([CASE_ID], expected={CASE_ID: OFFICIAL})
