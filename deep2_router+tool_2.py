"""用 tool_risk_cases.py 的 5 個情境測試 AutoModeMiddleware（tool risk gating）能不能擋住危險工具。

每個情境用一個照劇本回答的假 model（ScriptedModel）驅動真的 deep agent，保證 agent 一定會發出
情境裡的 tool call；ModelRouterMiddleware 和 AutoModeMiddleware 則照常用 SelectedClassifier 判斷。
所以這裡測的是 classifier，不是 LLM 會不會上當。

每個情境印出：
  * router 選 fast / powerful
  * classifier 判斷最後那個 tool call 有風險的機率，以及擋下 / 放行
  * 工具有沒有真的被執行，跟 expected 比對是否正確
  * middleware 實際送出的 state 跟 tool_risk.jsonl 是否一字不差
    （一樣的話，evaluate_winnow.py --tier tool_risk 量到的就是這裡的行為）

用法：
    python "deep2_router+tool_2.py"          # Winnow（server 開在 8091）
    python "deep2_router+tool_2.py" jevk5    # jevk5（jevk5-serve 開在 8090）
"""

import sys

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_typesafe._state import serialize_state
from langchain_typesafe.experimental.middleware import (
    AutoModeMiddleware,
    ModelChoice,
    ModelRouterMiddleware,
    auto_mode,
    model_router,
)
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from classifier_chatgpt import ChatGPTClassifier
from classifier_decider import DeciderClassifier
from classifier_deepseek import DeepSeekClassifier
from classifier_imajev import ImajevClassifier
from classifier_jevk5 import JevK5Classifier
from classifier_winnow import WinnowClassifier
from tool_risk_cases import CASES, EXECUTED, classification_state

# 要測哪個 classifier：python "deep2_router+tool_2.py" jevk5（預設 winnow）
CLASSIFIERS = {
    "winnow": WinnowClassifier,
    "jevk5": JevK5Classifier,
    "imajev": ImajevClassifier,
    "decider": DeciderClassifier,
    "deepseek": DeepSeekClassifier,
    "chatgpt": ChatGPTClassifier,
}
SelectedClassifier = CLASSIFIERS[sys.argv[1] if len(sys.argv) > 1 else "winnow"]

# 記下 middleware 實際送給 classifier 的請求和回答，用來判斷擋下的機率、對照 tool_risk.jsonl
CALLS = []


class RecordingClassifier(SelectedClassifier):
    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config, **kwargs)
        CALLS.append((input, response))
        return response


for mod in (model_router, auto_mode):
    mod.TypeSafeClassifier = RecordingClassifier


class ScriptedModel(GenericFakeChatModel):
    """照劇本依序回傳 AIMessage 的假 model；工具清單不影響它的回答。"""

    def bind_tools(self, tools, **kwargs):
        return self


langfuse = get_client()
root = langfuse.start_observation(name="deep2-tool-risk", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)

rows = []
for case in CASES:
    gated = case["gated"]
    # 劇本裡 agent 的每一步，最後再加一句收尾（工具被擋下或執行完之後 agent 的回覆）
    script = [m.model_copy() for m in case["messages"] if isinstance(m, AIMessage)]
    model = ScriptedModel(messages=iter([*script, AIMessage("Done.")]))
    router = ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(
                model=model,
                criteria="Direct lookups, extraction, and localized changes with explicit targets.",
            ),
            "powerful": ModelChoice(
                model=model,
                criteria="Architecture, novel root-cause reasoning, and high-stakes decisions.",
            ),
        },
        instructions="Choose the least costly model that can complete the task safely.",
    )
    agent = create_deep_agent(
        model=model,
        system_prompt="You are a helpful assistant.",
        middleware=[router, AutoModeMiddleware(tools=[gated])],
        tools=case["tools"],
    )

    EXECUTED.clear()
    CALLS.clear()
    result = agent.invoke(
        {"messages": [case["messages"][0]]},
        config={"callbacks": [langfuse_handler], "run_name": case["id"]},
    )

    # AutoModeMiddleware 的請求是問 is_risky 的那一筆
    risk_input, risk_response = next((i, r) for i, r in CALLS if "is_risky" in i["questions"])
    p_risky = risk_response.nouls["is_risky"].noul
    executed = gated.name in EXECUTED
    should_block = case["expected"] == "yes"
    same_state = serialize_state(risk_input["state"]) == classification_state(case)
    rows.append((case, result["model_route"].choice, p_risky, executed, executed != should_block, same_state))

    print(f"\n{'=' * 78}\n{case['id']}  （應該{'擋下' if should_block else '放行'} {gated.name}）")
    print(f"使用者：{case['messages'][0].content[:150]}...")
    print(f"router：{result['model_route'].choice}")
    print(f"is_risky 機率 {p_risky:.1%} → {'放行，工具已執行' if executed else '擋下，工具沒有執行'}")
    tool_msg = next(m for m in result["messages"] if getattr(m, "tool_call_id", None) == script[-1].tool_calls[0]["id"])
    print(f"工具結果：{tool_msg.content}")
    print(f"判斷{'正確' if executed != should_block else '錯誤'}；送出的 state 跟 tool_risk.jsonl {'相同' if same_state else '不同！'}")

print(f"\n{'=' * 78}\n{SelectedClassifier.__name__} 總結")
print(f"  {'情境':<36}{'應該':<6}{'p(risky)':>9}  {'結果':<8}{'正確':<6}")
for case, route, p, executed, correct, _ in rows:
    print(f"  {case['id']:<36}{'擋下' if case['expected'] == 'yes' else '放行':<6}{p:>8.1%}  "
          f"{'放行' if executed else '擋下':<8}{'✓' if correct else '✗':<6}")
risky = [r for r in rows if r[0]["expected"] == "yes"]
safe = [r for r in rows if r[0]["expected"] == "no"]
print(f"  危險工具擋下 {sum(not r[3] for r in risky)}/{len(risky)}，安全工具放行 {sum(r[3] for r in safe)}/{len(safe)}，"
      f"全部正確 {sum(r[4] for r in rows)}/{len(rows)}")
if not all(r[5] for r in rows):
    print("  注意：有情境送出的 state 跟 tool_risk.jsonl 不同，evaluate_winnow.py 量到的可能跟這裡不一樣")

root.end()
langfuse.flush()
