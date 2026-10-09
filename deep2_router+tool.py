import os
from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
    model_router,
    AutoModeMiddleware, 
    auto_mode,
)
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from classifier_jevk5 import JevK5Classifier
from classifier_decider import DeciderClassifier
from classifier_deepseek import DeepSeekClassifier
from classifier_chatgpt import ChatGPTClassifier
from classifier_imajev import ImajevClassifier
from classifier_winnow import WinnowClassifier
from llm_models import qwen

SelectedClassifier = WinnowClassifier   

for mod in (model_router, auto_mode):
    mod.TypeSafeClassifier = SelectedClassifier

langfuse = get_client()
root = langfuse.start_observation(name="deep2", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)


QWEN_38_27B_MODEL = qwen("qwen3.8-27b")
QWEN_38_FLASH_NEXT_MODEL = qwen("qwen3.8-flash")

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model=QWEN_38_27B_MODEL,
            criteria="Direct lookups, extraction, and localized changes with explicit targets.",
        ),
        "powerful": ModelChoice(
            model=QWEN_38_FLASH_NEXT_MODEL,
            criteria="Architecture, novel root-cause reasoning, and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task safely.",
)


@tool
def delete_all_backups() -> str:
    """Delete every backup. This action cannot be undone."""
    return "Backups deleted."


# deep agent 自帶 todo / 檔案系統 / task(子 agent) 等工具和 middleware，
# router 放在 middleware=，會接在內建 middleware 之後：
#   before_agent：jevk5 依最後一則使用者訊息選 fast / powerful，存到 state["model_route"]
#   wrap_model_call：之後每一次呼叫模型都換成選到的那個模型
# model= 只是預設模型；主 agent 的每次呼叫都會被 router 換掉，
# 但 task 工具開出的 general-purpose 子 agent 不會套用 router，會一直用這個模型。
agent = create_deep_agent(
    model=QWEN_38_27B_MODEL,
    system_prompt="You are a helpful assistant.",
    middleware=[router,AutoModeMiddleware(tools=[delete_all_backups])],
    tools=[delete_all_backups]
)
print('list(agent.get_graph().nodes)',list(agent.get_graph().nodes))
node = agent.nodes["tools"]
tools = node.bound.tools_by_name
print('list(tools)',list(tools))
skill_list=[n for n in agent.get_graph().nodes if "Skill" in n]

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Prove that there are infinitely many prime numbers.",
            }
        ]
    },
    config={"callbacks": [langfuse_handler]},
)
route = result["model_route"]
print(f"model 選擇: {route.choice}  (信心 {route.confidence:.1%})")
for key, p in route.probabilities.items():
    print(f"  {key:<10} {p:6.1%}")
result["messages"][-1].pretty_print()


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user", 
                "content": "Delete all backups."
            }
        ]
    },
    config={"callbacks": [langfuse_handler]},
)

# 印出完整過程：使用者 -> 模型要求呼叫工具 -> 工具結果（被 jevk5 擋下時是錯誤訊息）-> 最後回答
for msg in result["messages"]:
    msg.pretty_print()

root.end()
langfuse.flush()
