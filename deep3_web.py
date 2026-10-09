import ast
import json
import os
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
    model_router,
    AutoModeMiddleware, 
    auto_mode,
)
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from deepagents import create_deep_agent
from deepagents_code.tools import web_search, fetch_url

from classifier_jevk5 import JevK5Classifier
from classifier_decider import DeciderClassifier
from classifier_deepseek import DeepSeekClassifier
from classifier_chatgpt import ChatGPTClassifier
from classifier_imajev import ImajevClassifier
from llm_models import qwen

SelectedClassifier = ImajevClassifier   

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
    tools=[web_search, fetch_url, delete_all_backups]
)


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


# ===== 測試 web_search / fetch_url =====
# 這兩個工具失敗時不會丟例外，而是回傳 {"error": ...}，所以要檢查回傳內容，不能只看有沒有報錯。
# web_search 需要環境變數 TAVILY_API_KEY；fetch_url 會擋內網 IP，所以用公開網址測。
test_results = []


def check(name, ok, detail):
    test_results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def as_dict(content):
    """ToolMessage 的內容是字串，轉回 dict；轉不了就回傳 None。"""
    if isinstance(content, dict):
        return content
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(content)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


# 1) 直接呼叫工具（不經過模型），確認工具本身能用
r = fetch_url("https://example.com", timeout=15)
check(
    "fetch_url 直接呼叫",
    "error" not in r
    and r.get("status_code") == 200
    and "Example Domain" in r.get("markdown_content", ""),
    r.get("error") or f"status {r.get('status_code')}，內容 {r.get('content_length')} 字",
)

r = web_search("LangChain deepagents GitHub repository", max_results=3)
hits = r.get("results", []) if isinstance(r, dict) else []
check(
    "web_search 直接呼叫",
    isinstance(r, dict) and "error" not in r and len(hits) > 0,
    r.get("error")
    if isinstance(r, dict) and "error" in r
    else f"{len(hits)} 筆結果，第一筆 {hits[0].get('url') if hits else '-'}",
)

# 2) 透過 agent 呼叫，確認模型真的會用這兩個工具，而且工具回傳成功
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use fetch_url to read https://example.com and tell me its title. "
                    "Then use web_search to find the GitHub repository of LangChain "
                    "deepagents and give me its URL."
                ),
            }
        ]
    },
    config={"callbacks": [langfuse_handler]},
)
for tool_name in ("fetch_url", "web_search"):
    msgs = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == tool_name
    ]
    if not msgs:
        check(f"{tool_name} 透過 agent", False, "模型沒有呼叫這個工具")
        continue
    data = as_dict(msgs[-1].content)
    ok = msgs[-1].status != "error" and data is not None and "error" not in data
    detail = (
        (data or {}).get("error")
        or (f"呼叫 {len(msgs)} 次，回傳成功" if ok else str(msgs[-1].content)[:200])
    )
    check(f"{tool_name} 透過 agent", ok, detail)
result["messages"][-1].pretty_print()

passed = sum(ok for _, ok in test_results)
print(f"\n測試結果：{passed}/{len(test_results)} 通過")

root.end()
langfuse.flush()
