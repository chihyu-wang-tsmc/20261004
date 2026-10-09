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
from langchain_typesafe.experimental.middleware.auto_mode import (
    _PROBABILITY_THRESHOLD,
    _QUESTION_ID,
    _risk_questions,
)
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command, interrupt
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents_code.tools import web_search, fetch_url

from classifier_jevk5 import JevK5Classifier
from classifier_decider import DeciderClassifier
from llm_models import deepseek, qwen

model_router.TypeSafeClassifier = JevK5Classifier
auto_mode.TypeSafeClassifier = JevK5Classifier

langfuse = get_client()
root = langfuse.start_observation(name="deep2", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)


DEEPSEEK_V41_FLASH_MODEL = deepseek("deepseek-flash")
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


class RiskApprovalMiddleware(AutoModeMiddleware):
    """跟 AutoModeMiddleware 一樣先讓 classifier 判斷工具呼叫的風險，但有風險時不直接拒絕，
    而是用 interrupt() 暫停、交給人核准（HITL）；核准才執行，拒絕就回錯誤訊息給模型。

    暫停的內容和繼續的格式跟 LangChain 內建的 HumanInTheLoopMiddleware 相同：
    interrupt 的值是 {"action_requests": [...], "review_configs": [...]}，
    用 Command(resume={"decisions": [{"type": "approve"}]}) 或 {"type": "reject", "message": ...} 繼續。
    agent 需要 checkpointer 才能暫停後接著跑。
    """

    def _review(self, request, probability):
        """暫停等人決定；回傳 None 代表核准，否則回傳拒絕的 ToolMessage。"""
        tool_call = request.tool_call
        decision = interrupt(
            {
                "action_requests": [
                    {
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "description": f"classifier 判斷這個工具呼叫有風險（機率 {probability:.2f}），需要核准才會執行。",
                    }
                ],
                "review_configs": [
                    {"action_name": tool_call["name"], "allowed_decisions": ["approve", "reject"]}
                ],
            }
        )["decisions"][0]
        if decision["type"] == "approve":
            return None
        return ToolMessage(
            content=decision.get("message")
            or f"The tool call `{tool_call['name']}` was rejected by a human reviewer. The tool was not executed.",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            status="error",
        )

    def _risk(self, response):
        return response.nouls[_QUESTION_ID].noul

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self._tool_names:
            return handler(request)
        probability = self._risk(
            self.classifier.invoke(
                {"state": self._classification_state(request), "questions": _risk_questions(self.config)}
            )
        )
        if probability >= _PROBABILITY_THRESHOLD:
            rejected = self._review(request, probability)
            if rejected is not None:
                return rejected
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self._tool_names:
            return await handler(request)
        probability = self._risk(
            await self.classifier.ainvoke(
                {"state": self._classification_state(request), "questions": _risk_questions(self.config)}
            )
        )
        if probability >= _PROBABILITY_THRESHOLD:
            rejected = self._review(request, probability)
            if rejected is not None:
                return rejected
        return await handler(request)


# skill 是從 backend 讀的：/skills/ 導到硬碟上的 ~/skills_anthropic（virtual_mode 擋掉 .. 跳出去），
# 其他路徑維持預設的虛擬檔案系統，agent 寫檔不會碰到真的硬碟。
SKILLS_DIR = os.path.expanduser("~/skills_anthropic")
backend = CompositeBackend(
    default=StateBackend(),
    routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
)

agent = create_deep_agent(
    model=DEEPSEEK_V41_FLASH_MODEL,
    system_prompt="You are a helpful assistant.",
    # 有風險的呼叫改成暫停等人核准，而不是直接拒絕
    middleware=[router, RiskApprovalMiddleware(tools=[delete_all_backups])],
    tools=[web_search, fetch_url, delete_all_backups],
    backend=backend,
    skills=["/skills/"],
    # skill 檔案只能讀，不能被 agent 改寫或刪除
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")
    ],
    # 暫停後要能接著跑，需要 checkpointer 保存進度。router 把 ChoiceAnswer 存在 state 裡，
    # 要先登記成允許還原的型別，否則 LangGraph 會警告、未來版本會直接擋掉
    checkpointer=MemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[("langchain_typesafe.types", "ChoiceAnswer")]
        )
    ),
)

config = {"configurable": {"thread_id": "deep5"}, "callbacks": [langfuse_handler]}
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete all backups."
            }
        ]
    },
    config=config,
)

# 有風險的工具呼叫會停在這裡：印出來等你決定，再用 Command(resume=...) 繼續
while result.get("__interrupt__"):
    resume = {}
    for pending in result["__interrupt__"]:
        decisions = []
        for action in pending.value["action_requests"]:
            print("\n" + "=" * 60)
            print(f"agent 想執行：{action['name']}({action['args']})")
            print(action["description"])
            if input("核准執行？[y/N] ").strip().lower() == "y":
                decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "reject", "message": "使用者拒絕執行這個工具。"})
        resume[pending.id] = {"decisions": decisions}
    # 只有一個暫停時直接給值；同時有多個（平行的工具呼叫）時用 interrupt id 對應
    result = agent.invoke(
        Command(resume=next(iter(resume.values())) if len(resume) == 1 else resume),
        config=config,
    )

# 印出完整過程：使用者 -> 模型要求呼叫工具 -> 工具結果（被拒絕時是錯誤訊息）-> 最後回答
for msg in result["messages"]:
    msg.pretty_print()

root.end()
langfuse.flush()
