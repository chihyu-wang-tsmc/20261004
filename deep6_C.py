import functools
import os
import sys
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents_code.tools import fetch_url, web_search
from langchain_typesafe.experimental.middleware import AutoModeMiddleware, auto_mode, ModelChoice, ModelRouterMiddleware, model_router
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from classifier_jevk5 import JevK5Classifier
from classifier_decider import DeciderClassifier
from classifier_deepseek import DeepSeekClassifier
from llm_models import deepseek, gemini, glm, kimi, qwen

model_router.TypeSafeClassifier = JevK5Classifier
auto_mode.TypeSafeClassifier = functools.partial(JevK5Classifier, timeout=1800)

# model_router.TypeSafeClassifier = DeepSeekClassifier
# auto_mode.TypeSafeClassifier = functools.partial(DeepSeekClassifier, thinking=False)


langfuse = get_client()
root = langfuse.start_observation(name="deep6_C", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)


DEEPSEEK_V41_FLASH_MODEL = deepseek("deepseek-flash")
GEMINI_37_FLASH_MODEL = gemini("gemini-3.7-flash")
KIMI_K3_MODEL = kimi("kimi-k3")
GLM_53_MODEL = glm("glm-5.3")
GLM_47_FLASH_MODEL = glm("glm-4.7-flash")
QWEN_38_27B_MODEL = qwen("qwen3.8-27b")
QWEN_38_FLASH_NEXT_MODEL = qwen("qwen3.8-flash")

SKILLS_DIR = os.path.expanduser("~/skills_anthropic")
DECK_DIR = os.path.expanduser("~/jevk5/deck")

backend = CompositeBackend(
    default=LocalShellBackend(
        root_dir=DECK_DIR,
        virtual_mode=True,
        inherit_env=False,
        env={"PATH": f"{os.path.dirname(sys.executable)}:/usr/bin:/bin", "HOME": DECK_DIR},
        timeout=180,
    ),
    routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
)

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model=QWEN_38_27B_MODEL,
            criteria="Direct lookups, extraction, and localized changes with explicit targets.",
        ),
        "powerful": ModelChoice(
            model=KIMI_K3_MODEL,
            criteria="Architecture, novel root-cause reasoning, and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task safely.",
)



SYSTEM_PROMPT = f"""You are a helpful assistant"""

TASK = (
    "上網搜尋 TypeSafe AI 的 Jev做一份 6～8 頁的繁體中文投影片"
)

agent = create_deep_agent(
    # model=DEEPSEEK_V41_FLASH_MODEL,
    model=GEMINI_37_FLASH_MODEL,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search, fetch_url],
    backend=backend,
    skills=["/skills/"],
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")
    ],
    # middleware=[AutoModeMiddleware(tools=["execute"])],
    # middleware=[router,AutoModeMiddleware(tools=[delete_all_backups])],
    middleware=[router],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    config={"callbacks": [langfuse_handler]},
)

# router 在 before_agent 就決定好，結果存在 state["model_route"]；之後主 agent 每次呼叫模型都用它
route = result["model_route"]
# 用 .model 取模型名稱：ChatOpenAI 和 ChatGoogleGenerativeAI 都有 .model（Gemini 沒有 .model_name）
print(f"router 選擇: {route.choice} -> {router.models[route.choice].model}  (信心 {route.confidence:.1%})")
for key, p in route.probabilities.items():
    print(f"  {key:<10} {router.models[key].model:<15} {p:6.1%}")

result["messages"][-1].pretty_print()
out_file = os.path.join(DECK_DIR, "out", "jev_C.pptx")
print(f"投影片：{out_file}" if os.path.exists(out_file) else "沒有產生 out/jev_C.pptx")

root.end()
langfuse.flush()
