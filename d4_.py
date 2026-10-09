import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
    model_router,
)

from classifier_decider import DeciderClassifier

model_router.TypeSafeClassifier = DeciderClassifier


# Qwen via Alibaba Cloud DashScope's OpenAI-compatible API (same as test_agent.py)
def qwen(model_name):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        temperature=0,
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

agent = create_agent(QWEN_38_27B_MODEL, middleware=[router])

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Prove that there are infinitely many prime numbers.",
            }
        ]
    }
)
route = result["model_route"]
print(f"jevk5 選擇: {route.choice}  (信心 {route.confidence:.1%})")
for key, p in route.probabilities.items():
    print(f"  {key:<10} {p:6.1%}")
result["messages"][-1].pretty_print()
