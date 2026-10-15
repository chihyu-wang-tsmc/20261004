"""jev4.py 的 H2O-Lightning-4B 版：路由問題改問本機的 h2o-lightning 服務。

跟 jev4.py 只有一處不同：換掉的 classifier 從 JevK5Classifier 變成 H2OClassifier。
被路由的兩顆 qwen 模型、instructions、prompt 都沒改，輸出可以直接和 jev4.py 對照。

為什麼是 monkeypatch 模組屬性，而不是建完再改 router.classifier：
ModelRouterMiddleware.__init__ 的最後一行是 self.classifier = TypeSafeClassifier()
（model_router.py:152，無參數），而 TypeSafeClassifier 的 api_key 驗證器在沒有
TYPESAFE_API_KEY 時就會丟 ValidationError，所以 router 根本建不起來，輪不到事後替換。
先把模組看到的那個名字換掉，__init__ 裡那行就變成 self.classifier = H2OClassifier()。

要用非預設的網址或 timeout，建完之後再覆蓋一次就行（這時候實例已經存在了）：
    router.classifier = H2OClassifier(base_url="http://127.0.0.1:8741", timeout=120)

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    cd ~/20261004 && python jev4_h2o.py
"""

import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
    model_router,
)
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from classifier_h2o import H2OClassifier

langfuse_handler = CallbackHandler()
model_router.TypeSafeClassifier = H2OClassifier


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
    },
    config={"callbacks": [langfuse_handler]},
)
route = result["model_route"]
print(f"h2o-lightning 選擇: {route.choice}  (信心 {route.confidence:.1%})")
for key, p in route.probabilities.items():
    print(f"  {key:<10} {p:6.1%}")
result["messages"][-1].pretty_print()

get_client().flush()
