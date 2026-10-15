"""jev6.py 的 H2O-Lightning-4B 版：分派問題改問本機的 h2o-lightning 服務。

跟 jev6.py 只有一處不同：TriageMiddleware.__init__ 裡的 self.classifier 從
JevK5Classifier() 換成 H2OClassifier()。問題、三個 criteria、被包的模型和 prompt
都沒改，輸出可以直接和 jev6.py 對照。

這支不像 jev4.py 需要 monkeypatch：middleware 是這個檔案自己定義的，classifier 就在
自己的 __init__ 裡建，沒有經過 TypeSafeClassifier 的 api_key 驗證器。

要用非預設的網址或 timeout 就直接傳進去：
    self.classifier = H2OClassifier(base_url="http://127.0.0.1:8741", timeout=120)

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    cd ~/20261004 && python jev6_h2o.py
"""

import os

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain_openai import ChatOpenAI
from langchain_typesafe import Choice, ChoiceAnswer
from typing_extensions import NotRequired

from classifier_h2o import H2OClassifier


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


class TriageState(AgentState):
    triage: NotRequired[ChoiceAnswer]


class TriageMiddleware(AgentMiddleware[TriageState]):
    state_schema = TriageState

    def __init__(self) -> None:
        self.classifier = H2OClassifier()

    def before_agent(
        self, state: TriageState, runtime: Runtime
    ) -> dict[str, ChoiceAnswer]:
        response = self.classifier.invoke(
            {
                "state": state["messages"],
                "questions": {
                    "triage": Choice(
                        instructions="Which team should handle this conversation?",
                        criteria={
                            "billing": "Payments, invoices, and subscriptions.",
                            "infra": "Deploys, availability, and incidents.",
                            "other": "Requests that belong to another team.",
                        },
                    )
                },
            }
        )
        return {"triage": response.choices["triage"]}


agent = create_agent(
    QWEN_38_FLASH_NEXT_MODEL,
    middleware=[TriageMiddleware()],
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Customers are seeing 500 errors."}]}
)
triage = result["triage"]
print(f"h2o-lightning 分派: {triage.choice}  (信心 {triage.confidence:.1%})")
for team, p in triage.probabilities.items():
    print(f"  {team:<8} {p:6.1%}")
result["messages"][-1].pretty_print()
