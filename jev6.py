import os

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain_openai import ChatOpenAI
from langchain_typesafe import Choice, ChoiceAnswer
from typing_extensions import NotRequired

from classifier_jevk5 import JevK5Classifier


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
        self.classifier = JevK5Classifier()

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
print(f"jevk5 分派: {triage.choice}  (信心 {triage.confidence:.1%})")
for team, p in triage.probabilities.items():
    print(f"  {team:<8} {p:6.1%}")
result["messages"][-1].pretty_print()
