from langchain_typesafe import Choice, Noul, Score
from classifier_jevk5 import JevK5Classifier

classifier = JevK5Classifier()

response = classifier.invoke(
    {
        "state": (
            "The deploy failed twice and customers are seeing 500s. "
            "Can someone look now?"
        ),
        "questions": {
            "urgent": Noul(instructions="Does this need attention right now?"),
            "team": Choice(
                instructions="Which team should pick this up?",
                criteria={
                    "infra": "Deploys, availability, and on-call incidents.",
                    "billing": "Payments, invoices, and subscriptions.",
                },
            ),
            "severity": Score(
                instructions="How severe is the impact?",
                criteria=["Cosmetic.", "Degraded for some users.", "Full outage."],
            ),
        },
    }
)

severity_levels = ["Cosmetic.", "Degraded for some users.", "Full outage."]
urgent = response.nouls["urgent"]
team = response.choices["team"]
severity = response.scores["severity"]

print(f"緊急嗎?   {'是' if urgent.noul >= 0.5 else '否'}  (是的機率 {urgent.noul:.1%})")
print(f"負責團隊: {team.choice}  (信心 {team.confidence:.1%})")
top_level = int(max(severity.probabilities, key=severity.probabilities.get))
print(
    f"嚴重程度: {severity.score:.2f} / 2  → 最可能是「{severity_levels[top_level]}」"
    f" (信心 {severity.confidence:.1%})"
)
