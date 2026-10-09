from huggingface_hub import snapshot_download

from decider.infer import Decider

# JevBench v1.4.2 第一名測的是 decider-4b v2（decider_config version 4b-v2, T=1.935）。
# Hub 上 main 現在是 v2.1，所以要釘住官方測的那個 commit；Decider 沒有 revision 參數，先下載到本機再給資料夾路徑。
DECIDER_4B_V2 = "7ab294cbdf6be6ac17fc818c10cdead744393d92"

path = snapshot_download("Mapika/decider-4b", revision=DECIDER_4B_V2)   # 第一次會下載約 8.4 GB
d = Decider(path)                                                        # CUDA, bf16
print(f"模型: {d.name}  溫度 T={d.T}")

state = {
    "ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
    "refund_policy": "Duplicate charges are eligible for a refund.",
}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "billing": {"what": "Charges, invoices", "not_for": "delivery"},
            "other": None,
        },
    },
    "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]},
}

result = d.system_one(state, questions)


def bar(p, width=30):
    return "█" * round(p * width)


print("=" * 60)
for name, ans in result["answers"].items():
    print(f"問題: {questions[name]['instructions']}")
    if "noul" in ans:
        p_true = ans["noul"]
        print(f"判斷: {'是' if p_true >= 0.5 else '否'}")
        print(f"  是 {p_true:6.1%} {bar(p_true)}")
        print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
    else:
        answer = ans.get("choice", ans.get("score"))
        print(f"判斷: {answer}  (信心 {ans['confidence']:.1%})")
        probs = sorted(ans["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        width = max(len(k) for k, _ in probs)
        for option, p in probs:
            print(f"  {option:<{width}} {p:6.1%} {bar(p)}")
    print("-" * 60)

# 簡單格式：一段文字 + 選項清單
print(d.decide("My card was charged twice.", [{"question": "Which team?", "options": ["billing", "technical", "sales"]}]))
