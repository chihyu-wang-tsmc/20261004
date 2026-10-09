import requests

# imajev 服務（scripts/playground/server.py 開在 port 8765）；decider 是 8000，jevk5 是 8090
URL = "http://127.0.0.1:8765/v1/systemone"

# 跟 d1.py 相同的問題（decider 的 README 範例）。imajev 的 criteria 格式跟 Jev 一樣：
# choice 用 dict（說明可以是字串、None，或像 billing 那樣的結構化 dict，imajev 會攤平成 "what: ...; not_for: ..."），
# score 用由低到高的 list；instructions 裡的 `ticket.messages[0].text` 指向 state 裡的欄位
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


def ask(state, questions):
    """沒有照片，直接送 JSON（有照片時才用 multipart：files={"image": ...}, data={"request": ...}）。"""
    resp = requests.post(URL, json={"state": state, "questions": questions}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def bar(p, width=30):
    return "█" * round(p * width)


def show(questions, result):
    print("=" * 60)
    for name, ans in result["answers"].items():
        print(f"問題: {questions[name]['instructions']}")
        # imajev 每個答案都多了 unknown_probability（模型判斷「看不出來」的機率）和 abstained（是否放棄作答）
        unknown = f"看不出來 {ans['unknown_probability']:.1%}" + ("，放棄作答" if ans["abstained"] else "")
        if ans["type"] == "noul":
            p_true = ans["noul"]
            print(f"判斷: {'是' if p_true >= 0.5 else '否'}  ({unknown})")
            print(f"  是 {p_true:6.1%} {bar(p_true)}")
            print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
        else:
            # confidence 是 Jev 的定義（均勻分布為 0），再乘上 (1 - unknown_probability)
            answer = ans.get("choice", ans.get("score"))
            answer = f"{answer:.2f}" if ans["type"] == "score" else answer
            print(f"判斷: {answer}  (信心 {ans['confidence']:.1%}，{unknown})")
            probs = sorted(ans["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
            # score 的選項是等級編號 "0"、"1"…，用 legend 換回文字
            legend = ans.get("legend", {})
            labels = {k: f"{k} {legend[k]}" if k in legend else k for k, _ in probs}
            width = max(len(v) for v in labels.values())
            for option, p in probs:
                print(f"  {labels[option]:<{width}} {p:6.1%} {bar(p)}")
        print("-" * 60)
    usage = result["usage"]
    print(f"模型 {result['model']}，耗時 {usage['total_ms']:.0f} ms，輸入 {usage['input_tokens']} tokens")


show(questions, ask(state, questions))

# d1.py 最後的「簡單格式」（decider 的 d.decide）：imajev 沒有這個 API，改用同一個 /v1/systemone 問
simple = {"team": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": None, "technical": None, "sales": None}}}
print()
show(simple, ask("My card was charged twice.", simple))
