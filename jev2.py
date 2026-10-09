import requests

URL = "http://localhost:8090/v1/systemone"

payload = {
    "state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
    "questions": {
        "what": {
            "type": "choice",
            "instructions": "What happened to the parcel?",
            "criteria": ["delivered", "misdelivered", "unknown"],
        }
    },
}

resp = requests.post(URL, json=payload, timeout=60)
resp.raise_for_status()

result = resp.json()


def bar(p, width=30):
    return "█" * round(p * width)


print(f"情境: {payload['state']}")
print("=" * 60)

for name, ans in result["answers"].items():
    q = payload["questions"][name]
    print(f"問題: {q['instructions']}")

    if ans["type"] == "noul":
        p_true = ans["noul"]
        verdict = "是" if p_true >= 0.5 else "否"
        print(f"判斷: {verdict}  (信心 {ans['confidence']:.1%})")
        print(f"  是 {p_true:6.1%} {bar(p_true)}")
        print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
    else:
        print(f"判斷: {ans.get(ans['type'])}  (信心 {ans['confidence']:.1%})")
        probs = sorted(ans["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        width = max(len(k) for k, _ in probs)
        for option, p in probs:
            print(f"  {option:<{width}} {p:6.1%} {bar(p)}")
    print("-" * 60)

print(f"耗時 {result['latency_ms']:.0f} ms，輸入 {result['usage']['input_tokens']} tokens")
