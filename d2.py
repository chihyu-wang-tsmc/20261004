import requests

# decider-4b v2 服務（scripts/serve.sh 開在 port 8000）；jevk5 是 8090
URL = "http://localhost:8000/v1/systemone"

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
# decider 的回應沒有 latency_ms，改用 client 端量到的時間
latency_ms = resp.elapsed.total_seconds() * 1000


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
        # decider 的 noul 答案沒有 confidence，noul 本身就是「是」的機率
        print(f"判斷: {verdict}")
        print(f"  是 {p_true:6.1%} {bar(p_true)}")
        print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
    else:
        # confidence 是 TypeSafe 的定義（均勻分布為 0）；x_p_max 才是最高選項的機率
        print(
            f"判斷: {ans.get(ans['type'])}  "
            f"(信心 {ans['confidence']:.1%}，最高機率 {ans['x_p_max']:.1%})"
        )
        probs = sorted(ans["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        width = max(len(k) for k, _ in probs)
        for option, p in probs:
            print(f"  {option:<{width}} {p:6.1%} {bar(p)}")
    print("-" * 60)

print(f"模型 {result['model']}，耗時 {latency_ms:.0f} ms，輸入 {result['usage']['input_tokens']} tokens")
