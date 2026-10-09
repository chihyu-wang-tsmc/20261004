import requests

# Winnow-12B Q8 服務（winnow-inference 的 scripts/serve.py 開在 port 8091）
URL = "http://127.0.0.1:8091/v1/systemone"

payload = {
    "model": "Winnow-12B",
    "state": {"paid": True, "priority": "high"},
    "questions": {
        "paid": {"type": "noul", "instructions": "Has the customer paid?"},
        "route": {
            "type": "choice",
            "instructions": "Which priority is recorded?",
            "criteria": {"low": None, "high": None},
        },
        "rating": {
            "type": "score",
            "instructions": "How urgent is the recorded priority?",
            "criteria": ["not urgent", "moderately urgent", "very urgent"],
        },
    },
}

resp = requests.post(URL, json=payload, timeout=60)
resp.raise_for_status()

result = resp.json()
# Winnow 的回應沒有 latency_ms，改用 client 端量到的時間
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
        # Winnow 的 noul 答案沒有 confidence，noul 本身就是「是」的機率
        print(f"判斷: {verdict}")
        print(f"  是 {p_true:6.1%} {bar(p_true)}")
        print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
    else:
        # confidence = 1 - entropy/log(K)（均勻分布為 0），不是最高選項的機率
        probs = ans["probabilities"]
        if ans["type"] == "score":
            # score 的機率用 "0"、"1"… 當 key，換成 legend 裡的文字比較好讀
            probs = {ans["legend"][k]: p for k, p in probs.items()}
            answer = f"{ans['score']:.2f}（0~{len(probs) - 1} 的期望值）"
        else:
            answer = ans["choice"]
        print(
            f"判斷: {answer}  "
            f"(信心 {ans['confidence']:.1%}，最高機率 {max(probs.values()):.1%})"
        )
        probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        width = max(len(k) for k, _ in probs)
        for option, p in probs:
            print(f"  {option:<{width}} {p:6.1%} {bar(p)}")
    print("-" * 60)

print(f"模型 {result['model']}，耗時 {latency_ms:.0f} ms，輸入 {result['usage']['input_tokens']} tokens")
