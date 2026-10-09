import requests

# imajev 服務（scripts/playground/server.py 開在 port 8765）；decider 是 8000，jevk5 是 8090
URL = "http://127.0.0.1:8765/v1/systemone"

# 跟 d2.py 相同的問題。imajev 的 choice criteria 必須是 dict（選項 -> 說明），
# 不能像 d2 那樣用 list，所以改寫成 dict、說明留 None
payload = {
    "state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
    "questions": {
        "what": {
            "type": "choice",
            "instructions": "What happened to the parcel?",
            "criteria": {"delivered": None, "misdelivered": None, "unknown": None},
        }
    },
}

# 沒有照片，直接送 JSON（有照片時才用 multipart：files={"image": ...}, data={"request": ...}）
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

    # imajev 每個答案都多了 unknown_probability（模型判斷「看不出來」的機率）和 abstained（是否放棄作答）
    unknown = f"看不出來 {ans['unknown_probability']:.1%}" + ("，放棄作答" if ans["abstained"] else "")
    if ans["type"] == "noul":
        p_true = ans["noul"]
        verdict = "是" if p_true >= 0.5 else "否"
        print(f"判斷: {verdict}  ({unknown})")
        print(f"  是 {p_true:6.1%} {bar(p_true)}")
        print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
    else:
        # confidence 是 Jev 的定義（均勻分布為 0），再乘上 (1 - unknown_probability)
        print(f"判斷: {ans.get(ans['type'])}  (信心 {ans['confidence']:.1%}，{unknown})")
        probs = sorted(ans["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        width = max(len(k) for k, _ in probs)
        for option, p in probs:
            print(f"  {option:<{width}} {p:6.1%} {bar(p)}")
    print("-" * 60)

usage = result["usage"]
print(f"模型 {result['model']}，耗時 {usage['total_ms']:.0f} ms，輸入 {usage['input_tokens']} tokens")
