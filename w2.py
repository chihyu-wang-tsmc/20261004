import time

from langchain_typesafe import Choice, Noul, Score

from classifier_winnow import WinnowClassifier

# Winnow-12B Q8 服務（winnow-inference 的 scripts/serve.py 開在 port 8091，也是預設值）
classifier = WinnowClassifier(base_url="http://127.0.0.1:8091", model="Winnow-12B")

state = {"paid": True, "priority": "high"}
questions = {
    "paid": Noul(instructions="Has the customer paid?"),
    "route": Choice(
        instructions="Which priority is recorded?",
        # 描述給 None 時，Winnow 直接用 key 當選項文字
        criteria={"low": None, "high": None},
    ),
    "rating": Score(
        instructions="How urgent is the recorded priority?",
        criteria=["not urgent", "moderately urgent", "very urgent"],
    ),
}

start = time.perf_counter()
response = classifier.invoke({"state": state, "questions": questions})
# ClassifierResponse 沒有 latency 欄位，改用 client 端量到的時間
latency_ms = (time.perf_counter() - start) * 1000


def bar(p, width=30):
    return "█" * round(p * width)


print(f"情境: {state}")
print("=" * 60)

for name, noul in response.nouls.items():
    p_true = noul.noul
    print(f"問題: {questions[name].instructions}")
    # Winnow 的 noul 答案沒有 confidence，noul 本身就是「是」的機率
    print(f"判斷: {'是' if p_true >= 0.5 else '否'}")
    print(f"  是 {p_true:6.1%} {bar(p_true)}")
    print(f"  否 {1 - p_true:6.1%} {bar(1 - p_true)}")
    print("-" * 60)


def print_probs(probs):
    probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    width = max(len(k) for k, _ in probs)
    for option, p in probs:
        print(f"  {option:<{width}} {p:6.1%} {bar(p)}")


# confidence = 1 - entropy/log(K)（均勻分布為 0），不是最高選項的機率
for name, choice in response.choices.items():
    print(f"問題: {questions[name].instructions}")
    print(
        f"判斷: {choice.choice}  "
        f"(信心 {choice.confidence:.1%}，最高機率 {max(choice.probabilities.values()):.1%})"
    )
    print_probs(choice.probabilities)
    print("-" * 60)

for name, score in response.scores.items():
    # score 的機率用 0、1… 當 key，換成 legend 裡的文字比較好讀
    probs = {str(score.legend[k]): p for k, p in score.probabilities.items()}
    print(f"問題: {questions[name].instructions}")
    print(
        f"判斷: {score.score:.2f}（0~{len(probs) - 1} 的期望值）  "
        f"(信心 {score.confidence:.1%}，最高機率 {max(probs.values()):.1%})"
    )
    print_probs(probs)
    print("-" * 60)

print(f"模型 {response.model}，耗時 {latency_ms:.0f} ms，輸入 {response.usage.input_tokens} tokens")
