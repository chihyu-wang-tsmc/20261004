from jevk5 import JevK5

model = JevK5("alibiserikbay/JevK5")          # ~9 GB of GPU memory in bf16
result = model.decide(
    "Refunds need a receipt and a purchase within 30 days. "
    "The customer bought 12 days ago and has no receipt.",
    {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
)
# {'type': 'noul', 'confidence': ..., 'noul': <probability of true>, 'input_tokens': ...}

print("原始結果:", result)
p_true = result["noul"]
answer = "可以退款 (是)" if p_true >= 0.5 else "不可退款 (否)"
print(f"判斷: {answer}")
print(f"「是」的機率: {p_true:.1%}")
print(f"信心程度: {result['confidence']:.1%}")
