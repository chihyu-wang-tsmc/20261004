"""Gemini 當 classifier：每個問題送一次 chat 請求，請模型用 JSON 寫出每個選項的機率
（verbalized，不是 logprobs），再組成和 classifier_h2o.py 一樣的答案格式。

做法照 JevBench 的 openai_compat adapter（和舊的 classifier_gemini.py 相同）：system prompt、
選項標籤（noul 是 yes / no，score 是 "0".."n-1"）、機率的檢查規則（key 剛好是選項、每個值在 [0, 1]、
加總偏離 1 在 2% 以內就重新正規化，超過就重問）。

連線用 models.py 的設定（.env 的 GEMINI_37_FLASH_BASE_URL / GEMINI_37_FLASH_TOKEN），
所以公司 AI gateway 設定一次，作答和當 classifier 都能用。

    python classifier_gemini.py       # 問一題試試看連線通不通
"""

import json
from concurrent.futures import ThreadPoolExecutor

from models import create_model

SYSTEM = (
    "You are a calibration engine. You never answer in prose. You output only "
    "a JSON object with the key 'probabilities' mapping every given option to "
    "a probability, all options included, values in [0,1], summing to 1."
)
RENORMALIZE_BAND = 0.02
MAX_ATTEMPTS = 4
"""答案格式不對（不是 JSON、少選項、加總不對）時最多問幾次；連線錯誤的重試由 openai 套件處理。"""
REASONING_EFFORT = "low"
"""思考深度（舊版的 thinkingLevel low）。gateway 不認這個參數的話改成 None。"""
JSON_SCHEMA = True
"""用 response_format 限定回 JSON。gateway 不支援的話改成 False，只靠 system prompt 要求。"""


def state_text(state):
    """使用者訊息送原文（[user] 換行接內容），程式碼的換行才不會變成一整排 \\n；其他照 JSON。"""
    if isinstance(state, dict) and isinstance(state.get("role"), str) and isinstance(state.get("content"), str):
        return f"[{state['role']}]\n{state['content']}"
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def labels_of(question):
    if question["type"] == "noul":
        return ["yes", "no"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def user_message(state, question):
    """題目的文字（和 JevBench / 舊的 classifier_gemini.py 相同）。"""
    labels = labels_of(question)
    instructions = _text(question["instructions"])
    if question["type"] == "score":
        legend = "\n".join(f"{i}: {_text(c)}" for i, c in enumerate(question["criteria"]))
        body = (f"{instructions}\n\nLevels:\n{legend}\n\n"
                f"Rate the state. Output probabilities over the level indices: {json.dumps(labels)}.")
    else:
        if question["type"] == "noul":
            criteria = question.get("criteria") or {}
            described = {"yes": criteria.get("true"), "no": criteria.get("false")}
        else:
            described = question["criteria"]
        lines = [f"- {label}: {_text(described[label])}" if described.get(label) is not None else f"- {label}"
                 for label in labels]
        body = (f"{instructions}\n\nOptions:\n" + "\n".join(lines)
                + f"\n\nOutput probabilities over exactly these keys: {json.dumps(labels)}.")
    return f"State:\n{state_text(state)}\n\n{body}"


def parse_probabilities(text, labels):
    """取出最後一個只有 probabilities 這個 key 的 JSON 物件並檢查；不合規則丟 ValueError。"""
    found = None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and set(obj) == {"probabilities"}:
                found = obj["probabilities"]
    if not isinstance(found, dict) or set(found) != set(labels):
        raise ValueError(f"選項要剛好是 {labels}，回的是 {text[:200]!r}")
    if not all(isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1 for p in found.values()):
        raise ValueError(f"機率超出 [0, 1]：{found}")
    total = sum(found.values())
    if abs(total - 1) > RENORMALIZE_BAND:
        raise ValueError(f"機率加總是 {total}")
    return {label: found[label] / total for label in labels}


def to_answer(question, probs):
    """機率 → 和 h2o 一樣格式的答案。"""
    if question["type"] == "noul":
        return {"type": "noul", "noul": probs["yes"]}
    if question["type"] == "choice":
        return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
    levels = [probs[str(i)] for i in range(len(question["criteria"]))]
    return {"type": "score", "score": sum(i * p for i, p in enumerate(levels)),
            "probabilities": {str(i): p for i, p in enumerate(levels)}}


class GeminiClassifier:
    def __init__(self, model="gemini-3.7-flash"):
        self.llm = create_model(model, timeout=120)

    def classify(self, state, questions):
        """一個問題一次請求，同一則 state 的幾個問題同時送出。"""
        with ThreadPoolExecutor(max(1, min(8, len(questions)))) as pool:
            futures = {name: pool.submit(self._ask, state, q) for name, q in questions.items()}
            return {name: f.result() for name, f in futures.items()}

    def _ask(self, state, question):
        labels = labels_of(question)
        params = {}
        if REASONING_EFFORT:
            params["reasoning_effort"] = REASONING_EFFORT
        if JSON_SCHEMA:
            schema = {"type": "object", "required": ["probabilities"], "additionalProperties": False,
                      "properties": {"probabilities": {
                          "type": "object", "required": labels, "additionalProperties": False,
                          "properties": {label: {"type": "number"} for label in labels}}}}
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(state, question)}]
        for _ in range(MAX_ATTEMPTS):
            reply = self.llm.chat(messages, **params)
            try:
                return to_answer(question, parse_probabilities(reply.text, labels))
            except ValueError as e:
                error = ValueError(f"{e}（finish_reason={reply.finish_reason}）")
        raise error


if __name__ == "__main__":
    from classifiers import CODING, user_state

    print(GeminiClassifier().classify(user_state("Write a Python function that reverses a linked list."),
                                      {"coding": CODING}))
