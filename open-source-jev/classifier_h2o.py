"""H2O-Lightning-4B（https://huggingface.co/h2oai/h2o-lightning-4b）：本機 vLLM 加作者的 h2o_lightning_shim.py。

shim 提供 TypeSafe 格式的 POST /v1/systemone，一個請求可以問同一則 state 的好幾題：
    請求  {"state": ..., "questions": {名稱: 題目}}            題目格式見 classifiers.py
    答案  noul    {"type": "noul", "noul": P(是)}
          choice  {"type": "choice", "choice": 機率最高的選項, "probabilities": {選項: 機率}}
          score   {"type": "score", "score": 期望等級, "probabilities": {"0": 機率, "1": ...}}
不需要 API key。網址預設 http://localhost:8741，可用 .env 的 H2O_BASE_URL 改。服務怎麼起來見 README。

    python classifier_h2o.py          # 問一題試試看服務通不通
"""

import os
import time

import httpx

import common  # noqa: F401  讀 .env

H2O_MODEL = "h2oai/h2o-lightning-4b"
DEFAULT_URL = "http://localhost:8741"
RETRY_STATUS = {429, 502, 503, 504}
"""暫時性的錯誤才重試（shim 還在暖機時回 503）；422 是題目格式不對，重試也沒用。"""


class H2OClassifier:
    def __init__(self, base_url=None, timeout=600.0, retries=3):
        base = base_url or os.environ.get("H2O_BASE_URL") or DEFAULT_URL
        self.url = base.rstrip("/") + "/v1/systemone"
        self.client = httpx.Client(timeout=timeout)  # 可以同時從多個 thread 用
        self.retries = retries

    def classify(self, state, questions):
        """問同一則 state 的幾個問題，回傳 {名稱: 答案}。"""
        for attempt in range(self.retries + 1):
            try:
                r = self.client.post(self.url, json={"state": state, "questions": questions})
                if r.status_code not in RETRY_STATUS:
                    r.raise_for_status()
                    return r.json()["answers"]
                error = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            except httpx.TransportError as e:  # 連不上、逾時
                error = e
            if attempt < self.retries:
                time.sleep(2 ** attempt)
        raise error


if __name__ == "__main__":
    from classifiers import CODING, MODEL_ROUTE, user_state

    clf = H2OClassifier()
    print(f"問 {clf.url}")
    print(clf.classify(user_state("Write a Python function that reverses a linked list."),
                       {"coding": CODING, "model_route": MODEL_ROUTE}))
