"""用本機 H2O-Lightning-4B 服務取代 TypeSafeClassifier，介面同 classifier_jevk5.py。

模型 https://huggingface.co/h2oai/h2o-lightning-4b（Qwen3.5-4B 基底，Apache-2.0）。
服務是 stock vLLM 加作者的 h2o_lightning_shim.py，shim 原生提供 TypeSafe 的
POST /v1/systemone，所以請求格式和 jevk5 完全一樣，不需要 API key，也不送 model 欄位：

    vllm serve ~/models/h2o-lightning-4b --served-model-name h2oai/h2o-lightning-4b \
      --host 127.0.0.1 --port 8000 --max-model-len 40960 --gpu-memory-utilization 0.90 \
      --limit-mm-per-prompt '{"image": 4, "video": 0}' \
      --mm-processor-kwargs '{"max_pixels": 1605632}'
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

跟 JevK5Classifier 的差別只有網址、模型名稱，和下面兩處回應的補正：

  1. shim 的 confidence 是「正規化最大值」(p_max - 1/n) / (1 - 1/n)（shim 的 confidence()）。
     choice 的那條和 https://docs.typesafe.ai/confidence 的 (n * p_max - 1) / (n - 1) 恆等，
     但 score 的等級有順序，TypeSafe 是用離峰值的平均距離算的，所以 score 在這裡照文件重算。
     三個 *_confidence 函式和 classifier_jevk5.py 相同，兩顆模型的數字才能並排比。
  2. shim 的 noul 答案沒有 confidence，照 |2p - 1| 補上（NoulAnswerWithConfidence），同 jevk5。

shim 比 jevk5 多給的欄位也留著：usage.latency_ms、usage.images、probability_source、
effort_used，以及 noul 問題帶 threshold 時的 threshold / decision。

**noul_floor 要注意。** serve_config.json 設 0.801，shim 的 commit_noul() 會把
max(p, 1-p) < 0.801 的 yes/no 機率推到邊界，所以 noul 不會落在 (0.199, 0.801) 之間，
noul_confidence 也不會低於 0.602。答案（哪一邊贏）不變，但拿機率去比 llm_guardrails.py
route() 那種門檻時，這個夾鉗會讓 h2o 和 JevK5 不能直接對照。要原始機率就把
serve_config.json 的 noul_floor 改成 0（0 等於關掉），改完要重啟 shim。

用法：
    from classifier_h2o import H2OClassifier
    classifier = H2OClassifier()                       # 預設 http://localhost:8741
    python classifier_h2o.py                           # 跑一個三題的範例
    python classifier_h2o.py http://127.0.0.1:8741     # 指定服務網址
"""

import logging
from typing import Annotated, Any

import httpx2
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_typesafe._state import serialize_state
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import (
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

H2O_BASE_URL = "http://localhost:8741"
H2O_MODEL = "h2oai/h2o-lightning-4b"

logger = logging.getLogger(__name__)


# ---- confidence ----
# 和 classifier_jevk5.py 的三個函式相同，照 https://docs.typesafe.ai/confidence 的公式。


def choice_confidence(probabilities: list[float]) -> float:
    """Choice：(n * p_max - 1) / (n - 1)，平均分配是 0、全押一個是 1。"""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return min(1.0, max(0.0, (n * max(probabilities) - 1) / (n - 1)))


def score_confidence(probabilities: list[float]) -> float:
    """Score：1 - 離峰值的平均距離 / 平均分配時的平均距離，最低到 0。

    等級有順序，機率落在隔壁等級只扣一點，落在遠端扣很多。probabilities 要照等級 0..n-1 排好。
    """
    n = len(probabilities)
    m = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)


def noul_confidence(p: float) -> float:
    """Noul：|2p - 1|，也就是 n=2 的 Choice。"""
    return choice_confidence([p, 1 - p])


class NoulAnswerWithConfidence(NoulAnswer):
    """NoulAnswer 多一個 confidence，以及 shim 的 threshold / decision。

    TypeSafe 的 NoulAnswer 沒有這些欄位，直接塞進 JSON 會被 pydantic 丟掉，所以用子類別；
    isinstance(ans, NoulAnswer) 照樣成立，response.nouls 拿得到。
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """|2p - 1|，見 noul_confidence。"""

    threshold: float | None = None
    """問題帶 threshold 時 shim 原樣回傳的門檻，沒帶就是 None。"""

    decision: bool | None = None
    """shim 用上面的門檻判出來的 noul >= threshold，沒帶門檻就是 None。"""


class H2OUsage(Usage):
    """Usage 多兩個 shim 會回的欄位。"""

    latency_ms: float | None = None
    """shim 自己量的處理時間（毫秒），不含網路。"""

    images: int | None = None
    """這次請求帶了幾張圖，沒帶圖就不回這個欄位。"""


class H2OResponse(ClassifierResponse):
    """ClassifierResponse 換成 shim 實際回的形狀。

    answers 的型別要跟著換，不然 model_dump / LangSmith trace 會照 NoulAnswer 輸出、
    漏掉 confidence。
    """

    answers: dict[
        str,
        Annotated[
            NoulAnswerWithConfidence | ChoiceAnswer | ScoreAnswer,
            Field(discriminator="type"),
        ],
    ]

    usage: H2OUsage = Field(default_factory=H2OUsage)

    probability_source: str | None = None
    """shim 回的機率來源，自架這條路固定是 "native"（模型自己的 softmax）。"""

    effort_used: dict[str, JsonValue] | None = None
    """shim 回的 effort 說明；這顆模型只有 fast 這一檔，一題一次 forward pass。"""


class H2OClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 H2O-Lightning-4B 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing）。
    不需要 API key，也不送 model 欄位：shim 兩者都不使用。
    """

    base_url: str = H2O_BASE_URL
    """shim 的網址，請求送到底下的 /v1/systemone。"""

    model: str = H2O_MODEL
    """只用在 LangSmith trace 顯示的模型名稱，不會送給服務（回應裡的 model 由 shim 決定）。"""

    timeout: float = Field(default=30.0, gt=0)
    """自動建立的 sync / async client 的 timeout（秒）。"""

    client: httpx2.Client | None = Field(default=None, exclude=True, repr=False)
    """invoke / batch 用的 httpx2.Client；不給就自動建立，可重複使用連線。"""

    async_client: httpx2.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )
    """ainvoke / abatch 用的 httpx2.AsyncClient；不給就自動建立。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _build_clients(self) -> Self:
        if self.client is None:
            self.client = httpx2.Client(timeout=self.timeout)
        if self.async_client is None:
            self.async_client = httpx2.AsyncClient(timeout=self.timeout)
        return self

    def invoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        return self._call_with_config(
            self._classify, input, self._traced_config(config), run_type="llm"
        )

    async def ainvoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        return await self._acall_with_config(
            self._aclassify, input, self._traced_config(config), run_type="llm"
        )

    def _classify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        try:
            response = self.client.post(self._endpoint, json=payload)
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the h2o-lightning service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._parse(response, payload)

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        try:
            response = await self.async_client.post(self._endpoint, json=payload)
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the h2o-lightning service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._parse(response, payload)

    def _parse(
        self, response: httpx2.Response, payload: dict[str, JsonValue]
    ) -> ClassifierResponse:
        # confidence 改成 TypeSafe 文件的公式（見上面三個 *_confidence 函式），Noul 也補上
        if response.is_success:
            data = response.json()
            for name, ans in data.get("answers", {}).items():
                kind = ans.get("type")
                if kind == "noul":
                    ans["confidence"] = noul_confidence(ans["noul"])
                elif kind == "choice":
                    ans["confidence"] = choice_confidence(
                        list(ans["probabilities"].values())
                    )
                elif kind == "score":
                    # shim 自己就會回 legend，這裡只是萬一沒有時照問題的 criteria 補
                    if "legend" not in ans:
                        criteria = payload["questions"].get(name, {}).get("criteria")
                        ans["legend"] = dict(enumerate(criteria or []))
                    probs = ans["probabilities"]
                    ans["confidence"] = score_confidence(
                        [probs[k] for k in sorted(probs, key=int)]
                    )
            response = httpx2.Response(
                response.status_code, json=data, request=response.request
            )
        # 錯誤處理和 request_id 照 TypeSafe 的 parse_response；失敗的回應在這裡就 raise 了
        parsed = parse_response(response)
        # 再用 H2OResponse 驗一次，shim 多給的欄位才留得住（見 H2OResponse）
        return self._record_usage(
            H2OResponse.model_validate({**data, "request_id": parsed.request_id})
        )

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """讓 LangSmith trace 顯示 provider 和模型名稱。"""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": "h2o-lightning",
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config

    def _record_usage(self, response: ClassifierResponse) -> ClassifierResponse:
        """把 token 用量記到目前的 LangSmith run（沒開 tracing 就什麼都不做）。"""
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                run_tree.extra.setdefault("metadata", {})["usage_metadata"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
        except Exception:  # noqa: BLE001 - tracing must not break classification
            logger.debug("Could not attach h2o-lightning usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    def _payload(self, request: ClassifierRequest) -> dict[str, JsonValue]:
        return {
            "state": serialize_state(request["state"]),
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in request["questions"].items()
            },
        }


if __name__ == "__main__":
    import json
    import sys

    from langchain_typesafe import Choice, Noul, NoulCriteria, Score

    base_url = sys.argv[1] if len(sys.argv) > 1 else H2O_BASE_URL
    classifier = H2OClassifier(base_url=base_url, timeout=120)
    response = classifier.invoke(
        {
            "state": (
                "Checkout has been returning HTTP 500 for every customer for the last "
                "20 minutes. Payments are not going through at all."
            ),
            "questions": {
                "urgent": Noul(
                    instructions="Does this report require an urgent response?",
                    criteria=NoulCriteria(
                        true="it needs attention right now",
                        false="it can wait for the next working day",
                    ),
                ),
                "team": Choice(
                    instructions="Which team should own this report?",
                    criteria={
                        "payments": "checkout, billing and payment providers",
                        "infra": "servers, networking and deployments",
                        "mobile": "the iOS and Android apps",
                    },
                ),
                "severity": Score(
                    instructions="How severe is this incident?",
                    criteria=[
                        "No customer impact.",
                        "Some customers affected.",
                        "Every customer affected.",
                    ],
                ),
            },
        }
    )
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=1))
