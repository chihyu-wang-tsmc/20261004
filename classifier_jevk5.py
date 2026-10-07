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
)
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

JEVK5_BASE_URL = "http://localhost:8090"
JEVK5_MODEL = "alibiserikbay/JevK5"

logger = logging.getLogger(__name__)


# ---- confidence ----
# jevk5 服務回傳的 confidence 是 p_max（jevk5/prompt.py:166 的 max(probs.values())），
# 不是 https://docs.typesafe.ai/confidence 的公式。_parse 用下面三個函式照文件重算。
# Noul 在 TypeSafe 本來沒有 confidence，這裡也照 |2p - 1| 補上（NoulAnswerWithConfidence）。


def choice_confidence(probabilities: list[float]) -> float:
    """Choice：(n * p_max - 1) / (n - 1)，平均分配是 0、全押一個是 1。"""
    n = len(probabilities)
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
    """NoulAnswer 多一個 confidence。

    TypeSafe 的 NoulAnswer 沒有 confidence 欄位，直接塞進 JSON 會被 pydantic 丟掉，
    所以用子類別；isinstance(ans, NoulAnswer) 照樣成立，response.nouls 拿得到。
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """|2p - 1|，見 noul_confidence。"""


class JevK5Response(ClassifierResponse):
    """ClassifierResponse，只是 Noul 答案換成 NoulAnswerWithConfidence。

    answers 的型別要跟著換，不然 model_dump / LangSmith trace 會照 NoulAnswer 輸出、漏掉 confidence。
    """

    answers: dict[
        str,
        Annotated[
            NoulAnswerWithConfidence | ChoiceAnswer | ScoreAnswer,
            Field(discriminator="type"),
        ],
    ]


class JevK5Classifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 jevk5 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing）。
    不需要 API key，也不送 model 欄位：jevk5 服務兩者都不使用。
    """

    base_url: str = JEVK5_BASE_URL
    """jevk5 服務的網址，請求送到底下的 /v1/systemone。"""

    model: str = JEVK5_MODEL
    """只用在 LangSmith trace 顯示的模型名稱，不會送給服務。"""

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
            message = "Unable to connect to the jevk5 service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._parse(response, payload)

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        try:
            response = await self.async_client.post(self._endpoint, json=payload)
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the jevk5 service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._parse(response, payload)

    def _parse(
        self, response: httpx2.Response, payload: dict[str, JsonValue]
    ) -> ClassifierResponse:
        # jevk5 的 score 答案沒有 legend，照 TypeSafe 的格式補上；
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
                    criteria = payload["questions"][name]["criteria"]
                    ans.setdefault("legend", dict(enumerate(criteria)))
                    probs = ans["probabilities"]
                    ans["confidence"] = score_confidence(
                        [probs[k] for k in sorted(probs, key=int)]
                    )
            response = httpx2.Response(
                response.status_code, json=data, request=response.request
            )
        # 錯誤處理和 request_id 照 TypeSafe 的 parse_response；失敗的回應在這裡就 raise 了
        parsed = parse_response(response)
        # 再用 JevK5Response 驗一次，Noul 的 confidence 才留得住（見 NoulAnswerWithConfidence）
        parsed = JevK5Response.model_validate(
            {**parsed.model_dump(), "answers": data["answers"]}
        )
        return self._record_usage(parsed)

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """讓 LangSmith trace 顯示 provider 和模型名稱。"""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": "jevk5",
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
            logger.debug("Could not attach jevk5 usage.", exc_info=True)
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



