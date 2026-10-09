"""用本機 imajev 服務（mohit67890/imajev-4b）做分類的 LangChain Runnable。

參考 langchain_typesafe.classifier.TypeSafeClassifier 改寫。imajev 的 /v1/systemone 就是
TypeSafe（Jev）的 wire format：一次請求可以問多題，機率是模型直接讀出來的（native），
所以跟 TypeSafeClassifier 一樣整包送出、回應交給 parse_response 驗證就好。
JevBench 測 Imajev-4B 用的是 server 端設定：--rotations 1 --calibration calibration.json。
"""

import logging
from typing import Any

import httpx2
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env
from langchain_typesafe._state import serialize_state
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import ClassifierRequest, ClassifierResponse
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self, override

_DEFAULT_BASE_URL = "http://127.0.0.1:8765"
_DEFAULT_MODEL = "mohit67890/imajev-4b"
_DEFAULT_TIMEOUT = 60.0
_LS_PROVIDER = "imajev"
# imajev 的限制（vision_decision/jev_api.py）：一次請求最多 8 題
_MAX_QUESTIONS = 8

logger = logging.getLogger(__name__)


class ImajevClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 imajev 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing），回傳同樣的
    ClassifierResponse，所以 `response.nouls` / `.choices` / `.scores` 都能用，
    也可以直接換進 ModelRouterMiddleware / AutoModeMiddleware。

    跟 TypeSafeClassifier（Jev）的差別：
      * 不需要 API key，也不送 Authorization header：imajev 服務沒有驗證。
      * base_url 預設 http://127.0.0.1:8765，可用 IMAJEV_BASE_URL 環境變數覆寫。
      * 服務用哪個模型、要不要 --calibration / --rotations，是啟動 server 時決定的；
        `model` 只是 trace 名稱，實際回答的模型看 `response.model`。
      * imajev 每個答案還多了 unknown_probability 和 abstained（「看不出來」），
        ClassifierResponse 沒有這兩個欄位，所以會被略過；需要的話直接打 /v1/systemone。
      * 一次請求最多 8 題；choice 最多 254 個選項，score 2~10 個等級。

    ??? example "用法"

        ```python
        from langchain_typesafe import Choice, Noul, Score

        from classifier_imajev import ImajevClassifier

        classifier = ImajevClassifier()
        response = classifier.invoke(
            {
                "state": "The deploy failed twice and customers are seeing 500s.",
                "questions": {
                    "urgent": Noul(instructions="Does this need attention right now?"),
                    "team": Choice(
                        instructions="Which team should pick this up?",
                        criteria={"infra": "Deploys.", "billing": "Payments."},
                    ),
                    "severity": Score(
                        instructions="How severe is the impact?",
                        criteria=["Cosmetic.", "Degraded.", "Full outage."],
                    ),
                },
            }
        )
        print(response.model)  # imajev-4b（server 的 --model-name）
        print(response.nouls["urgent"].noul)
        ```
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """LangSmith trace 顯示的模型名稱；imajev server 會忽略請求裡的 model 欄位。"""

    base_url: str = Field(
        default_factory=from_env("IMAJEV_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """imajev 服務的網址，請求送到底下的 /v1/systemone。

    順序：建構時給的 `base_url` > `IMAJEV_BASE_URL` 環境變數 > http://127.0.0.1:8765。
    """

    timeout: float = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """自動建立的 sync / async client 的 timeout（秒）；不影響外部傳入的 client。"""

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

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str) -> str:
        model = model.strip()
        if not model:
            message = "imajev model must not be empty."
            raise ValueError(message)
        return model

    @model_validator(mode="after")
    def _build_clients(self) -> Self:
        if self.client is None:
            self.client = httpx2.Client(timeout=self.timeout)
        if self.async_client is None:
            self.async_client = httpx2.AsyncClient(timeout=self.timeout)
        return self

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "classifiers", "imajev"]

    @override
    def invoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        return self._call_with_config(
            self._classify, input, self._traced_config(config), run_type="llm"
        )

    @override
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
        if self.client is None:  # pragma: no cover - guaranteed by model validation
            message = "Synchronous imajev client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the imajev service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous imajev client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the imajev service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """讓 LangSmith trace 顯示 provider 和模型名稱。"""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": _LS_PROVIDER,
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
            logger.debug("Could not attach imajev usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    @property
    def _request_headers(self) -> dict[str, str]:
        # imajev 服務沒有驗證，所以不送 Authorization
        return {"Content-Type": "application/json"}

    def _payload(self, request: ClassifierRequest) -> dict[str, JsonValue]:
        questions = request["questions"]
        if len(questions) > _MAX_QUESTIONS:
            message = (
                f"imajev accepts at most {_MAX_QUESTIONS} questions per request; "
                f"got {len(questions)}. Split them into several requests."
            )
            raise ValueError(message)
        return {
            "state": serialize_state(request["state"]),
            "model": self.model,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in questions.items()
            },
        }


__all__ = ["ImajevClassifier"]
