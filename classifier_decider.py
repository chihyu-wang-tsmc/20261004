"""用本機 decider 服務（Mapika/decider-4b）做分類的 LangChain Runnable。

參考 langchain_typesafe.classifier.TypeSafeClassifier 改寫。
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

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_MODEL = "Mapika/decider-4b"
_DEFAULT_TIMEOUT = 30.0
_LS_PROVIDER = "decider"

logger = logging.getLogger(__name__)


class DeciderClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 decider 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing），回傳同樣的
    ClassifierResponse，所以 `response.nouls` / `.choices` / `.scores` 都能用。

    decider 的 `/v1/systemone` 就是 TypeSafe 的 wire format（score 答案也自帶
    legend），所以回應直接交給 parse_response 驗證，不用另外補欄位。
    跟 TypeSafeClassifier 的差別：
      * 不需要 API key，也不送 Authorization header：decider 服務沒有驗證。
      * base_url 預設 http://localhost:8000，可用 DECIDER_BASE_URL 環境變數覆寫。
      * 服務用哪個權重是啟動 server 時決定的（scripts/serve.sh <模型>），
        `model` 只是送出去的欄位和 trace 名稱；實際回答的模型看 `response.model`。

    ??? example "用法"

        ```python
        from langchain_typesafe import Choice, Noul, Score

        from classifier_decider import DeciderClassifier

        classifier = DeciderClassifier()
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
        print(response.model)  # decider-4b-v2
        print(response.nouls["urgent"].noul)
        ```
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """送給服務的 model 欄位，也是 LangSmith trace 顯示的模型名稱。"""

    base_url: str = Field(
        default_factory=from_env("DECIDER_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """decider 服務的網址，請求送到底下的 /v1/systemone。

    順序：建構時給的 `base_url` > `DECIDER_BASE_URL` 環境變數 > http://localhost:8000。
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
            message = "Decider model must not be empty."
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
        return ["langchain", "classifiers", "decider"]

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
            message = "Synchronous decider client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the decider service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous decider client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the decider service."
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
            logger.debug("Could not attach decider usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    @property
    def _request_headers(self) -> dict[str, str]:
        # decider 服務沒有驗證，所以不送 Authorization
        return {"Content-Type": "application/json"}

    def _payload(self, request: ClassifierRequest) -> dict[str, JsonValue]:
        return {
            "state": serialize_state(request["state"]),
            "model": self.model,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in request["questions"].items()
            },
        }


__all__ = ["DeciderClassifier"]
