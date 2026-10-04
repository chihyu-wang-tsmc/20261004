"""用本機 Winnow 服務（EldanRing/Winnow-12B，Q8_0 GGUF）當 Jev classifier 的 LangChain Runnable。

參考 langchain_typesafe.classifier.TypeSafeClassifier 改寫。Winnow 的推論 server
（github.com/EldanRing/winnow-inference，建在 llama.cpp 上）直接實作 TypeSafe（Jev）的
/v1/systemone wire format：一次請求可以問多題，機率是讀候選選項 label 的 logits 再 softmax
（native），score 答案也自帶 legend，所以跟 TypeSafeClassifier 一樣整包送出、回應交給
parse_response 驗證就好。
JevBench v1.2 測 Winnow-12B Q8 的設定（~/jevbench/docs/v1.2-additions-winnow.md）：
Q8_0 GGUF、8,192 context、4 個 decision branch、Q8 KV cache、全部 offload 到 GPU，
不做 retry / 選項順序 ensemble / 額外 calibration，temperature 用預設 1.0。
"""

import logging
from typing import Any

import httpx2
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langchain_typesafe._state import serialize_state
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import ClassifierRequest, ClassifierResponse
from langsmith.run_helpers import get_current_run_tree
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self, override

_DEFAULT_BASE_URL = "http://127.0.0.1:8091"
_DEFAULT_MODEL = "Winnow-12B"
_DEFAULT_TIMEOUT = 60.0
_LS_PROVIDER = "winnow"
# Winnow 的限制（winnow-inference/docs/API.md）：一次 1~256 題，每題 2~64 個選項
_MAX_QUESTIONS = 256

logger = logging.getLogger(__name__)


class WinnowClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 Winnow-12B Q8 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing），回傳同樣的
    ClassifierResponse，所以 `response.nouls` / `.choices` / `.scores` 都能用，
    也可以直接換進 ModelRouterMiddleware / AutoModeMiddleware。

    跟 TypeSafeClassifier（Jev）的差別：
      * API key 可有可無：server 沒開 `--api-key-file` 就不用驗證。有設 `api_key`
        或 `WINNOW_API_KEY` 環境變數時才送 `Authorization: Bearer <key>`。
      * base_url 預設 http://127.0.0.1:8091（scripts/serve.py 的預設 port），
        可用 WINNOW_BASE_URL 環境變數覆寫。
      * 用哪個 GGUF、context、decision branch 數是啟動 server 時決定的；
        `model` 預設 "Winnow-12B"（server 的 --alias），實際回答的模型看 `response.model`。
      * server 一次只處理一個 decision 請求（其他排隊，最多 128 個，超過回 429），
        所以 batch / abatch 的並行不會讓單一 server 變快。
      * 一次請求最多 256 題；choice / score 每題 2~64 個選項。

    啟動 server（Q8_0 是 winnow-inference 的預設 release 檔）：

        ```bash
        git clone https://github.com/EldanRing/winnow-inference && cd winnow-inference
        python3 scripts/setup.py      # 下載並驗證 Winnow-12B-Q8_0.gguf、編譯 server
        python3 scripts/serve.py      # 聽 http://127.0.0.1:8091
        ```

    ??? example "用法"

        ```python
        from langchain_typesafe import Choice, Noul, Score

        from classifier_winnow import WinnowClassifier

        classifier = WinnowClassifier()
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
        print(response.model)  # Winnow-12B（server 的 --alias）
        print(response.nouls["urgent"].noul)
        ```
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """送給 server 的模型名稱，也是 LangSmith trace 顯示的名稱。

    Winnow server 只載入一個模型；請求裡的 model 用 server 的 --alias（預設
    "Winnow-12B"），也接受 "jev-latest" 當別名。
    """

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("WINNOW_API_KEY", default=None),
        exclude=True,
        repr=False,
    )
    """Winnow server 的 API key（選填）。

    只有 server 用 `--api-key-file` 啟動時才需要。順序：建構時給的 `api_key` >
    `WINNOW_API_KEY` 環境變數 > 不送 Authorization。
    """

    base_url: str = Field(
        default_factory=from_env("WINNOW_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """Winnow 服務的網址，請求送到底下的 /v1/systemone。

    順序：建構時給的 `base_url` > `WINNOW_BASE_URL` 環境變數 > http://127.0.0.1:8091。
    """

    timeout: float = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """自動建立的 sync / async client 的 timeout（秒）；不影響外部傳入的 client。

    server 是一次處理一個請求，排隊時間也算在 timeout 裡。
    """

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
            message = "Winnow model must not be empty."
            raise ValueError(message)
        return model

    @field_validator("api_key", mode="before")
    @classmethod
    def _validate_api_key(cls, api_key: SecretStr | str | None) -> SecretStr | None:
        # 空字串當成沒設定，不送 Authorization
        if api_key is None:
            return None
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        return secret if secret.get_secret_value().strip() else None

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
        return ["langchain", "classifiers", "winnow"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "WINNOW_API_KEY"}

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
            message = "Synchronous Winnow client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the Winnow service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous Winnow client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the Winnow service."
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
        """把 token 用量記到目前的 LangSmith run（沒開 tracing 就什麼都不做）。

        Winnow 只讀 logits、不產生答案 token，所以 output_tokens 固定是 0。
        """
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
            logger.debug("Could not attach Winnow usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    @property
    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def _payload(self, request: ClassifierRequest) -> dict[str, JsonValue]:
        questions = request["questions"]
        if len(questions) > _MAX_QUESTIONS:
            message = (
                f"Winnow accepts at most {_MAX_QUESTIONS} questions per request; "
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


__all__ = ["WinnowClassifier"]
