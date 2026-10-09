"""用 DeepSeek V4.1 Flash 當 Jev classifier 的 LangChain Runnable。

參考 langchain_typesafe.classifier.TypeSafeClassifier 改寫。做法照 JevBench 測
「DeepSeek V4.1 Flash (thinking default)」用的 openai_compat adapter
（~/jevbench/jevbench/adapters/openai_compat.py）：每個問題送一次請求，請模型用 JSON
寫出每個選項的機率（verbalized，不是 logprobs），再組成跟 TypeSafe 一樣的 ClassifierResponse。
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx2
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langchain_typesafe._state import serialize_state
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
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

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-flash"
_DEFAULT_TIMEOUT = 120.0
_LS_PROVIDER = "deepseek"
# JevBench 的規則：機率加總偏離 1 在 2% 以內就重新正規化，超過就是無效答案
_RENORMALIZE_BAND = 0.02
# 跟 JevBench openai_compat adapter 一字不差的 system prompt
_SYSTEM = (
    "You are a calibration engine. You never answer in prose. You output only "
    "a JSON object with the key 'probabilities' mapping every given option to "
    "a probability, all options included, values in [0,1], summing to 1."
)

logger = logging.getLogger(__name__)


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class DeepSeekClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用 DeepSeek V4.1 Flash 取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing），回傳同樣的
    ClassifierResponse，所以 `response.nouls` / `.choices` / `.scores` 都能用，
    也可以直接換進 ModelRouterMiddleware / AutoModeMiddleware。

    跟 TypeSafeClassifier（Jev）的差別：
      * DeepSeek 不會一次回答多個問題，每個問題各送一次請求（同時送出，平行進行）。
      * 機率是模型「寫出來的」（verbalized），不是模型內部的分布；
        choice / score / confidence 由這裡依 TypeSafe 的定義算出來。
      * DeepSeek 不支援嚴格的 json_schema，只能用 json_object 要求 JSON 輸出。
      * API key 讀 DEEPSEEK_API_KEY；base_url 可用 DEEPSEEK_BASE_URL 覆寫。

    ??? example "用法"

        ```python
        from langchain_typesafe import Choice, Noul, Score

        from classifier_deepseek import DeepSeekClassifier

        classifier = DeepSeekClassifier()           # thinking=False 會快很多
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
        print(response.nouls["urgent"].noul)
        ```
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """DeepSeek 模型名稱，預設 deepseek-flash（DeepSeek V4.1 Flash）。"""

    api_key: SecretStr | str = Field(
        default_factory=secret_from_env("DEEPSEEK_API_KEY", default=""),
        exclude=True,
        repr=False,
    )
    """DeepSeek API key；不給就讀環境變數 DEEPSEEK_API_KEY。"""

    base_url: str = Field(
        default_factory=from_env("DEEPSEEK_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """DeepSeek API 網址，請求送到底下的 /chat/completions。"""

    thinking: bool = True
    """是否開啟思考模式。JevBench 測的是預設開啟；關掉通常快很多、token 少很多。"""

    max_tokens: int = Field(default=8192, gt=0)
    """每個問題的輸出上限（含思考內容）。"""

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
            message = "DeepSeek model must not be empty."
            raise ValueError(message)
        return model

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, api_key: SecretStr | str) -> SecretStr:
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not secret.get_secret_value().strip():
            message = (
                "DeepSeek API key is required. Pass `api_key` or set "
                "`DEEPSEEK_API_KEY`."
            )
            raise ValueError(message)
        return secret

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
        return ["langchain", "classifiers", "deepseek"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "DEEPSEEK_API_KEY"}

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
        if self.client is None:  # pragma: no cover - guaranteed by model validation
            message = "Synchronous DeepSeek client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        state_text = self._state_text(request)
        questions = request["questions"]
        # 每個問題一次請求，同時送出
        with ThreadPoolExecutor(max_workers=min(8, len(questions))) as pool:
            responses = list(
                pool.map(
                    lambda q: self._post(self._body(state_text, q)), questions.values()
                )
            )
        return self._record_usage(self._assemble(questions, responses))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous DeepSeek client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        state_text = self._state_text(request)
        questions = request["questions"]
        responses = await asyncio.gather(
            *(self._apost(self._body(state_text, q)) for q in questions.values())
        )
        return self._record_usage(self._assemble(questions, responses))

    def _post(self, body: dict[str, JsonValue]) -> httpx2.Response:
        try:
            response = self.client.post(
                self._endpoint, json=body, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the DeepSeek API."
            raise TypeSafeAPIConnectionError(message) from error
        return self._check_status(response)

    async def _apost(self, body: dict[str, JsonValue]) -> httpx2.Response:
        try:
            response = await self.async_client.post(
                self._endpoint, json=body, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the DeepSeek API."
            raise TypeSafeAPIConnectionError(message) from error
        return self._check_status(response)

    @staticmethod
    def _check_status(response: httpx2.Response) -> httpx2.Response:
        # 非 2xx 交給 parse_response 丟出對應的 TypeSafeAPIError（401 / 429 / 5xx…）
        if not response.is_success:
            parse_response(response)
        return response

    @staticmethod
    def _state_text(request: ClassifierRequest) -> str:
        state = serialize_state(request["state"])
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)

    @staticmethod
    def _labels(question: Any) -> list[str]:
        if question.type == "noul":
            return ["yes", "no"]
        if question.type == "choice":
            return list(question.criteria)
        return [str(level) for level in range(len(question.criteria))]

    def _body(self, state_text: str, question: Any) -> dict[str, JsonValue]:
        """照 JevBench openai_compat adapter 的格式組出一個問題的請求。"""
        labels = self._labels(question)
        instructions = _text(question.instructions)
        if question.type == "score":
            legend = "\n".join(
                f"{label}: {_text(question.criteria[int(label)])}" for label in labels
            )
            user = (
                f"{instructions}\n\nLevels:\n{legend}\n\n"
                "Rate the state. Output probabilities over the level indices: "
                f"{json.dumps(labels)}."
            )
        else:
            if question.type == "noul":
                criteria = question.criteria
                descriptions = {
                    "yes": criteria.true if criteria else None,
                    "no": criteria.false if criteria else None,
                }
            else:
                descriptions = question.criteria
            lines = [
                f"- {label}: {_text(descriptions[label])}"
                if descriptions.get(label) is not None
                else f"- {label}"
                for label in labels
            ]
            user = (
                f"{instructions}\n\nOptions:\n" + "\n".join(lines)
                + f"\n\nOutput probabilities over exactly these keys: {json.dumps(labels)}."
            )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"State:\n{state_text}\n\n{user}"},
            ],
            # DeepSeek 不支援嚴格的 json_schema，只能用 json_object
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }

    def _assemble(
        self, questions: dict[str, Any], responses: list[httpx2.Response]
    ) -> ClassifierResponse:
        """把每個問題的回應轉成 TypeSafe 的答案格式，組成 ClassifierResponse。"""
        answers: dict[str, JsonValue] = {}
        input_tokens = output_tokens = 0
        model = self.model
        for (name, question), response in zip(questions.items(), responses):
            data = response.json()
            model = data.get("model", model)
            usage = data.get("usage") or {}
            input_tokens += usage.get("prompt_tokens") or 0
            output_tokens += usage.get("completion_tokens") or 0
            probs = self._probabilities(response, data, name, self._labels(question))
            answers[name] = self._answer(question, probs)
        return ClassifierResponse.model_validate(
            {
                "model": model,
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        )

    @staticmethod
    def _probabilities(
        response: httpx2.Response, data: dict, name: str, labels: list[str]
    ) -> dict[str, float]:
        """取出模型寫的機率並檢查；格式不對就丟 TypeSafeAPIResponseValidationError。"""

        def invalid(reason: str) -> TypeSafeAPIResponseValidationError:
            logger.debug("DeepSeek answer %s invalid: %s", name, reason)
            return TypeSafeAPIResponseValidationError(
                response.status_code, data, response.headers, f"answers.{name}"
            )

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise invalid("no message content") from error
        # 跟 JevBench 一樣：取最後一個只有 probabilities 這個 key 的 JSON 物件
        found = None
        for i, char in enumerate(content):
            if char != "{":
                continue
            try:
                candidate, _ = json.JSONDecoder().raw_decode(content[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and set(candidate) == {"probabilities"}:
                found = candidate["probabilities"]
        if not isinstance(found, dict) or set(found) != set(labels):
            raise invalid(f"expected keys {labels}, got {content[:200]!r}")
        if not all(
            isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1
            for p in found.values()
        ):
            raise invalid(f"probabilities out of range: {found}")
        total = sum(found.values())
        if abs(total - 1) > _RENORMALIZE_BAND:
            raise invalid(f"probabilities sum to {total}")
        return {label: found[label] / total for label in labels}

    @staticmethod
    def _answer(question: Any, probs: dict[str, float]) -> dict[str, JsonValue]:
        """依 TypeSafe 的定義算出 choice / score / confidence。"""
        if question.type == "noul":
            return {"type": "noul", "noul": probs["yes"]}
        labels = list(probs)
        n = len(labels)
        top = max(labels, key=probs.get)
        if question.type == "choice":
            # (n·p_max − 1)/(n − 1)：均勻分布為 0，全部集中在一個選項為 1
            confidence = (n * probs[top] - 1) / (n - 1)
            return {
                "type": "choice",
                "choice": top,
                "probabilities": probs,
                "confidence": min(1.0, max(0.0, confidence)),
            }
        # score：期望等級；confidence = max(0, 1 − Σ pᵢ·|i − k| / D)，
        # D = (1/n)·Σ |i − (n − 1)/2|，k 是機率最高的等級
        p = [probs[str(level)] for level in range(n)]
        k = int(top)
        spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        confidence = max(0.0, 1 - sum(p[i] * abs(i - k) for i in range(n)) / spread)
        return {
            "type": "score",
            "score": sum(i * p[i] for i in range(n)),
            "legend": dict(enumerate(question.criteria)),
            "probabilities": dict(enumerate(p)),
            "confidence": min(1.0, confidence),
        }

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
            logger.debug("Could not attach DeepSeek usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def _request_headers(self) -> dict[str, str]:
        api_key = (
            self.api_key.get_secret_value()
            if isinstance(self.api_key, SecretStr)
            else self.api_key
        )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


__all__ = ["DeepSeekClassifier"]
