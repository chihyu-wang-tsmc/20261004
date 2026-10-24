"""用 Qwen（阿里雲 DashScope）當 Jev classifier 的 LangChain Runnable，介面同 classifier_jevk5.py。

做法和 classifier_chatgpt.py / classifier_deepseek.py / classifier_gemini.py 一樣（照 JevBench 的
openai_compat adapter）：每個問題送一次請求，請模型用 JSON 寫出每個選項的機率（verbalized，不是
logprobs），再組成跟 TypeSafe 一樣的 ClassifierResponse。題目文字、選項標籤（noul 是 yes / no、
score 是 "0".."n-1"）、system prompt、機率的檢查規則都和那幾家相同，數字才能並排比。

DashScope 有 OpenAI 相容網址（llm_models.py 的 qwen() 走的就是它），所以請求是 /chat/completions，
API key 一樣讀 DASHSCOPE_API_KEY——deep20 第 2 步呼叫 qwen 本來就要這把 key，不用另外申請。

預設模型 qwen3.8-27b：deep20 的 fast 模型。一個 prompt 要問 5～8 次，用便宜、快的那顆；
要換成 qwen3.8-flash（deep20 的 powerful）就改 QWEN_MODEL 或傳 model=。

和 JevK5Classifier / H2OClassifier 的差別：
  * 一個問題一次請求（同時送出，平行進行）。deep20 第一道問 5 題，所以一個 prompt 是 5 次 Qwen 呼叫。
  * enable_thinking 固定 false：只是要一個 JSON，不需要思考鏈；而且 DashScope 的 Qwen3 開源模型
    開思考時只接受串流請求、json_object 模式也不能和思考一起用，關掉最單純，也比較便宜。
  * response_format 用 json_object（同 classifier_deepseek.py）。DashScope 的 json_object 要求
    提示裡出現 "JSON" 這個字，system prompt 裡有。
  * temperature 0（同 llm_models.py 的 qwen()）。
  * 429 / 5xx / 連線錯誤 / 格式不對的答案會自動重試（max_retries 次、指數退避）。
  * DashScope 自己有內容審核：輸入被判定不當時回 HTTP 400 data_inspection_failed，同一題再問也一樣，
    所以不重試，丟 QwenBlockedError，由呼叫端決定怎麼辦（deep20_qwen.py 是把那題略過、記在 meta）。
  * 400 / 401 / 404 這類設定錯誤，會把 DashScope 自己的錯誤訊息寫進 log（TypeSafe 的錯誤只印
    「404 Not Found」，看不出是 key 錯還是模型名稱錯）。
  * confidence 用 classifier_jevk5 的三個函式（https://docs.typesafe.ai/confidence），
    noul 也補上 |2p - 1|（NoulAnswerWithConfidence），和 jevk5 / h2o 的回應同一個形狀。
  * state 是對話訊息（HumanMessage 之類）時，送給 Qwen 的是訊息原文，不是 JSON 字串：
    程式碼的換行才不會變成一整排 \\n，判斷 coding 時看得比較清楚。

用法：
    from classifier_qwen import QwenClassifier
    classifier = QwenClassifier()                      # 預設 qwen3.8-27b
    python classifier_qwen.py                          # 跑一個三題的範例
    python classifier_qwen.py qwen3.8-flash            # 指定模型
"""

import asyncio
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

import httpx2
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langchain_typesafe._state import serialize_state
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import (
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    ScoreAnswer,
)
from langsmith.run_helpers import get_current_run_tree
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from classifier_jevk5 import (
    NoulAnswerWithConfidence,
    choice_confidence,
    noul_confidence,
    score_confidence,
)

# 和 llm_models.py 的 qwen() 同一個網址（國際站）
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.8-27b"

_LS_PROVIDER = "dashscope"
# JevBench 的規則：機率加總偏離 1 在 2% 以內就重新正規化，超過就是無效答案
_RENORMALIZE_BAND = 0.02
# 跟 classifier_chatgpt.py（JevBench openai_compat adapter）一字不差的 system prompt
_SYSTEM = (
    "You are a calibration engine. You never answer in prose. You output only "
    "a JSON object with the key 'probabilities' mapping every given option to "
    "a probability, all options included, values in [0,1], summing to 1."
)
# 暫時性的狀態碼才重試；400 / 401 / 403 / 404 是設定錯了（key、模型名稱…），重試也沒用
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_BACKOFF = 60.0

logger = logging.getLogger(__name__)


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _error_fields(body: Any) -> dict:
    """DashScope 錯誤回應的錯誤物件：相容網址包在 "error" 底下，原生格式直接在最外層。"""
    if not isinstance(body, dict):
        return {}
    return body["error"] if isinstance(body.get("error"), dict) else body


def _api_message(body: Any) -> str:
    """錯誤回應裡的 message（例如模型名稱打錯、key 無效），寫進 log 用。

    TypeSafeAPIError 的訊息刻意不含 body，只會印「404 Not Found」，看不出是哪裡錯。
    """
    message = _error_fields(body).get("message")
    if message:
        return str(message)[:300]
    return str(body)[:300] if body else ""


def _inspection_reason(body: Any) -> str | None:
    """DashScope 內容審核擋下（data_inspection_failed / DataInspectionFailed）時回傳原因，不是就回 None。"""
    error = _error_fields(body)
    code = str(error.get("code") or error.get("type") or "")
    if "inspection" in code.lower():
        return f"{code} ({str(error.get('message', ''))[:120]})"
    return None


class QwenBlockedError(TypeSafeAPIError):
    """DashScope 的內容審核不讓這題送進模型（或答案每次都被過濾掉）。

    這不是暫時的錯誤，同一題再問一次結果也一樣，所以輸入被擋時不重試。
    """

    def __init__(
        self, response: httpx2.Response, body: Any, question: str, reason: str
    ) -> None:
        super().__init__(
            response.status_code,
            body,
            response.headers,
            f"Qwen blocked question {question!r}: {reason}",
        )
        self.question = question
        self.reason = reason


class _Retry(Exception):
    """這次失敗可以重試；次數用完就丟出 error（reason 是寫進 log 的原因）。"""

    def __init__(self, error: Exception, reason: str) -> None:
        super().__init__(reason)
        self.error = error
        self.reason = reason


class QwenResponse(ClassifierResponse):
    """ClassifierResponse，Noul 答案換成帶 confidence 的版本（同 classifier_jevk5.JevK5Response）。"""

    answers: dict[
        str,
        Annotated[
            NoulAnswerWithConfidence | ChoiceAnswer | ScoreAnswer,
            Field(discriminator="type"),
        ],
    ]


class QwenClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用 Qwen（DashScope）取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing），回傳同樣的
    ClassifierResponse，所以 `response.nouls` / `.choices` / `.scores` 都能用。
    """

    model: str = Field(default=QWEN_MODEL, min_length=1)
    """DashScope 上的 Qwen 模型名稱，預設 qwen3.8-27b（deep20 的 fast 模型）。"""

    api_key: SecretStr | str = Field(
        default_factory=secret_from_env("DASHSCOPE_API_KEY", default=""),
        exclude=True,
        repr=False,
    )
    """DashScope API key；不給就讀環境變數 DASHSCOPE_API_KEY。"""

    base_url: str = Field(
        default_factory=from_env("DASHSCOPE_BASE_URL", default=QWEN_BASE_URL)
    )
    """DashScope OpenAI 相容網址，請求送到底下的 /chat/completions（中國站換成 dashscope.aliyuncs.com）。"""

    max_tokens: int = Field(default=1024, gt=0)
    """每個問題的輸出上限；沒開思考，答案只是一個小 JSON。"""

    max_retries: int = Field(default=6, ge=0)
    """暫時性失敗（429 / 5xx / 連線 / 格式不對）最多重試幾次。"""

    timeout: float = Field(default=120.0, gt=0)
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
            message = "Qwen model must not be empty."
            raise ValueError(message)
        return model

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, api_key: SecretStr | str) -> SecretStr:
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not secret.get_secret_value().strip():
            message = (
                "DashScope API key is required. Pass `api_key` or set "
                "`DASHSCOPE_API_KEY`."
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

    # ---- 一次 invoke：每個問題各問一次，同時送出 ----

    def _classify(self, request: ClassifierRequest) -> ClassifierResponse:
        state_text = self._state_text(request)
        questions = request["questions"]
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(questions)))) as pool:
            results = list(
                pool.map(lambda item: self._ask(state_text, *item), questions.items())
            )
        return self._record_usage(self._assemble(questions, results))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        state_text = self._state_text(request)
        questions = request["questions"]
        results = await asyncio.gather(
            *(self._aask(state_text, name, q) for name, q in questions.items())
        )
        return self._record_usage(self._assemble(questions, results))

    # ---- 一個問題：送出、讀回應，暫時性失敗就退避重試 ----

    def _ask(self, state_text: str, name: str, question: Any) -> tuple:
        body = self._body(state_text, question)
        for attempt in range(self.max_retries + 1):
            try:
                return self._read(self._post(body), name, question)
            except _Retry as retry:
                delay = self._retry_delay(retry, name, attempt)
            time.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _aask(self, state_text: str, name: str, question: Any) -> tuple:
        body = self._body(state_text, question)
        for attempt in range(self.max_retries + 1):
            try:
                return self._read(await self._apost(body), name, question)
            except _Retry as retry:
                delay = self._retry_delay(retry, name, attempt)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _retry_delay(self, retry: _Retry, name: str, attempt: int) -> float:
        """次數用完就丟出最後一次的錯誤；不然回傳要等幾秒（指數退避加抖動，最多 60 秒）。"""
        if attempt >= self.max_retries:
            logger.warning(
                "Qwen %s: giving up after %d attempts: %s", name, attempt + 1, retry.reason
            )
            raise retry.error from retry.__cause__
        delay = min(_MAX_BACKOFF, 2.0**attempt) * random.uniform(0.5, 1.0)
        logger.info(
            "Qwen %s: attempt %d failed (%s), retrying in %.1fs",
            name, attempt + 1, retry.reason, delay,
        )
        return delay

    def _post(self, body: dict[str, JsonValue]) -> httpx2.Response:
        try:
            return self.client.post(
                self._endpoint, json=body, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise _Retry(TypeSafeAPITimeoutError(self.client.timeout), "timeout") from error
        except httpx2.HTTPError as error:
            failure = TypeSafeAPIConnectionError("Unable to connect to the DashScope API.")
            raise _Retry(failure, f"{type(error).__name__}: {error}") from error

    async def _apost(self, body: dict[str, JsonValue]) -> httpx2.Response:
        try:
            return await self.async_client.post(
                self._endpoint, json=body, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise _Retry(
                TypeSafeAPITimeoutError(self.async_client.timeout), "timeout"
            ) from error
        except httpx2.HTTPError as error:
            failure = TypeSafeAPIConnectionError("Unable to connect to the DashScope API.")
            raise _Retry(failure, f"{type(error).__name__}: {error}") from error

    def _read(
        self, response: httpx2.Response, name: str, question: Any
    ) -> tuple[dict[str, float], dict, str | None]:
        """一次 /chat/completions 的回應 → (機率, usage, 模型名稱)。

        內容審核擋下丟 QwenBlockedError；暫時性的失敗（429 / 5xx、格式不對）丟 _Retry；
        其他 HTTP 錯誤照 TypeSafe 的 parse_response 丟對應的 TypeSafeAPIError（401 / 404…）。
        """
        if not response.is_success:
            try:
                parse_response(response)
            except TypeSafeAPIError as error:
                blocked = _inspection_reason(error.body)
                if blocked:
                    raise QwenBlockedError(response, error.body, name, blocked) from error
                detail = f"HTTP {response.status_code}: {_api_message(error.body)}"
                if response.status_code in _RETRY_STATUS:
                    raise _Retry(error, detail) from error
                logger.warning("Qwen %s: %s", name, detail)
                raise
        try:
            data = response.json()
        except ValueError as error:
            raise _Retry(
                self._invalid(response, None, name), "response is not JSON"
            ) from error
        if not isinstance(data, dict):
            raise _Retry(self._invalid(response, data, name), "response is not an object")

        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            choice, content = {}, ""
        try:
            probs = self._probabilities(content, self._labels(question))
        except ValueError as reason:
            finish = choice.get("finish_reason") if isinstance(choice, dict) else None
            if finish == "content_filter":
                error = QwenBlockedError(response, data, name, "answer filtered (content_filter)")
            else:
                error = self._invalid(response, data, name)
            raise _Retry(error, f"{reason} (finish_reason={finish})") from reason
        return probs, data.get("usage") or {}, data.get("model")

    @staticmethod
    def _invalid(
        response: httpx2.Response, data: Any, name: str
    ) -> TypeSafeAPIResponseValidationError:
        return TypeSafeAPIResponseValidationError(
            response.status_code, data, response.headers, f"answers.{name}"
        )

    # ---- 題目和答案的格式（和 classifier_chatgpt.py 相同） ----

    @staticmethod
    def _state_text(request: ClassifierRequest) -> str:
        """state 是一則或一串對話訊息時送原文（[user] 換行接內容），其他照 JSON。"""
        state = serialize_state(request["state"])
        messages = state if isinstance(state, list) else [state]
        if messages and all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), str)
            for m in messages
        ):
            return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)

    @staticmethod
    def _labels(question: Any) -> list[str]:
        if question.type == "noul":
            return ["yes", "no"]
        if question.type == "choice":
            return list(question.criteria)
        return [str(level) for level in range(len(question.criteria))]

    def _body(self, state_text: str, question: Any) -> dict[str, JsonValue]:
        """一個問題的 /chat/completions 請求；使用者訊息的文字和 classifier_chatgpt.py 相同。"""
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
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self.max_tokens,
            # DashScope 專用參數：思考關掉（理由見檔頭）
            "enable_thinking": False,
        }

    @staticmethod
    def _probabilities(text: str, labels: list[str]) -> dict[str, float]:
        """取出模型寫的機率並檢查（規則同 classifier_chatgpt.py / JevBench），不對就丟 ValueError。"""
        # 跟 JevBench 一樣：取最後一個只有 probabilities 這個 key 的 JSON 物件
        found = None
        for i, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = json.JSONDecoder().raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and set(candidate) == {"probabilities"}:
                found = candidate["probabilities"]
        if not isinstance(found, dict) or set(found) != set(labels):
            raise ValueError(f"expected keys {labels}, got {text[:200]!r}")
        if not all(
            isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1
            for p in found.values()
        ):
            raise ValueError(f"probabilities out of range: {found}")
        total = sum(found.values())
        if abs(total - 1) > _RENORMALIZE_BAND:
            raise ValueError(f"probabilities sum to {total}")
        return {label: found[label] / total for label in labels}

    def _assemble(
        self, questions: dict[str, Any], results: list[tuple]
    ) -> ClassifierResponse:
        """把每個問題的機率轉成 TypeSafe 的答案格式，組成一個 ClassifierResponse。"""
        answers: dict[str, JsonValue] = {}
        input_tokens = output_tokens = 0
        model = self.model
        for (name, question), (probs, usage, served) in zip(questions.items(), results):
            model = served or model
            input_tokens += usage.get("prompt_tokens") or 0
            output_tokens += usage.get("completion_tokens") or 0
            answers[name] = self._answer(question, probs)
        return QwenResponse.model_validate(
            {
                "model": model,
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        )

    @staticmethod
    def _answer(question: Any, probs: dict[str, float]) -> dict[str, JsonValue]:
        """依 TypeSafe 的定義算出 noul / choice / score，confidence 用 classifier_jevk5 的公式。"""
        if question.type == "noul":
            p = probs["yes"]
            return {"type": "noul", "noul": p, "confidence": noul_confidence(p)}
        if question.type == "choice":
            return {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
                "confidence": choice_confidence(list(probs.values())),
            }
        # score：期望等級；probabilities 的 key 換回 int
        p = [probs[str(level)] for level in range(len(question.criteria))]
        return {
            "type": "score",
            "score": sum(i * pi for i, pi in enumerate(p)),
            "legend": dict(enumerate(question.criteria)),
            "probabilities": dict(enumerate(p)),
            "confidence": score_confidence(p),
        }

    # ---- tracing 和連線設定 ----

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
            logger.debug("Could not attach Qwen usage.", exc_info=True)
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


__all__ = ["QWEN_MODEL", "QwenBlockedError", "QwenClassifier"]


if __name__ == "__main__":
    import sys

    from langchain_typesafe import Choice, Noul, NoulCriteria, Score

    model = sys.argv[1] if len(sys.argv) > 1 else QWEN_MODEL
    classifier = QwenClassifier(model=model)
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
