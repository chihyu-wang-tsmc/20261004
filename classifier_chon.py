"""用本機 Chon 服務（ninhnv/chon-1.0-12b）當 Jev classifier 的 LangChain Runnable。

參考 langchain_typesafe.classifier.TypeSafeClassifier 改寫。
模型 https://huggingface.co/ninhnv/chon-1.0-12b：google/gemma-4-12B-it 加 LoRA adapter
（Apache-2.0）。推論 server（github.com/xoai/chon，`python -m chon.serve`）直接實作 TypeSafe
（Jev）的 POST /v1/systemone：一次請求可以問多題，每題一次 forward pass，讀選項字母的 logits
再 softmax；比較沒把握的題目（chon_config.json 的 think.tau，約 11%）會先用 base model 寫
最多 512 token 的推理再讀一次，那些 token 算在 usage.output_tokens。

    pip install "chon[serve] @ git+https://github.com/xoai/chon@v1.1.1"
    python -m chon.serve --model ninhnv/chon-1.0-12b --port 8000

請求格式和 TypeSafe 相同，但回應有三處不合 TypeSafe 的 schema，這裡在 parse_response 前補正：

  1. score 答案沒有 confidence，legend 也只收有描述的等級（描述是 JSON 的話還會被轉成字串，
     chon/answers.py）。ScoreAnswer 這兩個欄位都是必填，不補的話每個 score 題都會被
     parse_response 擋成 TypeSafeAPIResponseValidationError。confidence 照文件公式算，
     legend 直接用請求裡的 criteria。
  2. choice 的 confidence 是 p_max，不是 https://docs.typesafe.ai/confidence 的
     (n * p_max - 1) / (n - 1)，照文件重算。
  3. noul 答案沒有 confidence，照 |2p - 1| 補上（ChonNoulAnswer），同 classifier_jevk5.py。

三個 *_confidence 函式和 classifier_jevk5.py / classifier_h2o.py 相同，幾顆模型的數字才能並排比。
chon 多給的欄位也留著：noul 的 probabilities（true / false）、score 的 level（機率最高的等級）、
usage 的 forward_s / latency_s / thought。

**noul_commit 要注意。** chon_config.json 設 [0.199, 0.801]，server 會把落在 (0.2, 0.8) 之間的
P(yes) 推到比較近的那個邊界，所以 noul 不會落在 (0.199, 0.801) 之間，noul_confidence 也不會
低於 0.602。答案（哪一邊贏）不變，但拿機率去比 llm_guardrails.py route() 那種門檻時，跟 JevK5
不能直接對照（classifier_h2o.py 的 noul_floor 也是同樣情形）。要沒被推到邊界的機率，就把模型
下載到本機、刪掉 chon_config.json 的 noul_commit，再用本機資料夾啟動：

    hf download ninhnv/chon-1.0-12b --local-dir ./chon-1.0-12b
    python -m chon.serve --model ./chon-1.0-12b --port 8000

用法：
    from classifier_chon import ChonClassifier
    classifier = ChonClassifier()                      # 預設 http://127.0.0.1:8000
    python classifier_chon.py                          # 跑一個三題的範例
    python classifier_chon.py http://127.0.0.1:8000    # 指定服務網址
"""

import logging
from typing import Annotated, Any

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
from langchain_typesafe.types import (
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
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
from typing_extensions import Self, override

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_MODEL = "ninhnv/chon-1.0-12b"
_DEFAULT_TIMEOUT = 60.0
_LS_PROVIDER = "chon"
# parse_response 從這個 header 拿 request_id；chon 不回，但重建回應時照樣帶過去
_REQUEST_ID_HEADER = "x-typesafe-request-id"

logger = logging.getLogger(__name__)


# ---- confidence ----
# 和 classifier_jevk5.py / classifier_h2o.py 的三個函式相同，照 https://docs.typesafe.ai/confidence。


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


# ---- 回應型別 ----
# TypeSafe 的型別沒有下面這些欄位，直接塞進 JSON 會被 pydantic 丟掉，所以用子類別；
# isinstance(ans, NoulAnswer) / ScoreAnswer 照樣成立，response.nouls / .scores 拿得到。


class ChonNoulAnswer(NoulAnswer):
    """NoulAnswer 多一個 confidence，以及 chon 回的 true / false 機率。"""

    confidence: float = Field(ge=0.0, le=1.0)
    """|2p - 1|，見 noul_confidence。"""

    probabilities: dict[str, float] | None = None
    """{"true": P(yes), "false": P(no)}，和 noul 是同一個數字。"""


class ChonScoreAnswer(ScoreAnswer):
    """ScoreAnswer 多一個 level。"""

    level: int | None = None
    """機率最高的等級。score 是期望值，可能是小數；要一個整數等級就用這個。"""


class ChonUsage(Usage):
    """Usage 多三個 chon 會回的欄位。"""

    forward_s: float | None = None
    """模型推論花的秒數（含推理），不含排隊和網路。"""

    latency_s: float | None = None
    """server 收到請求到回應的秒數，含排隊等 GPU lock。"""

    thought: int | None = None
    """這次請求有幾題先寫了推理（think when unsure）；server 沒開推理就不回。"""


class ChonResponse(ClassifierResponse):
    """ClassifierResponse 換成 chon 補正後的形狀。

    answers 的型別要跟著換，不然 model_dump / LangSmith trace 會照 NoulAnswer 輸出、
    漏掉 confidence。
    """

    answers: dict[
        str,
        Annotated[
            ChonNoulAnswer | ChoiceAnswer | ChonScoreAnswer,
            Field(discriminator="type"),
        ],
    ]

    usage: ChonUsage = Field(default_factory=ChonUsage)


class ChonClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 Chon-1.0-12B 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke /
    batch / abatch / `|` 串接 / callbacks / LangSmith tracing），回傳同樣的
    ClassifierResponse（實際是子類別 ChonResponse），所以 `response.nouls` /
    `.choices` / `.scores` 都能用，也可以直接換進 ModelRouterMiddleware /
    AutoModeMiddleware。

    跟 TypeSafeClassifier（Jev）的差別：
      * API key 可有可無：server 啟動時有設 `CHON_API_KEY` 才驗證。有設 `api_key`
        或 `CHON_API_KEY` 環境變數時才送 `Authorization: Bearer <key>`。
      * base_url 預設 http://127.0.0.1:8000（chon.serve 的預設 port），可用
        CHON_BASE_URL 環境變數覆寫。
      * server 只載入一個模型，請求不送 model 欄位；`model` 只用在 LangSmith trace，
        實際回答的模型看 `response.model`（chon_config.json 的 name，"Chon-1.0-12B"）。
      * server 一次只處理一個請求（GPU lock，其他排隊），所以 batch / abatch 的並行
        不會讓單一 server 變快，排隊時間也算在 timeout 裡。
      * 一次可以問多題。choice 2~256 個選項（超過 16 個分組讀，再比一輪各組贏家），
        score 2~16 個等級；超出範圍 server 回 400（TypeSafeBadRequestError）。
      * state 超過 6,000 token 會被截掉（保留開頭），不會報錯。
      * 回應的 confidence / legend 照 TypeSafe 補正，noul 被 noul_commit 推離
        (0.2, 0.8)，見模組說明。

    ??? example "用法"

        ```python
        from langchain_typesafe import Choice, Noul, Score

        from classifier_chon import ChonClassifier

        classifier = ChonClassifier()
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
        print(response.model)  # Chon-1.0-12B
        print(response.nouls["urgent"].noul, response.nouls["urgent"].confidence)
        print(response.scores["severity"].score, response.scores["severity"].level)
        ```
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """LangSmith trace 顯示的模型名稱，不會送給 server。

    換成同一套 server 跑的其他 Chon 模型（ninhnv/chon-1.0-12b-fast、chon-1.0-e4b、
    chon-1.0-e2b）時，把這裡改成對應的名稱，trace 才分得出來。
    """

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("CHON_API_KEY", default=None),
        exclude=True,
        repr=False,
    )
    """Chon server 的 API key（選填）。

    只有 server 啟動時設了 `CHON_API_KEY` 才需要。順序：建構時給的 `api_key` >
    `CHON_API_KEY` 環境變數 > 不送 Authorization。
    """

    base_url: str = Field(
        default_factory=from_env("CHON_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """Chon 服務的網址，請求送到底下的 /v1/systemone。

    順序：建構時給的 `base_url` > `CHON_BASE_URL` 環境變數 > http://127.0.0.1:8000。
    """

    timeout: float = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """自動建立的 sync / async client 的 timeout（秒）；不影響外部傳入的 client。

    一般一題約 0.1 秒，要寫推理的題目 p95 約 5 秒（vLLM）；server 沒裝 vLLM 時
    改用 static KV cache 或一般 generate，p95 會到 10~24 秒。排隊時間也算在內。
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
            message = "Chon model must not be empty."
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
        return ["langchain", "classifiers", "chon"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "CHON_API_KEY"}

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
            message = "Synchronous Chon client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the Chon service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(self._parse(response, payload))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous Chon client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the Chon service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(self._parse(response, payload))

    def _parse(
        self, response: httpx2.Response, payload: dict[str, JsonValue]
    ) -> ClassifierResponse:
        """補正 chon 的回應（見模組說明），再交給 TypeSafe 的 parse_response。

        錯誤處理和 request_id 都照 parse_response：失敗的 HTTP 狀態、不是 JSON、
        或補正後還是不合 schema，都在 parse_response 裡 raise 對應的 TypeSafe 例外。
        """
        data: Any = None
        if response.is_success:
            try:
                data = response.json()
            except ValueError:
                data = None  # 不是 JSON：原樣交給 parse_response 報錯
            if isinstance(data, dict) and isinstance(data.get("answers"), dict):
                _fix_answers(data["answers"], payload)
                headers = {}
                if _REQUEST_ID_HEADER in response.headers:
                    headers[_REQUEST_ID_HEADER] = response.headers[_REQUEST_ID_HEADER]
                response = httpx2.Response(
                    response.status_code,
                    headers=headers,
                    json=data,
                    request=response.request,
                )
        parsed = parse_response(response)
        # 再用 ChonResponse 驗一次，confidence 和 chon 多給的欄位才留得住
        return ChonResponse.model_validate({**data, "request_id": parsed.request_id})

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

        大部分題目只讀 logits，output_tokens 是 0；有寫推理的題目，推理的 token 算在這裡。
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
            logger.debug("Could not attach Chon usage.", exc_info=True)
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
        return {
            "state": serialize_state(request["state"]),
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in request["questions"].items()
            },
        }


def _fix_answers(answers: dict[str, Any], payload: dict[str, JsonValue]) -> None:
    """就地補 confidence 和 score 的 legend，見模組說明。

    欄位缺了或形狀不對就跳過，讓 parse_response 報出是哪個欄位不合 schema。
    """
    questions = payload["questions"]
    for name, ans in answers.items():
        if not isinstance(ans, dict):
            continue
        kind = ans.get("type")
        probs = ans.get("probabilities")
        if kind == "noul":
            p = ans.get("noul")
            if isinstance(p, (int, float)) and 0.0 <= p <= 1.0:
                ans["confidence"] = noul_confidence(p)
        elif kind == "choice":
            if isinstance(probs, dict) and probs:
                ans["confidence"] = choice_confidence(list(probs.values()))
        elif kind == "score":
            criteria = questions.get(name, {}).get("criteria")
            if isinstance(criteria, list):
                ans["legend"] = dict(enumerate(criteria))
            if isinstance(probs, dict) and probs:
                try:
                    levels = sorted(probs, key=int)
                except (TypeError, ValueError):
                    continue
                ans["confidence"] = score_confidence([probs[k] for k in levels])


__all__ = [
    "ChonClassifier",
    "ChonNoulAnswer",
    "ChonResponse",
    "ChonScoreAnswer",
    "ChonUsage",
]


if __name__ == "__main__":
    import json
    import sys

    from langchain_typesafe import Choice, Noul, NoulCriteria, Score

    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    classifier = (
        ChonClassifier(base_url=base_url, timeout=120)
        if base_url
        else ChonClassifier(timeout=120)
    )
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
