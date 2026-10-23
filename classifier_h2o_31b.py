"""用本機 H2O-Lightning-31B 服務當 Jev classifier，以 TypeSafeClassifier 為範本，介面同 classifier_h2o.py。

範本是 langchain-ai/langchain 的 libs/partners/typesafe/langchain_typesafe/classifier.py
（class TypeSafeClassifier）。模型 https://huggingface.co/h2oai/h2o-lightning-31b：
google/gemma-4-31B-it 基底、31B 參數，H2O Model Non-Commercial Licence v1.1（只限非商業、
非正式上線使用）。repo 有 gate，要先在模型頁接受授權再 `hf auth login`。

服務是 vLLM 加作者的 lightning_31b_shim.py（vLLM 在 8000，shim 在 8741）。shim 原生提供
TypeSafe 的 POST /v1/systemone，不需要 API key：

    hf download h2oai/h2o-lightning-31b --revision v1.1 --local-dir h2o-lightning-31b

    # stock vLLM 0.30.0，一張 H100 80GB（FP8）。只有 one-way；超過 3,000 token 的 state 用 BM25 挑段落
    bash h2o-lightning-31b/serve.sh
    # 作者的 vLLM build（per-request attention），預設 two-way，JevBench 用的設定
    bash h2o-lightning-31b/run_default.sh

    curl -s http://127.0.0.1:8741/health    # 暖機完成前回 503 和 "warming": true

跟 TypeSafeClassifier 的差別：

  1. 網址預設 http://localhost:8741（或環境變數 H2O_LIGHTNING_31B_BASE_URL），model 預設
     h2oai/h2o-lightning-31b，照範本放進請求的 model 欄位。沒有 api_key，也不送 Authorization。
  2. 多兩個 31B shim 才有的選項，有設才放進請求的 "kwargs"，不設就照 shim 的 profile 決定：
       attention_mode="two-way" / "one-way"   ->  {"attention_mode": ...}
       readout_orders="auto" / 1 / 2          ->  {"readout": {"orders": ...}}（只影響 choice 題）
     two-way 要作者的 vLLM build，stock vLLM（serve.sh）只能 one-way。
  3. confidence 照 https://docs.typesafe.ai/confidence 重算。三個 *_confidence 函式和
     classifier_h2o.py、classifier_jevk5.py 相同，幾顆模型的數字才能並排比。模型卡沒寫 shim 的
     confidence 公式，所以 shim 原本的值不直接用，留在 shim_confidence。
     Noul 在 TypeSafe 本來沒有 confidence，照 |2p - 1| 補上（NoulAnswerWithConfidence）。
  4. 回應換成 H2O31BResponse，shim 比 TypeSafe 多給的欄位留得住：usage.processed_input_tokens、
     effort_used（attention_mode / attention_reason / state_selection / image_reread）、
     choice 題換順序讀兩次時的 orders，以及模型卡沒列的其他欄位（extra="allow"）。

**noul_floor 要注意。** 和 4B 一樣，shim 預設把 yes/no 機率往選中的那邊推到至少 0.801，
所以 noul 不會落在 (0.199, 0.801) 之間，noul_confidence 也不會低於 0.602。答案（哪一邊贏）
不變，但拿機率去比 llm_guardrails.py route() 那種門檻時，這個夾鉗會讓 31B 和 JevK5 不能直接
對照。要原始機率就用 SHIM_NOUL_FLOOR=0 啟動 shim。

圖片：把 data:image/...;base64,... URI 放進 state 裡任何地方，一個請求最多 4 張。

用法：
    from classifier_h2o_31b import H2O31BClassifier
    classifier = H2O31BClassifier()                          # 預設 http://localhost:8741
    classifier = H2O31BClassifier(attention_mode="one-way")  # 強制 one-way
    python classifier_h2o_31b.py                             # 跑一個三題的範例
    python classifier_h2o_31b.py http://127.0.0.1:8741 one-way
"""

import contextlib
import logging
from typing import Annotated, Any, Literal

import httpx2
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env
from langchain_typesafe._state import serialize_state
from langchain_typesafe.client import (
    _REQUEST_ID_HEADER,
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
from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

H2O_31B_BASE_URL = "http://localhost:8741"
H2O_31B_MODEL = "h2oai/h2o-lightning-31b"
_LS_PROVIDER = "h2o-lightning"

logger = logging.getLogger(__name__)


# ---- confidence ----
# 和 classifier_h2o.py、classifier_jevk5.py 的三個函式相同，照 https://docs.typesafe.ai/confidence 的公式。


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


def _fill_answer(ans: dict[str, Any], question: dict[str, Any]) -> None:
    """一題的 confidence 換成上面的公式，shim 原本的值搬到 shim_confidence。

    欄位不對（缺 probabilities、型別錯）就不補，留給 parse_response 丟 TypeSafe 的驗證錯誤。
    """
    if "confidence" in ans:
        ans["shim_confidence"] = ans["confidence"]
    kind = ans.get("type")
    with contextlib.suppress(
        TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError
    ):
        if kind == "noul":
            ans["confidence"] = noul_confidence(ans["noul"])
        elif kind == "choice":
            ans["confidence"] = choice_confidence(list(ans["probabilities"].values()))
        elif kind == "score":
            # TypeSafe 的 ScoreAnswer 一定要有 legend，shim 沒回就照問題的 criteria 補
            if "legend" not in ans:
                ans["legend"] = dict(enumerate(question.get("criteria") or []))
            probs = ans["probabilities"]
            ans["confidence"] = score_confidence(
                [probs[k] for k in sorted(probs, key=int)]
            )


# ---- 回應型別 ----
# TypeSafe 的型別用 pydantic 預設設定，不認得的欄位會被丟掉，所以每一層都換成 extra="allow" 的子類別。


class NoulAnswerWithConfidence(NoulAnswer):
    """NoulAnswer 多一個 confidence。

    子類別所以 isinstance(ans, NoulAnswer) 照樣成立，response.nouls 拿得到。
    """

    model_config = ConfigDict(extra="allow")

    confidence: float = Field(ge=0.0, le=1.0)
    """|2p - 1|，見 noul_confidence。"""

    shim_confidence: float | None = None
    """shim 原本回的 confidence；shim 沒回就是 None。"""


class H2O31BChoiceAnswer(ChoiceAnswer):
    """ChoiceAnswer 多 shim 原本的 confidence 和讀了幾次。"""

    model_config = ConfigDict(extra="allow")

    shim_confidence: float | None = None
    """shim 原本回的 confidence；confidence 已照 TypeSafe 文件重算。"""

    orders: int | None = None
    """選項換順序讀兩次再平均時 shim 回 2（見 readout_orders），只讀一次就是 None。"""


class H2O31BScoreAnswer(ScoreAnswer):
    """ScoreAnswer 多 shim 原本的 confidence。"""

    model_config = ConfigDict(extra="allow")

    shim_confidence: float | None = None
    """shim 原本回的 confidence；confidence 已照 TypeSafe 文件重算。"""


class H2O31BUsage(Usage):
    """Usage 多 shim 會回的 processed_input_tokens。"""

    model_config = ConfigDict(extra="allow")

    processed_input_tokens: int | None = None
    """實際處理的 prompt token。choice 題讀兩次或 re-read 時比 input_tokens（計費用）多。"""


class H2O31BResponse(ClassifierResponse):
    """ClassifierResponse 換成 31B shim 實際回的形狀。

    answers 的型別要跟著換，不然 model_dump / LangSmith trace 會照 TypeSafe 的型別輸出、
    漏掉 Noul 的 confidence 和 shim 多給的欄位。
    """

    model_config = ConfigDict(extra="allow")

    answers: dict[
        str,
        Annotated[
            NoulAnswerWithConfidence | H2O31BChoiceAnswer | H2O31BScoreAnswer,
            Field(discriminator="type"),
        ],
    ]

    usage: H2O31BUsage = Field(default_factory=H2O31BUsage)

    effort_used: dict[str, JsonValue] | None = None
    """shim 怎麼讀這個請求：attention_mode（bidirectional = two-way，causal = one-way）、
    attention_reason、state_selection（state 太長被挑段落時），以及 image_reread。"""


class H2O31BClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """用本機 H2O-Lightning-31B 服務取代 TypeSafeClassifier，用法相同。

    跟 TypeSafeClassifier 一樣是 LangChain Runnable（支援 invoke / ainvoke / batch /
    abatch / `|` 串接 / callbacks / LangSmith tracing）。state 可以是字串、JSON 物件或陣列，
    LangChain 的 BaseMessage 會轉成 role/content JSON。

    驗證完設定後會建立 sync 和 async 兩個 httpx2 client；要自己控制連線就傳 client /
    async_client，傳進來的照原樣使用、不會被關掉，timeout 只套用在自動建立的 client。
    """

    model: str = Field(default=H2O_31B_MODEL, min_length=1)
    """放進請求 model 欄位的模型名稱，也用在 LangSmith trace。要和 vLLM 的 served-model-name 一樣。"""

    base_url: str = Field(
        default_factory=from_env("H2O_LIGHTNING_31B_BASE_URL", default=H2O_31B_BASE_URL)
    )
    """shim 的網址，請求送到底下的 /v1/systemone。

    順序：建構時給的 base_url > 環境變數 H2O_LIGHTNING_31B_BASE_URL > http://localhost:8741。
    """

    attention_mode: Literal["two-way", "one-way"] | None = None
    """每個請求強制用哪種讀法，送成 kwargs.attention_mode。

    None 就照 shim 的 profile：run_default.sh / run_decision_index.sh 預設 two-way，state 超過
    two-way window 或超過 latency cap 的請求改 one-way；serve.sh（stock vLLM）只有 one-way。
    """

    readout_orders: Literal["auto", 1, 2] | None = None
    """choice 題要不要換選項順序再讀一次，送成 kwargs.readout.orders。

    "auto"：第一次的最高機率低於 0.8 才讀第二次；2：每題都讀兩次；1：只讀一次。
    None 就照 shim 的 serve_config.json（預設 1）。yes/no 和 score 題一律只讀一次。
    """

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

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str) -> str:
        model = model.strip()
        if not model:
            message = "h2o-lightning model must not be empty."
            raise ValueError(message)
        return model

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
        if self.client is None:  # pragma: no cover - _build_clients 保證有
            message = "Synchronous h2o-lightning client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(self._endpoint, json=payload)
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the h2o-lightning-31b service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._parse(response, payload)

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        if self.async_client is None:  # pragma: no cover - _build_clients 保證有
            message = "Asynchronous h2o-lightning client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(self._endpoint, json=payload)
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the h2o-lightning-31b service."
            raise TypeSafeAPIConnectionError(message) from error
        return self._parse(response, payload)

    def _parse(
        self, response: httpx2.Response, payload: dict[str, JsonValue]
    ) -> ClassifierResponse:
        data: Any = None
        if response.is_success:
            with contextlib.suppress(ValueError):
                data = response.json()
        if isinstance(data, dict) and isinstance(data.get("answers"), dict):
            questions = payload["questions"]
            for name, ans in data["answers"].items():
                if isinstance(ans, dict):
                    _fill_answer(ans, questions.get(name) or {})
            request_id = response.headers.get(_REQUEST_ID_HEADER)
            response = httpx2.Response(
                response.status_code,
                json=data,
                headers={_REQUEST_ID_HEADER: request_id} if request_id else None,
                request=response.request,
            )
        # 錯誤狀態碼（例如暖機中的 503）、不是 JSON、缺欄位都照 TypeSafe 的 parse_response raise
        parsed = parse_response(response)
        # 再用 H2O31BResponse 驗一次，shim 多給的欄位才留得住（見 H2O31BResponse）
        return self._record_usage(
            H2O31BResponse.model_validate({**data, "request_id": parsed.request_id})
        )

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
            logger.debug("Could not attach h2o-lightning usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    def _payload(self, request: ClassifierRequest) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "state": serialize_state(request["state"]),
            "model": self.model,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in request["questions"].items()
            },
        }
        kwargs: dict[str, JsonValue] = {}
        if self.attention_mode is not None:
            kwargs["attention_mode"] = self.attention_mode
        if self.readout_orders is not None:
            kwargs["readout"] = {"orders": self.readout_orders}
        if kwargs:
            payload["kwargs"] = kwargs
        return payload


__all__ = [
    "H2O_31B_BASE_URL",
    "H2O_31B_MODEL",
    "H2O31BClassifier",
    "H2O31BResponse",
]


if __name__ == "__main__":
    import json
    import sys

    from langchain_typesafe import Choice, Noul, NoulCriteria, Score

    base_url = sys.argv[1] if len(sys.argv) > 1 else H2O_31B_BASE_URL
    attention_mode = sys.argv[2] if len(sys.argv) > 2 else None
    classifier = H2O31BClassifier(
        base_url=base_url, attention_mode=attention_mode, timeout=120
    )
    # 模型卡 "The API" 那段的範例
    response = classifier.invoke(
        {
            "state": (
                "Ticket 4471: since 09:10 the checkout page returns HTTP 500 for every "
                "customer; no orders have gone through in 40 minutes. Severity guide: "
                "P1 = a revenue path fully down; P2 = degraded but working; "
                "P3 = cosmetic."
            ),
            "questions": {
                "priority": Choice(
                    instructions="Which priority does the severity guide assign?",
                    criteria={
                        "p1": "Revenue path fully down",
                        "p2": "Degraded but working",
                        "p3": "Cosmetic",
                    },
                ),
                "all_users": Noul(
                    instructions="The problem affects every customer.",
                    criteria=NoulCriteria(
                        true="Affects everyone", false="Affects only some"
                    ),
                ),
                "urgency": Score(
                    instructions="How urgent is this ticket?",
                    criteria=["Low", "Medium", "High"],
                ),
            },
        }
    )
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=1))
