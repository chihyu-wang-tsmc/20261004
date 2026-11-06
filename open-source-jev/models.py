"""作答用的 LLM：一律走 OpenAI 相容的 chat completions API（公司 AI gateway、本機 vLLM 都是）。

    run_bench.py         讓 model 回答 GPQA / AIME / AI2 ARC / GSM8K
    guard_report.py run  第 2 步：守門放行的題目交給 model_route 選到的 model 回答
    classifier_gemini.py Gemini 當 classifier 時也走這裡的連線設定

每個 model 的連線設定放在 .env（範例見 .env.example），變數名稱是 MODELS 的 env 字首加上：
    <env>_BASE_URL   API 網址，例如 https://<gateway>/v1、http://localhost:8001/v1
    <env>_TOKEN      API token（本機 vLLM 不用填）
    <env>_MODEL      （選填）送給 API 的模型名稱；沒填就用 MODELS 裡的 api_model

價格是 $ / 百萬 token（輸入 / 輸出），routing_report.py 算花費用。要加 model 就在 MODELS 加一行。
"""

import os
import re
import sys
from dataclasses import dataclass

import common  # noqa: F401  讀 .env


@dataclass(frozen=True)
class ModelSpec:
    env: str
    """.env 變數的字首。"""
    api_model: str
    """送給 API 的模型名稱（.env 的 <env>_MODEL 可以覆寫）。"""
    price_in: float
    """$ / 百萬輸入 token。"""
    price_out: float
    """$ / 百萬輸出 token（思考的 token 也算輸出）。"""


MODELS = {
    "gemini-3.7-flash": ModelSpec("GEMINI_37_FLASH", "gemini-3.7-flash", 0.30, 2.50),
    "gemma-4-31b": ModelSpec("GEMMA_4_31B", "gemma-4-31b", 0.14, 0.42),
    "qwen3.5-0.8b": ModelSpec("QWEN35_08B", "qwen3.5-0.8b", 0.05, 0.08),
}

# 有些 model（例如沒開 reasoning parser 的 Qwen）會把思考過程用 <think>…</think> 寫在回覆開頭，拿掉才是答案
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


@dataclass
class Reply:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    """stop = 正常結束；length = 被輸出上限截斷（通常就抽不到答案）；content_filter = 被 API 的安全過濾擋掉。"""


class ChatModel:
    """一個 model 的連線。chat() 可以同時從多個 thread 呼叫。"""

    def __init__(self, name, timeout=1200.0, max_retries=3):
        from openai import OpenAI

        if name not in MODELS:
            sys.exit(f"models.py 沒有 {name!r}（有：{', '.join(MODELS)}）")
        spec = MODELS[name]
        base_url = os.environ.get(f"{spec.env}_BASE_URL")
        if not base_url:
            sys.exit(f"{name}：.env 沒有設 {spec.env}_BASE_URL（見 .env.example）")
        self.name = name
        self.api_model = os.environ.get(f"{spec.env}_MODEL") or spec.api_model
        # 429 / 5xx / 連線錯誤由 openai 套件自己退避重試 max_retries 次；
        # 沒設 timeout 的話，model 卡住時會一直等下去
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(f"{spec.env}_TOKEN") or "unused",
                             timeout=timeout, max_retries=max_retries)

    def chat(self, messages, **params):
        """messages 是 [{"role": ..., "content": ...}]；params 原樣交給 API（例如 response_format）。"""
        r = self.client.chat.completions.create(model=self.api_model, messages=messages, **params)
        choice = r.choices[0]
        usage = r.usage
        return Reply(
            text=_THINK.sub("", choice.message.content or ""),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
        )


def create_model(name, **kwargs):
    return ChatModel(name, **kwargs)


def cost(name, input_tokens, output_tokens):
    """一次呼叫的花費（美元）。"""
    spec = MODELS[name]
    return ((input_tokens or 0) * spec.price_in + (output_tokens or 0) * spec.price_out) / 1e6
