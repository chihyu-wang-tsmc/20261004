"""deep*.py 共用的模型建立函式。

用法：
    from llm_models import deepseek, qwen
    DEEPSEEK_V41_FLASH_MODEL = deepseek("deepseek-flash")

每個函式只在被呼叫時才讀對應的 API key 環境變數，所以只用到 qwen 的程式不需要設定其他家的 key。
其他參數（例如 timeout=600）可以用關鍵字傳進去，會直接交給 ChatOpenAI / ChatGoogleGenerativeAI。
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# Qwen via Alibaba Cloud DashScope's OpenAI-compatible API
def qwen(model_name, **kwargs):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        temperature=0,
        **kwargs,
    )


def deepseek(model_name, **kwargs):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0,
        # 輸出上限要放在 extra_body 的 max_tokens：ChatOpenAI 的 max_tokens= 會被改送成
        # max_completion_tokens，DeepSeek 不認，會停在預設的 8192，整支投影片程式被截斷成無效的 JSON
        extra_body={"thinking": {"type": "disabled"}, "max_tokens": 65536},
        **kwargs,
    )


# K3 的思考模式一定開啟、不能關；temperature / top_p 等是固定值，不能設定（ChatOpenAI 預設不送，所以不要加 temperature=）。
# 思考深度用 extra_body={"reasoning_effort": "low" | "high" | "max"} 調整，預設 max（最慢、最貴）。
def kimi(model_name, **kwargs):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url="https://api.moonshot.ai/v1",
        **kwargs,
    )


# GLM-4.7-Flash via 智譜 Z.ai 國際站（api.z.ai）的 OpenAI-compatible API，這個模型免費
# 思考模式預設開啟，開著時一次工具呼叫要幾十秒到幾分鐘，所以關掉（關掉後約 3 秒）。
# 輸出上限跟 DeepSeek 一樣要放在 extra_body 的 max_tokens：ChatOpenAI 送的 max_completion_tokens 它會忽略。
# 免費模型尖峰時常回 429「服務暫時過載」，max_retries 讓它自動重試。
def glm(model_name, **kwargs):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["ZAI_API_KEY"],
        base_url="https://api.z.ai/api/paas/v4/",
        temperature=0,
        max_retries=6,
        extra_body={"thinking": {"type": "disabled"}, "max_tokens": 65536},
        **kwargs,
    )


# Gemini via Google 官方的 ChatGoogleGenerativeAI（不是 ChatOpenAI）
# Gemini 3 是思考模型，多輪工具呼叫時要把 thought_signature 原樣傳回去；
# ChatOpenAI 走 OpenAI 相容網址時會丟掉它，第二輪就回 400，官方套件會自動處理。
# Google 建議 Gemini 3 維持預設 temperature 1.0，調低可能造成重複或迴圈，所以不設 temperature。
def gemini(model_name, **kwargs):
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=os.environ["GEMINI_API_KEY"],
        max_output_tokens=65536,
        **kwargs,
    )


__all__ = ["deepseek", "gemini", "glm", "kimi", "qwen"]
