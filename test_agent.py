"""Minimal LangChain create_agent example.

Install:
    pip install -U langchain langchain-openai

Env:
    export DASHSCOPE_API_KEY=...   # Alibaba Cloud Model Studio (international) key
    export MODEL_NAME=qwen3.8-27b  # optional
"""
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake_data = {"taipei": "Sunny, 30°C", "hsinchu": "Windy, 26°C"}
    return fake_data.get(city.lower(), f"No weather data for {city}")


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# Qwen via Alibaba Cloud DashScope's OpenAI-compatible API (international endpoint)
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen3.8-27b"),
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[get_weather, add],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Hsinchu? Also, what is 123 + 456?"}]}
    )

    # Print the full trace: user -> tool calls -> tool results -> final answer
    for msg in result["messages"]:
        msg.pretty_print()
