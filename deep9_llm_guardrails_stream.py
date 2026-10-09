"""deep9_llm_guardrails 的 streaming 版：只換掉 run()，agent 和所有設定都從 deep9_llm_guardrails import。

deep9_llm_guardrails.run() 用 agent.invoke()，整個 turn 跑完才一次印出結果。走 skill 的任務要兩三分鐘，
那段時間畫面上只有「使用者：…」一行，看起來像卡住（跑範例時第一題就是 xlsx，實測 2 分 46 秒）。

這裡改用 agent.stream()，同時收三種 stream_mode：
  updates   每個節點跑完吐一次 → 分類結果一拿到就印（約 2 秒），之後每個工具呼叫和結果都即時印
  values    每個 super-step 後的完整 state → 最後一筆等於 invoke() 原本的回傳值，用來印最後的回覆
  messages  模型產生中的 token → 單次模型呼叫可能幾十秒，拿來當心跳，每 0.5 秒印一個點

輸入端被 guardrails 擋下時 TriageMiddleware.before_agent 會帶 jump_to，這裡照樣印出來，
讓「一次模型都沒呼叫」這件事在畫面上看得到。

用法跟原本一樣：
    python deep9_llm_guardrails_stream.py                 # 跑範例
    python deep9_llm_guardrails_stream.py "你的要求"       # 跑一個自訂的要求
"""

import sys
import time

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langfuse import get_client

from deep9_llm_guardrails import (
    CODING_THRESHOLD,
    EXAMPLES,
    GUARD_POLICY,
    MODELS,
    agent,
    langfuse_handler,
    print_hazards,
)
from llm_guardrails import POLICIES

# 工具參數裡優先拿來當預覽的欄位，照 deep agent 內建工具的參數名排
PREVIEW_KEYS = ("command", "file_path", "path", "query", "description")


def shorten(text, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def tool_call_preview(tool_call) -> str:
    args = tool_call.get("args") or {}
    for key in PREVIEW_KEYS:
        if key in args:
            return shorten(args[key])
    return shorten(", ".join(f"{k}={v!r}" for k, v in args.items()))


class Progress:
    """印帶經過秒數的進度行；模型產生中改印點，當作還活著的心跳。"""

    DOT_INTERVAL = 0.5  # 秒

    def __init__(self) -> None:
        self.start = time.monotonic()
        self.last_dot = 0.0
        self.dots_open = False

    def _close_dots(self) -> None:
        if self.dots_open:
            print(flush=True)
            self.dots_open = False

    def line(self, text: str) -> None:
        self._close_dots()
        print(f"  [{time.monotonic() - self.start:6.1f}s] {text}", flush=True)

    def plain(self, text: str = "") -> None:
        self._close_dots()
        print(text, flush=True)

    def dot(self) -> None:
        now = time.monotonic()
        if now - self.last_dot < self.DOT_INTERVAL:
            return
        self.last_dot = now
        if not self.dots_open:
            print("  ", end="", flush=True)
        print("·", end="", flush=True)
        self.dots_open = True


def print_triage(progress: Progress, update: dict) -> None:
    """TriageMiddleware.before_agent 一跑完就印，不等整個 turn 結束。"""
    route_answer, policy, coding = update["model_route"], POLICIES[GUARD_POLICY], update["coding"]
    progress.line("jevk5 一次分類的結果：")
    progress.plain(
        f"      模型      {route_answer.choice} -> {MODELS[route_answer.choice].model_name}"
        f"  (信心 {route_answer.confidence:.1%})"
    )
    progress.plain(f"      coding    {coding:6.1%}  {'是' if coding >= CODING_THRESHOLD else '否'}")
    progress.plain(
        f"      輸入 guardrails  -> {update['guard'].upper()}  [policy={GUARD_POLICY}: review >= "
        f"{policy['review_threshold']:.2f}, action >= {policy['action_threshold']:.2f}, "
        f"severity blocks at {policy['severity_block']:.2f}]"
    )
    print_hazards(update["hazards"], update["severity"])


def log_message(progress: Progress, message) -> None:
    if isinstance(message, AIMessage):
        text = message.text.strip()
        if text and message.tool_calls:
            # 工具呼叫前的旁白，模型用來說明下一步要做什麼
            progress.line(f"… {shorten(text, 70)}")
        for tool_call in message.tool_calls or []:
            progress.line(f"→ {tool_call['name']}  {tool_call_preview(tool_call)}")
        if text and not message.tool_calls:
            progress.line(f"模型給出最後的回覆（{len(text)} 字）")
    elif isinstance(message, ToolMessage):
        progress.line(f"{'✓' if message.status == 'success' else '✗'} {message.name}  {message.status}")


def handle_update(progress: Progress, chunk: dict) -> None:
    for node, update in chunk.items():
        if not isinstance(update, dict):  # 沒有改 state 的 hook 回傳 None
            continue
        if node == "TriageMiddleware.before_agent":
            print_triage(progress, update)
            if update.get("jump_to"):
                progress.line("輸入端擋下，直接跳到 end，主 agent 一次模型都不呼叫")
        elif node == "TriageMiddleware.after_agent":
            progress.line(f"輸出 guardrails  -> {update['output_guard'].upper()}")
            print_hazards(update["output_hazards"], update["output_severity"])
        elif "skills_metadata" in update:
            progress.line(f"載入 {len(update['skills_metadata'] or ())} 個 skill")
        else:
            for message in update.get("messages") or []:
                log_message(progress, message)


def run(task):
    print("=" * 80)
    print(f"使用者：{task}", flush=True)
    progress = Progress()
    final_state = None

    for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": task}]},
        config={"callbacks": [langfuse_handler]},
        stream_mode=["updates", "values", "messages"],
    ):
        if mode == "values":
            final_state = chunk  # 最後一筆是 after_agent 跑完的完整 state
        elif mode == "messages":
            # messages 模式除了模型的 token，也會吐出節點直接寫進 state 的完整訊息
            # （輸入端擋下時的 REFUSAL_MESSAGE 就是一筆），只有 chunk 才算模型正在產生
            if isinstance(chunk[0], AIMessageChunk):
                progress.dot()
        else:
            handle_update(progress, chunk)

    progress.plain()
    if final_state and final_state.get("messages"):
        final_state["messages"][-1].pretty_print()
    else:  # 理論上不會發生，但不要讓 stream 的意外變成 KeyError
        progress.plain("stream 沒有吐出 values，拿不到最後的回覆")


if __name__ == "__main__":
    for task in sys.argv[1:] or EXAMPLES:
        run(task)
    get_client().flush()
