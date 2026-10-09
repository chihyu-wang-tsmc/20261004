"""選項 B：給 agent shell，但每一個 execute 指令都要你核准才執行。

agent 的工作目錄是 ~/jevk5/deck（LocalShellBackend），可以自己寫程式、執行 node、
跑 pptx skill 的 validate.py、看錯誤訊息自己修。shell 不繼承你的環境變數（拿不到 API key），
但指令是直接在主機上執行、沒有隔離，所以每個指令都會暫停，印出來等你輸入 y 才執行。
"""

import os
import sys

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents_code.tools import fetch_url, web_search
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from llm_models import deepseek

langfuse = get_client()
root = langfuse.start_observation(name="deep5_B", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)


DEEPSEEK_V41_FLASH_MODEL = deepseek("deepseek-flash")

SKILLS_DIR = os.path.expanduser("~/skills_anthropic")
DECK_DIR = os.path.expanduser("~/jevk5/deck")

# 檔案工具的 / 對應 ~/jevk5/deck，shell 指令也在這裡執行；/skills/ 導到硬碟上的 skill。
# shell 的 PATH 只放 node 和這個 conda 環境的 python（validate.py 要用），不帶任何 API key。
backend = CompositeBackend(
    default=LocalShellBackend(
        root_dir=DECK_DIR,
        virtual_mode=True,
        inherit_env=False,
        env={"PATH": f"{os.path.dirname(sys.executable)}:/usr/bin:/bin", "HOME": DECK_DIR},
        timeout=180,
    ),
    routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
)

SYSTEM_PROMPT = f"""You are a helpful assistant. 回答一律使用繁體中文。
工作目錄是 {DECK_DIR}，execute 的 shell 指令就在這個目錄執行。
檔案工具（read_file/write_file/edit_file/ls）的路徑以 / 開頭、相對於工作目錄：
寫到 /jev_slides.js 就是 {DECK_DIR}/jev_slides.js。檔案工具裡不要用 /home/... 的完整路徑，
否則會寫到錯的位置；shell 指令裡則直接用相對路徑（例如 node jev_slides.js）。
這裡已裝好 pptxgenjs、react、react-dom、react-icons、sharp，不要 npm install。
skill 檔案在檔案工具的 /skills/ 底下；但在 shell 指令裡要用實際路徑 {SKILLS_DIR}/
（例如 {SKILLS_DIR}/pptx/scripts/office/validate.py）。
做法：讀 /skills/pptx/SKILL.md → 把程式用 write_file 寫到 /jev_slides.js →
執行 `mkdir -p out && node jev_slides.js` 產生 out/jev_B.pptx →
執行 `python {SKILLS_DIR}/pptx/scripts/office/validate.py out/jev_B.pptx` 驗證，有錯就改程式重跑。
每個 shell 指令都要經過使用者核准，請盡量合併指令、減少次數。"""

TASK = (
    "上網搜尋 TypeSafe AI 的 Jev（一種不產生文字、直接回傳各選項機率的決策模型）的最新資料，"
    "然後依照 pptx skill 的做法，做一份 6～8 頁的繁體中文投影片，內容包含：封面、Jev 是什麼、"
    "三種題型（Noul / Choice / Score）、運作方式與特色、使用情境、評測表現、參考來源（列出網址）。"
)

agent = create_deep_agent(
    model=DEEPSEEK_V41_FLASH_MODEL,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search, fetch_url],
    backend=backend,
    skills=["/skills/"],
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")
    ],
    # 每個 execute 都暫停等人核准；暫停後要能接著跑，所以需要 checkpointer 保存進度
    interrupt_on={"execute": True},
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "deep5_B"}, "callbacks": [langfuse_handler]}
result = agent.invoke({"messages": [{"role": "user", "content": TASK}]}, config=config)

# agent 每次想執行指令就會停在這裡，印出指令等你決定，再用 Command(resume=...) 繼續
while result.get("__interrupt__"):
    request = result["__interrupt__"][0].value
    decisions = []
    for action in request["action_requests"]:
        print("\n" + "=" * 60)
        print("agent 想執行：", action["args"].get("command", action["args"]))
        answer = input("核准執行？[y/N] ").strip().lower()
        if answer == "y":
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "使用者拒絕執行這個指令。"})
    result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

result["messages"][-1].pretty_print()
out_file = os.path.join(DECK_DIR, "out", "jev_B.pptx")
print(f"投影片：{out_file}" if os.path.exists(out_file) else "沒有產生 out/jev_B.pptx")

root.end()
langfuse.flush()
