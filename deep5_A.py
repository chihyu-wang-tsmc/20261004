"""選項 A：agent 只寫程式，由這支程式在沙箱裡執行 node。

agent 上網查 Jev、讀 pptx skill，把 pptxgenjs 程式寫進「虛擬檔案」/jev_slides.js；
agent 沒有 shell，碰不到硬碟上的其他檔案。agent 結束後，這支程式把 /jev_slides.js
存到 ~/jevk5/deck，用 Node 的權限模式（--permission）執行：只能讀 deck 資料夾、
只能寫 deck/out，不能開子程序，也拿不到你的 API key。
"""

import os
import subprocess
import sys

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents_code.tools import fetch_url, web_search
from langgraph.checkpoint.memory import MemorySaver
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from llm_models import deepseek

langfuse = get_client()
root = langfuse.start_observation(name="deep5_A", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)


DEEPSEEK_V41_FLASH_MODEL = deepseek("deepseek-flash")

SKILLS_DIR = os.path.expanduser("~/skills_anthropic")
DECK_DIR = os.path.expanduser("~/jevk5/deck")
OUT_DIR = os.path.join(DECK_DIR, "out")
SCRIPT_FILE = os.path.join(DECK_DIR, "jev_slides_A.js")
OUT_FILE = os.path.join(OUT_DIR, "jev_A.pptx")

# 預設是虛擬檔案系統（agent 寫的檔案只存在 state 裡），只有 /skills/ 導到硬碟上的 skill
backend = CompositeBackend(
    default=StateBackend(),
    routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
)

SYSTEM_PROMPT = """You are a helpful assistant. 回答一律使用繁體中文。
你沒有 shell，不能執行任何程式。做投影片的方式：
1. 先用 read_file 讀 /skills/pptx/SKILL.md，照「Creating with pptxgenjs」的注意事項寫程式。
2. 把完整的 Node.js（CommonJS，require）程式用 write_file 寫到 /jev_slides.js。
3. 程式最後必須用 pres.writeFile({ fileName: process.env.OUT_FILE }) 輸出，不要寫死檔名或路徑。
4. 可用套件都已安裝，不要 npm install：pptxgenjs、react、react-dom、react-icons、sharp。
5. 程式只能寫自己的輸出檔；不能用 child_process、網路、也不能讀其他檔案（執行時會被沙箱擋下）。
程式寫好後，系統會在沙箱裡執行它並產生 .pptx，你不用也不能自己執行。"""

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
    # skill 檔案只能讀，不能被 agent 改寫或刪除
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")
    ],
    # 失敗時要把錯誤傳回給同一段對話修正，需要 checkpointer 保存對話和虛擬檔案
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "deep5_A"}, "callbacks": [langfuse_handler]}
result = agent.invoke({"messages": [{"role": "user", "content": TASK}]}, config=config)
result["messages"][-1].pretty_print()


# ===== agent 結束後：取出虛擬檔案 /jev_slides.js，在沙箱裡用 node 執行 =====
def file_text(file_data):
    """StateBackend 的檔案內容可能是字串或一行一行的 list。"""
    content = file_data["content"]
    return content if isinstance(content, str) else "\n".join(content)


def build_in_sandbox(script_text):
    """把程式存到 deck、在沙箱裡執行並驗證；成功回傳 None，失敗回傳錯誤訊息。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)
    with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(script_text)
    run = subprocess.run(
        [
            "node",
            "--permission",  # Node 權限模式：預設禁止讀寫檔案、開子程序
            "--allow-addons",  # sharp 是原生模組，需要這個才能載入
            f"--allow-fs-read={DECK_DIR}",  # 只能讀 deck（程式本身和 node_modules）
            f"--allow-fs-write={OUT_DIR}",  # 只能寫 deck/out
            SCRIPT_FILE,
        ],
        cwd=DECK_DIR,
        # 不繼承你的環境變數，node 程式拿不到任何 API key
        env={"PATH": "/usr/bin:/bin", "HOME": DECK_DIR, "OUT_FILE": OUT_FILE},
        capture_output=True,
        text=True,
        timeout=180,
    )
    if run.returncode != 0 or not os.path.exists(OUT_FILE):
        stderr = "\n".join(l for l in run.stderr.splitlines() if "--allow-addons" not in l and "trace-warnings" not in l)
        return f"node 執行失敗（exit {run.returncode}）：\n{stderr[-3000:]}"
    # 用 pptx skill 自帶的驗證工具檢查檔案
    check = subprocess.run(
        [sys.executable, "scripts/office/validate.py", OUT_FILE],
        cwd=os.path.join(SKILLS_DIR, "pptx"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if check.returncode != 0:
        return f"validate.py 檢查失敗：\n{(check.stdout + check.stderr)[-3000:]}"
    return None


# agent 自己不能執行程式，所以失敗時把錯誤訊息傳回給它修，最多試 3 次
for attempt in range(1, 4):
    files = agent.get_state(config).values.get("files", {})
    script = files.get("/jev_slides.js")
    if script is None:
        error = f"找不到 /jev_slides.js（虛擬檔案有：{sorted(files)}），請把完整程式寫到 /jev_slides.js。"
    else:
        print(f"第 {attempt} 次：程式存到 {SCRIPT_FILE}，在沙箱裡執行…")
        error = build_in_sandbox(file_text(script))
        if error is None:
            print(f"成功，validate.py 通過。投影片：{OUT_FILE}")
            break
    print(error)
    if attempt < 3:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"系統執行你的程式時失敗了：\n{error}\n請修正 /jev_slides.js（用 edit_file 或 write_file），不用重新搜尋資料。"}]},
            config=config,
        )
else:
    print("試了 3 次仍然失敗，最後的程式在：", SCRIPT_FILE)

root.end()
langfuse.flush()
