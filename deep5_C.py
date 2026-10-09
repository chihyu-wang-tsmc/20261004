"""選項 C：給 agent shell，全自動執行，不暫停核准。

跟 deep5_B.py 一樣，工作目錄是 ~/jevk5/deck、shell 拿不到你的 API key，
但 agent 的每個指令都直接在主機上執行、沒有人把關。唯一的防線是 AutoModeMiddleware：
每個 execute 先交給 jevk5 判斷風險，判定危險就擋下（這是模型判斷，不是沙箱，不保證擋得住）。
agent 會讀網頁內容，若網頁藏有提示注入，可能誘導它執行惡意指令，請只在能接受這個風險時使用。
"""

import functools
import os
import sys

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents_code.tools import fetch_url, web_search
from langchain_typesafe.experimental.middleware import AutoModeMiddleware, auto_mode
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from classifier_jevk5 import JevK5Classifier
from llm_models import deepseek

# AutoMode 會把最近 30 則對話（搜尋結果、SKILL.md、整支程式）一起送給 jevk5 判斷風險，
# 內容很長時 jevk5 要 20 秒以上，JevK5Classifier 預設的 30 秒 timeout 會不夠，所以拉長到 180 秒
auto_mode.TypeSafeClassifier = functools.partial(JevK5Classifier, timeout=1800)

langfuse = get_client()
root = langfuse.start_observation(name="deep5_C", as_type="span")
langfuse_handler = CallbackHandler(
    trace_context={"trace_id": root.trace_id, "parent_span_id": root.id}
)


DEEPSEEK_V41_FLASH_MODEL = deepseek("deepseek-flash")

SKILLS_DIR = os.path.expanduser("~/skills_anthropic")
DECK_DIR = os.path.expanduser("~/jevk5/deck")

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
執行 `mkdir -p out && node jev_slides.js` 產生 out/jev_C.pptx →
執行 `python {SKILLS_DIR}/pptx/scripts/office/validate.py out/jev_C.pptx` 驗證，有錯就改程式重跑。
只在工作目錄內操作，不要讀取或修改工作目錄以外的檔案。"""

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
    # 沒有人工核准；每個 execute 先給 jevk5 判斷風險，危險的就擋下（需要 jevk5-serve 開在 8090）
    middleware=[AutoModeMiddleware(tools=["execute"])],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    config={"callbacks": [langfuse_handler]},
)
result["messages"][-1].pretty_print()
out_file = os.path.join(DECK_DIR, "out", "jev_C.pptx")
print(f"投影片：{out_file}" if os.path.exists(out_file) else "沒有產生 out/jev_C.pptx")

root.end()
langfuse.flush()
