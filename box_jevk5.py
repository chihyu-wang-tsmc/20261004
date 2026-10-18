"""box_h2o.py 的 jevk5 版：照 https://docs.typesafe.ai/primitives/box 的範例，
把 Jev（api.typesafe.ai，"model": "jev-preview"）換成本機 jevk5（classifier_jevk5.py）。

**JevK5 答不了 Box，這支跑到第一個 request 就會停。** 原因有兩層：
  * 模型看不到圖：alibiserikbay/JevK5（v0.3）是純文字權重，config.json 是 Qwen3_5ForCausalLM /
    qwen3_5_text，沒有 vision tower（Hub 上標的是 text-generation，h2o-lightning-4b 標的是
    image-text-to-text）。jevk5 repo 的 bench/vision_probe.py 只是把文字權重接上 Qwen 視覺塔的實驗，
    不是釋出的模型。
  * 服務不收：jevk5 runtime（github.com/allebee/jevk5）的 jevk5/server.py，normalized() 只收
    noul / choice / score，box 題回 HTTP 400 {"error": "unknown question type 'box'"}；
    server 也只讀 request 的 state 和 questions，assets 整個被忽略，參照字串（6e1d0b7c-kitchen）
    會被當成一般文字送進模型。
所以 Request structure 會照樣印出來、存好 request.json，送出去之後印出服務回的 400 和原因就結束（exit 1）。
後面的 Noul / Choice 雖然 jevk5 收，但它看不到圖，答案沒有意義，所以不往下跑。要真的跑 Box 請用 box_h2o.py。

為了兩顆模型可以並排比，這支和 box_h2o.py 只差這幾處，其餘（每一段的程式、題目、圖、印法）一行都沒改：
  * client 是 JevK5Classifier 的子類別 JevK5BoxClassifier（classifier_jevk5.py 本身沒改），
    多送 assets、多收 box 答案，做法和 box_h2o.py 的 H2OBoxClassifier 一樣。
  * 網址用 JEVK5_ENDPOINT 改（預設 http://localhost:8090），輸出放在 box_jevk5/。
  * 400 時印的提示（SERVICE_HINT）。
官網和這兩支不同的地方（Box 題型和 box 答案怎麼補、圖為什麼用 base64 不用 multipart、box/ 裡三張
CC0 照片、ready() 放大小圖、Noul 看 >= 0.5）見 box_h2o.py 開頭，這裡做法相同；box/ 的圖也是同一份。

jevk5-serve 要開在 8090。用法：
    python box_jevk5.py
    JEVK5_ENDPOINT=http://127.0.0.1:8090 python box_jevk5.py   # 指定服務網址
"""

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

import httpx2
from langchain_typesafe import Choice, Noul
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from langchain_typesafe.types import ChoiceAnswer, ScoreAnswer
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, JsonValue

from classifier_jevk5 import (
    JEVK5_BASE_URL,
    JevK5Classifier,
    JevK5Response,
    NoulAnswerWithConfidence,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "box"  # kitchen.jpg、kitchen-no-mug.jpg、cabinet.jpg，h2o 版共用
OUT_DIR = HERE / "box_jevk5"  # 切出來的圖、畫了框的圖、request.json

# ---- Box 和圖片（langchain-typesafe 0.0.1a3 沒有，照官方 typesafe-sdk 0.7.5 的 wire 格式補上）----


class Box(BaseModel):
    """Box 題：{"type": "box", "instructions": ...}，不收 criteria（typesafe-sdk 的 BoxQuestion）。"""

    type: Literal["box"] = "box"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    """要找什麼、在哪張圖，例如 "the coffee mug in `photo`"。"""


class BoxCoordinates(BaseModel):
    """框的四邊，都是 0~1、從左上角量：top / bottom 是高度的比例，left / right 是寬度的比例。"""

    top: float
    bottom: float
    left: float
    right: float


class BoxAnswer(BaseModel):
    """Box 題的答案（typesafe-sdk 的 BoxAnswer），加上 h2o 多給的 present / found。"""

    type: Literal["box"]
    box: BoxCoordinates
    confidence: float | None = None
    """0~1，越高越有把握；模型不給時是 None。"""

    present: float | None = None
    """h2o 才有：shim 另外讀的「目標在不在圖裡」機率，不是框的分數。"""

    found: bool | None = None
    """h2o 才有：有沒有生出框；沒有時框是整張圖、confidence 0。"""


class JevK5BoxResponse(JevK5Response):
    """JevK5Response，answers 多收 box 答案（jevk5 服務不會回，留著讓兩版程式一樣）。"""

    answers: dict[
        str,
        Annotated[
            NoulAnswerWithConfidence | ChoiceAnswer | ScoreAnswer | BoxAnswer,
            Field(discriminator="type"),
        ],
    ]

    @property
    def boxes(self) -> dict[str, BoxAnswer]:
        """box 答案，key 是題目名稱（同官方 SDK 的 result.boxes）。"""
        return {n: a for n, a in self.answers.items() if isinstance(a, BoxAnswer)}


class JevK5BoxClassifier(JevK5Classifier):
    """JevK5Classifier 多送 assets、多收 box 答案；網址、confidence 重算、trace 都不變。"""

    def _payload(self, request):
        payload = super()._payload(request)
        if "assets" in request:
            payload["assets"] = request["assets"]
        return payload

    def _parse(self, response, payload):
        # langchain-typesafe 的 ClassifierResponse 不認得 box 答案，整份會驗證失敗：
        # 先把 box 答案拿出來，其他的照 JevK5Classifier._parse 處理，最後放回原來的位置
        if not response.is_success:
            return super()._parse(response, payload)  # 錯誤照 parse_response raise
        data = response.json()
        answers = data.get("answers", {})
        boxes = {n: a for n, a in answers.items() if a.get("type") == "box"}
        rest = httpx2.Response(
            response.status_code,
            json={
                **data,
                "answers": {n: a for n, a in answers.items() if n not in boxes},
            },
            request=response.request,
        )
        parsed = super()._parse(rest, payload).model_dump()
        parsed["answers"] = {
            n: boxes[n] if n in boxes else parsed["answers"][n] for n in answers
        }
        return JevK5BoxResponse.model_validate(parsed)


def image_asset(path: str) -> tuple[str, dict]:
    """Return a new reference and the asset entry for one image file."""
    # 官網 https://docs.typesafe.ai/images 的「Base64 in JSON」，逐字照抄
    fmt = "png" if path.lower().endswith(".png") else "jpg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return str(uuid.uuid4()), {"type": "image", "format": fmt, "data_base64": data}


# ---- Setup ----

PAD = 0.05  # Fraction of the image to add on each side. Tune for your images.
MIN_CONFIDENCE = 0.6  # Example threshold. Tune it on your own images.
YES = 0.5  # Noul 判斷「是」的門檻（官網沒給，見檔頭）

# 官網每個帶圖或 Box 題的 request 都要帶這個 header；jevk5 服務不看，照送
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
ENDPOINT = os.environ.get("JEVK5_ENDPOINT", JEVK5_BASE_URL)
SERVICE_HINT = (
    "JevK5 是純文字模型，jevk5 服務只收 noul / choice / score，box 題一定回 400（見檔頭）；"
    "要真的跑 Box 請用 box_h2o.py"
)
# JevK5BoxClassifier 是 LangChain Runnable，重試用 with_retry（第一次 + 重試 5 次，同 citation_check_jevk5.py）
client = JevK5BoxClassifier(
    base_url=ENDPOINT,
    timeout=120.0,
    client=httpx2.Client(timeout=120.0, headers=IMAGES_HEADER),
    async_client=httpx2.AsyncClient(timeout=120.0, headers=IMAGES_HEADER),
).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=6,
    exponential_jitter_params={"initial": 1.0, "max": 20.0},
)

EDGES = ("left", "top", "right", "bottom")


def ask(request: dict) -> dict:
    """送一次 /v1/systemone，回傳和官網一樣形狀的 JSON（dict），官網的 response["answers"]["mug"] 才能照抄。"""
    return client.invoke(request).model_dump(mode="json")


def without_none(value):
    """印出來用：拿掉值是 None 的欄位（request_id、沒帶門檻的 threshold / decision 等）。"""
    if isinstance(value, dict):
        return {k: without_none(v) for k, v in value.items() if v is not None}
    return value


def short_base64(value):
    """印 request 用：data_base64 只留開頭，後面標字數。"""
    if isinstance(value, dict):
        return {
            k: f"{v[:16]}...({len(v):,} chars)"
            if k == "data_base64"
            else short_base64(v)
            for k, v in value.items()
        }
    return value


def wire(request: dict) -> dict:
    """request 的 JSON 形狀（題目從 pydantic 物件換成 dict），印出來或存檔用。"""
    return {
        **request,
        "questions": {
            name: q.model_dump(mode="json", exclude_none=True)
            for name, q in request["questions"].items()
        },
    }


def pixels(image: Image.Image, edges: dict) -> tuple[int, int, int, int]:
    """四邊（0~1）→ Pillow crop 用的像素座標，同官網 Reading a Box 的算法。"""
    width, height = image.size
    return (
        round(edges["left"] * width),
        round(edges["top"] * height),
        round(edges["right"] * width),
        round(edges["bottom"] * height),
    )


def ready(image: Image.Image) -> Image.Image:
    """官網 Images 頁 Preparing images：最長邊 <= 1568；短邊不到 280 就放大，放大後太長再補白邊。"""
    image = image.convert("RGB")
    image.thumbnail((1568, 1568))
    if min(image.size) < 280:
        scale = 280 / min(image.size)
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
        image.thumbnail((1568, 1568))
        if min(image.size) < 280:
            size = (max(280, image.width), max(280, image.height))
            image = ImageOps.pad(image, size, color="white")
    return image


def asset_of(image: Image.Image, name: str) -> tuple[str, dict]:
    """切出來的圖照 Preparing images 處理、存到 OUT_DIR，再用 image_asset 變成 asset。"""
    path = OUT_DIR / name
    ready(image).save(path, "JPEG", quality=90)
    return image_asset(str(path))


def to_full(edges: dict, region: dict) -> dict:
    """region（整張圖的比例）裡量的框 → 整張圖的比例。"""
    width = region["right"] - region["left"]
    height = region["bottom"] - region["top"]
    return {
        "left": region["left"] + edges["left"] * width,
        "top": region["top"] + edges["top"] * height,
        "right": region["left"] + edges["right"] * width,
        "bottom": region["top"] + edges["bottom"] * height,
    }


def iou(a: dict, b: dict) -> float:
    """兩個框的 IoU，拿來比不同做法框出來的位置。"""
    w = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    h = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))

    def area(e: dict) -> float:
        return (e["right"] - e["left"]) * (e["bottom"] - e["top"])

    union = area(a) + area(b) - w * h
    return w * h / union if union > 0 else 0.0


def show(edges: dict) -> str:
    return "  ".join(f"{e} {edges[e]:.3f}" for e in ("top", "bottom", "left", "right"))


def show_box(answer: dict) -> str:
    """一個 box 答案印成一行：四邊、confidence，h2o 有的話加 present / found。"""
    conf = answer.get("confidence")
    text = f"{show(answer['box'])}  confidence " + (
        "-" if conf is None else f"{conf:.2f}"
    )
    if answer.get("present") is not None:
        text += f"  (h2o present {answer['present']:.2f}, found {answer['found']})"
    return text


# ---- ## Request structure ----


def request_structure() -> dict:
    print("## Request structure\n")
    # 官網：multipart，"assets": {"6e1d0b7c-kitchen": {"type": "image", "part": "kitchen-photo"}}，
    # 外加 "model": "jev-preview"。這裡改成 base64 放在 JSON 裡，model 拿掉（見檔頭）
    _, kitchen = image_asset(str(DATA_DIR / "kitchen.jpg"))
    request = {
        "state": {"photo": "6e1d0b7c-kitchen"},
        "assets": {"6e1d0b7c-kitchen": kitchen},
        "questions": {"mug": Box(instructions="the coffee mug in `photo`")},
    }
    print(json.dumps(short_base64(wire(request)), indent=2))

    # 官網的 curl：同一個 request 存成 JSON，可以直接重送
    path = OUT_DIR / "request.json"
    path.write_text(json.dumps(wire(request), indent=1) + "\n")
    print(f"\nthe same request, written to {path}:")
    print(
        f"  curl -s {ENDPOINT.rstrip('/')}/v1/systemone -H 'Content-Type: application/json'"
        f" -H 'TypeSafe-Danger-Zone-Images: true' -d @{path}"
    )
    return request


# ---- ## Response structure ----


def response_structure(request: dict) -> dict:
    print("\n## Response structure\n")
    response = ask(request)
    print(json.dumps(without_none(response), indent=2, ensure_ascii=False))
    return response


# ---- ## Reading a Box ----


def reading_a_box(response: dict) -> dict:
    print("\n## Reading a Box\n")
    # 官網的程式，只改了檔案的位置
    image = Image.open(DATA_DIR / "kitchen.jpg")
    width, height = image.size
    mug = response["answers"]["mug"]
    edges = mug["box"]

    box = (
        round(edges["left"] * width),
        round(edges["top"] * height),
        round(edges["right"] * width),
        round(edges["bottom"] * height),
    )
    image.crop(box).save(OUT_DIR / "mug.jpg")
    print(
        f"kitchen.jpg is {width} x {height}; pixel box {box} -> {OUT_DIR / 'mug.jpg'}"
    )

    # ### Using confidence
    print("\n### Using confidence\n")
    # 官網的程式（PAD、MIN_CONFIDENCE 定義在上面 Setup）
    if mug["confidence"] < MIN_CONFIDENCE:
        edges = {
            "left": max(0.0, edges["left"] - PAD),
            "top": max(0.0, edges["top"] - PAD),
            "right": min(1.0, edges["right"] + PAD),
            "bottom": min(1.0, edges["bottom"] + PAD),
        }
        # Then crop with the padded `edges` as in the code above.
        image.crop(pixels(image, edges)).save(OUT_DIR / "mug-padded.jpg")
        print(
            f"confidence {mug['confidence']:.2f} < {MIN_CONFIDENCE}: padded by {PAD} on each side\n"
            f"  {show(edges)} -> {OUT_DIR / 'mug-padded.jpg'}"
        )
    else:
        print(
            f"confidence {mug['confidence']:.2f} >= {MIN_CONFIDENCE}: the box is used as is"
        )
    return mug["box"]


# ---- ## Writing Box questions ----


def target_can_be_a_picture() -> None:
    print("\n## Writing Box questions\n")
    print("### The target can be a picture\n")
    # 官網的 JSON 片段：參考圖的參照直接寫在 instructions 裡
    instructions = "the person in `photo` who matches this badge photo: 9a21c7d3-badge"
    print(json.dumps({"instructions": instructions}))
    # 不送：h2o 的 box.py 生框時只把 `photo` 那一張圖給模型（image_for() 挑一張、generate() 只帶一張），
    # instructions 裡的參考圖只會變成文字 "[image 2]"，模型看不到；jevk5 連圖都看不到。兩版都只印不送
    print(
        "not sent: h2o's Box reads only the image that `photo` names, so a reference picture"
        " in the instructions never reaches the model; JevK5 reads no images at all"
    )


def always_a_box() -> None:
    print("\n### A Box answer is always a box\n")
    # 官網的 JSON 片段：同一個 request 先用 Noul 問在不在，Noul 低就不理那個框
    questions = {
        "mug_visible": Noul(instructions="Is there a coffee mug in `photo`?"),
        "mug": Box(instructions="the coffee mug in `photo`"),
    }
    for name in ("kitchen.jpg", "kitchen-no-mug.jpg"):
        ref, photo = image_asset(str(DATA_DIR / name))
        response = ask(
            {"state": {"photo": ref}, "assets": {ref: photo}, "questions": questions}
        )
        visible = response["answers"]["mug_visible"]["noul"]
        mug = response["answers"]["mug"]
        action = "use the box" if visible >= YES else "ignore the box"
        print(f"{name:<20} mug_visible {visible:.2f} -> {action}")
        print(f"{'':<20} mug  {show_box(mug)}")


def one_box_per_question() -> None:
    print("\n### One box per question\n")
    # 兩個一樣的杯子：描述清楚要哪一個（官網："the mug nearest the sink in `photo`"），
    # 或旁邊加一題 none / one / several 的 Choice
    image = Image.open(DATA_DIR / "cabinet.jpg")
    count = Choice(
        instructions="How many mugs are in `photo`?",
        criteria={
            "none": "No mug.",
            "one": "Exactly one mug.",
            "several": "Two or more mugs.",
        },
    )
    ref, photo = image_asset(str(DATA_DIR / "cabinet.jpg"))
    response = ask(
        {
            "state": {"photo": ref},
            "assets": {ref: photo},
            "questions": {
                "mug_count": count,
                "mug": Box(instructions="the mug in `photo`"),
                "mug_right": Box(instructions="the mug on the right in `photo`"),
            },
        }
    )
    answers = response["answers"]
    print(
        f"cabinet.jpg  mug_count {answers['mug_count']['choice']}  {answers['mug_count']['probabilities']}"
    )
    print(f"  mug        {show_box(answers['mug'])}")
    print(f"  mug_right  {show_box(answers['mug_right'])}")
    if answers["mug_count"]["choice"] != "several":
        return

    # several：把圖切成左右兩半，每一半再問一次 Choice + Box，框換算回整張圖
    print("\nseveral: split into left and right halves and ask again on each")
    halves = {
        "left half": {"left": 0.0, "top": 0.0, "right": 0.5, "bottom": 1.0},
        "right half": {"left": 0.5, "top": 0.0, "right": 1.0, "bottom": 1.0},
    }
    found = []
    for label, region in halves.items():
        ref, half = asset_of(
            image.crop(pixels(image, region)), f"cabinet-{label.split()[0]}.jpg"
        )
        response = ask(
            {
                "state": {"photo": ref},
                "assets": {ref: half},
                "questions": {
                    "mug_count": count,
                    "mug": Box(instructions="the mug in `photo`"),
                },
            }
        )
        choice = response["answers"]["mug_count"]["choice"]
        mug = response["answers"]["mug"]
        print(f"  {label:<11} mug_count {choice:<8} {show_box(mug)}")
        if choice == "one":
            found.append(to_full(mug["box"], region))
        elif choice == "several":
            print(f"  {'':<11} still several: split this half again")
    for i, edges in enumerate(found, 1):
        print(f"  mug {i} in the full image  {show(edges)}")


def small_targets(direct: dict) -> None:
    print("\n### Crop large images before looking for small things\n")
    # 先用 Choice 在 tile 裡挑一塊（選項就是 tile 的圖），再在那塊上問 Box，四邊換算回整張圖。
    # 2 x 2 塊、每塊佔寬高的 0.6，相鄰的 tile 會重疊，中等大小的目標才會整個落在某一塊裡。
    # h2o 一個 request 最多 4 張圖，剛好 4 塊
    image = Image.open(DATA_DIR / "kitchen.jpg")
    tile = 0.6
    tiles = {
        "top_left": {"left": 0.0, "top": 0.0},
        "top_right": {"left": 1 - tile, "top": 0.0},
        "bottom_left": {"left": 0.0, "top": 1 - tile},
        "bottom_right": {"left": 1 - tile, "top": 1 - tile},
    }
    for region in tiles.values():
        region.update(right=region["left"] + tile, bottom=region["top"] + tile)
    refs, assets = {}, {}  # tile 名稱 -> 參照字串；參照字串 -> asset
    for name, region in tiles.items():
        ref, asset = asset_of(image.crop(pixels(image, region)), f"tile-{name}.jpg")
        refs[name], assets[ref] = ref, asset
    response = ask(
        {
            "state": "Four overlapping tiles cut from one kitchen photo; each option is one tile.",
            "assets": assets,
            "questions": {
                "tile": Choice(
                    instructions="Which tile shows the coffee mug?", criteria=refs
                )
            },
        }
    )
    tile_answer = response["answers"]["tile"]
    chosen = tile_answer["choice"]
    probs = "  ".join(f"{k} {v:.2f}" for k, v in tile_answer["probabilities"].items())
    print(f"tile: {chosen}  ({probs})")

    ref = refs[chosen]
    response = ask(
        {
            "state": {"tile": ref},
            "assets": {ref: assets[ref]},
            "questions": {"mug": Box(instructions="the coffee mug in `tile`")},
        }
    )
    mug = response["answers"]["mug"]
    full = to_full(mug["box"], tiles[chosen])
    print(f"  in the tile        {show_box(mug)}")
    print(f"  in the full image  {show(full)}")
    print(f"  IoU with the box asked on the full image: {iou(full, direct):.2f}")


def check_edges(image: Image.Image, edges: dict, name: str, rounds: int = 3) -> dict:
    """切到框裡，每一邊問一題 Noul；超出的那幾邊各往外推 PAD 再問，最多 rounds 輪。"""
    questions = {
        f"past_{edge}": Noul(
            instructions=f"Does the mug in `crop` continue past the {edge} edge of the image?"
        )
        for edge in EDGES
    }
    for round_ in range(1, rounds + 1):
        ref, crop = asset_of(image.crop(pixels(image, edges)), f"{name}-{round_}.jpg")
        response = ask(
            {"state": {"crop": ref}, "assets": {ref: crop}, "questions": questions}
        )
        past = {edge: response["answers"][f"past_{edge}"]["noul"] for edge in EDGES}
        print(f"  round {round_}  {show(edges)}")
        print("           " + "  ".join(f"past {e} {p:.2f}" for e, p in past.items()))
        # 已經貼在圖邊的那一邊推不出去，不算
        limit = {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
        out = [e for e in EDGES if past[e] >= YES and edges[e] != limit[e]]
        if not out:
            print("           no edge cuts the mug: done")
            return edges
        print(f"           move out: {', '.join(out)}")
        edges = {
            **edges,
            **{e: max(0.0, edges[e] - PAD) for e in out if e in ("left", "top")},
            **{e: min(1.0, edges[e] + PAD) for e in out if e in ("right", "bottom")},
        }
    print(f"           stopped after {rounds} rounds")
    return edges


def edge_questions(direct: dict) -> dict:
    print("\n### Check a box with edge questions\n")
    image = Image.open(DATA_DIR / "kitchen.jpg")
    print("the box from Response structure:")
    checked = check_edges(image, direct, "edge-check")
    # 官網沒有這段的程式，照它最後一段的測法：故意切掉目標的一塊，看 edge questions 抓不抓得到。
    # 這裡把右邊往內收 40%，杯子的把手在右邊
    cut = {**direct, "right": direct["left"] + 0.6 * (direct["right"] - direct["left"])}
    print("\nthe same box with its right 40% cut off (the handle is on the right):")
    check_edges(image, cut, "edge-check-cut")
    return checked


# ---- 官網沒有：把幾種框畫在原圖上，方便用眼睛看 ----


def draw(direct: dict, checked: dict) -> None:
    image = Image.open(DATA_DIR / "kitchen.jpg").convert("RGB")
    pen = ImageDraw.Draw(image)
    for edges, color in (
        (checked, "lime"),
        (direct, "red"),
    ):  # 紅框後畫，重疊的邊看得到紅色
        pen.rectangle(pixels(image, edges), outline=color, width=3)
    path = OUT_DIR / "kitchen-boxes.jpg"
    image.save(path, "JPEG", quality=90)
    print(
        f"\nred: the box from Response structure, green: after the edge questions -> {path}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    request = request_structure()
    response = response_structure(request)
    direct = reading_a_box(response)
    target_can_be_a_picture()
    always_a_box()
    one_box_per_question()
    small_targets(direct)
    checked = edge_questions(direct)
    draw(direct, checked)


if __name__ == "__main__":
    try:
        main()
    except TypeSafeAPIError as error:
        body = json.dumps(error.body, ensure_ascii=False)
        print(f"\n{error.endpoint} returned HTTP {error.status}: {body}")
        print(SERVICE_HINT)
        raise SystemExit(1) from None
    except TypeSafeAPIConnectionError:
        print(
            f"\ncannot reach {ENDPOINT}: is the service running? (see the top of this file)"
        )
        raise SystemExit(1) from None
