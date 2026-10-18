"""照 https://docs.typesafe.ai/images 的範例，
把 Jev（api.typesafe.ai，"model": "jev-latest"）換成本機 h2o-lightning（classifier_h2o.py）。

官網這一頁的範例依序是（這支照順序全部跑一次，每段印出官網的小標題）：
  ## Turning on images            帶圖的 request 都送 TypeSafe-Danger-Zone-Images: true
  ## Sending images in a request  找到的狗放 state、尋狗海報放 same_dog 的 instructions，問 same_dog / has_collar / coat
     ### Asset fields、### References    欄位和參照字串的規則，Errors 那段逐條送一次
  ## Uploading images             SDK：Image 物件直接放進 state（door-camera.jpg，person_present / carrying）
     ### Over HTTP                multipart（官網建議）和 Base64 in JSON 各送一次
  ## Image placement
     ### State                    三格門口相機的 frames，`frames[2].frame` 指其中一格
     ### Instructions             門禁卡照片寫在題目的句子裡
     ### Criteria                 手勢 Choice 的選項用圖定義；Score 的等級、Noul 的 true / false 也用圖
     ### Structured data          `lineup[2]`、`faces.happy`、一個 criterion 放好幾張範例
  ## Patterns                     表格六種：前三種上面跑過了，這裡跑 Look closer when needed、
                                  Search a large image（tile 當 Choice 的選項，一層層往下）、Find where it is（Box）
  ## Limits                       官網的限制寫成送出前的檢查，images/ 每張圖檢查一次；h2o 自己的限制從 /health 讀
  ## Preparing images             官網的 Pillow 程式，處理 EXIF 轉向、太大、太小、WebP / GIF / 動態 PNG / progressive JPEG
  ## Image tokens                 usage.input_tokens：有圖沒圖、同一張圖放 state 一次 vs 放三題的 instructions
  ## Errors                       官網錯誤表的每一種都送一次，印官網寫的狀態碼和 h2o 實際回的
State、Instructions、Criteria、Structured data、Patterns 官網只有 JSON 片段和文字，這裡照著寫成程式。

跟官網不同的地方：
  * client 是 H2OImageClassifier：box_h2o.py 的 H2OBoxClassifier（H2OClassifier 的子類別，送 assets、收 box
    答案）再加上 SDK 的 Image 物件；classifier_h2o.py、box_h2o.py 本身都沒改。官方 typesafe-sdk 0.7.5 的
    Image 產生 <image:32 位 hex> 的參照字串、用 multipart 上傳；這裡參照字串一樣，但改送 base64。
  * 圖一律用 base64 放進 JSON 的 assets（官網 Base64 in JSON 的 image_asset()，box_h2o.py 裡逐字照抄的那個）。
    shim 的 multipart 把檔案都當成 top-level images（[image 1]、[image 2]…），不認 assets 的 part，所以官網的
    multipart 會 400：Over HTTP 那段照送一次給你看，再送 h2o 自己的 multipart 寫法。
  * 官網的 "model": "jev-latest" 和 Authorization 拿掉：shim 都不看。TypeSafe-Danger-Zone-Images: true 照送
    （自建的 httpx2.Client 帶上），shim 也不看。重試用 with_retry（同 box_h2o.py）。
  * h2o 一個 request 最多 4 張圖（vLLM 的 --limit-mm-per-prompt '{"image": 4, ...}' 和 serve_config.json 的
    image.max_images），第 5 張回 422。官網手勢那題有 5 張（camera、thumb、palm1、palm2、fist）：照送一次看 422，
    再把 open_palm 的範例減成一張、送 4 張。要 5 張以上就兩邊一起調高（shim 那邊也可以用 SHIM_MAX_IMAGES）。
  * 圖怎麼到模型（模型資料夾 assets.py 的 bind()、shim 的 image_prompt()）：每個不同的參照只送一次，request 裡
    所有的圖都排在每一題的文字前面，文字裡的參照換成 [image N]。所以官網這幾句在 h2o 不成立：
      - 「Each use adds the image again」、「放 instructions 的圖每題送一次」：同一張圖用幾次都只算一張
        （usage.images），放 state 一次和放三題的 instructions，token 一樣（見 Image tokens）。
      - 「An image in instructions is sent only with that question」：同一個 request 的其他題也看得到那張圖，
        只是那幾題的文字沒提到它。
      - 「Answers contain the reference」：Score 的 legend 是 "[image N]"，不是參照字串。
  * 尺寸：官網 280 ~ 1568 px 以外整個 request 拒絕（400 / 413）；h2o 照收，太大的縮到 1,048,576 px 以內
    （image.max_pixels），太小的也收。所以 Limits 那段照官網的表在送出前自己檢查。
  * 狀態碼：h2o 跟官網不一樣的地方（超過張數是 422、超過 16 MiB 是 400 不是 413 等），Errors 那段逐條對照。
  * EXIF 轉向：官網說 API 不套用 EXIF 的 orientation；h2o 的 vLLM 讀圖時會套（vllm/multimodal/image.py 的
    ImageOps.exif_transpose），所以帶 EXIF 的手機照片在 h2o 上本來就是正的。Preparing images 那段三張對照。
  * 答案後面的 ✓ / ✗ 是跟人看圖給的標準答案比（官網沒有），跑完印總共對幾題。官網 Criteria 的拼圖 Noul
    （true 用範例圖定義）在 h2o 上會把拼好的判成 no：照片只要跟範例圖有一點不同就判錯。旁邊加一題
    「跟這張圖的排列一樣嗎」，這個問法判得對。
  * Noul 判斷一律看 >= 0.5（官網沒給門檻）。classifier_h2o.py 開頭說的 noul_floor 會把 Noul 推到 0.199 以下
    或 0.801 以上，所以這裡只看哪一邊贏，不拿 Noul 的機率去比別的門檻。
  * Box（Find where it is）預設是關的：box 題回 400 時印出怎麼開（同 box_h2o.py 檔頭），然後繼續往下跑。
    官網說目標的參考圖可以放 instructions，h2o 的 Box 讀不到（見 box_h2o.py 的 target_can_be_a_picture）。
  * 圖：官網沒附圖。images/ 放 CC0 的照片和 images/make_synthetic.py 做的合成圖（門慢慢打開、拼圖），作者、網址、
    怎麼處理的見 images/SOURCES.md。官網寫出來的參照字串照用（a3f1c2e8-found、77b0d4a1-poster、e48a6f30-door、
    9a21c7d3-badge、e48a6f30-cam、0b6c51aa-thumb…），官網沒寫的照 References 的建議每張用新的 uuid4。

用法：
    python images_h2o.py
    H2O_ENDPOINT=http://127.0.0.1:8741 python images_h2o.py   # 指定服務網址
服務照 classifier_h2o.py 開頭起（serve_config.json 要有 "image" 那一塊，預設有）；Find where it is 那段另外要
LIGHTNING_BOX_ENABLED=1（見 box_h2o.py 檔頭），沒開只會跳過那一段。
切出來的圖、Preparing images 的輸入和輸出、畫了框的圖放在 images_h2o/。
"""

import base64
import io
import json
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
from langchain_typesafe import Choice, Noul, NoulCriteria, Score
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

# 官網 Preparing images 的 Image（Pillow）改叫 PILImage：Image 這個名字留給 Uploading images 的 SDK Image 物件
from PIL import Image as PILImage
from PIL import ImageDraw, ImageOps

from box_h2o import (
    SERVICE_HINT,
    Box,
    H2OBoxClassifier,
    image_asset,
    pixels,
    ready,
    short_base64,
    show_box,
    wire,
    without_none,
)
from classifier_h2o import H2O_BASE_URL

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "images"  # CC0 照片和合成圖（images/SOURCES.md）
OUT_DIR = HERE / "images_h2o"  # 切出來的圖、Preparing images 的輸入和輸出、畫了框的圖

YES = 0.5  # Noul 判斷「是」的門檻（官網沒給，見檔頭）
# Look closer：小圖的 confidence 低於這個就送大圖。範例值，用自己的圖調
LOOK_CLOSER = 0.6

# 官網每個帶圖或 Box 題的 request 都要帶這個 header；shim 不看，照送
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
ENDPOINT = os.environ.get("H2O_ENDPOINT", H2O_BASE_URL)
SYSTEMONE = f"{ENDPOINT.rstrip('/')}/v1/systemone"


# ---- SDK 的 Image 物件（照官方 typesafe-sdk 0.7.5 的 typesafe_sdk.Image，改成送 base64）----

REFERENCE = re.compile(r"<image:[0-9a-f]{32}>")  # 同 SDK 的 REFERENCE_PATTERN
# 參照字串 -> Image。用一般的 dict（強參照）：f-string 裡當場建的 Image 送出時也找得到
# （SDK 是要 attach_assets 或 assets= 才認得句子裡的參照）。一支跑完就結束的程式，留著不會怎樣
_PLACED: dict[str, "Image"] = {}


@dataclass(frozen=True, eq=False)
class Image:
    """一張圖：檔案路徑或 bytes，同 SDK 的 Image(Path("door-camera.jpg"))。

    放在 state 裡任何能放文字的地方，或用 str(image) / f-string 寫進題目的句子裡；H2OImageClassifier
    送出前把它換成參照字串（<image:32 位 hex>），再把用到的圖補進 assets。
    """

    data: bytes | os.PathLike
    _reference: str = field(
        default_factory=lambda: f"<image:{uuid.uuid4().hex}>", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _PLACED[self._reference] = self

    def __str__(self) -> str:
        """這張圖的參照字串，寫進句子用（同 SDK 的 str(image)）。"""
        return self._reference

    def asset(self) -> dict:
        """assets 裡的一筆：照檔頭判斷 png / jpg，內容 base64（同官網的 image_asset()）。"""
        if isinstance(self.data, os.PathLike):
            raw = Path(self.data).read_bytes()
        else:
            raw = bytes(self.data)
        fmt = "png" if raw.startswith(b"\x89PNG") else "jpg"
        data = base64.b64encode(raw).decode("ascii")
        return {"type": "image", "format": fmt, "data_base64": data}


def _place(value):
    """state 裡的 Image 物件換成參照字串（langchain-typesafe 的 serialize_state 不認得 Image）。"""
    if isinstance(value, Image):
        return str(value)
    if isinstance(value, dict):
        return {k: _place(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_place(v) for v in value]
    return value


class H2OImageClassifier(H2OBoxClassifier):
    """H2OBoxClassifier（多送 assets、多收 box 答案）再收 SDK 的 Image 物件；其他都不變。"""

    def _payload(self, request):
        payload = super()._payload({**request, "state": _place(request["state"])})
        # 參照字串是 ASCII，在 JSON 裡和在文字裡長得一樣：整份掃一次，用到哪張補哪張（同 SDK 的 prepare_system_one）
        used = dict.fromkeys(REFERENCE.findall(json.dumps(payload)))
        placed = {ref: _PLACED[ref].asset() for ref in used if ref in _PLACED}
        if placed:
            payload["assets"] = {**payload.get("assets", {}), **placed}
        return payload


# ---- Setup ----

SERVICE_HINT_IMAGES = (
    '這個 shim 只讀文字：serve_config.json 要有 "image" 那一塊，vLLM 要用 --limit-mm-per-prompt 起'
    "（見 classifier_h2o.py 開頭）"
)


def make_classifier(headers: dict[str, str]) -> H2OImageClassifier:
    return H2OImageClassifier(
        base_url=ENDPOINT,
        timeout=120.0,
        client=httpx2.Client(timeout=120.0, headers=headers),
        async_client=httpx2.AsyncClient(timeout=120.0, headers=headers),
    )


classifier = make_classifier(IMAGES_HEADER)
# H2OImageClassifier 是 LangChain Runnable，重試用 with_retry（第一次 + 重試 5 次，同 box_h2o.py）
client = classifier.with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=6,
    exponential_jitter_params={"initial": 1.0, "max": 20.0},
)
# 不經過 classifier 的 POST：multipart、Errors 那段要看原始的狀態碼
RAW = httpx2.Client(timeout=120.0)


def ask(request: dict) -> dict:
    """送一次 /v1/systemone，回傳和官網一樣形狀的 JSON（dict），官網的 response["answers"] 才能照抄。"""
    return client.invoke(request).model_dump(mode="json")


def detail(body) -> str:
    """錯誤回應的 detail，印出來用（太長就截掉）。"""
    text = body.get("detail", body) if isinstance(body, dict) else body
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return text if len(text) <= 120 else text[:117] + "..."


def try_ask(request: dict) -> dict | None:
    """同 ask，失敗時印出狀態碼和原因、回傳 None（官網的 request 在 h2o 上會失敗的那幾個用）。"""
    try:
        return ask(request)
    except TypeSafeAPIError as error:
        print(f"  -> HTTP {error.status}: {detail(error.body)}")
        return None


def post(payload=None, *, headers=IMAGES_HEADER, files=None) -> tuple[int, dict]:
    """直接 POST /v1/systemone，回傳狀態碼和 JSON。"""
    response = RAW.post(SYSTEMONE, json=payload, files=files, headers=headers)
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    return response.status_code, body


def asset(name: str) -> dict:
    """images/ 裡一張圖的 asset（參照字串另外給：官網有寫的照用）。"""
    return image_asset(str(DATA_DIR / name))[1]


def asset_of(image: PILImage.Image, name: str) -> tuple[str, dict]:
    """切出來的圖照 Preparing images 處理（box_h2o.ready）、存到 OUT_DIR，再用 image_asset 變成 asset。"""
    path = OUT_DIR / name
    ready(image).save(path, "JPEG", quality=90)
    return image_asset(str(path))


def brief(answer: dict) -> str:
    """一個答案印成一行。"""
    kind = answer["type"]
    if kind == "noul":
        return f"noul {answer['noul']:.2f}"
    if kind == "box":
        return show_box(answer)
    probs = "  ".join(f"{k} {v:.2f}" for k, v in answer["probabilities"].items())
    if kind == "choice":
        return f"{answer['choice']}  (confidence {answer['confidence']:.2f};  {probs})"
    return f"score {answer['score']:.2f}  (confidence {answer['confidence']:.2f};  {probs})"


CHECKED = {"ok": 0, "wrong": 0}  # 有標準答案的題目，main() 最後印總數


def mark(answer: dict, expected) -> str:
    """跟標準答案比：Noul 看 >= YES，Choice 看選項。expected 是人看圖給的答案。"""
    got = answer["noul"] >= YES if answer["type"] == "noul" else answer.get("choice")
    ok = got == expected
    CHECKED["ok" if ok else "wrong"] += 1
    if ok:
        return "✓"
    return f"✗ expected {({True: 'yes', False: 'no'}).get(expected, expected)}"


def show_answers(
    response: dict, indent: str = "  ", expected: dict | None = None
) -> None:
    """每題印一行；expected（題目名稱 -> 標準答案）有給的題目後面加 ✓ / ✗。"""
    for name, answer in response["answers"].items():
        line = f"{indent}{name:<16} {brief(answer)}"
        if expected and name in expected:
            line += f"  {mark(answer, expected[name])}"
        print(line)


def show_marks(response: dict, expected: dict, indent: str = "  ") -> None:
    """官網原樣印出答案之後，另外每題印一行 ✓ / ✗。"""
    for name, want in expected.items():
        print(f"{indent}{name:<16} {mark(response['answers'][name], want)}")


def h2o_images() -> dict:
    """/health 的 images 那一塊（max_images、max_pixels…）；shim 只讀文字時是空的。"""
    return RAW.get(f"{ENDPOINT.rstrip('/')}/health").json().get("images") or {}


# ---- ## Turning on images ----


def turning_on_images() -> int:
    print("## Turning on images\n")
    print(
        f"every request below that has an image or a Box question sends {IMAGES_HEADER}"
    )
    print(
        "(the h2o shim does not read this header: see Errors for a request without it)"
    )
    images = h2o_images()
    if not images:
        raise SystemExit(f"\n{ENDPOINT} reads text only. {SERVICE_HINT_IMAGES}")
    print(
        f"h2o {ENDPOINT}: up to {images['max_images']} images per request, "
        f"each scaled to at most {images['max_pixels'] or 'vLLM default'} pixels"
    )
    return images["max_images"]


# ---- ## Sending images in a request ----


def sending_images() -> None:
    print("\n## Sending images in a request\n")
    # 官網的 request，"model": "jev-latest" 拿掉（見檔頭）。題目用 langchain-typesafe 的 Noul / Choice，送出去的 JSON 一樣
    questions = {
        "same_dog": Noul(
            instructions="Is the dog in `photo` the same dog as in this picture: 77b0d4a1-poster?"
        ),
        "has_collar": Noul(instructions="Is the dog in `photo` wearing a collar?"),
        "coat": Choice(
            instructions="What is the coat of the dog in `photo` like?",
            criteria={
                "short": "Short and smooth.",
                "long": "Long or feathered.",
                "curly": "Curly or wiry.",
            },
        ),
    }
    request = {
        "state": {"photo": "a3f1c2e8-found"},
        "assets": {
            "a3f1c2e8-found": asset("found-dog.jpg"),
            "77b0d4a1-poster": asset("poster-dog.jpg"),
        },
        "questions": questions,
    }
    print(json.dumps(short_base64(wire(request)), indent=2))
    response = ask(request)
    print("\nanswers (found-dog.jpg in `photo`, the same husky as the poster):")
    print(json.dumps(without_none(response["answers"]), indent=2))
    print(f"usage.images {response['usage'].get('images')}")
    show_marks(response, {"same_dog": True, "has_collar": True})

    # 同一張海報，`photo` 換成另一隻狗：same_dog 和 coat 應該跟著變
    response = ask(
        {
            **request,
            "assets": {**request["assets"], "a3f1c2e8-found": asset("other-dog.jpg")},
        }
    )
    print("\nthe same poster, other-dog.jpg (a dachshund) in `photo`:")
    show_answers(
        response, expected={"same_dog": False, "has_collar": True, "coat": "short"}
    )

    print("\n### Asset fields\n### References\n")
    print("the rules in these two sections are sent one by one under Errors below")


# ---- ## Uploading images ----

# 三張門口照片的標準答案（person_present、carrying）
DOORS = {
    "door-camera.jpg": {"person_present": True, "carrying": "package"},
    "door-bags.jpg": {"person_present": True, "carrying": "bag"},
    "door-empty.jpg": {"person_present": False, "carrying": "nothing"},
}


def uploading_images() -> None:
    print("\n## Uploading images\n")
    # 官網的 SDK 程式：TypeSafeClient().system_one(...) 換成 client.invoke(...)，extra_headers 已經在 client 裡
    questions = {
        "person_present": Noul(instructions="Is there a person in `door`?"),
        "carrying": Choice(
            instructions="What is the person in `door` carrying?",
            criteria={
                "package": "A box or parcel.",
                "bag": "A bag of groceries or shopping.",
                "nothing": "Empty hands.",
            },
        ),
    }
    result = client.invoke(
        {"state": {"door": Image(DATA_DIR / "door-camera.jpg")}, "questions": questions}
    )
    print(result.answers)
    show_marks(result.model_dump(mode="json"), DOORS["door-camera.jpg"])

    # 官網沒有：另外兩張門口的照片（提著購物袋、門口沒人）
    for name in ("door-bags.jpg", "door-empty.jpg"):
        result = client.invoke(
            {"state": {"door": Image(DATA_DIR / name)}, "questions": questions}
        )
        print(f"\n{name}")
        show_answers(result.model_dump(mode="json"), expected=DOORS[name])

    # 官網：To put an image inside a sentence ... see the SDK guides。str(image) 就是參照字串，可以寫進 f-string
    door, badge = Image(DATA_DIR / "train-door.jpg"), Image(DATA_DIR / "badge.jpg")
    request = {
        "state": {"door": door},
        "questions": {
            "is_badge_holder": Noul(
                instructions=f"Is the person in `door` the same person as in this badge photo: {badge}?"
            )
        },
    }
    payload = classifier._payload(request)
    print("\nan image inside a sentence (f-string with str(image)):")
    print(
        json.dumps(
            short_base64(
                {"state": payload["state"], "questions": payload["questions"]}
            ),
            indent=2,
        )
    )
    print(f"assets: {list(payload['assets'])}")
    show_answers(ask(request), expected={"is_badge_holder": True})
    print("(the JavaScript example is the same request; this file is Python only)")


def over_http() -> None:
    print("\n### Over HTTP\n")
    print("#### Multipart (recommended)\n")
    raw = (DATA_DIR / "door-camera.jpg").read_bytes()
    # 官網的 curl：JSON 放第一個 part（request），圖放 door-photo，assets 用 part 指它
    request = {
        "state": {"door": "a3f1c2e8-door"},
        "assets": {"a3f1c2e8-door": {"type": "image", "part": "door-photo"}},
        "questions": {
            "person_present": {
                "type": "noul",
                "instructions": "Is there a person in `door`?",
            }
        },
    }
    files = [
        ("request", (None, json.dumps(request), "application/json")),
        ("door-photo", ("door-camera.jpg", raw, "image/jpeg")),
    ]
    status, body = post(files=files)
    print(f"the docs' multipart request -> HTTP {status}: {detail(body)}")
    # shim 的 multipart：每個檔案 part 照順序變成 top-level images 的 [image 1]、[image 2]…，不收 assets，
    # 所以 state 直接寫 "[image 1]"（模型資料夾 README 的 Images 那段）
    request = {"state": {"door": "[image 1]"}, "questions": request["questions"]}
    files[0] = ("request", (None, json.dumps(request), "application/json"))
    response = classifier.client.post(SYSTEMONE, files=files)
    parsed = classifier._parse(response, request).model_dump(mode="json")
    print('h2o\'s own multipart (no assets, "door": "[image 1]"):')
    show_answers(parsed, expected={"person_present": True})

    print("\n#### Base64 in JSON\n")
    # 官網的程式：image_asset() 逐字照抄（box_h2o.py），requests.post 換成 client
    photo_key, photo = image_asset(str(DATA_DIR / "door-camera.jpg"))
    response = ask(
        {
            "state": {"door": photo_key},
            "assets": {photo_key: photo},
            "questions": {
                "person_present": Noul(instructions="Is there a person in `door`?"),
                "carrying": Choice(
                    instructions="What is the person in `door` carrying?",
                    criteria={
                        "package": "A box or parcel.",
                        "bag": "A bag of groceries or shopping.",
                        "nothing": "Empty hands.",
                    },
                ),
            },
        }
    )
    print(response["answers"])
    show_marks(response, DOORS["door-camera.jpg"])


# ---- ## Image placement ----


def placement_state() -> None:
    print("\n## Image placement\n")
    print("### State\n")
    # 官網的 frames：最舊的在前面，參照字串照官網。第二題用路徑指最後一格
    refs = ["5c0e9a12-f1", "1d7be0c4-f2", "e48a6f30-f3"]
    state = {
        "frames": [
            {"seconds_ago": 2, "frame": refs[0]},
            {"seconds_ago": 1, "frame": refs[1]},
            {"seconds_ago": 0, "frame": refs[2]},
        ]
    }
    print(json.dumps({"state": state}, indent=2))
    questions = {
        "door_opened": Noul(
            instructions="Did the door in `frames` open during these frames?"
        ),
        "open_now": Noul(instructions="Is the door in `frames[2].frame` open?"),
    }
    for label, names, opened in (
        (
            "the door opens",
            ["door-frame-1.jpg", "door-frame-2.jpg", "door-frame-3.jpg"],
            True,
        ),
        (
            "the door stays shut",
            ["door-still-1.jpg", "door-still-2.jpg", "door-still-3.jpg"],
            False,
        ),
    ):
        assets = {ref: asset(name) for ref, name in zip(refs, names, strict=True)}
        response = ask({"state": state, "assets": assets, "questions": questions})
        print(f"\n{label} ({', '.join(names)})")
        show_answers(response, expected={"door_opened": opened, "open_now": opened})


def placement_instructions() -> None:
    print("\n### Instructions\n")
    # 官網的 request：門口的照片在 state，門禁卡照片的參照直接寫在題目的句子裡
    question = Noul(
        instructions="Is the person in `door` the same person as in this badge photo: 9a21c7d3-badge?"
    )
    print(json.dumps({"instructions": question.instructions}))
    for badge, label, same in (
        ("badge.jpg", "the same man, from another photo", True),
        ("badge-other.jpg", "a different man", False),
    ):
        response = ask(
            {
                "state": {"door": "e48a6f30-door"},
                "assets": {
                    "e48a6f30-door": asset("train-door.jpg"),
                    "9a21c7d3-badge": asset(badge),
                },
                "questions": {"is_badge_holder": question},
            }
        )
        answer = response["answers"]["is_badge_holder"]
        print(
            f"  train-door.jpg + {badge:<16} ({label}): {brief(answer)}  {mark(answer, same)}"
        )
    print(
        "several questions about one image: put it in the state once (token counts under Image tokens)"
    )


def gesture(examples: list[str]) -> Choice:
    """官網 Criteria 的手勢 Choice：選項用範例圖定義。examples 是 open_palm 的範例圖。"""
    return Choice(
        instructions="Which gesture is the hand in `camera` making?",
        criteria={
            "thumbs_up": "0b6c51aa-thumb",
            "open_palm": {
                "description": "A hand held up with the palm facing the camera.",
                "examples": examples,
            },
            "fist": "a7e94d02-fist",
            "none": "No hand, or a hand that matches none of the examples.",
        },
    )


def placement_criteria(max_images: int) -> None:
    print("\n### Criteria\n")
    examples = {
        "0b6c51aa-thumb": asset("thumb.jpg"),
        "4fd2e7b9-palm1": asset("palm1.jpg"),
        "c83a0e15-palm2": asset("palm2.jpg"),
        "a7e94d02-fist": asset("fist.jpg"),
    }
    cameras = {  # 照片 -> 標準答案
        "camera-thumbs-up.jpg": "thumbs_up",
        "camera-palms.jpg": "open_palm",
        "camera-flex.jpg": "fist",
        "camera-ok.jpg": "none",
    }
    question = gesture(["4fd2e7b9-palm1", "c83a0e15-palm2"])
    print(json.dumps(question.model_dump(mode="json"), indent=2))
    print(
        f"\nthe docs' request has {1 + len(examples)} images; this h2o reads {max_images}:"
    )
    response = try_ask(
        {
            "state": {"camera": "e48a6f30-cam"},
            "assets": {"e48a6f30-cam": asset(next(iter(cameras))), **examples},
            "questions": {"gesture": question},
        }
    )
    if response is None:
        # 4 張：open_palm 只留一張範例（要 5 張就調高 vLLM 的 --limit-mm-per-prompt 和 SHIM_MAX_IMAGES）
        print(
            '  sending it again with one open_palm example: "examples": ["4fd2e7b9-palm1"]'
        )
        question = gesture(["4fd2e7b9-palm1"])
        examples.pop("c83a0e15-palm2")
    else:
        first = next(iter(cameras))
        answer = response["answers"]["gesture"]
        print(f"  {first:<22} {brief(answer)}  {mark(answer, cameras.pop(first))}")
    for name, want in cameras.items():
        response = ask(
            {
                "state": {"camera": "e48a6f30-cam"},
                "assets": {"e48a6f30-cam": asset(name), **examples},
                "questions": {"gesture": question},
            }
        )
        answer = response["answers"]["gesture"]
        print(f"  {name:<22} {brief(answer)}  {mark(answer, want)}")

    # 官網：Score levels can each include an example picture（握拳、半張開、張開三級）
    print("\nScore levels as pictures: how open is the hand?")
    levels = {
        str(uuid.uuid4()): asset("fist.jpg"),
        str(uuid.uuid4()): asset("hand-half.jpg"),
        str(uuid.uuid4()): asset("palm1.jpg"),
    }
    openness = Score(
        instructions="How open is the hand in `camera`?", criteria=list(levels)
    )
    for name in ("camera-flex.jpg", "camera-reach.jpg", "camera-palms.jpg"):
        cam = str(uuid.uuid4())
        response = ask(
            {
                "state": {"camera": cam},
                "assets": {cam: asset(name), **levels},
                "questions": {"openness": openness},
            }
        )
        print(f"  {name:<22} {brief(response['answers']['openness'])}")
    # 官網：Answers contain the reference, not the image。h2o 的 legend 是 bind() 換好的 [image N]
    print(f"  legend {response['answers']['openness']['legend']}")

    # 官網：Noul criteria can define true and false with pictures
    print(
        "\nNoul true / false defined with a picture: is the puzzle correctly assembled?"
    )
    reference = str(uuid.uuid4())
    assembled = Noul(
        instructions="Is the puzzle in `photo` correctly assembled?",
        criteria=NoulCriteria(
            true={"description": "Correctly assembled", "examples": [reference]},
            false="Parts missing or in the wrong place.",
        ),
    )
    # 官網沒有：同一件事改問「跟這張拼好的圖排列一樣嗎」。h2o 上官網的問法只要照片跟範例圖有一點不同
    # （換底色、轉一點角度）就判 no，這個問法拼好、拼錯都判得對，兩種一起印出來對照
    same = Noul(
        instructions=f"Does the puzzle in `photo` show the same arrangement as this assembled puzzle: {reference}?"
    )
    for name, done in (("puzzle-done.jpg", True), ("puzzle-broken.jpg", False)):
        ref = str(uuid.uuid4())
        response = ask(
            {
                "state": {"photo": ref},
                "assets": {ref: asset(name), reference: asset("puzzle-assembled.jpg")},
                "questions": {"assembled": assembled, "same_as_picture": same},
            }
        )
        print(f"  {name}")
        show_answers(
            response,
            indent="    ",
            expected={"assembled": done, "same_as_picture": done},
        )


def structured_data() -> None:
    print("\n### Structured data\n")
    # 官網：A question can refer to one picture in an array as `lineup[2]`
    lineup = [str(uuid.uuid4()) for _ in range(3)]
    poster = str(uuid.uuid4())
    response = ask(
        {
            "state": {"lineup": lineup},
            "assets": {
                lineup[0]: asset("other-dog.jpg"),
                lineup[1]: asset("husky-path.jpg"),
                lineup[2]: asset("frenchie.jpg"),
                poster: asset("poster-dog.jpg"),
            },
            "questions": {
                "match": Choice(
                    instructions=f"Which dog in `lineup` is the same dog as in this poster photo: {poster}?",
                    criteria={
                        "first": "The dog in `lineup[0]`.",
                        "second": "The dog in `lineup[1]`.",
                        "third": "The dog in `lineup[2]`.",
                    },
                ),
                "dressed": Noul(
                    instructions="Is the dog in `lineup[2]` wearing clothes?"
                ),
            },
        }
    )
    print(
        "lineup: other-dog.jpg, husky-path.jpg (the poster's husky), frenchie.jpg (in a hoodie)"
    )
    show_answers(response, expected={"match": "second", "dressed": True})

    # 官網：... or to one in a labeled object as `faces.happy`
    happy, serious = str(uuid.uuid4()), str(uuid.uuid4())
    response = ask(
        {
            "state": {"faces": {"happy": happy, "serious": serious}},
            "assets": {happy: asset("laughing.jpg"), serious: asset("badge-other.jpg")},
            "questions": {
                "laughing": Noul(
                    instructions="Is the person in `faces.happy` laughing?"
                ),
                "tie": Noul(
                    instructions="Is the person in `faces.serious` wearing a tie?"
                ),
            },
        }
    )
    print("\nfaces: happy = laughing.jpg, serious = badge-other.jpg")
    show_answers(response, expected={"laughing": True, "tie": True})

    # 官網：A criterion can hold several example pictures as an array
    palm1, palm2 = str(uuid.uuid4()), str(uuid.uuid4())
    palm = Noul(
        instructions="Is the hand in `camera` held up like in the examples?",
        criteria=NoulCriteria(
            true={"description": "Palm facing the camera.", "examples": [palm1, palm2]},
            false="Any other hand position, or no hand.",
        ),
    )
    print("\na criterion with two example pictures (palm1.jpg, palm2.jpg):")
    for name, want in (("camera-palms.jpg", True), ("camera-thumbs-up.jpg", False)):
        cam = str(uuid.uuid4())
        response = ask(
            {
                "state": {"camera": cam},
                "assets": {
                    cam: asset(name),
                    palm1: asset("palm1.jpg"),
                    palm2: asset("palm2.jpg"),
                },
                "questions": {"palm": palm},
            }
        )
        answer = response["answers"]["palm"]
        print(f"  {name:<22} {brief(answer)}  {mark(answer, want)}")


# ---- ## Patterns ----

PATTERNS = (
    (
        "Compare against a known picture",
        "above: Sending images (same_dog), Instructions",
    ),
    ("Define options by example", "above: Criteria (gesture)"),
    ("What changed", "above: State (frames)"),
    ("Look closer when needed", "below"),
    ("Search a large image", "below"),
    ("Find where it is", "below (Box)"),
)
TARGET = "the doorbell button"  # 門右邊牆上那個小小的方形按鈕


def look_closer() -> None:
    print("\n### Look closer when needed\n")
    # 先問小圖（短邊 280，官網的下限），confidence 不夠才送大圖（door-empty.jpg 原檔 960 x 1440）
    image = PILImage.open(DATA_DIR / "door-empty.jpg")
    small = image.copy()
    small.thumbnail((10_000, round(280 * image.height / image.width)))
    house_number = Choice(
        instructions="What is the house number on the door in `photo`?",
        criteria={
            **{str(n): str(n) for n in range(1, 10)},
            "none": "No number is visible.",
        },
    )
    for label, (ref, photo) in (
        (
            f"small copy {small.width} x {small.height}",
            asset_of(small, "door-small.jpg"),
        ),
        (
            f"full image {image.width} x {image.height}",
            image_asset(str(DATA_DIR / "door-empty.jpg")),
        ),
    ):
        response = ask(
            {
                "state": {"photo": ref},
                "assets": {ref: photo},
                "questions": {"house_number": house_number},
            }
        )
        answer = response["answers"]["house_number"]
        print(
            f"  {label:<24} {answer['choice']}  confidence {answer['confidence']:.2f}"
            f"  {mark(answer, '2')}"
        )
        if answer["confidence"] >= LOOK_CLOSER:
            print(f"  confidence >= {LOOK_CLOSER}: done")
            return
        print(f"  confidence < {LOOK_CLOSER}: look closer")
    print(
        "  still unsure on the full image: crop the region and ask again, or ask a person"
    )


def tiles_of(region: dict, tile: float = 0.6) -> dict[str, dict]:
    """region 切成 2 x 2 塊，每塊佔寬高的 tile（相鄰的會重疊，目標才不會剛好被切開）。"""
    width = region["right"] - region["left"]
    height = region["bottom"] - region["top"]
    out = {}
    for name, (fx, fy) in {
        "top_left": (0.0, 0.0),
        "top_right": (1 - tile, 0.0),
        "bottom_left": (0.0, 1 - tile),
        "bottom_right": (1 - tile, 1 - tile),
    }.items():
        left, top = region["left"] + fx * width, region["top"] + fy * height
        out[name] = {
            "left": left,
            "top": top,
            "right": left + tile * width,
            "bottom": top + tile * height,
        }
    return out


def search_large_image(max_depth: int = 6) -> dict:
    print("\n### Search a large image\n")
    # 官網：Cut the image into tiles and ask a Choice with the tiles as its options.
    # Repeat on the chosen tile until the target takes up most of the tile.
    # 每一層 tile 是上一層的 0.6 倍，門鈴只有圖寬的 5% 左右，大概第 5 層才佔得滿 tile
    image = PILImage.open(DATA_DIR / "door-empty.jpg")
    region = {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
    for depth in range(1, max_depth + 1):
        tiles = tiles_of(region)
        refs, assets = {}, {}  # tile 名稱 -> 參照字串；參照字串 -> asset
        for name, tile in tiles.items():
            ref, tile_asset = asset_of(
                image.crop(pixels(image, tile)), f"search-{depth}-{name}.jpg"
            )
            refs[name], assets[ref] = ref, tile_asset
        response = ask(
            {
                "state": "Four overlapping tiles cut from one photo of a front door; each option is one tile.",
                "assets": assets,
                "questions": {
                    "tile": Choice(
                        instructions=f"Which tile shows {TARGET}?", criteria=refs
                    )
                },
            }
        )
        chosen = response["answers"]["tile"]["choice"]
        region = tiles[chosen]
        ref = refs[chosen]
        response = ask(
            {
                "state": {"tile": ref},
                "assets": {ref: assets[ref]},
                "questions": {
                    "fills": Noul(instructions=f"Does {TARGET} take up most of `tile`?")
                },
            }
        )
        fills = response["answers"]["fills"]["noul"]
        box = pixels(image, region)
        print(f"  depth {depth}: {chosen:<12} pixels {box}  fills the tile {fills:.2f}")
        if fills >= YES:
            print("  the target takes up most of the tile: done")
            break
    else:
        print(f"  stopped after {max_depth} levels")
    return region


def find_where_it_is() -> dict | None:
    print("\n### Find where it is\n")
    ref, photo = image_asset(str(DATA_DIR / "door-empty.jpg"))
    response = try_ask(
        {
            "state": {"photo": ref},
            "assets": {ref: photo},
            "questions": {"doorbell": Box(instructions=f"{TARGET} in `photo`")},
        }
    )
    if response is None:
        print(f"  {SERVICE_HINT}")
        return None
    answer = response["answers"]["doorbell"]
    print(f"  doorbell  {show_box(answer)}")
    print(
        "  (a picture of the target in the instructions, as the docs allow, never reaches h2o's Box:"
        " see box_h2o.py)"
    )
    return answer["box"]


def patterns() -> None:
    print("\n## Patterns\n")
    for name, where in PATTERNS:
        print(f"  {name:<34} {where}")
    look_closer()
    region = search_large_image()
    box = find_where_it_is()
    # 官網沒有：把兩種做法找到的位置畫在原圖上（藍：tile，紅：Box）
    image = PILImage.open(DATA_DIR / "door-empty.jpg").convert("RGB")
    pen = ImageDraw.Draw(image)
    pen.rectangle(pixels(image, region), outline="blue", width=4)
    if box is not None:
        pen.rectangle(pixels(image, box), outline="red", width=4)
    path = OUT_DIR / "door-doorbell.jpg"
    image.save(path, "JPEG", quality=90)
    print(f"\nblue: the last tile, red: the Box answer -> {path}")


# ---- ## Limits ----

MIN_SIDE, MAX_SIDE = 280, 1568
MIN_PIXELS, MAX_PIXELS = 280 * 280, 1568 * 1568  # 78,400 ~ 2,458,624
MAX_FILE = 16 * 1024 * 1024  # 每張圖 16 MiB（base64 是 decode 之後的大小）
MAX_REQUEST = 32 * 1024 * 1024  # request body、所有的圖、展開後的 request 各 32 MiB
MAX_REF, MAX_REFS = 1024, 65536  # 每個參照 1,024 bytes、全部加起來 65,536 bytes


def check_image(raw: bytes) -> list[str]:
    """官網 Limits 表的單張圖限制，回傳不符合的地方（空的就是可以送）。兩端都算在範圍內。"""
    if not raw.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        return ["not PNG or JPEG"]
    problems = []
    if len(raw) > MAX_FILE:
        problems.append(f"{len(raw) / 2**20:.1f} MiB > 16 MiB")
    try:
        with PILImage.open(io.BytesIO(raw)) as image:
            frames = getattr(image, "n_frames", 1)
            image.load()  # 讀到最後，壞掉或不完整的檔案在這裡出錯
            width, height = image.size
    except Exception as error:  # noqa: BLE001 - Pillow 讀不了就是壞檔
        return [f"damaged or incomplete ({type(error).__name__})"]
    if frames > 1:
        problems.append("animated")
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        problems.append(f"{width} x {height}: each side must be 280 to 1568")
    if not MIN_PIXELS <= width * height <= MAX_PIXELS:
        problems.append(f"{width * height:,} pixels: must be 78,400 to 2,458,624")
    return problems


def _uses(value, pattern: re.Pattern, counts: dict[str, int]) -> None:
    """數每個參照在字串裡出現幾次：由左到右、不重疊、同一處最長的優先；object key 不算（同官網 References）。"""
    if isinstance(value, str):
        for match in pattern.findall(value):
            counts[match] += 1
    elif isinstance(value, dict):
        for v in value.values():
            _uses(v, pattern, counts)
    elif isinstance(value, list):
        for v in value:
            _uses(v, pattern, counts)


def check_request(payload: dict) -> list[str]:
    """官網 Limits 表和 References 的 request 層級限制（payload 是送出去的 JSON）。"""
    problems = []
    body = len(json.dumps(payload).encode())
    if body > MAX_REQUEST:
        problems.append(f"request body {body / 2**20:.1f} MiB > 32 MiB")
    assets = payload.get("assets", {})
    if not assets:
        return problems
    sizes = {
        ref: len(base64.b64decode(a.get("data_base64", "")))
        for ref, a in assets.items()
    }
    for ref, a in assets.items():
        problems += [
            f"{ref[:20]}: {p}"
            for p in check_image(base64.b64decode(a.get("data_base64", "")))
        ]
    if any(not ref.strip() for ref in assets):
        problems.append("an empty reference")
    if any(len(ref.encode()) > MAX_REF for ref in assets):
        problems.append("a reference over 1,024 bytes")
    if sum(len(ref.encode()) for ref in assets) > MAX_REFS:
        problems.append("references over 65,536 bytes together")
    pattern = re.compile(
        "|".join(re.escape(r) for r in sorted(assets, key=len, reverse=True) if r)
    )
    counts = dict.fromkeys(assets, 0)
    for key in ("state", "questions"):
        _uses(payload.get(key), pattern, counts)
    problems += [f"{ref[:20]}: not referenced" for ref, n in counts.items() if n == 0]
    # 處理過的圖可能比原檔大，這裡只能用原檔的大小估
    if sum(sizes.values()) > MAX_REQUEST:
        problems.append("all images over 32 MiB (estimated from the files)")
    text = len(json.dumps({k: v for k, v in payload.items() if k != "assets"}).encode())
    expanded = text + sum(sizes[ref] * n for ref, n in counts.items())
    if expanded > MAX_REQUEST:
        problems.append(
            f"expanded request {expanded / 2**20:.1f} MiB > 32 MiB (estimated)"
        )
    return problems


def limits(max_images: int) -> None:
    print("\n## Limits\n")
    print(
        "the docs' limits, checked before sending (h2o itself rejects none of the size limits):"
    )
    for path in sorted(DATA_DIR.glob("*.jpg")):
        raw = path.read_bytes()
        with PILImage.open(path) as image:
            size = f"{image.width} x {image.height}"
        problems = check_image(raw)
        print(
            f"  {path.name:<22} {size:>11}  {len(raw) / 1024:6.0f} KiB  {'; '.join(problems) or 'ok'}"
        )

    # request 層級：同一張圖用四次，展開後的大小算四次（官網：Each use adds the image again）
    ref, door = image_asset(str(DATA_DIR / "door-empty.jpg"))
    payload = {
        "state": {"a": ref, "b": ref, "c": ref, "d": ref},
        "assets": {ref: door},
        "questions": {
            "q": {"type": "noul", "instructions": "Is there a person in `a`?"}
        },
    }
    expanded = len(base64.b64decode(door["data_base64"])) * 4
    print(
        f"\none image used four times: expanded request about {expanded / 2**20:.2f} MiB of images,"
        f" problems: {check_request(payload) or 'none'}"
    )

    images = h2o_images()
    print(f"\nh2o's own limits ({ENDPOINT}/health and the model folder's README):")
    print(f"  images per request      at most {max_images} (one more is a 422)")
    print(
        f"  pixels per image        scaled down to {images.get('max_pixels') or 'the vLLM default'} pixels,"
        " never rejected; no minimum size"
    )
    print(
        "  file size               16 MiB per asset (assets.py, a 400), 20 MB per image otherwise"
    )
    print("  formats                 PNG and JPEG for assets")
    print("  the same image twice    sent once (each distinct reference is bound once)")


# ---- ## Preparing images ----


def prepare(path: Path, out: Path) -> None:
    """官網 Preparing images 的程式，只把檔名換成參數（Image 換成 PILImage，見 import）。"""
    with PILImage.open(path) as original:
        img = ImageOps.exif_transpose(original)  # rotate the pixels upright
        img.thumbnail((1568, 1568))  # longest side at most 1568, same aspect ratio
        # thumbnail only shrinks: upscale or pad an image under the minimum separately
        img.convert("RGB").save(out, "JPEG", quality=90)  # baseline JPEG


def describe(path: Path) -> str:
    with PILImage.open(path) as image:
        text = f"{image.format} {image.width} x {image.height}"
        if getattr(image, "n_frames", 1) > 1:
            text += f", {image.n_frames} frames"
        if image.info.get("progressive"):
            text += ", progressive"
        orientation = image.getexif().get(0x0112)
        if orientation and orientation != 1:
            text += f", EXIF orientation {orientation}"
    return text


def make_inputs(folder: Path) -> list[Path]:
    """Preparing images 要處理的幾種檔案，從 camera-thumbs-up.jpg 做出來。"""
    base = PILImage.open(DATA_DIR / "camera-thumbs-up.jpg").convert("RGB")
    # 手機照片：4032 x 3024（4:3），像素存成躺著的，EXIF 說看的時候順時針轉 90 度（Orientation 6）
    width = round(base.height * 4 / 3)
    left = (base.width - width) // 2
    upright = base.crop((left, 0, left + width, base.height)).resize((4032, 3024))
    exif = PILImage.Exif()
    exif[0x0112] = 6
    upright.rotate(90, expand=True).save(
        folder / "phone.jpg", "JPEG", quality=90, exif=exif
    )
    base.save(folder / "progressive.jpg", "JPEG", quality=90, progressive=True)
    base.resize((210, 140)).save(folder / "small.jpg", "JPEG", quality=90)
    base.save(folder / "webp-photo.webp", "WEBP", quality=90)
    base.save(folder / "gif-photo.gif", "GIF")
    second = ImageOps.mirror(base)
    base.save(
        folder / "animated.png",
        "PNG",
        save_all=True,
        append_images=[second],
        duration=300,
        loop=0,
    )
    return [
        folder / n
        for n in (
            "phone.jpg",
            "progressive.jpg",
            "small.jpg",
            "webp-photo.webp",
            "gif-photo.gif",
            "animated.png",
        )
    ]


def preparing_images() -> None:
    print("\n## Preparing images\n")
    folder = OUT_DIR / "prepare"
    folder.mkdir(exist_ok=True)
    for path in make_inputs(folder):
        out = folder / f"{path.stem}-ready.jpg"
        prepare(path, out)
        problems = check_image(out.read_bytes())
        print(
            f"  {path.name:<16} {describe(path):<44} -> {describe(out):<20} {'; '.join(problems) or 'ok'}"
        )
        if problems:
            # 官網：Scale up, or pad, an image smaller than the minimum（box_h2o.ready）
            with PILImage.open(out) as image:
                ready(image).save(out, "JPEG", quality=90)
            print(
                f"  {'':<16} {'scaled up (box_h2o.ready)':<44} -> {describe(out):<20} {'; '.join(check_image(out.read_bytes())) or 'ok'}"
            )

    # 官網：The API ... does not apply the EXIF orientation。h2o 不一樣：vLLM 讀圖時會套
    # ImageOps.exif_transpose（vllm/multimodal/image.py），所以帶 EXIF 的 phone.jpg 進模型時已經是正的。
    # 另外存一張同樣像素、拿掉 EXIF 的，才是 Jev 看到的樣子
    with PILImage.open(folder / "phone.jpg") as image:
        image.convert("RGB").save(folder / "phone-no-exif.jpg", "JPEG", quality=90)
    print("\nthe phone photo: is the person upright?")
    upright = Noul(
        instructions="Is the person in `photo` upright, with the head at the top of the picture?"
    )
    for name, want, note in (
        ("phone.jpg", True, "pixels sideways, EXIF says rotate: vLLM rotates it"),
        ("phone-no-exif.jpg", False, "the same pixels without EXIF: what Jev sees"),
        ("phone-ready.jpg", True, "after the docs' Pillow code"),
    ):
        ref, photo = image_asset(str(folder / name))
        response = ask(
            {
                "state": {"photo": ref},
                "assets": {ref: photo},
                "questions": {"upright": upright},
            }
        )
        answer = response["answers"]["upright"]
        print(f"  {name:<18} {brief(answer)}  {mark(answer, want)}  ({note})")
    print(
        "  (phone.jpg is 3024 x 4032: Jev rejects it with 413, h2o scales it down and reads it)"
    )


# ---- ## Image tokens ----


def image_tokens() -> None:
    print("\n## Image tokens\n")
    ref, door = image_asset(str(DATA_DIR / "door-camera.jpg"))
    person = {"person_present": Noul(instructions="Is there a person in `door`?")}

    def tokens(response: dict) -> str:
        usage = response["usage"]
        return f"input_tokens {usage['input_tokens']:>6}  usage.images {usage.get('images') or 0}"

    text = ask(
        {"state": {"door": "A photo from the front-door camera."}, "questions": person}
    )
    image = ask({"state": {"door": ref}, "assets": {ref: door}, "questions": person})
    print(f"  text only                         {tokens(text)}")
    print(f"  door-camera.jpg                   {tokens(image)}")
    with PILImage.open(DATA_DIR / "door-camera.jpg") as photo:
        small, size = photo.copy(), f"{photo.width} x {photo.height}"
    print(
        f"  -> about {image['usage']['input_tokens'] - text['usage']['input_tokens']} input tokens"
        f" for one {size} image"
    )

    # 官網：When several questions use the same image, put it in the state once and refer to it by path
    in_state = {
        "person": Noul(instructions="Is there a person in `door`?"),
        "box": Noul(instructions="Is someone in `door` holding a box?"),
        "daylight": Noul(instructions="Was `door` taken in daylight?"),
    }
    in_instructions = {
        "person": Noul(instructions=f"Is there a person in this photo: {ref}?"),
        "box": Noul(instructions=f"Is someone in this photo holding a box: {ref}?"),
        "daylight": Noul(instructions=f"Was this photo taken in daylight: {ref}?"),
    }
    once = ask({"state": {"door": ref}, "assets": {ref: door}, "questions": in_state})
    three = ask(
        {
            "state": "Three questions about one front-door camera photo.",
            "assets": {ref: door},
            "questions": in_instructions,
        }
    )
    print("\nthree questions about the same image:")
    print(f"  in the state once                 {tokens(once)}")
    print(f"  in each question's instructions   {tokens(three)}")
    print(
        "  (Jev reads the image three times in the second; h2o binds each reference once)"
    )

    # 官網：Each use adds the image again
    twice = ask(
        {
            "state": {"before": ref, "now": ref, "after": ref},
            "assets": {ref: door},
            "questions": {
                "person_present": Noul(instructions="Is there a person in `now`?")
            },
        }
    )
    print(f"  one reference used three times    {tokens(twice)}")

    # 小圖 token 少：短邊 280 的版本 vs 原圖
    small.thumbnail((10_000, 280))
    small_ref, small_asset = asset_of(small, "door-camera-small.jpg")
    smaller = ask(
        {
            "state": {"door": small_ref},
            "assets": {small_ref: small_asset},
            "questions": person,
        }
    )
    small_size = f"{small.width} x {small.height}"
    print(f"\n  door-camera.jpg at {small_size:<14} {tokens(smaller)}")
    print(f"  door-camera.jpg at {size:<14} {tokens(image)}")


# ---- ## Errors ----


def encode(image: PILImage.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def entry(raw: bytes, fmt: str = "jpg", **extra) -> dict:
    """assets 裡的一筆（format 可以故意給錯）。"""
    return {
        "type": "image",
        "format": fmt,
        "data_base64": base64.b64encode(raw).decode("ascii"),
        **extra,
    }


def errors() -> None:
    print("\n## Errors\n")
    photo = (DATA_DIR / "door-camera.jpg").read_bytes()
    base = PILImage.open(DATA_DIR / "door-camera.jpg").convert("RGB")
    ref = str(uuid.uuid4())
    person = {
        "person_present": {
            "type": "noul",
            "instructions": "Is there a person in `door`?",
        }
    }

    def one(asset_entry: dict, reference: str = ref) -> dict:
        return {
            "state": {"door": reference},
            "assets": {reference: asset_entry},
            "questions": person,
        }

    def multipart(request: dict, parts: list[tuple[str, bytes]]) -> tuple[int, dict]:
        files = [("request", (None, json.dumps(request), "application/json"))]
        files += [(name, (f"{name}.jpg", raw, "image/jpeg")) for name, raw in parts]
        return post(files=files)

    # 超過 16 MiB 的 PNG：雜訊、不壓縮（2400 x 2400 x 3 bytes 大約 16.5 MiB）
    noise = random.Random(0).randbytes(2400 * 2400 * 3)
    huge = encode(
        PILImage.frombytes("RGB", (2400, 2400), noise), "PNG", compress_level=0
    )
    animated = encode(
        base, "PNG", save_all=True, append_images=[ImageOps.mirror(base)], duration=300
    )
    five = {
        str(uuid.uuid4()): asset(n)
        for n in (
            "door-camera.jpg",
            "door-bags.jpg",
            "found-dog.jpg",
            "poster-dog.jpg",
            "thumb.jpg",
        )
    }
    long_ref = "door-" + "x" * 1100
    part_request = {
        "state": {"door": "a3f1c2e8-door"},
        "assets": {"a3f1c2e8-door": {"type": "image", "part": "door-photo"}},
        "questions": person,
    }

    # (官網錯誤表裡的原因, 官網寫的狀態碼, 送出去)
    cases = [
        (
            "no TypeSafe-Danger-Zone-Images header",
            "400",
            lambda: post(one(entry(photo)), headers={}),
        ),
        (
            "not PNG or JPEG (a GIF)",
            "400",
            lambda: post(one(entry(encode(base, "GIF"), "png"))),
        ),
        (
            "damaged or incomplete (half a JPEG)",
            "400",
            lambda: post(one(entry(photo[: len(photo) // 2]))),
        ),
        ("animated PNG", "400", lambda: post(one(entry(animated, "png")))),
        (
            "under the minimum (210 x 140)",
            "400",
            lambda: post(one(entry(encode(base.resize((210, 140)), "JPEG")))),
        ),
        (
            "an asset that is not referenced",
            "400",
            lambda: post(
                {
                    **one(entry(photo)),
                    "assets": {ref: entry(photo), str(uuid.uuid4()): entry(photo)},
                }
            ),
        ),
        (
            "invalid base64",
            "400",
            lambda: post(
                one(
                    {
                        "type": "image",
                        "format": "jpg",
                        "data_base64": "not base64 at all!",
                    }
                )
            ),
        ),
        ('invalid format ("gif")', "400", lambda: post(one(entry(photo, "gif")))),
        (
            'format "jpg", PNG bytes',
            "400",
            lambda: post(one(entry(encode(base, "PNG"), "jpg"))),
        ),
        ("an empty reference", "400", lambda: post(one(entry(photo), reference=""))),
        (
            "a reference only as an object key",
            "400",
            lambda: post(
                {
                    "state": "A front-door camera photo.",
                    "assets": {ref: entry(photo)},
                    "questions": {
                        "pick": {
                            "type": "choice",
                            "instructions": "Which option?",
                            "criteria": {ref: "this photo", "none": "no photo"},
                        }
                    },
                }
            ),
        ),
        (
            "multipart: a file part no asset names",
            "400",
            lambda: multipart(
                {"state": {"door": "a photo"}, "questions": person},
                [("door-photo", photo)],
            ),
        ),
        (
            "multipart: an asset whose part is missing",
            "400",
            lambda: multipart(part_request, []),
        ),
        (
            "multipart, as the docs send it",
            "200",
            lambda: multipart(part_request, [("door-photo", photo)]),
        ),
        (
            "an asset without type (optional)",
            "200",
            lambda: post(
                one({"format": "jpg", "data_base64": entry(photo)["data_base64"]})
            ),
        ),
        (
            "part and data_base64 in a JSON request",
            "400",
            lambda: post(one(entry(photo, part="door-photo"))),
        ),
        (
            "wider than 1568 (2000 x 1335)",
            "413",
            lambda: post(one(entry(encode(base.resize((2000, 1335)), "JPEG")))),
        ),
        ("an image file over 16 MiB", "413", lambda: post(one(entry(huge, "png")))),
        (
            "a reference over 1,024 bytes",
            "413",
            lambda: post(one(entry(photo), reference=long_ref)),
        ),
        (
            "5 images in one request",
            "200",
            lambda: post(
                {
                    "state": {f"cam{i}": r for i, r in enumerate(five)},
                    "assets": five,
                    "questions": {
                        "q": {
                            "type": "noul",
                            "instructions": "Is there a person in `cam0`?",
                        }
                    },
                }
            ),
        ),
    ]
    print(f"  {'cause':<44} {'docs':<5} h2o")
    for cause, documented, send in cases:
        status, body = send()
        note = "" if status == 200 else f"  {detail(body)}"
        print(f"  {cause:<44} {documented:<5} {status}{note}")
    print(
        "\n  not sent: max_tokens_exceeded (h2o reads at most 4 images, each under its pixel cap, so images alone"
        " stay under the context; a long state is cut, not refused) and a request body over 32 MiB (the shim"
        " has no body limit)"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_images = turning_on_images()
    sending_images()
    uploading_images()
    over_http()
    placement_state()
    placement_instructions()
    placement_criteria(max_images)
    structured_data()
    patterns()
    limits(max_images)
    preparing_images()
    image_tokens()
    errors()
    total = CHECKED["ok"] + CHECKED["wrong"]
    print(
        f"\n{CHECKED['ok']} / {total} answers match the expected answer"
        f" ({CHECKED['wrong']} marked ✗ above)"
    )


if __name__ == "__main__":
    try:
        main()
    except TypeSafeAPIError as error:
        body = json.dumps(error.body, ensure_ascii=False)
        print(f"\n{error.endpoint} returned HTTP {error.status}: {body}")
        raise SystemExit(1) from None
    except (TypeSafeAPIConnectionError, httpx2.ConnectError):
        print(
            f"\ncannot reach {ENDPOINT}: is the service running? (see the top of classifier_h2o.py)"
        )
        raise SystemExit(1) from None
