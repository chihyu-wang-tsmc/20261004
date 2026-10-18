"""產生 images_h2o.py 用的合成圖：門慢慢打開的三格畫面、拼圖。

官網 https://docs.typesafe.ai/images 的 State（frames 歷史）和 Criteria（Noul 用圖定義「組裝正確」）
兩段需要的圖網路上找不到 CC0 的現成照片，所以從 repo 裡已有的照片做出來：

  door-frame-{1,2,3}.jpg       door-empty.jpg 的門往內打開 0°、40°、75°（門鉸在左邊）
  door-still-{1,2,3}.jpg       同一扇門三格都關著，只有亮度差一點點（模擬相機曝光）
  puzzle-assembled.jpg         box/kitchen.jpg 切成 3 x 3 拼好（Noul criteria 的範例圖）
  puzzle-done.jpg              同一幅拼圖拼好，換底板顏色、轉 1.5 度（另一張照片）
  puzzle-broken.jpg            中間那塊不見、左上和右下對調

用法（在 repo 根目錄）：
    python images/make_synthetic.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
BOX = HERE.parent / "box"

# ---- 門 ----
# door-empty.jpg（960 x 1440）裡門洞的位置：左右 190 ~ 752，底 1325，拱頂中間 315、兩側 352
X0, X1, BOTTOM = 190, 752, 1325
ARCH_SIDE, ARCH_TOP = 352, 315


def arch_y(x: float) -> float:
    """拱形上緣在 x 的高度（橢圓弧）。"""
    half = (X1 - X0) / 2
    t = min(1.0, abs(x - (X0 + half)) / half)
    return ARCH_SIDE - (ARCH_SIDE - ARCH_TOP) * (1 - t * t) ** 0.5


def doorway() -> list[tuple[float, float]]:
    """門洞的多邊形：拱形上緣加長方形。"""
    top = [(x, arch_y(x)) for x in range(X0, X1 + 1, 4)]
    return top + [(X1, BOTTOM), (X0, BOTTOM)]


def perspective_coeffs(dst, src):
    """Image.transform(PERSPECTIVE) 的 8 個係數：輸出的四點 dst 對到輸入的四點 src。"""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    # 8 x 8 高斯消去，不想為了這個裝 numpy
    m = [r + [b] for r, b in zip(rows, rhs, strict=True)]
    for i in range(8):
        p = max(range(i, 8), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(8):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [a - f * b for a, b in zip(m[r], m[i], strict=True)]
    return [m[i][8] / m[i][i] for i in range(8)]


def open_door(base: Image.Image, degrees: float) -> Image.Image:
    """門鉸在左邊、往屋內轉 degrees 度：右邊那一側往後退，看起來變窄變矮，露出裡面的暗處。"""
    import math

    if degrees == 0:
        return base.copy()
    theta = math.radians(degrees)
    width, height = X1 - X0, BOTTOM - ARCH_TOP
    cx, cy = base.width / 2, 800  # 消失點：畫面中間、大約門把的高度
    scale = 1 / (1 + 0.45 * math.sin(theta))  # 越往後越小
    free_x = cx + (X0 + width * math.cos(theta) - cx) * scale

    # 門板（含拱形）做成 RGBA，貼到透視後的四邊形上
    panel = base.crop((X0, ARCH_TOP, X1, BOTTOM)).convert("RGBA")
    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).polygon(
        [(x - X0, y - ARCH_TOP) for x, y in doorway()], fill=255
    )
    panel.putalpha(mask)
    panel = ImageEnhance.Brightness(panel).enhance(1 - 0.3 * math.sin(theta))

    dst = [
        (X0, ARCH_TOP),
        (free_x, cy + (ARCH_TOP - cy) * scale),
        (free_x, cy + (BOTTOM - cy) * scale),
        (X0, BOTTOM),
    ]
    src = [(0, 0), (width, 0), (width, height), (0, height)]
    warped = panel.transform(
        base.size,
        Image.Transform.PERSPECTIVE,
        perspective_coeffs(dst, src),
        Image.Resampling.BICUBIC,
    )

    # 門洞裡面：上暗下亮一點（地板），再把門板疊上去
    out = base.convert("RGB")
    inside = Image.new("RGB", base.size)
    pen = ImageDraw.Draw(inside)
    for y in range(base.height):
        t = max(0.0, min(1.0, (y - ARCH_TOP) / (BOTTOM - ARCH_TOP)))
        shade = int(22 + 38 * t * t)
        pen.line([(0, y), (base.width, y)], fill=(shade + 6, shade + 2, shade))
    hole = Image.new("L", base.size, 0)
    ImageDraw.Draw(hole).polygon(doorway(), fill=255)
    out.paste(inside, (0, 0), hole.filter(ImageFilter.GaussianBlur(1)))
    out.paste(warped, (0, 0), warped)
    # 門板的厚度：自由端畫一條亮一點的邊
    edge = ImageDraw.Draw(out)
    edge.line([dst[1], dst[2]], fill=(150, 140, 128), width=max(2, round(7 * scale)))
    return out


def save(image: Image.Image, name: str, longest: int = 960) -> None:
    """照官網 Preparing images：最長邊縮到 longest、存 baseline JPEG（quality 90）。"""
    image = image.convert("RGB")
    image.thumbnail((longest, longest))
    image.save(HERE / name, "JPEG", quality=90)
    print(f"{name:<22} {image.width} x {image.height}")


def doors() -> None:
    base = Image.open(HERE / "door-empty.jpg").convert("RGB")
    for i, degrees in enumerate((0, 40, 75), 1):
        save(open_door(base, degrees), f"door-frame-{i}.jpg")
    for i, brightness in enumerate((0.97, 0.99, 1.02), 1):
        save(ImageEnhance.Brightness(base).enhance(brightness), f"door-still-{i}.jpg")


# ---- 拼圖 ----


def puzzle(
    tiles: dict[tuple[int, int], tuple[int, int] | None],
    board: tuple[int, int, int],
    rotate: float = 0.0,
) -> Image.Image:
    """box/kitchen.jpg 切成 3 x 3，tiles[(列, 行)] = 放哪一塊原圖（None 是缺一塊）。"""
    photo = Image.open(BOX / "kitchen.jpg").convert("RGB")
    tw, th = photo.width // 3, photo.height // 3
    gap, margin = 8, 36
    out = Image.new(
        "RGB", (3 * tw + 2 * gap + 2 * margin, 3 * th + 2 * gap + 2 * margin), board
    )
    pen = ImageDraw.Draw(out)
    for (r, c), piece in tiles.items():
        x, y = margin + c * (tw + gap), margin + r * (th + gap)
        if piece is None:  # 空格：底板上畫一個淡淡的框
            pen.rectangle(
                (x, y, x + tw, y + th), outline=tuple(v - 25 for v in board), width=2
            )
            continue
        pr, pc = piece
        shadow = (x + 4, y + 4, x + tw + 4, y + th + 4)
        pen.rectangle(shadow, fill=tuple(max(0, v - 60) for v in board))
        out.paste(photo.crop((pc * tw, pr * th, (pc + 1) * tw, (pr + 1) * th)), (x, y))
        pen.rectangle((x, y, x + tw, y + th), outline=(60, 50, 40), width=2)
    if rotate:
        out = out.rotate(rotate, Image.Resampling.BICUBIC, expand=True, fillcolor=board)
    return out


def puzzles() -> None:
    solved = {(r, c): (r, c) for r in range(3) for c in range(3)}
    save(puzzle(solved, (232, 226, 216)), "puzzle-assembled.jpg")
    save(puzzle(solved, (205, 214, 222), rotate=1.5), "puzzle-done.jpg")
    broken = {**solved, (1, 1): None, (0, 0): (2, 2), (2, 2): (0, 0)}
    save(puzzle(broken, (205, 214, 222), rotate=1.5), "puzzle-broken.jpg")


if __name__ == "__main__":
    doors()
    puzzles()
