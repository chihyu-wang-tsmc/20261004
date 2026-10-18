"""把 hierarchical_*.py 畫好的 <slug>_tree.svg 翻成中文，另存 <slug>_tree_zh.svg（英文版不動）。

CPC、Shopify、MeSH 的 jevk5 / h2o 共六支都在 main() 裡 render_svg 之後呼叫一次：

    write_zh_svg(hierarchy.slug, svg_path)

翻譯的來源：
  * 節點名稱：各資料集自己的 zh_titles.json（和 datasets/ 放在一起，jevk5 / h2o 共用）
        hierarchical_cpc/zh_titles.json
        hierarchical_shopify/zh_titles.json
        hierarchical_biomedical/zh_titles.json
    格式是 {"documents": {英文題目: 中文}, "titles": {英文節點名: 中文 或 {"short": 框內短名, "full": 完整中文}}}。
    只有 "full" 的話，框內放不下就截斷加「…」。滑鼠移到框上會顯示完整中文和英文原文。
  * 標題區的固定句子（Greedy vs parallel beam search、圖例…）在下面的 _header()，照格式翻，
    答案、K、top/second 這些數字照原圖帶過來。

字典只先翻了「標準答案那條路、官網 greedy 那條路，以及現有結果」上每一層的所有選項。換模型或換題目
走到字典裡沒有的節點時，那個節點照樣顯示英文，跑完會印出還沒翻的節點，補進 zh_titles.json 就好。

單獨跑（例如字典補完後重做中文版，不用重問模型）：
    python hierarchical_svg_zh.py cpc hierarchical_cpc_h2o/cpc_tree.svg
"""

import html
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE_FILES = {
    "cpc": HERE / "hierarchical_cpc" / "zh_titles.json",
    "shopify": HERE / "hierarchical_shopify" / "zh_titles.json",
    "mesh": HERE / "hierarchical_biomedical" / "zh_titles.json",
}
# Hierarchy.name（根節點、aria-label 用）和眉標上的大寫版本
NAMES = {
    "CPC patents": "CPC 專利",
    "Shopify products": "Shopify 商品",
    "MeSH biomedical subjects": "MeSH 生醫主題",
}
UPPER_NAMES = {en.upper(): zh for en, zh in NAMES.items()}

# render_svg 的版面：框寬 300、文字從 x+11 起；字型大小見 render_svg 的 <style>
NODE_TEXT_WIDTH = 300 - 11 - 8
CJK_FONTS = '"Noto Sans CJK TC","PingFang TC","Microsoft JhengHei"'


def _width(text: str, size: float) -> float:
    """估計文字寬度（px）：全形字算 1 個字高，其他算 0.6 個。"""
    return sum(
        size if unicodedata.east_asian_width(c) in "WF" else size * 0.6 for c in text
    )


def _fit(text: str, limit: float, size: float) -> str:
    if _width(text, size) <= limit:
        return text
    while text and _width(text + "…", size) > limit:
        text = text[:-1]
    return text + "…"


def _wrap(text: str, lines: int, limit: float, size: float) -> list[str]:
    """把中文題目切成剛好 lines 行（版面的高度是照英文行數排的，行數不能變多）。

    英文字（例如 infliximab）當成一整塊，不會被切到兩行。
    """
    out = [""]
    for piece in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*|.", text):
        if _width(out[-1] + piece, size) > limit and len(out) < lines:
            out.append("")
            piece = piece.lstrip()
        out[-1] += piece
    return out + [""] * (lines - len(out))


class _Titles:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.documents: dict[str, str] = data.get("documents", {})
        self.titles: dict = data.get("titles", {})
        self.missing: list[str] = []

    def full(self, label: str) -> str | None:
        value = self.titles.get(label)
        if value is None and label.endswith("…"):  # render_svg 截斷過的名字
            prefix = label[:-1]
            value = next(
                (v for k, v in self.titles.items() if k.startswith(prefix)), None
            )
        if value is None:
            if label not in self.missing:
                self.missing.append(label)
            return None
        return value["full"] if isinstance(value, dict) else value

    def short(self, label: str) -> str | None:
        value = self.titles.get(label)
        if isinstance(value, dict):
            return value.get("short", value["full"])
        return self.full(label)


def _header(cls: str, text: str, titles: _Titles) -> str:
    """標題區每一行的翻譯；照格式翻，數字和答案照原圖帶過來。"""
    if cls == "title" and text == "Greedy vs parallel beam search":
        return "貪婪搜尋 vs 平行集束搜尋（beam search）"
    if cls == "document-label" and text == "DOCUMENT":
        return "輸入文件"
    if cls == "eyebrow":
        m = re.fullmatch(r"(.+?) · (.+) · (.+?) · ([\d,]+) NODES", text)
        if m:
            model, name, version, nodes = m.groups()
            return f"{model} · {UPPER_NAMES.get(name, name)} · {version} · {nodes} 個節點"
    if cls in ("result greedy-result", "result beam-result"):
        m = re.fullmatch(r"(GREEDY TOP-1|BEAM K=(\d+)) → (.*)", text)
        if m:
            head = "貪婪搜尋 第一名" if m.group(2) is None else f"集束搜尋 K={m.group(2)}"
            leaf = m.group(3)
            leaf_zh = "沒有葉節點" if leaf == "no leaf" else titles.full(leaf) or leaf
            return f"{head} → {_fit(leaf_zh, 1400, 14)}"
    if cls == "copy":
        m = re.fullmatch(
            r"parallel sibling Choices → keep (\d+) by geometric mean p → top/second = (.+)",
            text,
        )
        if m:
            return (
                f"同一層的子類別平行問 Choice → 依幾何平均機率保留前 {m.group(1)} 條路徑"
                f" → 第一名／第二名 = {m.group(2)}"
            )
        if text.startswith("orange = greedy"):
            return (
                "橘色 = 貪婪搜尋   綠色 = 集束搜尋勝出的路徑   "
                "紫色 = 保留在集束中的路徑   虛線 = 被剪掉"
            )
    raise ValueError(f"不認得的標題文字（render_svg 改過了？）：{cls}: {text}")


def translate_svg(svg: str, titles: _Titles) -> str:
    """回傳翻好的 SVG 文字；節點查不到中文就留英文，記在 titles.missing。"""

    # 1. 輸入文件：幾行英文合起來查字典，中文照原本的行數重排
    doc_pattern = r'(<text class="document"[^>]*>)([^<]*)(</text>)'
    doc_lines = re.findall(doc_pattern, svg)
    document = " ".join(html.unescape(line) for _, line, _ in doc_lines)
    document_zh = titles.documents.get(document)
    if document_zh is not None:
        width = float(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
        zh_lines = iter(_wrap(document_zh, len(doc_lines), min(width - 48, 1100), 17))
        svg = re.sub(
            doc_pattern,
            lambda m: m.group(1) + html.escape(next(zh_lines)) + m.group(3),
            svg,
        )
    else:
        print(f"  [中文版] 題目沒有翻譯，留英文：{document}")

    # 2. 標題區其他行
    def header(m: re.Match) -> str:
        cls, attrs, text = m.group(1), m.group(2), html.unescape(m.group(3))
        zh = _header(cls, text, titles)
        return f'<text class="{cls}"{attrs}>{html.escape(zh)}</text>'

    svg = re.sub(
        r'<text class="(eyebrow|title|document-label|result greedy-result|'
        r'result beam-result|copy)"([^>]*)>([^<]*)</text>',
        header,
        svg,
    )

    # 3. 節點：框內放短名，tooltip 放完整中文和英文原文
    def node(m: re.Match) -> str:
        kind, title, middle, text_tag = m.group(1), html.unescape(m.group(2)), m.group(3), m.group(4)
        if kind == "root":
            short = full = NAMES.get(title)
        else:
            full = titles.full(title)
            short = titles.short(title) if full is not None else None
        if full is None:
            return m.group(0)
        tip = html.escape(f"{full}\n（原文：{title}）", quote=True)
        shown = html.escape(_fit(short, NODE_TEXT_WIDTH, 13))
        return f'<g class="node {kind}"><title>{tip}</title>{middle}{text_tag}{shown}</text></g>'

    svg = re.sub(
        r'<g class="node (\w+)"><title>([^<]*)</title>(<rect[^>]*/>)(<text[^>]*>)[^<]*</text></g>',
        node,
        svg,
    )

    # 4. aria-label、補上中文字型（有些系統的等寬字型沒有中文）
    svg = re.sub(
        r'aria-label="(.+?) taxonomy beam search"',
        lambda m: f'aria-label="{html.escape(NAMES.get(html.unescape(m.group(1)), m.group(1)))}分類的集束搜尋"',
        svg,
    )
    for old, new in [
        ("Menlo,Consolas,monospace", f'Menlo,Consolas,"Noto Sans Mono CJK TC",{CJK_FONTS},monospace'),
        ('system-ui,-apple-system,"Segoe UI",sans-serif', f'system-ui,-apple-system,"Segoe UI",{CJK_FONTS},sans-serif'),
    ]:
        if old not in svg:
            raise ValueError(f"找不到字型設定（render_svg 改過了？）：{old}")
        svg = svg.replace(old, new)
    return svg


def write_zh_svg(slug: str, svg_path: Path) -> Path:
    """讀 svg_path，在旁邊寫 <stem>_zh.svg，印出還沒翻譯的節點。回傳中文版的路徑。"""
    svg_path = Path(svg_path)
    titles = _Titles(TITLE_FILES[slug])
    zh = translate_svg(svg_path.read_text(encoding="utf-8"), titles)
    zh_path = svg_path.with_name(f"{svg_path.stem}_zh{svg_path.suffix}")
    zh_path.write_text(zh, encoding="utf-8")
    print(f"中文版：{zh_path}")
    if titles.missing:
        print(
            f"  [中文版] {len(titles.missing)} 個節點還沒翻譯，先顯示英文；"
            f"要翻的話補進 {TITLE_FILES[slug].relative_to(HERE)}："
        )
        for label in titles.missing:
            print(f"    {label}")
    return zh_path


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in TITLE_FILES:
        sys.exit(f"用法：python hierarchical_svg_zh.py {{{'|'.join(TITLE_FILES)}}} <slug>_tree.svg")
    write_zh_svg(sys.argv[1], Path(sys.argv[2]))
