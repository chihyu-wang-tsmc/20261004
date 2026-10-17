"""hierarchical_shopify_h2o.py 的「帶路徑」版：每一題的 instructions 前面加上已經走過的路徑。

原版不管走到哪個節點，h2o 看到的都是同一段摘要和同一句 instructions，只有選項換成這一層的子節點，
不知道上一層選了什麼。Shopify 的選項只有名稱、沒有編號（例如 Pet Chairs），h2o 完全看不出上一層選了
什麼，這一版把路徑補進題目，看 h2o 會不會判斷得更準。

跟 hierarchical_shopify_h2o.py 只差這幾處：
  * child_question(labels, path)：path 不是空的時候，instructions 變成
        "This document has already been classified under: Animals & Pet Supplies > Pet Supplies > Cat Supplies.
         Which direct child category best matches this document?"
    根節點沒有路徑，instructions 和原版一字不差。
  * choose(state, path, labels)：多一個 path，beam_search / greedy_search 呼叫時把目前的路徑傳進去，
    快取 key 也因此帶著路徑。樹裡每個節點只有一條路徑，greedy 走到 beam 問過的節點照樣走快取。
  * state 還是只有摘要：TypeSafe 的 state 是「被判斷的內容」，路徑是題目的背景，所以放在 instructions。
  * 快取、SVG、results.md 放在 hierarchical_shopify_h2o_path/，和原版分開；SVG 左上角寫 H2O-LIGHTNING + PATH。
其餘（categories.txt、題目、標準答案 Cat Window Beds & Perches、greedy / K=3 beam search、表格）和原版完全相同，兩個版本的 results.md
可以直接對照。

要注意的風險：beam search 能救回早期的錯，靠的是走錯分支時 h2o 的機率會散掉、分數下降。題目先說
「已經分到某一類」，等於告訴 h2o 前提成立，它在錯的分支裡也可能很有把握，錯的路徑就不容易被剪掉。
所以比較時除了看答對與否，也看 top/second 和沒選上的路徑分數有沒有變高。另外每個資料集只有一題，
一題的對錯說明不了什麼，要下結論需要更多有標準答案的題目。

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    python hierarchical_shopify_h2o_path.py
"""

import functools
import html
import json
import shutil
import textwrap
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, TypeAlias

from langchain_typesafe import Choice
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from classifier_h2o import H2OClassifier

OUT_DIR = Path(__file__).resolve().with_name("hierarchical_shopify_h2o_path")
# categories.txt 和其他版本共用這份（已入版控）
DATA_DIR = Path(__file__).resolve().with_name("hierarchical_shopify") / "datasets"

Tree: TypeAlias = dict[str, "Tree"]


class Hierarchy(NamedTuple):
    """One query and a complete hierarchy.

    :param slug: filename-safe taxonomy name.
    :param name: display name.
    :param version: pinned dataset version.
    :param source_url: hierarchy source.
    :param node_count: number of loaded hierarchy nodes.
    :param document: unstructured text classified by TypeSafe.
    :param expected_leaf: expected final classification.
    :param tree: nested direct-child menus.
    """

    slug: str
    name: str
    version: str
    source_url: str
    node_count: int
    document: str
    expected_leaf: str
    tree: Tree


SHOPIFY_URL = (
    "https://raw.githubusercontent.com/Shopify/product-taxonomy/"
    "v2026-02/dist/en/categories.txt"
)


def _download(url: str, path: Path) -> Path:
    """Download a pinned dataset once.

    :param url: official dataset URL.
    :param path: local cache path.
    :returns: local dataset path.
    """

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(
        url, headers={"User-Agent": "typesafe-taxonomy/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary_path.open("wb") as file:
            shutil.copyfileobj(response, file)
    temporary_path.replace(path)
    return path


def _insert(tree: Tree, path: tuple[str, ...]) -> None:
    subtree_value: Tree = tree
    for label in path:
        subtree_value = subtree_value.setdefault(label, {})


def _load_shopify(path: Path) -> tuple[Tree, int]:
    tree: Tree = {}
    category_count: int = 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        _, path_text = line.split(" : ", maxsplit=1)
        category_path: tuple[str, ...] = tuple(path_text.strip().split(" > "))
        _insert(tree, category_path)
        category_count += 1
    return tree, category_count


def load_hierarchies(
    data_directory: Path = DATA_DIR,
) -> tuple[Hierarchy, ...]:
    """Load the Shopify product taxonomy.

    :param data_directory: cache directory for official raw files.
    :returns: Shopify example.
    """

    shopify_tree, shopify_nodes = _load_shopify(
        _download(SHOPIFY_URL, data_directory / "shopify_categories_2026-02.txt")
    )
    return (
        Hierarchy(
            slug="shopify",
            name="Shopify products",
            version="2026-02",
            source_url=SHOPIFY_URL,
            node_count=shopify_nodes,
            document=(
                "Furniture listing: a wall-mounted window shelf bed. This padded floating shelf "
                "uses suction cups and a washable cushion as a sunny perch for one cat."
            ),
            expected_leaf="Cat Window Beds & Perches",
            tree=shopify_tree,
        ),
    )


NODE_W, NODE_H = 300, 38
COL_W, ROW_H = 360, 50
PAD_X = 28
EDGE_TOP_K = 5


def subtree(tree: Tree, path: tuple[str, ...]) -> Tree:
    """Return the direct-child menu below ``path``.

    :param tree: taxonomy root.
    :param path: path from the taxonomy root.
    :returns: child mapping at the path.
    """

    subtree_value: Tree = tree
    for label in path:
        subtree_value = subtree_value[label]
    return subtree_value


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _truncate(value: str, length: int = 33) -> str:
    return value if len(value) <= length else value[: length - 1] + "…"


def _build_nodes(hierarchy: Hierarchy, result: dict) -> dict:
    records: dict[tuple[str, ...], dict] = {
        tuple(record["parent"]): record for record in result["records"]
    }
    best_path: tuple[str, ...] = tuple(result["beam"][0]["path"])
    greedy_path: tuple[str, ...] = tuple(result["greedy"]["path"])
    retained: set[tuple[str, ...]] = {tuple(path) for path in result["retained_paths"]}

    def grow(path: tuple[str, ...]) -> list[dict]:
        record: dict | None = records.get(path)
        if record is None:
            return []
        children: list[dict] = []
        probabilities: dict[str, float] = record["probabilities"]
        ranked: list[tuple[str, float]] = sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )
        shown_labels: set[str] = {label for label, _ in ranked[:EDGE_TOP_K]}
        shown_labels.update(
            label
            for label, _ in ranked
            if path + (label,) in retained
            or path + (label,) == best_path[: len(path) + 1]
            or path + (label,) == greedy_path[: len(path) + 1]
        )
        for label, probability in ranked:
            if label not in shown_labels:
                continue
            child_path: tuple[str, ...] = path + (label,)
            on_best_path: bool = child_path == best_path[: len(child_path)]
            on_greedy_path: bool = child_path == greedy_path[: len(child_path)]
            kind: str = (
                "winner"
                if on_best_path
                else "greedy"
                if on_greedy_path
                else "beam"
                if child_path in retained
                else "alt"
            )
            children.append(
                {
                    "label": label,
                    "probability": probability,
                    "kind": kind,
                    "children": grow(child_path),
                }
            )
        return children

    return {
        "label": hierarchy.name,
        "probability": None,
        "kind": "root",
        "children": grow(()),
    }


def _layout(root: dict) -> tuple[int, int]:
    rows: list[int] = [0]
    maximum_depth: list[int] = [0]

    def walk(node: dict, depth: int) -> None:
        node["depth"] = depth
        maximum_depth[0] = max(maximum_depth[0], depth)
        if node["children"]:
            for child in node["children"]:
                walk(child, depth + 1)
            node["row"] = (node["children"][0]["row"] + node["children"][-1]["row"]) / 2
        else:
            node["row"] = rows[0]
            rows[0] += 1

    walk(root, 0)
    return maximum_depth[0], rows[0]


def render_svg(hierarchy: Hierarchy, result: dict, path: Path) -> None:
    """Write a standalone traversal SVG matching the Customer_ProdX visual language.

    :param hierarchy: taxonomy demonstration.
    :param result: beam-search result from the notebook.
    :param path: output SVG path.
    """

    root: dict = _build_nodes(hierarchy, result)
    maximum_depth, row_count = _layout(root)
    document_lines: list[str] = textwrap.wrap(
        hierarchy.document,
        width=105,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    document_y: int = 124
    greedy_y: int = document_y + (len(document_lines) - 1) * 21 + 34
    beam_y: int = greedy_y + 25
    method_y: int = beam_y + 29
    legend_y: int = method_y + 23
    header_height: int = legend_y + 32
    width: int = PAD_X * 2 + maximum_depth * COL_W + NODE_W
    height: int = header_height + max(row_count, 1) * ROW_H + 34
    edges: list[str] = []
    nodes: list[str] = []

    def node_x(node: dict) -> float:
        return PAD_X + node["depth"] * COL_W

    def node_y(node: dict) -> float:
        return header_height + node["row"] * ROW_H

    def walk(node: dict) -> None:
        x_value, y_value = node_x(node), node_y(node)
        for child in node["children"]:
            child_x, child_y = node_x(child), node_y(child)
            x1, y1 = x_value + NODE_W, y_value + NODE_H / 2
            x2, y2 = child_x, child_y + NODE_H / 2
            bend: float = COL_W * 0.38
            edges.append(
                f'<path class="edge {child["kind"]}" '
                f'd="M{x1:.0f},{y1:.0f} C{x1 + bend:.0f},{y1:.0f} '
                f'{x2 - bend:.0f},{y2:.0f} {x2:.0f},{y2:.0f}"/>'
            )
            edges.append(
                f'<text class="prob" x="{x2 - 7:.0f}" y="{y2 - 5:.0f}" '
                f'text-anchor="end">{child["probability"]:.2f}</text>'
            )
            walk(child)

        kind: str = node["kind"]
        label: str = _truncate(node["label"], 40)
        nodes.append(
            f'<g class="node {kind}"><title>{_escape(node["label"])}</title>'
            f'<rect x="{x_value:.0f}" y="{y_value:.0f}" width="{NODE_W}" '
            f'height="{NODE_H}" rx="7"/>'
            f'<text x="{x_value + 11:.0f}" y="{y_value + 24:.0f}">'
            f"{_escape(label)}</text></g>"
        )

    walk(root)
    best: dict = result["beam"][0]
    best_path: tuple[str, ...] = tuple(best["path"])
    greedy_path: tuple[str, ...] = tuple(result["greedy"]["path"])
    beam_leaf: str = best_path[-1] if best_path else "no leaf"
    greedy_leaf: str = greedy_path[-1] if greedy_path else "no leaf"
    beam_width: int = result["beam_width"]
    separation_ratio: float = result["separation_ratio"]
    document_text: str = "".join(
        f'<text class="document" x="24" y="{document_y + index * 21}">'
        f"{_escape(line)}</text>"
        for index, line in enumerate(document_lines)
    )
    separation_text: str = (
        ">999×" if separation_ratio > 999 else f"{separation_ratio:.2f}×"
    )

    svg: str = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
  width="{width}" height="{height}" role="img" aria-label="{_escape(hierarchy.name)} taxonomy beam search">
<style>
  .bg {{ fill:#f6f7fb }}
  text {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace }}
  .eyebrow {{ font-size:14px; font-weight:700; letter-spacing:1.2px; fill:#4f46e5 }}
  .title {{ font:700 28px system-ui,-apple-system,"Segoe UI",sans-serif; fill:#181b28 }}
  .document-label {{ font:700 12px system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:1px; fill:#777c91 }}
  .document {{ font:500 17px system-ui,-apple-system,"Segoe UI",sans-serif; fill:#303449 }}
  .copy {{ font-size:14px; fill:#5c6178 }}
  .result {{ font-size:14px; font-weight:700 }}
  .greedy-result {{ fill:#c2410c }}
  .beam-result {{ fill:#15803d }}
  .edge {{ fill:none; stroke:#c9cee0; stroke-width:2 }}
  .edge.winner {{ stroke:#15803d; stroke-width:2.5 }}
  .edge.greedy {{ stroke:#ea580c; stroke-width:2.5 }}
  .edge.beam {{ stroke:#4f46e5; stroke-width:2.2 }}
  .edge.alt {{ opacity:.48 }}
  .prob {{ font-size:12px; font-weight:600; fill:#5c6178 }}
  .node rect {{ stroke-width:1.7 }}
  .node text {{ font-size:13px }}
  .node.root rect {{ fill:#f0f2f8; stroke:#e2e5f0 }}
  .node.root text {{ fill:#5c6178 }}
  .node.winner rect {{ fill:#e4f5ea; stroke:#15803d }}
  .node.winner text {{ fill:#15803d; font-weight:700 }}
  .node.greedy rect {{ fill:#fff0e8; stroke:#ea580c }}
  .node.greedy text {{ fill:#c2410c; font-weight:700 }}
  .node.beam rect {{ fill:#ecebfd; stroke:#4f46e5 }}
  .node.beam text {{ fill:#181b28 }}
  .node.alt rect {{ fill:#fff; stroke:#e2e5f0; stroke-dasharray:3 3 }}
  .node.alt text {{ fill:#5c6178 }}
</style>
<rect class="bg" width="{width}" height="{height}" rx="14"/>
<text class="eyebrow" x="24" y="32">H2O-LIGHTNING + PATH · {hierarchy.name.upper()} · {hierarchy.version.upper()} · {hierarchy.node_count:,} NODES</text>
<text class="title" x="24" y="69">Greedy vs parallel beam search</text>
<text class="document-label" x="24" y="99">DOCUMENT</text>
{document_text}
<text class="result greedy-result" x="24" y="{greedy_y}">GREEDY TOP-1 → {_escape(_truncate(greedy_leaf, 105))}</text>
<text class="result beam-result" x="24" y="{beam_y}">BEAM K={beam_width} → {_escape(_truncate(beam_leaf, 105))}</text>
<text class="copy" x="24" y="{method_y}">parallel sibling Choices → keep {beam_width} by geometric mean p → top/second = {separation_text}</text>
<text class="copy" x="24" y="{legend_y}">orange = greedy   green = beam winner   purple = retained beam   dashed = pruned</text>
{"".join(edges)}{"".join(nodes)}
</svg>'''
    path.write_text(svg)


class JsonCache:
    """取代 cooksafe.JsonCache：函式結果依參數存進一個 JSON 檔，重跑時直接讀，不再問 h2o-lightning。

    beam search 會從多個 thread 同時呼叫，讀寫都加鎖；每存一筆就整檔重寫（Shopify 一次只有幾十筆）。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, function):
        @functools.wraps(function)
        def wrapper(*args):
            key = json.dumps([function.__name__, *args], ensure_ascii=False)
            with self.lock:
                if key in self.data:
                    return self.data[key]
            value = function(*args)
            with self.lock:
                self.data[key] = value
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.path.with_suffix(".tmp")
                temporary_path.write_text(json.dumps(self.data, ensure_ascii=False))
                temporary_path.replace(self.path)
            return value

        return wrapper


# 官網的 MODEL = "jev-1.12" 拿掉：h2o-lightning 的 shim 不看 model 欄位
BEAM_WIDTH, MAX_DEPTH, EPSILON = 3, 12, 1e-9
# 官網是 TypeSafeClient(retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))；
# H2OClassifier 是 LangChain Runnable，用 with_retry 做同樣的事（第一次 + 重試 5 次）。
# Shopify 有些節點的子類別很多（最多 81 個），回答要比較久，timeout 放寬到 120 秒（同 hierarchical_cpc_h2o.py）
client = H2OClassifier(timeout=120).with_retry(
    retry_if_exception_type=(
        TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    ),
    stop_after_attempt=6,
    exponential_jitter_params={"initial": 1.0, "max": 20.0},
)
json_cache = JsonCache(OUT_DIR / "json_cache.json")


@json_cache
def choose(
    state: str, path: tuple[str, ...], labels: tuple[str, ...]
) -> dict[str, float]:
    """Ask one atomic direct-child question and return its distribution."""
    # 和原版不同：多一個 path（已經走過的節點），所以快取 key 也帶著路徑
    if len(labels) == 1:
        return {labels[0]: 1.0}
    question, keys = child_question(labels, path)
    # 官網：client.system_one(state=state, questions={"child": question}, model=MODEL)
    response = client.invoke({"state": state, "questions": {"child": question}})
    probabilities = response.answers["child"].probabilities
    return {label: probabilities[key] for key, label in keys.items()}


def child_question(
    labels: tuple[str, ...], path: tuple[str, ...] = ()
) -> tuple[Choice, dict[str, str]]:
    """Build the direct-child Choice and its reversible option mapping."""
    keys = {f"c{i}": label for i, label in enumerate(labels)}
    instructions = "Which direct child category best matches this document?"
    # 和原版唯一的差別：instructions 前面加上已經走過的路徑；根節點沒有路徑，和原版一字不差
    if path:
        instructions = (
            f"This document has already been classified under: {' > '.join(path)}. "
            + instructions
        )
    question = Choice(instructions=instructions, criteria=keys)
    return question, keys


def extend_candidate(
    candidate: dict, label: str, probabilities: dict[str, float]
) -> dict:
    """Append one edge and recompute its geometric-mean path score."""
    is_decision: bool = len(probabilities) > 1
    # Use log space for very deep trees to avoid floating-point precision loss.
    probability_product: float = candidate["probability_product"] * (
        max(probabilities[label], EPSILON) if is_decision else 1.0
    )
    decision_count: int = candidate["decision_count"] + is_decision
    return {
        "path": candidate["path"] + (label,),
        "probability_product": probability_product,
        "decision_count": decision_count,
        "score": probability_product ** (1 / decision_count) if decision_count else 1.0,
    }


def choice_record(path: tuple[str, ...], probabilities: dict[str, float]) -> dict:
    """Package one sibling decision for the traversal diagram."""
    return {"parent": path, "probabilities": probabilities}


def beam_search(hierarchy: Hierarchy) -> dict:
    """Parallel width-three beam search using geometric-mean probability."""
    beam = [{"path": (), "probability_product": 1.0, "decision_count": 0, "score": 1.0}]
    records, retained_paths = [], {()}

    for _ in range(MAX_DEPTH):
        expandable = [
            candidate
            for candidate in beam
            if subtree(hierarchy.tree, candidate["path"])
        ]
        finished = [
            candidate
            for candidate in beam
            if not subtree(hierarchy.tree, candidate["path"])
        ]
        if not expandable:
            break
        with ThreadPoolExecutor(max_workers=BEAM_WIDTH) as executor:
            distributions = list(
                executor.map(
                    lambda candidate: choose(
                        hierarchy.document,
                        candidate["path"],
                        tuple(subtree(hierarchy.tree, candidate["path"])),
                    ),
                    expandable,
                )
            )

        expanded = []
        round_records = []
        for candidate, probabilities in zip(expandable, distributions, strict=True):
            round_records.append(choice_record(candidate["path"], probabilities))
            candidate_expanded = []
            for label in probabilities:
                candidate_expanded.append(
                    extend_candidate(candidate, label, probabilities)
                )
            expanded.extend(candidate_expanded)
        beam = sorted(
            finished + expanded,
            key=lambda candidate: candidate["score"],
            reverse=True,
        )[:BEAM_WIDTH]
        retained_paths.update(candidate["path"] for candidate in beam)
        records.extend(round_records)

    beam = sorted(beam, key=lambda candidate: candidate["score"], reverse=True)
    return {
        "beam": beam,
        "records": records,
        "retained_paths": sorted(retained_paths, key=lambda path: (len(path), path)),
    }


def greedy_search(hierarchy: Hierarchy) -> dict:
    """Follow only the locally highest-probability child."""
    path, probability_product, decision_count, records = (), 1.0, 0, []
    for _ in range(MAX_DEPTH):
        labels = tuple(subtree(hierarchy.tree, path))
        if not labels:
            break
        probabilities = choose(hierarchy.document, path, labels)
        records.append(choice_record(path, probabilities))
        label = max(probabilities, key=probabilities.get)
        if len(probabilities) > 1:
            probability_product *= max(probabilities[label], EPSILON)
            decision_count += 1
        path += (label,)
    score: float = (
        probability_product ** (1 / decision_count) if decision_count else 1.0
    )
    return {"path": path, "score": score, "records": records}


def compare_searches(hierarchy: Hierarchy) -> dict:
    """Run beam and greedy, then merge their queried nodes for rendering."""
    result = beam_search(hierarchy)
    greedy = greedy_search(hierarchy)
    recorded_paths = {tuple(record["parent"]) for record in result["records"]}
    result["records"].extend(
        record
        for record in greedy["records"]
        if tuple(record["parent"]) not in recorded_paths
    )
    result["greedy"] = greedy
    result["beam_width"] = BEAM_WIDTH
    top_score: float = result["beam"][0]["score"]
    second_score: float = result["beam"][1]["score"]
    result["separation_ratio"] = top_score / max(second_score, EPSILON)
    return result


def main() -> None:
    HIERARCHIES = load_hierarchies()

    with ThreadPoolExecutor(max_workers=len(HIERARCHIES)) as executor:
        results = list(executor.map(compare_searches, HIERARCHIES))

    rows: list[dict[str, str | int | bool]] = []
    for hierarchy, result in zip(HIERARCHIES, results, strict=True):
        svg_path: Path = OUT_DIR / f"{hierarchy.slug}_tree.svg"
        render_svg(hierarchy, result, svg_path)
        beam_path: tuple[str, ...] = tuple(result["beam"][0]["path"])
        greedy_path: tuple[str, ...] = tuple(result["greedy"]["path"])
        beam_leaf: str = beam_path[-1]
        greedy_leaf: str = greedy_path[-1]
        rows.append(
            {
                "hierarchy": hierarchy.name,
                "nodes": hierarchy.node_count,
                "expected leaf": hierarchy.expected_leaf,
                "greedy leaf": greedy_leaf,
                "beam K=3 leaf": beam_leaf,
                "greedy correct": greedy_leaf == hierarchy.expected_leaf,
                "beam correct": beam_leaf == hierarchy.expected_leaf,
                "mean p": f"{result['beam'][0]['score']:.2f}",
                "top/second": f"{result['separation_ratio']:.2f}×",
            }
        )

    greedy_correct_count: int = sum(bool(row["greedy correct"]) for row in rows)
    beam_correct_count: int = sum(bool(row["beam correct"]) for row in rows)
    recovered_names: str = ", ".join(
        str(row["hierarchy"])
        for row in rows
        if not row["greedy correct"] and row["beam correct"]
    )
    table_lines: list[str] = [
        "| Hierarchy | Expected leaf | Greedy leaf | Beam K=3 leaf | Greedy correct | Beam correct |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    table_lines.extend(
        "| "
        + " | ".join(
            (
                str(row["hierarchy"]),
                str(row["expected leaf"]),
                str(row["greedy leaf"]),
                str(row["beam K=3 leaf"]),
                "yes" if row["greedy correct"] else "no",
                "yes" if row["beam correct"] else "no",
            )
        )
        + " |"
        for row in rows
    )
    # 官網是 IPython 的 display(Markdown(...))；這裡印出來，也寫成 results.md（圖片路徑相對於它）
    markdown: str = (
        "## Results\n\n"
        "Each example has a known expected leaf. "
        f"Beam search matched {beam_correct_count} of {len(rows)} expected leaves; "
        f"greedy search matched {greedy_correct_count} of {len(rows)}. "
        f"Keeping three paths recovered the expected classification for {recovered_names}.\n\n"
        + "\n".join(table_lines)
        + "\n\nThe diagrams show why the methods differ. Orange marks the greedy route, "
        "green marks the winning beam route, purple marks other retained paths, and "
        "dashed edges were pruned.\n\n"
        + "\n\n".join(
            f"### {hierarchy.name}\n\n![]({hierarchy.slug}_tree.svg)"
            for hierarchy in HIERARCHIES
        )
    )
    print(markdown)
    (OUT_DIR / "results.md").write_text(markdown + "\n")


if __name__ == "__main__":
    main()
