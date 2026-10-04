"""用 JevBench 的題目評測 jevk5（jevk5-serve 要開在 8090）。

評測流程在 evaluate_jevbench.py；這裡只決定用 JevK5Classifier、預設輸出到 ~/jb_runs/<tier>_py。

用法（在 ~/jevk5 底下）：
    python evaluate_jevk5.py                          # hard，輸出到 ~/jb_runs/hard_py
    python evaluate_jevk5.py --tier easy              # easy / standard / hard / hard_cn
    python evaluate_jevk5.py --tier all               # 上面四個 tier 依序全部跑
    python evaluate_jevk5.py --tier routing           # 自己出的 model-routing 題（routing.jsonl）
    python evaluate_jevk5.py --tier tool_risk         # 自己出的 tool-risk-gating 題（tool_risk.jsonl）
    python evaluate_jevk5.py --tier gpqa              # gpqa / aime / lcb / gsm8k / ai2arc：各 *_routing.py report 產生的 ground truth
    python evaluate_jevk5.py --limit 5                # 只跑前 5 題試試看
"""

from classifier_jevk5 import JEVK5_BASE_URL, JevK5Classifier
from evaluate_jevbench import main


def make_classifier(args):
    # jevbench 的 hard 題很長，jevk5 最久要一兩秒；timeout 放寬一點
    return JevK5Classifier(base_url=args.base_url or JEVK5_BASE_URL, timeout=120)


if __name__ == "__main__":
    main(
        "用 JevBench 題目評測 jevk5",
        make_classifier,
        default_out="~/jb_runs/{tier}_py",
        base_url_help=f"jevk5 服務網址，預設 {JEVK5_BASE_URL}",
    )
