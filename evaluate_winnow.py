"""用 JevBench 的題目評測 Winnow-12B Q8（winnow-inference 的 scripts/serve.py 要開在 8091）。

評測流程在 evaluate_jevbench.py；這裡只決定用 WinnowClassifier、預設輸出到 ~/jb_runs/winnow_<tier>。

用法（在 ~/jevk5 底下）：
    python evaluate_winnow.py                         # hard，輸出到 ~/jb_runs/winnow_hard
    python evaluate_winnow.py --tier easy             # easy / standard / hard / hard_cn
    python evaluate_winnow.py --tier all              # 上面四個 tier 依序全部跑
    python evaluate_winnow.py --tier routing          # 自己出的 model-routing 題（routing.jsonl）
    python evaluate_winnow.py --tier tool_risk        # 自己出的 tool-risk-gating 題（tool_risk.jsonl）
    python evaluate_winnow.py --tier gpqa             # gpqa / aime / lcb / gsm8k / ai2arc：各 *_routing.py report 產生的 ground truth
    python evaluate_winnow.py --limit 5               # 只跑前 5 題試試看
"""

from classifier_winnow import WinnowClassifier
from evaluate_jevbench import main


def make_classifier(args):
    # jevbench 的 hard 題很長；server 一次只處理一個請求，timeout 放寬一點
    kwargs = {"base_url": args.base_url} if args.base_url else {}
    return WinnowClassifier(model=args.model, timeout=120, **kwargs)


def add_arguments(ap):
    ap.add_argument("--model", default="Winnow-12B", help="送給 server 的模型名稱（server 的 --alias）")


if __name__ == "__main__":
    main(
        "用 JevBench 題目評測 Winnow-12B Q8",
        make_classifier,
        default_out="~/jb_runs/winnow_{tier}",
        base_url_help="Winnow 服務網址；預設 WINNOW_BASE_URL 環境變數，沒有就 http://127.0.0.1:8091",
        add_arguments=add_arguments,
    )
