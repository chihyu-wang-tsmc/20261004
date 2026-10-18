"""function_calling_jevk5.py 的 H2O-Lightning-4B 版：照
https://docs.typesafe.ai/cookbooks/function_calling 的範例，
把 Jev（TypeSafe client，TYPESAFE_MODEL = "jev-1.12"）換成本機 h2o-lightning（classifier_h2o.py）。

跟 function_calling_jevk5.py 只差這幾處，其餘（十個函式、spec、54 題、14 句指令、每一段的印法）
一行都沒改，結果可以直接和 jevk5 版、官網的 Jev 對照：
  * client 問的是 H2OClassifier（預設 http://localhost:8741），不是 JevK5Classifier。
  * 快取、圖、playground_questions.json 放在 function_calling_h2o/。快取一定要分開：key 只有
    句子和題目，跟 jevk5 共用同一個檔會把 jevk5 的答案當成 h2o 的。
  * 最後的對照表標 h2o。

官網跟 function_calling_jevk5.py 不同的地方（trader / dispatch / spec 照頁面重寫、模擬的 1 分 K、
快取版的 client、display 換成存 PNG、playground 連結換成 JSON）見 function_calling_jevk5.py 開頭，
這裡做法相同，三個共用檔（function_calling_trader.py / _dispatch.py / _spec.json）也是同一份。

**noul_floor 要注意**（見 classifier_h2o.py 開頭）：shim 預設把 max(p, 1-p) < 0.801 的 Noul
推到邊界。這裡 stated、bool、set 的每個成員都是 Noul，所以：
  * 填不填、True / False、成員選不選（看落在 0.5 哪一邊）都不受影響；
  * 但每個 Noul 判斷的機率至少 0.801，confidence 和引數的 p 會比 jevk5 版偏高，
    省略的引數 p 也不會低於 0.80。要比 confidence 就把 serve_config.json 的 noul_floor 改成 0、
    重啟 shim，再刪掉 function_calling_h2o/json_cache.json 重跑。
Choice（選函式、選值）不受影響。

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    pip install polars matplotlib numpy
    python function_calling_h2o.py                          # 照官網跑一遍
    python function_calling_h2o.py "show tsla candles"      # 另外分派自己的句子（可以給好幾句）
"""

import json
import sys
from pathlib import Path

from classifier_h2o import H2OClassifier
from function_calling_dispatch import ROUTE, Call, Dispatcher, closed_sets
from function_calling_trader import TOOLS, load, make_client, show

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "function_calling_h2o"

# 官網：from trader import TOOLS, client, load（client 在 trader.py 裡建好）。
# trader 這裡兩版共用，client 就在各自的檔案建。54 題一次送，timeout 放寬到 120 秒（同 hierarchical_cpc.py）
client = make_client(H2OClassifier(timeout=120), OUT_DIR / "json_cache.json")
SPEC = json.loads((HERE / "function_calling_spec.json").read_text())

COMMANDS = [
    "show nvda 1h",
    "plot rolling correlation between nvda and spy for the past month",
    "when during the day does nvda trade the most",
    "what moved today",
    "what tickers do you have",
    "how did the market do this week",
    "candles for tesla with a 20 period moving average",
    "compare nvda amd and msft over the past three months",
    "how volatile is tsla",
    "biggest losers today",
    "worst drawdown for nvda this quarter, and chart it please",
    "spy stats for the last month",
    "show me apple daily with volume",
    "is amd tracking nvidia lately",
]

# 官網（Jev, jev-1.12）的結果：呼叫、confidence、選函式那題的機率
OFFICIAL = {
    "show nvda 1h": ("plot_price(symbol='NVDA', resolution='1h')", 0.78, 1.00),
    "plot rolling correlation between nvda and spy for the past month": (
        "rolling_correlation(symbol='NVDA', benchmark='SPY', window='1mo')",
        0.91,
        1.00,
    ),
    "when during the day does nvda trade the most": (
        "intraday_pattern(symbol='NVDA')",
        0.53,
        1.00,
    ),
    "what moved today": ("top_movers(window='1d', direction='gainers')", 0.90, 0.90),
    "what tickers do you have": ("list_symbols()", 1.00, 1.00),
    "how did the market do this week": ("market_summary(window='1w')", 0.96, 0.99),
    "candles for tesla with a 20 period moving average": (
        "plot_price(symbol='TSLA', style='candles', moving_average='20')",
        0.69,
        0.97,
    ),
    "compare nvda amd and msft over the past three months": (
        "compare_returns(symbols=['NVDA', 'AMD', 'MSFT'], window='3mo')",
        0.94,
        1.00,
    ),
    "how volatile is tsla": ("volatility(symbol='TSLA')", 0.96, 1.00),
    "biggest losers today": ("top_movers(window='1d', direction='losers')", 0.98, 0.98),
    "worst drawdown for nvda this quarter, and chart it please": (
        "drawdown(symbol='NVDA', window='3mo', plot=True)",
        0.84,
        0.84,
    ),
    "spy stats for the last month": (
        "summary_stats(symbol='SPY', window='1mo')",
        0.88,
        0.88,
    ),
    "show me apple daily with volume": (
        "plot_price(symbol='AAPL', resolution='1d', include_volume=True)",
        0.75,
        0.85,
    ),
    "is amd tracking nvidia lately": (
        "rolling_correlation(symbol='AMD', benchmark='NVDA')",
        0.82,
        0.82,
    ),
}


def explain(command: str, call: Call) -> None:
    """官網「Read the confidence」那段：每個引數的值、p、機率最高的三個選項。"""
    print(f'"{command}"  ->  {call}   confidence {call.confidence:.2f}')
    for name, argument in call.arguments.items():
        top = sorted(argument.distribution.items(), key=lambda kv: -kv[1])[:3]
        shown = "omitted, default stands" if argument.omitted else repr(argument.value)
        print(
            f"  {name:<11} {shown:<25} p {argument.probability:.2f}   "
            + "  ".join(f"{k} {v:.2f}" for k, v in top)
        )
    weakest = call.weakest()
    # 官網直接 call.weakest().name；選到沒有引數的函式（list_symbols）時 weakest() 是 None
    print(f"  weakest argument: {weakest.name if weakest else '-'}")


def dispatch_own(commands: list[str]) -> None:
    """python function_calling_h2o.py "句子" ...：分派自己的句子、拆解 confidence、執行。"""
    assistant = Dispatcher(SPEC, TOOLS, client)
    for i, command in enumerate(commands, 1):
        call = assistant(command)
        explain(command, call)
        print(f"  tool {call.tool.probability:.2f}")
        show(call.run(), OUT_DIR / f"own_{i}_{call.name}.png")
        print()


def main() -> None:
    if len(sys.argv) > 1:
        dispatch_own(sys.argv[1:])
        return

    # ## Setup
    print(f"{len(TOOLS)} functions over {load().height:,} one-minute bars")

    # ## Find the closed sets in the signatures
    print("\n## Find the closed sets in the signatures\n")
    for name, fn in TOOLS.items():
        shapes = closed_sets(fn)
        print(
            f"  {name:<20}{len(shapes)}  "
            + ", ".join(f"{a}:{s}" for a, (s, _) in shapes.items())
        )
    print(
        f"\n{sum(len(closed_sets(fn)) for fn in TOOLS.values())} fillable arguments in total"
    )

    # ## Write the spec
    print("\n## Write the spec\n")
    for argument in ("style", "moving_average"):
        print(
            json.dumps(
                {argument: SPEC["functions"]["plot_price"]["arguments"][argument]},
                indent=2,
            )
        )

    # ## Turn the spec into questions
    print("\n## Turn the spec into questions\n")
    assistant = Dispatcher(SPEC, TOOLS, client)
    print(f"{len(assistant.questions)} questions per command, among them:")
    for qid in (
        "__tool__",
        "plot_price.style",
        "plot_price.style?",
        "compare_returns.symbols.NVDA",
    ):
        question = assistant.questions[qid]
        print(f"  {qid:<30}{question['type']:<8}{str(question['instructions'])[:64]}")

    # ## Run fourteen commands
    print("\n## Run fourteen commands\n")
    CALLS = {command: assistant(command) for command in COMMANDS}
    for command, call in CALLS.items():
        print(f'  "{command}"')
        print(
            f"      {str(call):<65} confidence {call.confidence:.2f}"
            f"   tool {call.tool.probability:.2f}"
        )

    # Running three of them（官網用 display 顯示圖，這裡存成 PNG）
    print("\n## Running three of them\n")
    for i, command in enumerate(
        (
            "plot rolling correlation between nvda and spy for the past month",
            "compare nvda amd and msft over the past three months",
            "when during the day does nvda trade the most",
        ),
        1,
    ):
        print(f'"{command}"  ->  {CALLS[command]}')
        show(CALLS[command].run(), OUT_DIR / f"chart_{i}_{CALLS[command].name}.png")

    # And the ones that answer in text
    print("\n## And the ones that answer in text\n")
    for command in ("how did the market do this week", "biggest losers today"):
        print(f'"{command}"  ->  {CALLS[command]}')
        print(CALLS[command].run(), "\n")

    # ## Read the confidence
    print("## Read the confidence\n")
    explain("is amd tracking nvidia lately", CALLS["is amd tracking nvidia lately"])

    # ## Open it in the playground（官網產生 TypeSafe playground 連結；這裡把同一組題目存成 JSON）
    print("\n## Open it in the playground\n")
    COMMAND = "plot rolling correlation between nvda and spy for the past month"
    picked = CALLS[COMMAND]
    playground_questions = {ROUTE: assistant.questions[ROUTE]} | {
        q: v for q, v in assistant.questions.items() if q.startswith(f"{picked.name}.")
    }
    path = OUT_DIR / "playground_questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"state": COMMAND, "questions": playground_questions},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"{len(playground_questions)} questions for {COMMAND!r} written to {path}")

    # 官網沒有這段：14 句和官網（Jev）的結果並排
    print("\n## 跟官網（Jev）對照\n")
    same = 0
    for command, call in CALLS.items():
        official, official_confidence, official_tool = OFFICIAL[command]
        matched = str(call) == official
        same += matched
        print(f'  {"same" if matched else "DIFF"}  "{command}"')
        print(
            f"        h2o   {str(call):<65} confidence {call.confidence:.2f}"
            f"   tool {call.tool.probability:.2f}"
        )
        print(
            f"        jev   {official:<65} confidence {official_confidence:.2f}"
            f"   tool {official_tool:.2f}"
        )
    print(f"\n{same} of {len(CALLS)} calls match the official Jev calls")


if __name__ == "__main__":
    main()
