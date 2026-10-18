"""官網 function calling cookbook 的 dispatch.py：讀函式簽名和 spec，把一句話變成一個函式呼叫。

原文 https://docs.typesafe.ai/cookbooks/function_calling 只說 dispatch.py「holds the code that reads
a signature and a spec and makes the call」，原始檔（CookSafe repo）沒有公開，這裡照頁面上的
用法和輸出重寫：

  closed_sets(fn)   讀型別註記，把引數分成 choice（Literal）、set（list[Literal]）、flag（bool）。
                    其他型別（int、str、日期…）不出題，函式自己的預設值照用。
  Dispatcher        spec 一次轉成全部題目（assistant.questions）。每句話送一次請求，
                    裡面同時有「選哪個函式」和每個函式每個引數的題目，只讀選中那個函式的答案。

題目 id 照官網：
  __tool__                        Choice，十個函式的 description 當選項
  plot_price.style                choice 引數：Choice，選項 key 就是 Literal 的值
  plot_price.style?               有 stated 的引數多一題 Noul：使用者有沒有提到這個引數
  plot_price.include_volume       flag 引數：Noul
  compare_returns.symbols.NVDA    set 引數：每個成員一題 Noul，question 裡的 {} 換成成員名稱

答案怎麼組回來（從官網 "is amd tracking nvidia lately" 那段拆解反推）：
  * stated 的 Noul < 0.5 → 引數省略（omitted），用函式的預設值；這個引數的 p 是 1 - P(stated)。
  * 否則填值：choice 取機率最高的選項，flag 取 P(yes) >= 0.5，set 取 P(yes) >= 0.5 的成員。
    引數的 p = P(stated) × P(選到的值)（沒有 stated 的就是 P(選到的值)）。
    官網 benchmark 那行 p 0.78、NVDA 0.92，就是 0.92 × P(stated) ≈ 0.85。
  * call.confidence = 整個呼叫背後最沒把握的那一個判斷（選函式那題、每題 stated、每個值），
    不是全部相乘。官網那句 confidence 0.82 = 選函式那題的 0.82，比 benchmark 的 0.78 高，
    因為 0.78 是兩個判斷（0.85、0.92）乘起來的，各自都比 0.82 有把握。
  * call.weakest() = p 最低的引數。
"""

import inspect
import math
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

ROUTE = "__tool__"
"""「選哪個函式」那題的題目 id。"""

Shape = Literal["choice", "set", "flag"]


def closed_sets(fn: Callable) -> dict[str, tuple[Shape, tuple]]:
    """讀 fn 的型別註記，回傳 {引數名稱: (形狀, 可能的值)}，順序照簽名。

    Literal[...]             → ("choice", Literal 的值)
    Literal[...] | None      → ("choice", Literal 的值)，None 只是「沒指定」的預設值
    list[Literal[...]]       → ("set", Literal 的值)
    bool                     → ("flag", (True, False))
    其他型別不算 closed set，不回傳。
    """
    hints = typing.get_type_hints(fn)
    shapes: dict[str, tuple[Shape, tuple]] = {}
    for name in inspect.signature(fn).parameters:
        hint = hints.get(name)
        if hint is bool:
            shapes[name] = ("flag", (True, False))
            continue
        origin = typing.get_origin(hint)
        if origin in (typing.Union, types.UnionType):
            members = [a for a in typing.get_args(hint) if a is not type(None)]
            if len(members) != 1:
                continue
            hint = members[0]
            origin = typing.get_origin(hint)
        if origin is Literal:
            shapes[name] = ("choice", typing.get_args(hint))
        elif origin is list:
            (item,) = typing.get_args(hint)
            if typing.get_origin(item) is Literal:
                shapes[name] = ("set", typing.get_args(item))
    return shapes


class Client(Protocol):
    """Dispatcher 需要的 client：送一句話和一組題目，回 {題目 id: 答案的 JSON}。"""

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]: ...


@dataclass(frozen=True)
class Tool:
    """選函式那題的結果。"""

    name: str
    probability: float
    distribution: dict[str, float]


@dataclass(frozen=True)
class Argument:
    """一個 closed-set 引數的結果。"""

    name: str
    shape: Shape
    value: Any
    """填進去的值；omitted 時是函式自己的預設值（不會出現在呼叫裡）。"""
    omitted: bool
    """stated 那題說使用者沒提到這個引數，呼叫時不傳，用函式的預設值。"""
    probability: float
    """填值：P(stated) × P(選到的值)；省略：1 - P(stated)。"""
    distribution: dict
    """choice：每個選項的機率；set：每個成員的 P(yes)；flag：{True: p, False: 1-p}；省略時是空的。"""
    judgements: tuple[float, ...]
    """這個引數背後每一個判斷的機率（stated 一個、值一個或 set 的每個成員一個）。"""


@dataclass(frozen=True)
class Call:
    """一句話分派出來的函式呼叫。"""

    name: str
    function: Callable
    tool: Tool
    arguments: dict[str, Argument]
    """這個函式全部的 closed-set 引數（含省略的），順序照簽名。"""

    @property
    def kwargs(self) -> dict[str, Any]:
        return {
            name: argument.value
            for name, argument in self.arguments.items()
            if not argument.omitted
        }

    @property
    def confidence(self) -> float:
        """最沒把握的那一個判斷，不是全部相乘：錯一個引數就足以讓結果不對。"""
        return min(
            [self.tool.probability]
            + [p for argument in self.arguments.values() for p in argument.judgements]
        )

    def weakest(self) -> Argument | None:
        """p 最低的引數；函式沒有 closed-set 引數時是 None。"""
        return min(
            self.arguments.values(),
            key=lambda argument: argument.probability,
            default=None,
        )

    def run(self) -> Any:
        return self.function(**self.kwargs)

    def __str__(self) -> str:
        filled = ", ".join(f"{name}={value!r}" for name, value in self.kwargs.items())
        return f"{self.name}({filled})"


class Dispatcher:
    """spec 轉成題目；每句話問一次 client，組成 Call。

    assistant = Dispatcher(SPEC, TOOLS, client)
    call = assistant("show nvda 1h")       # → plot_price(symbol='NVDA', resolution='1h')
    call.run()
    """

    def __init__(self, spec: dict, tools: dict[str, Callable], client: Client) -> None:
        self.spec = spec
        self.tools = tools
        self.client = client
        self.shapes = {name: closed_sets(fn) for name, fn in tools.items()}
        self.questions = self._build_questions()

    def _build_questions(self) -> dict[str, dict]:
        functions = self.spec["functions"]
        missing = [name for name in self.tools if name not in functions]
        if missing:
            raise ValueError(f"spec 沒有這些函式：{missing}")
        questions: dict[str, dict] = {
            ROUTE: {
                "type": "choice",
                "instructions": self.spec["question"],
                "criteria": {
                    name: functions[name]["description"] for name in self.tools
                },
            }
        }
        for name, shapes in self.shapes.items():
            arguments = functions[name]["arguments"]
            for argument, (shape, values) in shapes.items():
                if argument not in arguments:
                    raise ValueError(f"spec 沒有 {name}.{argument}")
                entry = arguments[argument]
                qid = f"{name}.{argument}"
                if shape == "choice":
                    options = entry["options"]
                    if set(options) != {str(v) for v in values}:
                        raise ValueError(
                            f"{qid} 的選項 {sorted(options)} 跟 Literal 的值 "
                            f"{sorted(map(str, values))} 不一樣"
                        )
                    questions[qid] = {
                        "type": "choice",
                        "instructions": entry["question"],
                        "criteria": dict(options),
                    }
                elif shape == "flag":
                    questions[qid] = {"type": "noul", "instructions": entry["question"]}
                else:
                    for member in values:
                        questions[f"{qid}.{member}"] = {
                            "type": "noul",
                            "instructions": entry["question"].format(member),
                        }
                if "stated" in entry:
                    questions[f"{qid}?"] = {
                        "type": "noul",
                        "instructions": entry["stated"],
                    }
        return questions

    def __call__(self, command: str) -> Call:
        answers = self.client.ask(command, self.questions)
        route = answers[ROUTE]["probabilities"]
        name = max(route, key=route.get)
        function = self.tools[name]
        defaults = {
            parameter.name: parameter.default
            for parameter in inspect.signature(function).parameters.values()
        }
        arguments = {
            argument: self._argument(
                f"{name}.{argument}",
                argument,
                shape,
                values,
                defaults[argument],
                answers,
            )
            for argument, (shape, values) in self.shapes[name].items()
        }
        return Call(
            name=name,
            function=function,
            tool=Tool(name, route[name], dict(route)),
            arguments=arguments,
        )

    @staticmethod
    def _argument(
        qid: str,
        name: str,
        shape: Shape,
        values: tuple,
        default: Any,
        answers: dict[str, dict],
    ) -> Argument:
        stated, judgements = 1.0, []
        if f"{qid}?" in answers:
            stated = answers[f"{qid}?"]["noul"]
            if stated < 0.5:
                return Argument(
                    name=name,
                    shape=shape,
                    value=default,
                    omitted=True,
                    probability=1 - stated,
                    distribution={},
                    judgements=(1 - stated,),
                )
            judgements.append(stated)

        if shape == "choice":
            probabilities = answers[qid]["probabilities"]
            key = max(probabilities, key=probabilities.get)
            value = {str(v): v for v in values}[key]
            likelihood = probabilities[key]
            distribution = dict(probabilities)
            judgements.append(likelihood)
        elif shape == "flag":
            p = answers[qid]["noul"]
            value = p >= 0.5
            likelihood = max(p, 1 - p)
            distribution = {True: p, False: 1 - p}
            judgements.append(likelihood)
        else:
            distribution = {
                member: answers[f"{qid}.{member}"]["noul"] for member in values
            }
            value = [member for member, p in distribution.items() if p >= 0.5]
            members = [max(p, 1 - p) for p in distribution.values()]
            likelihood = math.prod(members)
            judgements.extend(members)

        return Argument(
            name=name,
            shape=shape,
            value=value,
            omitted=False,
            probability=stated * likelihood,
            distribution=distribution,
            judgements=tuple(judgements),
        )
