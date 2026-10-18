"""basic_memory_jevk5.py 的 H2O-Lightning-4B 版：照
https://docs.typesafe.ai/cookbooks/basic_memory（Simple memory system）的範例，
把 Jev（TypeSafeClient.system_one，TYPESAFE_MODEL = "jev-1.13.0"）換成本機 h2o-lightning（classifier_h2o.py）。

跟 basic_memory_jevk5.py 只差這幾處，其餘（DATA、ROUTE、GATE、ask_with_memory()、每一段的印法）
一行都沒改，結果可以直接和 jevk5 版、官網的 Jev 對照：
  * client 問的是 H2OClassifier（預設 http://localhost:8741），回應是 H2OResponse。
  * 快取和 playground.json 放在 basic_memory_h2o/。
  * 印出來的名稱是 h2o。

官網跟 basic_memory_jevk5.py 不同的地方（client 換成本機分類器加 with_retry、cooksafe.JsonCache
換成這裡的 JsonCache、playground 連結換成 JSON、最後多一段跟官網對照）見 basic_memory_jevk5.py 開頭，
這裡做法相同。as_memory() 的 "jev_answered" 一樣照官網不改名（GATE 的題目點名這個欄位）。

**noul_floor 要注意**（見 classifier_h2o.py 開頭）：gate 是一題 Noul，shim 預設把
max(p, 1-p) < 0.801 的機率推到邊界，所以 P(remember) 只會 <= 0.199 或 >= 0.801。
存不存（看落在 0.5 哪一邊）不受影響，但印出來的 P(remember) 會比原始機率極端，
不能直接拿去跟官網、jevk5 版的 0.07、0.91 這些數字比，也沒辦法拿來調 REMEMBER_AT。
要原始機率就把 serve_config.json 的 noul_floor 改成 0、重啟 shim，再刪掉
basic_memory_h2o/json_cache.json 重跑。分派那題是 Choice，不受影響。

要先把服務起起來（見 classifier_h2o.py 開頭）：
    vllm serve ~/models/h2o-lightning-4b ... --port 8000
    python ~/models/h2o-lightning-4b/h2o_lightning_shim.py --vllm http://127.0.0.1:8000 --port 8741

用法：
    python basic_memory_h2o.py

官網的結果（Jev, jev-1.13.0, 2026-09-28）：存下 T-101、T-102、T-105 三張（P(remember) 0.91 ~ 0.92），
T-103、T-104 沒存（0.07、0.10）；六則新訊息不帶記憶分對 2/6，帶記憶 6/6。
"""

import functools
import hashlib
import json
import threading
from pathlib import Path

from langchain_typesafe import Choice, Noul, NoulCriteria, Question
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from pydantic import TypeAdapter

from classifier_h2o import H2O_MODEL, H2OClassifier, H2OResponse

NAME = "h2o"  # 印出來取代官網的 "Jev"
OUT_DIR = Path(__file__).resolve().with_name("basic_memory_h2o")


class JsonCache:
    """取代 cooksafe.JsonCache：函式結果依參數存進一個 JSON 檔，重跑時直接讀，不再問 h2o。

    key 的算法跟 cooksafe 相同：函式名稱和參數用 | 接起來，超過 200 字就換成 sha256。
    回傳的是用 JSON 複製出來的新物件，呼叫端改了也不會動到快取。
    """

    MAX_KEY_CHARS = 200

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, function):
        @functools.wraps(function)
        def wrapper(*args):
            key = "|".join([function.__name__, *map(str, args)])
            if len(key) > self.MAX_KEY_CHARS:
                key = hashlib.sha256(key.encode()).hexdigest()
            if key not in self.data:
                result = function(*args)
                with self.lock:
                    self.data[key] = result
                    ordered = {k: self.data[k] for k in sorted(self.data)}
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = self.path.with_suffix(".tmp")
                    temporary_path.write_text(json.dumps(ordered, indent=2))
                    temporary_path.replace(self.path)
            return json.loads(json.dumps(self.data[key]))

        return wrapper


# ## Setup

MODEL = H2O_MODEL  # 官網是 TYPESAFE_MODEL = "jev-1.13.0"；這裡只用在快取的 key
REMEMBER_AT = 0.5  # save when P(lasting customer fact) reaches this starting threshold
MEMORY_KEY = "earlier_tickets_from_this_customer"  # where saved tickets go in the state

# 官網：TypeSafeClient(api_key=..., base_url=os.environ.get("TYPESAFE_ENDPOINT"), timeout=120.0)。
# h2o 不需要 API key；重試用 LangChain 的 with_retry（第一次 + 重試 5 次，同 hierarchical_*_h2o.py）
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


# ## Define the tickets

DATA = {
    "customers": {
        "C-1042": {"id": "C-1042", "company": "Harbor Logistics"},
        "C-2210": {"id": "C-2210", "company": "Fernwood Studio"},
    },
    "earlier_tickets": [
        {"id": "T-101", "customer_id": "C-1042", "received_on": "2026-08-04",
         "subject": "Adding an admin",
         "text": "Hi, we run your self-hosted edition on Windows Server 2022 in our own data "
                 "centre. How do I add a new admin user?"},
        {"id": "T-102", "customer_id": "C-2210", "received_on": "2026-08-11",
         "subject": "Exporting to PDF",
         "text": "We're a small design studio on your cloud edition, paying monthly on the "
                 "self-serve Team plan. How do I export a dashboard as a PDF?"},
        {"id": "T-103", "customer_id": "C-1042", "received_on": "2026-08-19",
         "subject": "Invoice download",
         "text": "Sorry for the typos, typing this on my phone at the airport. Where can I "
                 "download last month's invoice?"},
        {"id": "T-104", "customer_id": "C-2210", "received_on": "2026-09-02",
         "subject": "Dark mode?",
         "text": "Our whole team is out on Friday for a public holiday, so no rush on this "
                 "one. Is there a dark mode?"},
        {"id": "T-105", "customer_id": "C-1042", "received_on": "2026-09-10",
         "subject": "Sandbox environment",
         "text": "For context, we signed an Enterprise contract with you in March. Can we get "
                 "a sandbox environment for testing?"},
    ],
    "later_messages": [
        {"subject": "Slow dashboards",
         "text": "Dashboards have been really slow since this morning."},
        {"subject": "Upgrade failing",
         "text": "The upgrade to version 5.2 keeps failing halfway through."},
        {"subject": "More seats",
         "text": "We'd like to add 20 more seats before next quarter."},
    ],
    "later_received_on": "2026-09-28",
    # Labelled by hand from what each customer said in the earlier tickets.
    "expected_team": {
        "C-1042": {"Slow dashboards": "selfhosted_windows",
                   "Upgrade failing": "selfhosted_windows", "More seats": "enterprise_desk"},
        "C-2210": {"Slow dashboards": "cloud_ops", "Upgrade failing": "cloud_ops",
                   "More seats": "billing"},
    },
}
CUSTOMERS = DATA["customers"]


def ticket_state(customer_id: str, ticket_id: str, received_on: str, subject: str,
                 text: str) -> dict:
    """The state Jev sees for one ticket: a simplified ticket, like one from a helpdesk."""
    return {
        "customer": CUSTOMERS[customer_id],
        "ticket": {
            "id": ticket_id,
            "received_on": received_on,
            "subject": subject,
            "messages": [{"from": "customer", "text": text}],
        },
    }


EARLIER = [
    ticket_state(t["customer_id"], t["id"], t["received_on"], t["subject"], t["text"])
    for t in DATA["earlier_tickets"]
]


# ## Route a ticket

ROUTE = {
    "team": Choice(
        instructions="Which support team should handle `ticket`?",
        criteria={
            "cloud_ops": "Cloud team: problems with the hosted cloud edition, run by us",
            "selfhosted_linux": "Self-hosted team, Linux: installs, upgrades and "
            "performance of the self-hosted edition on Linux servers",
            "selfhosted_windows": "Self-hosted team, Windows: installs, upgrades and "
            "performance of the self-hosted edition on Windows Server",
            "billing": "Billing: invoices, payments, seats and plan changes for "
            "self-serve plans",
            "enterprise_desk": "Enterprise desk: seats, contracts and account changes for "
            "customers on an Enterprise contract (their technical problems go to the "
            "technical teams)",
            "general": "General product questions: features, how-to, settings",
            "none_suitable": "None of these teams fits the ticket",
        },
    ),
}

_QUESTION = TypeAdapter(Question)


@json_cache
def _system_one(model: str, state: str, questions: str) -> dict:
    # 官網：client.system_one(state=..., questions=..., model=model)。
    # shim 不看 model，model 只是快取 key 的一部分
    response = client.invoke(
        {
            "state": json.loads(state),
            "questions": {
                qid: _QUESTION.validate_python(question)
                for qid, question in json.loads(questions).items()
            },
        }
    )
    return response.model_dump(mode="json")


def ask(state: dict, questions: dict) -> H2OResponse:
    """client.system_one, with the model pinned and the call cached."""
    as_json = {qid: q.model_dump(mode="json", exclude_none=True) for qid, q in questions.items()}
    raw = _system_one(MODEL, json.dumps(state), json.dumps(as_json))
    return H2OResponse.model_validate(raw)


def later_state(customer_id: str, message_index: int) -> dict:
    """One of the later messages, sent by this customer, with a fixed ticket ID."""
    n = 201 + 3 * list(CUSTOMERS).index(customer_id) + message_index
    message = DATA["later_messages"][message_index]
    return ticket_state(customer_id, f"T-{n}", DATA["later_received_on"], message["subject"],
                        message["text"])


# ## Ask Jev whether the ticket is worth remembering

GATE = {
    "remember": Noul(
        instructions=(
            "In `request.ticket`, does the thread state a lasting fact about the customer that "
            "should change how their future, unrelated tickets are handled? Count what the "
            "customer or our support staff wrote in `request.ticket`. `jev_answered` shows how "
            "the ticket was routed, which may rely on things learned earlier: do not count "
            "anything it says or implies."
        ),
        criteria=NoulCriteria(
            true=(
                "Yes: in `request.ticket` the customer or our support staff state something "
                "about the customer that stays true after this ticket, such as the product "
                "edition or operating system they run, their plan or contract, or a standing "
                "preference."
            ),
            false=(
                "No: `request.ticket` only contains things specific to this one ticket or "
                "moment, such as a one-off request, today's schedule, or the device they "
                "happen to be typing on; or the lasting fact appears only in `jev_answered`."
            ),
        ),
    ),
}


def answer_text(question, answer):
    """The readable part of an answer: the chosen option's description, or the number."""
    if answer.type == "choice":
        return question.criteria[answer.choice]
    return answer.noul if answer.type == "noul" else answer.score


def as_memory(state: dict, questions: dict, response: H2OResponse) -> dict:
    # "jev_answered" 照官網，不改成 h2o：GATE 的題目點名這個欄位
    return {
        "request": state,
        "jev_answered": {
            q.instructions: answer_text(q, response.answers[qid])
            for qid, q in questions.items()
        },
    }


def remember_probability(memory: dict) -> float:
    """P(the request states an important fact about the customer), with the answer as context."""
    return ask(memory, GATE).answers["remember"].noul


# ## Wrap the call so it remembers

MEMORIES: dict[str, list[dict]] = {}  # customer ID -> that customer's saved memories


def customer_id(state: dict) -> str:
    return state["customer"]["id"]


def ask_with_memory(state: dict, questions: dict) -> H2OResponse:
    """ask(), plus memory: same arguments, same return value."""
    # 1. Answer the call with this customer's existing memories.
    saved = MEMORIES.get(customer_id(state), [])
    response = ask({**state, MEMORY_KEY: saved} if saved else state, questions)

    # 2. Ask Jev whether the original request states an important fact worth keeping.
    memory = as_memory(state, questions, response)
    if remember_probability(memory) >= REMEMBER_AT:
        MEMORIES.setdefault(customer_id(state), []).append(memory)
    return response


# ## Send the earlier tickets and watch the answers change

PROBES = {"Slow dashboards": 0, "More seats": 2}  # later messages to send after each ticket


def memory_ids(cid: str) -> str:
    return ", ".join(m["request"]["ticket"]["id"] for m in MEMORIES.get(cid, [])) or "none"


def send_probes(cid: str) -> None:
    for subject, i in PROBES.items():
        team = ask_with_memory(later_state(cid, i), ROUTE).answers["team"]
        print(f"      {subject:<16} -> {team.choice:<19} {team.probabilities[team.choice]:.2f}")


# ## Route the same messages for both customers

EXPECTED = DATA["expected_team"]


def cell(answer, expected: str) -> str:
    mark = "✓" if answer.choice == expected else "✗"
    return f"{mark} {answer.choice} {answer.probabilities[answer.choice]:.2f}"


# ## 跟官網（Jev）對照：官網頁面上印出來的結果（jev-1.13.0, 2026-09-28）

OFFICIAL_FIRST = ("cloud_ops", 0.80)  # C-1042 的 Slow dashboards，不帶記憶
OFFICIAL_GATE = {  # 舊單 -> (分派到哪一組, P(remember))
    "T-101": ("selfhosted_windows", 0.91),
    "T-102": ("general", 0.91),
    "T-103": ("billing", 0.07),
    "T-104": ("general", 0.10),
    "T-105": ("enterprise_desk", 0.92),
}
OFFICIAL_MEMORIES = {"C-1042": "T-101, T-105", "C-2210": "T-102"}  # 全部跑完後存下的記憶
OFFICIAL_TABLE = {  # (客戶, 新訊息) -> (不帶記憶, 帶記憶)，各是 (分派到哪一組, 機率)
    ("C-1042", "Slow dashboards"): (("cloud_ops", 0.80), ("selfhosted_windows", 0.94)),
    ("C-1042", "Upgrade failing"): (("none_suitable", 0.60), ("selfhosted_windows", 1.00)),
    ("C-1042", "More seats"): (("enterprise_desk", 0.62), ("enterprise_desk", 0.99)),
    ("C-2210", "Slow dashboards"): (("cloud_ops", 0.75), ("cloud_ops", 0.91)),
    ("C-2210", "Upgrade failing"): (("none_suitable", 0.63), ("cloud_ops", 0.86)),
    ("C-2210", "More seats"): (("enterprise_desk", 0.52), ("billing", 0.98)),
}


def official_cell(answer: tuple[str, float], expected: str) -> str:
    """cell() 的官網版：answer 是 (分派到哪一組, 機率)。"""
    choice, p = answer
    return f"{'✓' if choice == expected else '✗'} {choice} {p:.2f}"


def gate_cell(choice: str, p: float) -> str:
    verdict = "saved" if p >= REMEMBER_AT else "not saved"
    return f"{choice:<19} P(remember) {p:.2f}  {verdict}"


def compare(first, gates: dict, rows: dict) -> None:
    """官網沒有這段：把這裡跑出來的結果和官網（Jev）的並排。"""
    print("\n## 跟官網（Jev）對照\n")
    choice, p = OFFICIAL_FIRST
    print("C-1042 Slow dashboards, no memory")
    print(f"    {'Jev':<7}{choice} {p:.2f}")
    print(f"    {NAME:<7}{first.choice} {first.probabilities[first.choice]:.2f}")

    print(f"\nEarlier tickets through the gate (saved when P(remember) >= {REMEMBER_AT})")
    for ticket_id, official in OFFICIAL_GATE.items():
        print(f"  {ticket_id}  {'Jev':<7}{gate_cell(*official)}")
        print(f"         {NAME:<7}{gate_cell(*gates[ticket_id])}")

    print("\nMemories at the end")
    for cid in CUSTOMERS:
        print(f"  {cid}  {'Jev':<7}{OFFICIAL_MEMORIES[cid]}")
        print(f"          {NAME:<7}{memory_ids(cid)}")

    print("\nLater messages")
    print(f"{'customer':<9}{'message':<17}{'expected':<20}{'':<7}{'plain':<27}{'with memory':<27}")
    totals = {"Jev": [0, 0], NAME: [0, 0]}
    for (cid, subject), (official_plain, official_memory) in OFFICIAL_TABLE.items():
        expected = EXPECTED[cid][subject]
        plain, remembered = rows[cid, subject]
        totals["Jev"][0] += official_plain[0] == expected
        totals["Jev"][1] += official_memory[0] == expected
        totals[NAME][0] += plain.choice == expected
        totals[NAME][1] += remembered.choice == expected
        print(f"{cid:<9}{subject:<17}{expected:<20}{'Jev':<7}"
              f"{official_cell(official_plain, expected):<27}"
              f"{official_cell(official_memory, expected):<27}")
        print(f"{'':<46}{NAME:<7}{cell(plain, expected):<27}{cell(remembered, expected):<27}")
    print()
    for name, (plain_correct, memory_correct) in totals.items():
        print(f"{name:<7}expected routes: {plain_correct}/{len(rows)} without memory, "
              f"{memory_correct}/{len(rows)} with memory.")


def main() -> None:
    print("## Define the tickets\n")
    print(f"{len(CUSTOMERS)} customers, {len(EARLIER)} earlier tickets, "
          f"{len(DATA['later_messages'])} later messages. The first ticket as {NAME} sees it:\n")
    print(json.dumps(EARLIER[0], indent=2))

    print("\n## Route a ticket\n")
    slow = later_state("C-1042", 0)
    team = ask(slow, ROUTE).answers["team"]
    print(f"C-1042: {slow['ticket']['messages'][0]['text']!r}")
    print(f"  plain {NAME} -> {team.choice} {team.probabilities[team.choice]:.2f}")

    print("\n## Send the earlier tickets and watch the answers change\n")
    gates = {}  # 官網沒有：記下每張舊單的分派和 P(remember)，給最後的對照用
    for cid in CUSTOMERS:
        print(f"{cid}, memories: {memory_ids(cid)}")
        send_probes(cid)

    for state in EARLIER:
        cid = customer_id(state)
        response = ask_with_memory(state, ROUTE)
        p = remember_probability(as_memory(state, ROUTE, response))  # cached: the wrapper's call
        verdict = "saved" if p >= REMEMBER_AT else "not saved"
        text = state["ticket"]["messages"][0]["text"]
        print(f"\n{cid} {state['ticket']['id']}  {response.answers['team'].choice:<19} "
              f"P(remember) {p:.2f}  {verdict}")
        print(f"    {text[:84]}")
        print(f"{cid}, memories: {memory_ids(cid)}")
        send_probes(cid)
        gates[state["ticket"]["id"]] = (response.answers["team"].choice, p)

    print("\n## Route the same messages for both customers\n")
    plain_correct = memory_correct = total = 0
    example_state = None
    rows = {}  # 官網沒有：記下每一列的兩個答案，給最後的對照用

    print(f"{'customer':<9}{'message':<17}{'expected':<20}{'plain ' + NAME:<27}{'with memory':<27}")
    for cid in CUSTOMERS:
        for i, message in enumerate(DATA["later_messages"]):
            state = later_state(cid, i)
            expected = EXPECTED[cid][message["subject"]]
            plain = ask(state, ROUTE).answers["team"]
            if example_state is None:
                # Keep the exact state used for this call for the playground link.
                saved = MEMORIES.get(cid, [])
                example_state = json.loads(json.dumps({**state, MEMORY_KEY: saved} if saved else state))
            response = ask_with_memory(state, ROUTE)
            remembered = response.answers["team"]
            plain_correct += plain.choice == expected
            memory_correct += remembered.choice == expected
            total += 1
            rows[cid, message["subject"]] = (plain, remembered)
            print(f"{cid:<9}{message['subject']:<17}{expected:<20}{cell(plain, expected):<27}"
                  f"{cell(remembered, expected):<27}")

    print(f"\nExpected routes: {plain_correct}/{total} without memory, "
          f"{memory_correct}/{total} with memory.")

    # 官網用 make_playground_link 產生 TypeSafe playground 的連結（雲端的 Jev），
    # 這裡把同一組 state（C-1042 第一次問 Slow dashboards 時帶的記憶）和題目寫成 JSON
    print("\n## Open it in the playground\n")
    path = OUT_DIR / "playground.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "state": example_state,
                "questions": {
                    qid: q.model_dump(mode="json", exclude_none=True)
                    for qid, q in ROUTE.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"C-1042's ticket with its memories and the routing question written to {path}")

    compare(team, gates, rows)


if __name__ == "__main__":
    main()
