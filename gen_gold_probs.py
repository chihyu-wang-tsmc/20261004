"""產生帶 gold_probs 的機率題，格式和 JevBench 的 probability family 一致。

為什麼需要這種題：
    一般的評測資料集只給硬標籤（「這題有害」），所以只能算 ECE（需要對/錯就夠）。
    要算 TVD / probability fidelity，必須知道「這題正確的機率分佈是多少」。
    JevBench 的 111 題公開 hard 題裡只有 10 題做到，全在 probability family。

和 JevBench 作法的差別（刻意的改進）：
    JevBench 是 author_model=claude-opus-5 設計情境並自己算機率，再由另一個引擎盲審
    （provenance.label_basis: "Authored with rationale; cross-model review before any system run"）。
    這支程式把順序反過來：
        1. Python 先用 fractions.Fraction 算出精確機率（分數，零浮點誤差），算式由程式產生
        2. 模型只負責把這些數字包裝成自然語言情境，不准改動任何數字
        3. 另一家廠牌的模型盲審：只看情境和問題，看不到 gold 和算式，自己重算一次
        4. 盲審的數字和 Python 的不符就丟掉這題
    模型算錯的可能性因此從根本上消除——gold 永遠是 Python 算的，模型只負責寫字和驗算。

三種題型都是封閉形式可精確計算的：
    hypergeometric   N 件中有 D 件瑕疵，抽 n 件不放回，問「至少一件瑕疵」。Noul。
                     干擾：已失效的舊抽樣計畫用不同的 n，算出來會跨過 50% 的另一邊。
    bayes_test       盛行率 + 敏感度 + 特異度，問「檢驗陽性時真的有病」。Noul。
                     干擾：情境裡有人把敏感度當成答案講出來（base rate fallacy）。
    finite_three_way 有限母體依條件分成三類，問落在哪一類。Choice（3 個標籤）。
                     干擾：未經條件篩選的原始人數比例。

用法：
    python gen_gold_probs.py -n 12                      # 產生 12 題
    python gen_gold_probs.py -n 6 --family bayes_test   # 只出某一種
    python gen_gold_probs.py --author gemini --reviewer deepseek
    python gen_gold_probs.py --no-review                # 跳過盲審（只有在趕時間時用）

輸出：JSONL，每行一題，欄位和 JevBench 的 datasets/public/hard.jsonl 相同，
      可以直接餵給 deep9_calibration.py 之類的評測程式算 TVD。
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from fractions import Fraction
from math import comb
from pathlib import Path

from pydantic import BaseModel, Field

from llm_models import deepseek, gemini, glm, kimi, qwen

MODELS = {
    "gemini": lambda: gemini("gemini-3.7-flash"),
    "deepseek": lambda: deepseek("deepseek-flash"),
    "qwen": lambda: qwen("qwen3.8-27b"),
    "glm": lambda: glm("glm-4.7-flash"),
    "kimi": lambda: kimi("kimi-k3"),
}
MODEL_NAMES = {
    "gemini": "gemini-3.7-flash",
    "deepseek": "deepseek-flash",
    "qwen": "qwen3.8-27b",
    "glm": "glm-4.7-flash",
    "kimi": "kimi-k3",
}
# DeepSeek 和 GLM 的端點不吃 json_schema 格式的 response_format（回 400
# "This response_format type is unavailable now"），要改走 tool call。
STRUCTURED_METHOD = {"deepseek": "function_calling", "glm": "function_calling"}


def structured(llm, key, schema):
    method = STRUCTURED_METHOD.get(key)
    return llm.with_structured_output(schema, method=method) if method else llm.with_structured_output(schema)

# gold_probs 和 JevBench 一樣四捨五入到小數 4 位
ROUND = 4


# ---------------------------------------------------------------- 題目結構產生器
# 每個產生器回傳 dict：labels / exact（label -> Fraction）/ expected / rationale /
# surface_answer / why_hard / facts（要寫進情境的數字）/ question（criteria 與 instructions）


def gen_hypergeometric(rng):
    """N 件中 D 件瑕疵，抽 n 件不放回，問至少一件瑕疵。

    P(無瑕疵) = C(N-D, n) / C(N, n)，用 Fraction 保持精確。
    干擾用的舊計畫 n_old 會讓答案跨過 50%，所以抄錯計畫一定答錯。
    """
    for _ in range(200):
        N = rng.randint(10, 24)
        D = rng.randint(2, max(2, N // 3))
        n_new = rng.randint(3, 6)
        n_old = rng.randint(2, n_new - 1)  # 舊計畫至少抽 2 件，干擾才像真的計畫
        if n_new > N - D:
            continue
        p_none_new = Fraction(comb(N - D, n_new), comb(N, n_new))
        p_none_old = Fraction(comb(N - D, n_old), comb(N, n_old))
        p_new, p_old = 1 - p_none_new, 1 - p_none_old
        # 要求兩個計畫落在 50% 的兩側，干擾才有力道
        if (p_new > Fraction(1, 2)) == (p_old > Fraction(1, 2)):
            continue
        if not (Fraction(11, 20) < p_new < Fraction(9, 10) or Fraction(1, 10) < p_new < Fraction(9, 20)):
            continue
        break
    else:
        raise RuntimeError("hypergeometric 參數抽樣失敗")

    exact = {"yes": p_new, "no": p_none_new}
    expected = "yes" if p_new > Fraction(1, 2) else "no"
    surface = "no" if expected == "yes" else "yes"
    rationale = (
        f"The plan in force draws {n_new} units. "
        f"P(no defective) = C({N - D},{n_new})/C({N},{n_new}) = {comb(N - D, n_new)}/{comb(N, n_new)} "
        f"= {p_none_new} = {float(p_none_new):.4f}, so P(at least one) = {p_new} = {float(p_new):.4f}. "
        f"Under the superseded plan ({n_old} units) it would be "
        f"1 - C({N - D},{n_old})/C({N},{n_old}) = {p_old} = {float(p_old):.4f}, "
        f"which is what the inspector's note matches."
    )
    return {
        "family": "probability",
        "kind": "hypergeometric",
        "labels": ["no", "yes"],
        "exact": exact,
        "expected": expected,
        "rationale": rationale,
        "surface_answer": surface,
        "why_hard": (
            "The superseded plan lands on the other side of 50 %; only the in-force plan gives the "
            "right answer, and the arithmetic is sampling without replacement."
        ),
        "facts": {
            "total_units": N,
            "defective_units": D,
            "units_drawn_in_force": n_new,
            "units_drawn_superseded": n_old,
            "misleading_quote_value": f"{float(p_old):.1%}",
        },
        "question": {
            "type": "noul",
            "instructions": (
                "Will the inspection sample contain at least one defective unit (so that the lot is "
                "rejected)? Give probabilities that reflect the evidence in the state."
            ),
            "criteria": {
                "true": "At least one of the sampled units is defective.",
                "false": "None of the sampled units is defective.",
            },
        },
    }


def gen_bayes_test(rng):
    """盛行率 + 敏感度 + 特異度，問陽性時真的有病的機率（典型的 base rate fallacy）。

    P(有病|陽性) = prev·sens / (prev·sens + (1-prev)·(1-spec))
    """
    for _ in range(200):
        prev = Fraction(rng.choice([1, 2, 3, 5]), rng.choice([100, 200, 500, 1000]))
        sens = Fraction(rng.choice([90, 92, 95, 98]), 100)
        spec = Fraction(rng.choice([90, 92, 95, 96, 98]), 100)
        tp = prev * sens
        fp = (1 - prev) * (1 - spec)
        post = tp / (tp + fp)
        # 要有明顯的 base rate fallacy：後驗遠低於敏感度
        if post > Fraction(1, 2) or post < Fraction(1, 50):
            continue
        break
    else:
        raise RuntimeError("bayes_test 參數抽樣失敗")

    exact = {"yes": post, "no": 1 - post}
    rationale = (
        f"Base rate {prev} = {float(prev):.4f}. "
        f"P(positive and affected) = {prev} x {sens} = {tp} = {float(tp):.6f}. "
        f"P(positive and unaffected) = {1 - prev} x {1 - spec} = {fp} = {float(fp):.6f}. "
        f"P(affected | positive) = {tp}/({tp} + {fp}) = {post} = {float(post):.4f}. "
        f"The quoted sensitivity {float(sens):.0%} is not the answer: most positives come from the "
        f"much larger unaffected group."
    )
    return {
        "family": "probability",
        "kind": "bayes_test",
        "labels": ["no", "yes"],
        "exact": exact,
        "expected": "yes" if post > Fraction(1, 2) else "no",
        "rationale": rationale,
        "surface_answer": "yes",
        "why_hard": (
            "The scenario quotes the sensitivity prominently and someone in the state treats it as the "
            "answer. The base rate makes the true posterior far lower."
        ),
        "facts": {
            "prevalence": f"{prev.numerator} in {prev.denominator}",
            "sensitivity": f"{float(sens):.0%}",
            "specificity": f"{float(spec):.0%}",
            "misleading_quote_value": f"{float(sens):.0%}",
        },
        "question": {
            "type": "noul",
            "instructions": (
                "Given the positive screening result described in the state, is this unit actually "
                "affected? Give probabilities that reflect the evidence in the state."
            ),
            "criteria": {
                "true": "The unit is actually affected.",
                "false": "The unit is not affected; the positive result is a false positive.",
            },
        },
    }


def gen_finite_three_way(rng):
    """有限母體先依條件篩選，再分成三類，問落在哪一類。Choice（3 個標籤）。

    機率是篩選後的人數比例，分母是篩選後的總數（不是原始母體）。
    """
    names = rng.choice([
        (["early", "on_time", "late"], "delivery outcome"),
        (["upheld", "modified", "overturned"], "appeal outcome"),
        (["resolved_first_contact", "escalated", "reopened"], "ticket outcome"),
    ])
    labels, topic = names
    for _ in range(200):
        counts = [rng.randint(1, 9) for _ in labels]
        excluded = rng.randint(2, 12)  # 被條件排除掉的件數
        total = sum(counts)
        if total < 10 or total > 40:
            continue
        fr = [Fraction(c, total) for c in counts]
        if max(fr) < Fraction(2, 5) or min(fr) < Fraction(1, 25):
            continue
        break
    else:
        raise RuntimeError("finite_three_way 參數抽樣失敗")

    exact = dict(zip(labels, fr))
    expected = max(exact, key=lambda k: exact[k])
    # 干擾：把被排除的件數也算進分母
    wrong_total = total + excluded
    wrong = [Fraction(c, wrong_total) for c in counts]
    rationale = (
        f"After applying the stated eligibility filter, {total} cases remain "
        f"({', '.join(f'{c} {l}' for c, l in zip(counts, labels))}); {excluded} cases are excluded and "
        f"must not enter the denominator. "
        + ", ".join(f"P({l}) = {c}/{total} = {f} = {float(f):.4f}" for l, c, f in zip(labels, counts, fr))
        + f". Using the unfiltered total of {wrong_total} instead would give "
        + ", ".join(f"{float(w):.4f}" for w in wrong)
        + ", which is the trap."
    )
    return {
        "family": "probability",
        "kind": "finite_three_way",
        "labels": list(labels),
        "exact": exact,
        "expected": expected,
        "rationale": rationale,
        "surface_answer": expected,
        "why_hard": (
            "The state reports a larger headline population. Only the cases passing the stated filter "
            "belong in the denominator."
        ),
        "facts": {
            "topic": topic,
            "counts_after_filter": dict(zip(labels, counts)),
            "excluded_count": excluded,
            "headline_total": wrong_total,
        },
        "question": {
            "type": "choice",
            "instructions": (
                f"For the next eligible case described in the state, which {topic} will it have? "
                "Give probabilities that reflect the evidence in the state."
            ),
            "criteria": {l: f"The case ends up {l.replace('_', ' ')}." for l in labels},
        },
    }


GENERATORS = {
    "hypergeometric": gen_hypergeometric,
    "bayes_test": gen_bayes_test,
    "finite_three_way": gen_finite_three_way,
}


# ---------------------------------------------------------------- 模型：寫情境、盲審
class Prose(BaseModel):
    state: str = Field(description="The scenario text shown to the system under test. Plain prose, no answer, no probabilities computed for the reader.")


class Review(BaseModel):
    probabilities: dict[str, float] = Field(description="Your independently computed probability for every label. Must sum to 1.")
    derivation: str = Field(description="The arithmetic you used, shown step by step.")


AUTHOR_PROMPT = """You write scenarios for a decision benchmark. Turn the facts below into one realistic business scenario.

Hard requirements:
- Use EVERY number in `facts` exactly as given. Do not round, rename or add numbers that change the arithmetic.
- The scenario must make the probability exactly computable from the text alone, with no outside assumptions. State explicitly anything a solver needs (sampling without replacement, counts are exact, tests are perfectly accurate, which rule is in force, which cases the filter excludes).
- Include the distractor: quote a person in the scenario asserting the WRONG answer ({surface_hint}), phrased as an offhand opinion. Never say it is wrong.
- Do NOT state the correct probability, do NOT show any calculation, and do NOT hint at the answer.
- Write 150-320 words. Open with an ALL-CAPS header line naming the organisation and the case, like a real internal document. Use short labelled lines or a couple of short paragraphs.
- Invent a plausible company, place and reference code. Keep it neutral and non-sensitive.

question the system will be asked: {instructions}
labels: {labels}
why this item is hard (for your framing, do not restate): {why_hard}

facts (JSON):
{facts}
"""

REVIEW_PROMPT = """You are blind-reviewing a probability item for a benchmark. Compute the answer yourself from the scenario alone.

Read the scenario, then give the probability of each label. Work the arithmetic exactly: use fractions, sampling without replacement where stated, and only the population the scenario's rules allow. Watch for a superseded rule or an excluded subgroup; opinions voiced by people in the scenario may be wrong.

labels: {labels}
question: {instructions}
{criteria}

scenario:
{state}
"""


def author_state(llm, llm_key, spec):
    prompt = AUTHOR_PROMPT.format(
        surface_hint=spec["facts"].get("misleading_quote_value", f"that the answer is {spec['surface_answer']}"),
        instructions=spec["question"]["instructions"],
        labels=", ".join(spec["labels"]),
        why_hard=spec["why_hard"],
        facts=json.dumps(spec["facts"], ensure_ascii=False, indent=1),
    )
    return structured(llm, llm_key, Prose).invoke(prompt).state.strip()


def blind_review(llm, llm_key, spec, state):
    crit = "\n".join(f"  {k}: {v}" for k, v in spec["question"]["criteria"].items())
    prompt = REVIEW_PROMPT.format(
        labels=", ".join(spec["labels"]),
        instructions=spec["question"]["instructions"],
        criteria=f"label meanings:\n{crit}",
        state=state,
    )
    return structured(llm, llm_key, Review).invoke(prompt)


def build_item(spec, state, author_key, reviewer_key, reviewed):
    gold = {l: round(float(spec["exact"][l]), ROUND) for l in spec["labels"]}
    item_id = f"gen-{spec['kind']}-{abs(hash(state)) % 10**6:06d}"
    return {
        "expected": spec["expected"],
        "family": spec["family"],
        "group": item_id,
        "id": item_id,
        "labels": spec["labels"],
        "provenance": {
            "approx_state_tokens": len(state) // 4,
            "author_model": MODEL_NAMES[author_key],
            "exact_gold_probs": {l: str(spec["exact"][l]) for l in spec["labels"]},  # 精確分數，未捨入
            "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gold_probs": gold,
            "generator": f"gen_gold_probs.py:{spec['kind']}",
            "label_basis": (
                f"Probabilities computed exactly in Python with fractions.Fraction; prose authored by "
                f"{MODEL_NAMES[author_key]} from fixed facts; "
                + (f"blind cross-model review by {MODEL_NAMES[reviewer_key]} reproduced the gold"
                   if reviewed else "NOT REVIEWED")
            ),
            "rationale": spec["rationale"],
            "surface_answer": spec["surface_answer"],
            "why_hard": spec["why_hard"],
        },
        "question": spec["question"],
        "split": "generated",
        "state": state,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=9, help="要產生幾題（通過盲審的）")
    ap.add_argument("--family", choices=list(GENERATORS), action="append", help="只出這些題型，可重複")
    ap.add_argument("--author", choices=list(MODELS), default="gemini", help="寫情境的模型")
    ap.add_argument("--reviewer", choices=list(MODELS), default="deepseek", help="盲審的模型（請用不同廠牌）")
    ap.add_argument("--tolerance", type=float, default=0.01, help="盲審機率和 gold 的容許差距")
    ap.add_argument("--no-review", action="store_true", help="跳過盲審")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="generated_gold_probs.jsonl")
    args = ap.parse_args()

    if args.author == args.reviewer and not args.no_review:
        sys.exit("出題和盲審要用不同的模型，交叉複核才有意義")

    kinds = args.family or list(GENERATORS)
    rng = random.Random(args.seed)
    author = MODELS[args.author]()
    reviewer = None if args.no_review else MODELS[args.reviewer]()

    items, rejected, attempts = [], [], 0
    while len(items) < args.n and attempts < args.n * 4:
        attempts += 1
        kind = kinds[len(items) % len(kinds)]
        try:
            spec = GENERATORS[kind](rng)
            state = author_state(author, args.author, spec)
        except Exception as e:
            print(f"[出題失敗] {kind}: {type(e).__name__}: {e}", flush=True)
            continue

        gold = {l: float(spec["exact"][l]) for l in spec["labels"]}
        if reviewer is None:
            items.append(build_item(spec, state, args.author, args.reviewer, False))
            print(f"[未審]  {kind}  gold={ {k: round(v, 4) for k, v in gold.items()} }", flush=True)
            continue

        try:
            rev = blind_review(reviewer, args.reviewer, spec, state)
        except Exception as e:
            print(f"[盲審失敗] {kind}: {type(e).__name__}: {e}", flush=True)
            continue
        worst = max(abs(rev.probabilities.get(l, 0.0) - gold[l]) for l in spec["labels"])
        if worst <= args.tolerance:
            items.append(build_item(spec, state, args.author, args.reviewer, True))
            print(f"[通過]  {kind}  最大誤差 {worst:.4f}  gold={ {k: round(v, 4) for k, v in gold.items()} }", flush=True)
        else:
            rejected.append({"kind": kind, "worst": worst, "gold": gold,
                             "reviewer": rev.probabilities, "derivation": rev.derivation[:400], "state": state})
            print(f"[退回]  {kind}  最大誤差 {worst:.4f}  盲審={ {k: round(v, 4) for k, v in rev.probabilities.items()} }", flush=True)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n通過 {len(items)} 題 / 嘗試 {attempts} 次，退回 {len(rejected)} 題 -> {out}")
    if rejected:
        rej = out.with_suffix(".rejected.json")
        rej.write_text(json.dumps(rejected, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"退回的題目（含盲審的推導，可以看是誰算錯）-> {rej}")


if __name__ == "__main__":
    main()
