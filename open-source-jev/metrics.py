"""分類指標（coding_report.py 和 guard_report.py 共用）。分母是 0 的指標一律回 None，不要變成假的 0 或 1。"""

import math
from collections import defaultdict


def confusion(pred, gold):
    """(TP, FN, FP, TN)；pred / gold 都是 bool 的 list，正類 = True。"""
    tp = sum(p and g for p, g in zip(pred, gold))
    fn = sum(not p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    tn = sum(not p and not g for p, g in zip(pred, gold))
    return tp, fn, fp, tn


def metrics(tp, fn, fp, tn):
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "accuracy": (tp + tn) / total if total else None,
    }


def _by_score(scores, gold):
    """由高到低掃過每個不同的分數，產生 (分數, 累計 TP, 累計 FP)：門檻 = 這個分數時判成正類的題數。"""
    pairs = sorted(zip(scores, gold), key=lambda x: -x[0])
    tp = fp = i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        yield pairs[i][0], tp, fp
        i = j


def sweep(scores, gold):
    """[(門檻, TP, FP)]，門檻由高到低：分數 ≥ 門檻就判成正類。"""
    return list(_by_score(scores, gold))


def auroc(scores, gold):
    """Mann-Whitney U（同分給平均名次）。任一類沒有樣本回 None。"""
    n_pos = sum(gold)
    n_neg = len(gold) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_pos = sum(r for r, g in zip(ranks, gold) if g)
    return (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(scores, gold):
    """Average precision（sklearn 的定義）。要和隨機基線（= 正類比例）比，不是和 0.5 比。"""
    n_pos = sum(gold)
    if not n_pos or n_pos == len(gold):
        return None
    ap = prev = 0.0
    for _, tp, fp in _by_score(scores, gold):
        recall = tp / n_pos
        ap += (recall - prev) * tp / (tp + fp)
        prev = recall
    return ap


def tpr_at_fpr(scores, gold, budget):
    """把門檻調到 FPR ≤ budget 時能拿到的最高 TPR，回傳 (TPR, 門檻)。"""
    n_pos = sum(gold)
    n_neg = len(gold) - n_pos
    if not n_pos or not n_neg:
        return None, None
    best = (0.0, None)
    for t, tp, fp in _by_score(scores, gold):
        if fp / n_neg > budget + 1e-12:
            break
        best = (tp / n_pos, t)
    return best


def ece(probs, gold, n_bins=10):
    """二元校準誤差：P(正類) 分 n_bins 箱，每箱 |平均機率 − 實際正類比例| 依題數加權。"""
    if not probs:
        return None
    bins = defaultdict(list)
    for p, g in zip(probs, gold):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, g))
    return sum(len(b) / len(probs) * abs(sum(p for p, _ in b) / len(b) - sum(g for _, g in b) / len(b))
               for b in bins.values())


def wilson(k, n, z=1.96):
    """比例 k/n 的 Wilson 95% 信賴區間；題數少時光看比例會誤判。"""
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
