"""Compare your DecisionBench run against the official imajev-4b record on the leaderboard.

usage:
  python compare_to_official.py <your_run_dir>/summary.json <decision-bench-results>/results/mohit67890__imajev-4b/c9e5f132465da85d31735ec502d5557982671a7d/DecisionBench.json
"""
import json
import sys


def load(path):
    text = open(path).read()
    # the harness may prepend DECISION_BENCH_PROGRESS lines when stdout was redirected
    text = "\n".join(l for l in text.splitlines() if not l.startswith("DECISION_BENCH_PROGRESS"))
    return json.loads(text)


mine, official = load(sys.argv[1]), load(sys.argv[2])
m_views, o_views = mine.get("metrics", mine.get("views", {})), official["views"]

mine_primary = mine.get("benchmark_accuracy_counting_unsupported_as_incorrect", mine.get("primary_accuracy"))
print(f"{'metric':<34}{'mine':>10}{'official':>10}{'diff':>9}")
for label, a, b in [
    ("primary accuracy", mine_primary, official["primary_accuracy"]),
    ("coverage", mine.get("coverage"), official["coverage"]),
    ("expected calibration error", m_views.get("overall", {}).get("expected_calibration_error", mine.get("expected_calibration_error")), official["expected_calibration_error"]),
]:
    if a is None:
        print(f"{label:<34}{'-':>10}{b:>10.4f}")
    else:
        print(f"{label:<34}{a:>10.4f}{b:>10.4f}{a - b:>+9.4f}")
print(f"{'error / unsupported rows':<34}{str(mine.get('error_rows')) + ' / ' + str(mine.get('ineligible_rows', mine.get('unsupported_rows'))):>10}"
      f"{str(official['error_rows']) + ' / ' + str(official['unsupported_rows']):>10}")

print(f"\n{'task':<36}{'rows':>6}{'mine':>8}{'official':>10}{'diff':>8}")
flagged = []
for key in sorted(k for k in o_views if k.startswith("task:")):
    o = o_views[key]
    m = m_views.get(key)
    if m is None:
        print(f"{key[5:]:<36}{o['rows']:>6}{'-':>8}{o['accuracy'] * 100:>10.1f}")
        continue
    d = (m["accuracy"] - o["accuracy"]) * 100
    # a 100-row task moves ~±1 point per row; flag gaps that are more than row noise
    if abs(d) > max(3.0, 200 / o["rows"]):
        flagged.append(key[5:])
    print(f"{key[5:]:<36}{m['rows']:>6}{m['accuracy'] * 100:>8.1f}{o['accuracy'] * 100:>10.1f}{d:>+8.1f}")
print("\ntasks worth a look:", ", ".join(flagged) if flagged else "none")
