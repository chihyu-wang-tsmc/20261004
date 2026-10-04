"""把 GSM8K 和 AI2 ARC 的 test split 從 HuggingFace 下載成 jsonl，給 gsm8k_routing.py / ai2arc_routing.py 用。

在 ~/jevk5 底下、jevk5 環境執行（只需要跑一次）：

    python download_benchmarks.py              # 兩個都下載
    python download_benchmarks.py --only gsm8k
    python download_benchmarks.py --only ai2arc

輸出：
    gsm8k/gsm8k_test.jsonl              1319 題（openai/gsm8k 的 main config）
    ai2arc/arc_challenge_test.jsonl     1172 題（allenai/ai2_arc 的 ARC-Challenge）
    ai2arc/arc_easy_test.jsonl          2376 題（allenai/ai2_arc 的 ARC-Easy）

兩個資料集都是 open license（GSM8K、ARC 都是 CC BY-SA 4.0），不像 GPQA 有不准公開的條款。
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

GSM8K_REPO = "openai/gsm8k"
ARC_REPO = "allenai/ai2_arc"
ARC_CONFIGS = {"challenge": "ARC-Challenge", "easy": "ARC-Easy"}


def read_parquet(repo, filename):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(repo, filename, repo_type="dataset")
    return pq.read_table(path).to_pylist()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {len(rows)} 題 → {path}")


def download_gsm8k():
    """GSM8K：answer 欄位是解題過程，最後一行 '#### 18' 是答案（整數）。

    順便存 steps（解題過程的行數，也就是要幾步才算得出來），routing 報表用它分組。
    """
    print(f"下載 {GSM8K_REPO}（main config、test split）")
    rows = []
    for i, row in enumerate(read_parquet(GSM8K_REPO, "main/test-00000-of-00001.parquet")):
        solution, _, final = row["answer"].rpartition("####")
        rows.append(
            {
                "id": f"gsm8k-{i:04d}",
                "question": row["question"].strip(),
                # 去掉 <<16-3-4=9>> 這種 calculator annotation，只留人看得懂的解法
                "solution": re.sub(r"<<[^>]*>>", "", solution).strip(),
                "answer": int(final.strip().replace(",", "")),
                "steps": len([l for l in solution.strip().splitlines() if l.strip()]),
            }
        )
    write_jsonl(os.path.join(HERE, "gsm8k", "gsm8k_test.jsonl"), rows)


def download_ai2arc():
    """AI2 ARC：choices 是 {text, label}，選項數 3–5，label 有 ABCD 也有 1234。

    這裡一律照順序換成 A/B/C/D/E，answerKey 也跟著換成字母，所以題目格式都一致。
    """
    for subset, config in ARC_CONFIGS.items():
        print(f"下載 {ARC_REPO}（{config}、test split）")
        rows = []
        for row in read_parquet(ARC_REPO, f"{config}/test-00000-of-00001.parquet"):
            labels = list(row["choices"]["label"])
            options = [t.strip() for t in row["choices"]["text"]]
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"].strip(),
                    "options": options,
                    "answer": "ABCDE"[labels.index(row["answerKey"])],
                }
            )
        write_jsonl(os.path.join(HERE, "ai2arc", f"arc_{subset}_test.jsonl"), rows)


DOWNLOADERS = {"gsm8k": download_gsm8k, "ai2arc": download_ai2arc}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=DOWNLOADERS, default=None, help="只下載其中一個（預設兩個都下載）")
    args = ap.parse_args()
    for name in [args.only] if args.only else DOWNLOADERS:
        DOWNLOADERS[name]()
