"""有標準答案的四個 benchmark：run_bench.py 讓 model 作答、routing_report.py 算成本／正確率。"""

from .ai2arc import AI2ARC
from .aime import AIME
from .base import Benchmark
from .gpqa import GPQA
from .gsm8k import GSM8K

BENCHMARKS = {b.name: b for b in (GPQA, AI2ARC, AIME, GSM8K)}

__all__ = ["BENCHMARKS", "Benchmark"]
