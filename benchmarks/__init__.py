"""
Backward compatibility for ``uqlab.benchmarks`` (use ``uqlab.4_evaluation.benchmarks``).
"""

import importlib

_bench = importlib.import_module("uqlab.4_evaluation.benchmarks")
globals().update({k: v for k, v in vars(_bench).items() if not k.startswith("_")})

