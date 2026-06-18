"""
benchmarks/_helpers.py
──────────────────────
Shared fixtures and utilities used by all benchmark scripts.

  make_batch_job(job_id, cpu_cores, mem_gb)  → dict for submit_job()
  placement_cost(svc, result)                → cost_per_hour_usd of chosen node
  AZURE_JOB_SIZES                            → 500-item reproducible list (seed=42)
  TRACE_CSV                                  → Path to alibaba_machine_usage_300s.csv
  RESULTS_DIR                                → benchmarks/results/ (created on import)
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import List

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRACE_CSV = _HERE.parent / "tests" / "fixtures" / "alibaba_machine_usage_300s.csv"


# ── Azure VM size distribution (AzureTracesForPacking2020, Protean OSDI'20) ───
# Empirical weights derived from 114M VM requests in packing_trace_zone_a_v1.sqlite.
# Reference machine: 48 vCPUs (fraction × 48 → core count).
# Sub-1-core micro-VMs (23.2% of all requests) excluded — not relevant for
# compute scheduling. Remaining ≥1-core requests normalised to 100%.
#
# Observed bimodal distribution (peaks at 2c and 8c):
#   1 core  → 18%   (frac ≈ 0.021)
#   2 core  → 45%   (frac ≈ 0.042)   ← dominant
#   4 core  →  8%   (frac ≈ 0.083)
#   8 core  → 24%   (frac ≈ 0.167)   ← second peak
#  16 core  →  3%   (frac ≈ 0.333)
#  32 core  →  2%   (frac ≈ 0.667)
#
# Citation: Hadary et al., "Protean: VM Allocation Service at Scale",
#           OSDI 2020. Dataset: AzureTracesForPacking2020.

_rng = random.Random(42)
_SIZES: List[int] = [1, 2, 4, 8, 16, 32]
_WEIGHTS: List[float] = [0.18, 0.45, 0.08, 0.24, 0.03, 0.02]

AZURE_JOB_SIZES: List[int] = _rng.choices(_SIZES, weights=_WEIGHTS, k=500)
"""
Pre-generated 500-item list of CPU core counts drawn from the empirical Azure VM
size distribution (AzureTracesForPacking2020, Protean OSDI'20).
Benchmarks index into this list (e.g. AZURE_JOB_SIZES[:30]) for reproducibility.
Memory is always cpu_cores * 2.0 GB.
"""


# ── Job factory ───────────────────────────────────────────────────────────────

def make_batch_job(
    job_id: str,
    cpu_cores: float,
    mem_gb: float,
    priority: int = 50,
) -> dict:
    """
    Return a submit_job()-compatible request dict for a BATCH job.

    Always sets gpu_count=1 (Pydantic ge=1 constraint) and gpu_required=False.
    Priority is clamped to [1, 100].
    """
    return {
        "job_id": job_id,
        "workload_type": "batch",
        "resources": {
            "cpu_cores_min": float(cpu_cores),
            "memory_gb_min": float(mem_gb),
            "gpu_required": False,
            "gpu_count": 1,
        },
        "priority": max(1, min(100, int(priority))),
        "preemptible": True,
    }


# ── Result persistence ────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def save_results(name: str, data: dict) -> None:
    """Persist benchmark numeric results to JSON for paper tables."""
    out = RESULTS_DIR / f"{name}.json"
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "results": data,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)
    print(f"Results saved: {out}")


# ── Cost lookup ───────────────────────────────────────────────────────────────

def placement_cost(svc, result: dict) -> float:
    """
    Return the cost_per_hour_usd of the node chosen in a submit_job() result.
    Returns 0.0 if node_id is missing or result status is not SCHEDULED.
    """
    node_id = result.get("node_id")
    if not node_id or result.get("status") not in ("SCHEDULED",):
        return 0.0
    node = svc.node_state.get(node_id)
    if node is None:
        return 0.0
    return node.cost_profile.cost_per_hour_usd
