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

import random
from pathlib import Path
from typing import List

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRACE_CSV = _HERE.parent / "tests" / "fixtures" / "alibaba_machine_usage_300s.csv"


# ── Azure VM size distribution (weighted, seed=42) ────────────────────────────
# Mirrors real Azure VM core count distribution:
#   Standard_B2s  →  2 cores  (40%)
#   Standard_B4ms →  4 cores  (30%)
#   Standard_D8s  →  8 cores  (20%)
#   Standard_D16s → 16 cores  ( 7%)
#   Standard_D32s → 32 cores  ( 3%)

_rng = random.Random(42)
_SIZES: List[int] = [2, 4, 8, 16, 32]
_WEIGHTS: List[float] = [0.40, 0.30, 0.20, 0.07, 0.03]

AZURE_JOB_SIZES: List[int] = _rng.choices(_SIZES, weights=_WEIGHTS, k=500)
"""
Pre-generated 500-item list of CPU core counts drawn from the Azure VM distribution.
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
