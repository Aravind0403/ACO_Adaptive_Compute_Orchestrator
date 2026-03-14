"""
T1.3 — P99 Scheduling Latency Under Burst Load
════════════════════════════════════════════════
"0.08ms at near-zero load" is meaningless without a curve. This benchmark
measures P50 / P90 / P99 scheduling latency across burst sizes of
1 / 10 / 50 / 100 / 200 jobs.

Jobs use fractional cores (0.1 CPU) to eliminate saturation as a confound and
isolate pure algorithm latency. OrchestratorService is reset between burst sizes.

Run:
    python -m benchmarks.tier1_p99_latency

Output:
    benchmarks/results/tier1_p99_latency.png
    Printed: P50 / P90 / P99 table for each burst size
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.control_plane.orchestration_service import OrchestratorService
from benchmarks._helpers import RESULTS_DIR


# ── Benchmark ─────────────────────────────────────────────────────────────────

BURST_SIZES = [1, 10, 50, 100, 200]


def _measure_burst(burst_size: int) -> Dict[str, float]:
    """
    Submit `burst_size` jobs sequentially to a fresh OrchestratorService.
    Returns {"p50", "p90", "p99", "mean", "max"} in milliseconds.

    Uses fractional CPU (0.1 core) so the cluster (104 cores total) can
    comfortably hold 200+ simultaneous allocations without rejection.
    """
    svc = OrchestratorService()
    latencies: List[float] = []

    for i in range(burst_size):
        req = {
            "job_id": f"lat-{i}",
            "workload_type": "batch",
            "resources": {
                "cpu_cores_min": 0.1,
                "memory_gb_min": 0.2,
                "gpu_required": False,
                "gpu_count": 1,
            },
            "priority": 50,
            "preemptible": True,
        }
        t0 = time.perf_counter()
        svc.submit_job(req)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000.0)  # → ms

    arr = np.array(latencies)
    return {
        "p50":  float(np.percentile(arr, 50)),
        "p90":  float(np.percentile(arr, 90)),
        "p99":  float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "max":  float(np.max(arr)),
        "n":    burst_size,
    }


def run() -> dict:
    results: Dict[int, Dict] = {}

    print(f"\n{'='*60}")
    print("T1.3 — P99 Scheduling Latency Under Burst Load")
    print(f"{'='*60}")
    print(f"{'Burst':>8}  {'P50 (ms)':>10}  {'P90 (ms)':>10}  {'P99 (ms)':>10}  {'Max (ms)':>10}")
    print("-" * 55)

    for burst in BURST_SIZES:
        stats = _measure_burst(burst)
        results[burst] = stats
        print(f"{burst:>8}  {stats['p50']:>10.3f}  {stats['p90']:>10.3f}  "
              f"{stats['p99']:>10.3f}  {stats['max']:>10.3f}")

    from aco_core.colony import N_ANTS, N_ITERATIONS, STAGNATION_LIMIT
    print(f"\nColony: N_ANTS={N_ANTS}, N_ITERATIONS={N_ITERATIONS}, "
          f"STAGNATION_LIMIT={STAGNATION_LIMIT}")
    print(f"Note: jobs use 0.1 CPU each (saturation-free) — pure algorithm latency")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("T1.3 — Scheduling Latency Under Burst Load", fontsize=13, fontweight="bold")

    bursts = BURST_SIZES
    p50 = [results[b]["p50"] for b in bursts]
    p90 = [results[b]["p90"] for b in bursts]
    p99 = [results[b]["p99"] for b in bursts]

    ax.plot(bursts, p50, "o-", color="#2ca02c",  linewidth=2, markersize=7, label="P50 (median)")
    ax.plot(bursts, p90, "s-", color="#ff7f0e",  linewidth=2, markersize=7, label="P90")
    ax.plot(bursts, p99, "^-", color="#d62728",  linewidth=2, markersize=7, label="P99")

    # Shaded P50–P99 band
    ax.fill_between(bursts, p50, p99, alpha=0.10, color="#d62728")

    # 10ms SLA reference
    ax.axhline(10.0, linestyle="--", color="purple", linewidth=1.0, alpha=0.7, label="10ms SLA budget")

    # Annotate P99 values
    for b, val in zip(bursts, p99):
        ax.annotate(f"{val:.2f}ms", xy=(b, val), xytext=(2, 6),
                    textcoords="offset points", fontsize=8, color="#d62728")

    ax.set_xscale("log")
    ax.set_xticks(bursts)
    ax.set_xticklabels([str(b) for b in bursts])
    ax.set_xlabel("Burst size (jobs submitted sequentially)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("P50 / P90 / P99 scheduling latency vs burst size")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = RESULTS_DIR / "tier1_p99_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out}")

    return results


if __name__ == "__main__":
    run()
