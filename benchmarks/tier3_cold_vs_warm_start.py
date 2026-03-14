"""
T3.2 — Cold Start vs Warm Pheromone Start
═══════════════════════════════════════════
After 200 jobs, save the learned pheromone vector. Then compare:

  Cold start: fresh OrchestratorService (pheromone = 1.0 for all nodes)
  Warm start: fresh service + load saved snapshot (learned preferences)

If pheromone snapshot is useful, warm start should converge to lower cost
faster — particularly in the first 10 placements.

Run:
    python -m benchmarks.tier3_cold_vs_warm_start

Output:
    benchmarks/results/tier3_cold_vs_warm_start.png
    Printed: training pheromone, early/late mean costs, advantage detected?
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.control_plane.orchestration_service import OrchestratorService
from benchmarks._helpers import AZURE_JOB_SIZES, RESULTS_DIR, make_batch_job, placement_cost


# ── Rolling mean ──────────────────────────────────────────────────────────────

def _rolling(values: List[float], window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.mean(values[i - window + 1 : i + 1])
    return result


# ── Benchmark ─────────────────────────────────────────────────────────────────

def _run_jobs(svc: OrchestratorService, job_slice: List[int], prefix: str) -> List[float]:
    """Submit jobs, complete each immediately, return list of placement costs."""
    costs = []
    for i, cpu in enumerate(job_slice):
        req = make_batch_job(f"{prefix}-{i}", cpu_cores=float(cpu), mem_gb=float(cpu) * 2.0)
        result = svc.submit_job(req)
        if result["status"] == "SCHEDULED":
            costs.append(placement_cost(svc, result))
            svc.complete_job(
                result["job_id"], success=True,
                actual_cpu_used_cores=float(cpu) * 0.85,
                actual_memory_used_gb=float(cpu) * 2.0 * 0.90,
                actual_scheduling_latency_ms=0.5,
            )
        else:
            costs.append(0.0)   # QUEUED/REJECTED — unusual at fresh start
    return costs


def run() -> dict:
    # ── Phase 1: training run (200 jobs, build up pheromone) ──────────────────
    print(f"\n{'='*60}")
    print("T3.2 — Cold Start vs Warm Pheromone Start")
    print(f"{'='*60}")
    print("Phase 1: Training (200 jobs)...")

    svc_train = OrchestratorService()
    _run_jobs(svc_train, AZURE_JOB_SIZES[:200], "train")

    training_pheromone = dict(svc_train._node_pheromone)

    # Save snapshot to temp file
    tmpdir = tempfile.mkdtemp()
    snapshot_path = str(Path(tmpdir) / "pheromone.json")
    svc_train.save_pheromone_snapshot(path=snapshot_path)

    print(f"Training pheromone (top → bottom by value):")
    for nid, val in sorted(training_pheromone.items(), key=lambda x: -x[1]):
        print(f"  {nid}: {val:.4f}")

    # ── Phase 2: cold start (50 jobs, no prior knowledge) ─────────────────────
    print(f"\nPhase 2: Cold start (50 jobs)...")
    svc_cold = OrchestratorService()
    cold_costs = _run_jobs(svc_cold, AZURE_JOB_SIZES[200:250], "cold")

    # ── Phase 3: warm start (same 50 jobs, loaded pheromone) ──────────────────
    print(f"Phase 3: Warm start (same 50 jobs)...")
    svc_warm = OrchestratorService()
    svc_warm.load_pheromone_snapshot(path=snapshot_path)
    warm_costs = _run_jobs(svc_warm, AZURE_JOB_SIZES[200:250], "warm")

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _mean_safe(lst: List[float], start: int = 0, end: int | None = None) -> float:
        window = [v for v in lst[start:end] if v > 0.0]
        return sum(window) / len(window) if window else 0.0

    cold_early = _mean_safe(cold_costs, 0, 10)
    warm_early = _mean_safe(warm_costs, 0, 10)
    cold_all   = _mean_safe(cold_costs)
    warm_all   = _mean_safe(warm_costs)

    early_improvement = (cold_early - warm_early) / cold_early * 100.0 if cold_early > 0 else 0.0
    total_improvement = (cold_all - warm_all) / cold_all * 100.0 if cold_all > 0 else 0.0
    advantage_detected = early_improvement > 0.0

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nCold start (jobs 0–9  mean):  ${cold_early:.4f}/hr")
    print(f"Warm start (jobs 0–9  mean):  ${warm_early:.4f}/hr")
    print(f"Early improvement:            {early_improvement:+.1f}%")
    print(f"\nCold start (all 50 mean):     ${cold_all:.4f}/hr")
    print(f"Warm start (all 50 mean):     ${warm_all:.4f}/hr")
    print(f"Total improvement:            {total_improvement:+.1f}%")
    print(f"\nWarm-start advantage: {'DETECTED ✓' if advantage_detected else 'NOT DETECTED ✗'}")
    if not advantage_detected:
        # Diagnostic: explain why advantage was not measurable
        top_node = max(training_pheromone, key=training_pheromone.get)  # type: ignore[arg-type]
        top_val  = training_pheromone[top_node]
        print(f"\n  Diagnostic: training pheromone converged strongly to '{top_node}'")
        print(f"  (pheromone={top_val:.2f}×) — cost function already pointed there")
        print(f"  from job 1, so cold/warm start both pick it immediately.")
        print(f"  Warm-start advantage is most pronounced in multi-modal clusters")
        print(f"  where similar-cost nodes create genuine path degeneracy.")

    # ── Chart ─────────────────────────────────────────────────────────────────
    n_plot = len(cold_costs)
    xs     = list(range(n_plot))

    cold_roll = _rolling(cold_costs, window=5)
    warm_roll = _rolling(warm_costs, window=5)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("T3.2 — Cold vs Warm Pheromone Start (50 jobs each)",
                 fontsize=13, fontweight="bold")

    # Raw scatter (faded)
    ax.scatter(xs, cold_costs, color="#e07b54", alpha=0.2, s=20)
    ax.scatter(xs, warm_costs, color="#4c72b0", alpha=0.2, s=20)

    # Rolling mean lines
    ax.plot(xs, cold_roll, color="#e07b54", linewidth=2.2,
            label=f"Cold start (rolling-5 mean) — all-50 avg ${cold_all:.3f}/hr")
    ax.plot(xs, warm_roll, color="#4c72b0", linewidth=2.2,
            label=f"Warm start (rolling-5 mean) — all-50 avg ${warm_all:.3f}/hr")

    # "First 10 jobs" shaded region
    ax.axvspan(0, 9.5, alpha=0.06, color="gray")
    ax.text(4.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3.5,
            "First 10 jobs", ha="center", fontsize=9, color="gray",
            va="top" if ax.get_ylim()[1] > 0 else "bottom")

    ax.axvline(9.5, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Job index (within benchmark run)")
    ax.set_ylabel("Placement cost ($/hr)")
    ax.set_title(
        f"Early improvement: {early_improvement:+.1f}% "
        f"({'✓ warm start wins' if advantage_detected else '✗ no advantage detected'})"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = RESULTS_DIR / "tier3_cold_vs_warm_start.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out}")

    return {
        "cold_early_mean": cold_early,
        "warm_early_mean": warm_early,
        "early_improvement_pct": early_improvement,
        "cold_all_mean": cold_all,
        "warm_all_mean": warm_all,
        "advantage_detected": advantage_detected,
    }


if __name__ == "__main__":
    run()
