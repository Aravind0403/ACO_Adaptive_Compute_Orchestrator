"""
T3.1 — Queue Drain Time Under Sustained Saturation
════════════════════════════════════════════════════
Fill the cluster to capacity, submit 30 more jobs (they queue up), then
drain the queue in cycles as nodes become free. Validates the back-pressure
fix isn't just theoretically correct — the queue actually empties.

Approach:
  1. Submit 1-core jobs until 5 consecutive QUEUED statuses → cluster full
  2. Submit 30 more 1-core jobs → they enter the pending queue
  3. In each drain cycle: complete 4 active jobs, call drain_pending_queue()
  4. Repeat until queue is empty

Run:
    python -m benchmarks.tier3_queue_drain

Output:
    benchmarks/results/tier3_queue_drain.png
    Printed: saturation depth, queue depth, drain cycles, timing
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.control_plane.orchestration_service import OrchestratorService
from benchmarks._helpers import RESULTS_DIR


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    svc = OrchestratorService()

    # ── Phase 1: saturate the cluster ─────────────────────────────────────────
    active_job_ids: List[str] = []
    consecutive_queued = 0
    job_idx = 0

    while consecutive_queued < 5:
        req = {
            "job_id": f"sat-{job_idx}",
            "workload_type": "batch",
            "resources": {
                "cpu_cores_min": 1.0,
                "memory_gb_min": 1.0,
                "gpu_required": False,
                "gpu_count": 1,
            },
            "priority": 50,
            "preemptible": True,
        }
        result = svc.submit_job(req)
        job_idx += 1

        if result["status"] == "SCHEDULED":
            active_job_ids.append(result["job_id"])
            consecutive_queued = 0
        elif result["status"] == "QUEUED":
            consecutive_queued += 1
        else:
            # REJECTED (permanent) or ERROR — stop trying
            break

    n_active = len(active_job_ids)
    print(f"\n{'='*60}")
    print("T3.1 — Queue Drain Under Saturation")
    print(f"{'='*60}")
    print(f"Saturation: {n_active} active jobs holding cluster resources")

    # ── Phase 2: fill the queue ────────────────────────────────────────────────
    queue_job_ids: List[str] = []
    n_queue_target = 30

    for i in range(n_queue_target):
        req = {
            "job_id": f"q-{i}",
            "workload_type": "batch",
            "resources": {
                "cpu_cores_min": 1.0,
                "memory_gb_min": 1.0,
                "gpu_required": False,
                "gpu_count": 1,
            },
            "priority": 50,
            "preemptible": True,
        }
        result = svc.submit_job(req)
        if result["status"] == "QUEUED":
            queue_job_ids.append(req["job_id"])

    initial_depth = svc.get_queue_status()["depth"]
    print(f"Queue filled to: {initial_depth} jobs")

    if initial_depth == 0:
        print("[WARN] Queue is empty — cluster may not be fully saturated. Check node capacity.")
        return {"initial_depth": 0, "drain_cycles": 0, "total_drain_ms": 0.0}

    # ── Phase 3: drain loop ────────────────────────────────────────────────────
    # Each cycle: complete 4 jobs → free 4 cores → drain_pending_queue()
    COMPLETE_PER_CYCLE = 4
    cycle_data: List[Tuple[int, int, float, int]] = []
    # (cycle_num, drained_count, drain_time_ms, remaining_depth)

    cycle = 0
    t_drain_start = time.perf_counter()
    cumulative_ms = 0.0

    while svc.get_queue_status()["depth"] > 0 and active_job_ids:
        cycle += 1

        # Complete some active jobs to free resources
        completed_ids = active_job_ids[:COMPLETE_PER_CYCLE]
        active_job_ids = active_job_ids[COMPLETE_PER_CYCLE:]

        for jid in completed_ids:
            try:
                svc.complete_job(
                    jid, success=True,
                    actual_cpu_used_cores=0.85,
                    actual_memory_used_gb=0.90,
                    actual_scheduling_latency_ms=0.5,
                )
            except Exception:
                pass  # job may not be in active_jobs if state already cleaned

        # Drain the pending queue
        t0 = time.perf_counter()
        outcomes = svc.drain_pending_queue()
        t1 = time.perf_counter()
        drain_ms = (t1 - t0) * 1_000.0
        cumulative_ms += drain_ms

        drained = sum(1 for o in outcomes if o.get("status") == "SCHEDULED")
        remaining = svc.get_queue_status()["depth"]
        cycle_data.append((cycle, drained, drain_ms, remaining))

        # Track newly scheduled jobs as active (so we can complete them later)
        for o in outcomes:
            if o.get("status") == "SCHEDULED" and o.get("job_id"):
                active_job_ids.append(o["job_id"])

        # Safety: avoid infinite loop if queue never drains
        if cycle > 50:
            print("[WARN] Drain loop exceeded 50 cycles — breaking")
            break

    total_drain_ms = (time.perf_counter() - t_drain_start) * 1_000.0
    final_depth = svc.get_queue_status()["depth"]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Drain cycles: {cycle}")
    print(f"{'  Cycle':>8}  {'Drained':>8}  {'Time (ms)':>10}  {'Remaining':>10}")
    print("-" * 45)
    for cyc, drained, t_ms, rem in cycle_data:
        print(f"  {cyc:>6}  {drained:>8}  {t_ms:>10.3f}  {rem:>10}")

    total_drained = initial_depth - final_depth
    drain_rate = total_drained / (total_drain_ms / 1_000.0) if total_drain_ms > 0 else 0.0
    print(f"\nTotal queue drain time:  {total_drain_ms:.1f} ms")
    print(f"Total jobs drained:      {total_drained}")
    print(f"Drain rate:              {drain_rate:.1f} jobs/second")
    print(f"Final queue depth:       {final_depth}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("T3.1 — Queue Drain Under Saturation", fontsize=13, fontweight="bold")

    # Left: queue depth staircase over cycles
    cycles     = [0] + [c[0] for c in cycle_data]
    depths     = [initial_depth] + [c[3] for c in cycle_data]
    drain_mss  = [0.0] + [c[2] for c in cycle_data]

    ax1.step(cycles, depths, where="post", color="#4c72b0", linewidth=2.0,
             label="Queue depth")
    ax1.fill_between(cycles, depths, step="post", alpha=0.15, color="#4c72b0")
    ax1.set_xlabel("Drain cycle")
    ax1.set_ylabel("Queue depth (pending jobs)")
    ax1.set_title(f"Queue depth per drain cycle\n({initial_depth} jobs → {final_depth})")
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=9)

    # Right: drained per cycle bar chart + drain time overlay
    cyc_nums  = [c[0] for c in cycle_data]
    drained_n = [c[1] for c in cycle_data]
    t_mss     = [c[2] for c in cycle_data]

    ax2_twin = ax2.twinx()
    ax2.bar(cyc_nums, drained_n, color="#2ca02c", alpha=0.7, label="Jobs drained")
    ax2_twin.plot(cyc_nums, t_mss, "o-", color="#d62728", linewidth=1.5,
                  markersize=5, label="Drain time (ms)")

    ax2.set_xlabel("Drain cycle")
    ax2.set_ylabel("Jobs drained", color="#2ca02c")
    ax2_twin.set_ylabel("Drain call time (ms)", color="#d62728")
    ax2.set_title(f"Jobs drained & timing per cycle\nRate: {drain_rate:.1f} jobs/s")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Combined legend
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "tier3_queue_drain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out}")

    return {
        "n_active_at_saturation": n_active,
        "initial_depth": initial_depth,
        "final_depth": final_depth,
        "drain_cycles": cycle,
        "total_drain_ms": total_drain_ms,
        "drain_rate_per_s": drain_rate,
    }


if __name__ == "__main__":
    run()
