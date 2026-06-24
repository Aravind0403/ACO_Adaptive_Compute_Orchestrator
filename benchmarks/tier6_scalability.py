"""
T6 — Scalability: Scheduling Latency vs Cluster Size (32–1024 nodes)

Measures ACO-Adaptive scheduling latency as the cluster scales from 32 to 1024 nodes.
Calls aco_schedule() directly to isolate pure scheduling time (no API, no async).

Design:
  - Cluster sizes: 32, 64, 128, 256, 512, 1024
  - Heterogeneous nodes: 60% ON_DEMAND x86_64, 25% SPOT ARM64, 15% ON_DEMAND GPU
  - Job mix: 50% LATENCY_CRITICAL (fast path), 50% BATCH (full colony)
  - 100 jobs per cluster size, 5 seeds → 500 scheduling calls per size
  - Metrics: P99 and mean latency per (cluster_size, job_type)
"""

import random
import time
import uuid
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks._helpers import RESULTS_DIR, save_results
from orchestrator.shared.models import (
    ComputeNode, NodeArch, NodeCostProfile, NodeState,
    InstanceType, JobRequest, ResourceRequest, WorkloadType,
)
from orchestrator.control_plane.scheduler import aco_schedule

# ── Constants ─────────────────────────────────────────────────────────────────

CLUSTER_SIZES  = [32, 64, 128, 256, 512, 1024]
N_JOBS_PER_RUN = 100
N_SEEDS        = 5


# ── Node factory ──────────────────────────────────────────────────────────────

def _make_cluster(n_nodes: int, rng: random.Random) -> List[ComputeNode]:
    """
    Build a heterogeneous synthetic cluster of n_nodes.

    Distribution:
      60% ON_DEMAND x86_64  (mid-range, $0.48–$1.20/hr, 16–64 cores)
      25% SPOT      ARM64   (cheap, $0.10–$0.30/hr, 16–32 cores)
      15% ON_DEMAND GPU     (expensive, $1.50–$4.00/hr, 16 cores, 1 A100)
    """
    nodes: List[ComputeNode] = []
    n_od    = int(n_nodes * 0.60)
    n_spot  = int(n_nodes * 0.25)
    n_gpu   = n_nodes - n_od - n_spot

    for i in range(n_od):
        cores = rng.choice([16, 32, 48, 64])
        nodes.append(ComputeNode(
            node_id=f"od-x86-{i:04d}",
            arch=NodeArch.X86_64,
            total_cpu_cores=float(cores),
            total_memory_gb=float(cores * 4),
            cost_profile=NodeCostProfile(
                instance_type=InstanceType.ON_DEMAND,
                cost_per_hour_usd=round(rng.uniform(0.48, 1.20), 2),
                interruption_prob=0.0,
            ),
        ))

    for i in range(n_spot):
        cores = rng.choice([16, 32])
        nodes.append(ComputeNode(
            node_id=f"spot-arm-{i:04d}",
            arch=NodeArch.ARM64,
            total_cpu_cores=float(cores),
            total_memory_gb=float(cores * 4),
            cost_profile=NodeCostProfile(
                instance_type=InstanceType.SPOT,
                cost_per_hour_usd=round(rng.uniform(0.10, 0.30), 2),
                interruption_prob=round(rng.uniform(0.05, 0.30), 2),
            ),
        ))

    for i in range(n_gpu):
        nodes.append(ComputeNode(
            node_id=f"od-gpu-{i:04d}",
            arch=NodeArch.X86_64,
            total_cpu_cores=16.0,
            total_memory_gb=128.0,
            gpu_inventory={"A100": 1},
            gpu_vram_gb={"A100": 80.0},
            cost_profile=NodeCostProfile(
                instance_type=InstanceType.ON_DEMAND,
                cost_per_hour_usd=round(rng.uniform(1.50, 4.00), 2),
                interruption_prob=0.0,
            ),
        ))

    rng.shuffle(nodes)
    return nodes


# ── Job factory ───────────────────────────────────────────────────────────────

def _make_lc_job() -> JobRequest:
    return JobRequest(
        job_id=str(uuid.uuid4()),
        workload_type=WorkloadType.LATENCY_CRITICAL,
        resources=ResourceRequest(
            cpu_cores_min=2.0,
            memory_gb_min=4.0,
            gpu_required=False,
        ),
        priority=90,
        latency_p99_ms=10,
    )


def _make_batch_job() -> JobRequest:
    return JobRequest(
        job_id=str(uuid.uuid4()),
        workload_type=WorkloadType.BATCH,
        resources=ResourceRequest(
            cpu_cores_min=4.0,
            memory_gb_min=8.0,
            gpu_required=False,
        ),
        priority=50,
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    results_by_size: Dict[int, Dict[str, list]] = {}

    for n_nodes in CLUSTER_SIZES:
        lc_latencies: List[float] = []
        batch_latencies: List[float] = []

        for seed in range(N_SEEDS):
            rng = random.Random(seed * 1000 + n_nodes)
            nodes = _make_cluster(n_nodes, rng)
            node_pheromone = {n.node_id: 1.0 for n in nodes}

            for _ in range(N_JOBS_PER_RUN):
                job = _make_lc_job() if rng.random() < 0.5 else _make_batch_job()
                t0 = time.perf_counter()
                try:
                    aco_schedule(
                        job_request=job,
                        available_nodes=nodes,
                        node_pheromone=node_pheromone,
                    )
                except Exception:
                    pass
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if job.workload_type == WorkloadType.LATENCY_CRITICAL:
                    lc_latencies.append(elapsed_ms)
                else:
                    batch_latencies.append(elapsed_ms)

        results_by_size[n_nodes] = {
            "lc_latencies_ms":    lc_latencies,
            "batch_latencies_ms": batch_latencies,
        }
        lc_p99    = float(np.percentile(lc_latencies, 99))    if lc_latencies    else 0.0
        batch_p99 = float(np.percentile(batch_latencies, 99)) if batch_latencies else 0.0
        print(f"  {n_nodes:>5} nodes | LC P99={lc_p99:.3f}ms | Batch P99={batch_p99:.3f}ms"
              f" | n_lc={len(lc_latencies)} n_batch={len(batch_latencies)}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for n_nodes in CLUSTER_SIZES:
        d = results_by_size[n_nodes]
        lc    = d["lc_latencies_ms"]
        batch = d["batch_latencies_ms"]
        rows.append({
            "n_nodes":           n_nodes,
            "lc_p99_ms":         float(np.percentile(lc, 99))   if lc    else None,
            "lc_mean_ms":        float(np.mean(lc))              if lc    else None,
            "batch_p99_ms":      float(np.percentile(batch, 99)) if batch else None,
            "batch_mean_ms":     float(np.mean(batch))           if batch else None,
            "n_lc_calls":        len(lc),
            "n_batch_calls":     len(batch),
        })

    # ── Plot ──────────────────────────────────────────────────────────────────
    xs        = [r["n_nodes"]    for r in rows]
    lc_p99s   = [r["lc_p99_ms"]    for r in rows]
    bat_p99s  = [r["batch_p99_ms"] for r in rows]
    lc_means  = [r["lc_mean_ms"]   for r in rows]
    bat_means = [r["batch_mean_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, lc_p99s,   "o-",  color="#1f77b4", label="LC P99 (fast path)")
    ax.plot(xs, bat_p99s,  "s--", color="#ff7f0e", label="Batch P99 (full colony)")
    ax.plot(xs, lc_means,  "o:",  color="#1f77b4", alpha=0.5, label="LC mean")
    ax.plot(xs, bat_means, "s:",  color="#ff7f0e", alpha=0.5, label="Batch mean")
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10 ms SLA target")
    ax.set_xlabel("Cluster size (nodes)")
    ax.set_ylabel("Scheduling latency (ms)")
    ax.set_title("T6 — ACO-Adaptive Scheduling Latency vs Cluster Scale")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = RESULTS_DIR / "tier6_scalability.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    result = {
        "cluster_sizes":  CLUSTER_SIZES,
        "n_jobs_per_run": N_JOBS_PER_RUN,
        "n_seeds":        N_SEEDS,
        "rows":           rows,
    }
    save_results("tier6_scalability", result)
    return result


if __name__ == "__main__":
    print("T6 — Scalability benchmark: 32 → 1024 nodes")
    print(f"  {N_SEEDS} seeds × {N_JOBS_PER_RUN} jobs × {len(CLUSTER_SIZES)} sizes"
          f" = {N_SEEDS * N_JOBS_PER_RUN * len(CLUSTER_SIZES)} total calls\n")
    data = run()
    print("\nSummary:")
    print(f"{'Nodes':>6}  {'LC P99':>8}  {'Batch P99':>10}  {'LC mean':>8}  {'Batch mean':>10}")
    for r in data["rows"]:
        print(f"{r['n_nodes']:>6}  {r['lc_p99_ms']:>8.3f}ms"
              f"  {r['batch_p99_ms']:>10.3f}ms"
              f"  {r['lc_mean_ms']:>8.3f}ms"
              f"  {r['batch_mean_ms']:>10.3f}ms")
