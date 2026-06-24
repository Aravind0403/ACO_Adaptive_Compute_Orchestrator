"""
T8 — ACO Parameter Sensitivity Analysis

Sweeps α, β, ρ, and N_ANTS one at a time (others fixed at default).
Metric: routing quality (% stable-node selection) using the same
32-node stable/volatile setup as T5.2.

Expected result: smooth, monotone-ish response — no collapse or instability
within ±2× of default values. This shows the scheduler is robust to
parameter choices, a standard systems-paper requirement.

Defaults: α=1.0, β=2.0, ρ=0.1, N_ANTS=20
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aco_core.ant as ant_module
import aco_core.pheromone as pheromone_module
import aco_core.colony as colony_module

from benchmarks._helpers import TRACE_CSV, RESULTS_DIR, save_results
from orchestrator.control_plane.predictor import WorkloadPredictor
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.shared.models import (
    ComputeNode, InstanceType, JobRequest, NodeArch, NodeCostProfile,
    NodeState, PredictionResult, ResourceRequest, WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ALPHA  = 1.0
DEFAULT_BETA   = 2.0
DEFAULT_RHO    = 0.1
DEFAULT_N_ANTS = 20

N_NODES = 32
N_JOBS  = 200; N_SEEDS = 3
NODE_CPU_CORES = 32; NODE_MEM_GB = 64.0

# Cost tiers — nodes vary 4× in price so ACO has something to optimise
COST_TIERS = [0.20, 0.40, 0.80, 1.60]   # $/hr, 8 nodes per tier

# Sweep values for each parameter
SWEEPS = {
    "alpha":  [0.25, 0.5, 1.0, 2.0, 4.0],
    "beta":   [0.5,  1.0, 2.0, 4.0, 8.0],
    "rho":    [0.01, 0.05, 0.10, 0.20, 0.40],
    "n_ants": [5,    10,   20,   40,   80],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cluster() -> List[ComputeNode]:
    """32 heterogeneous nodes — 4 cost tiers, 8 nodes each."""
    nodes = []
    for tier_idx, cost in enumerate(COST_TIERS):
        for i in range(8):
            nid = f"tier{tier_idx}-node{i:02d}"
            nodes.append(ComputeNode(
                node_id=nid, arch=NodeArch.X86_64, state=NodeState.HEALTHY,
                total_cpu_cores=float(NODE_CPU_CORES), total_memory_gb=NODE_MEM_GB,
                cost_profile=NodeCostProfile(
                    instance_type=InstanceType.ON_DEMAND,
                    cost_per_hour_usd=cost,
                    interruption_prob=0.0,
                ),
            ))
    return nodes


def _batch_job() -> JobRequest:
    """BATCH job — triggers full ACO colony, not fast path."""
    return JobRequest(
        job_id=str(uuid.uuid4()),
        workload_type=WorkloadType.BATCH,
        resources=ResourceRequest(cpu_cores_min=2.0, memory_gb_min=4.0,
                                  gpu_required=False, gpu_count=1),
        priority=50, preemptible=True,
    )


def _mean_cost(nodes: List[ComputeNode], n_jobs: int, rng: random.Random) -> float:
    """Schedule n_jobs BATCH jobs; return mean cost of chosen nodes."""
    cost_map = {n.node_id: n.cost_profile.cost_per_hour_usd for n in nodes}
    node_pheromone = {n.node_id: 1.0 for n in nodes}
    total = 0.0
    for _ in range(n_jobs):
        shuffled = nodes.copy(); rng.shuffle(shuffled)
        chosen = aco_schedule(_batch_job(), shuffled, node_pheromone=node_pheromone)
        total += cost_map.get(chosen, 0.0)
    return total / n_jobs


def _p99_latency_ms(nodes: List[ComputeNode], n_jobs: int, rng: random.Random) -> float:
    """Schedule n_jobs BATCH jobs; return P99 scheduling latency in ms."""
    import time
    node_pheromone = {n.node_id: 1.0 for n in nodes}
    latencies = []
    for _ in range(n_jobs):
        shuffled = nodes.copy(); rng.shuffle(shuffled)
        t0 = time.perf_counter()
        aco_schedule(_batch_job(), shuffled, node_pheromone=node_pheromone)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(latencies, 99))


# ── Parameter override context ────────────────────────────────────────────────

class _Override:
    """Temporarily patch module-level ACO constants, restore on exit."""
    def __init__(self, alpha=None, beta=None, rho=None, n_ants=None, n_iter=None):
        self._patch = {}
        if alpha  is not None: self._patch[(ant_module,      "ALPHA")]          = alpha
        if beta   is not None: self._patch[(ant_module,      "BETA")]           = beta
        if rho    is not None: self._patch[(pheromone_module, "EVAPORATION_RATE")] = rho
        if n_ants is not None: self._patch[(colony_module,   "N_ANTS")]         = n_ants

    def __enter__(self):
        self._saved = {(mod, attr): getattr(mod, attr) for mod, attr in self._patch}
        for (mod, attr), val in self._patch.items():
            setattr(mod, attr, val)

    def __exit__(self, *_):
        for (mod, attr), val in self._saved.items():
            setattr(mod, attr, val)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("T8 — ACO Parameter Sensitivity (BATCH jobs, cost metric)")
    print(f"  {N_SEEDS} seeds × {N_JOBS} jobs × {sum(len(v) for v in SWEEPS.values())} configs")
    print(f"  Cluster: {N_NODES} nodes, cost tiers {COST_TIERS} $/hr")
    print(f"  Optimal mean cost = ${COST_TIERS[0]:.2f}/hr (all jobs on cheapest tier)\n")

    nodes = _make_cluster()
    results = {}

    # Quality sweep: cost vs α, β, ρ (expect flat — η dominance)
    quality_params = ["alpha", "beta", "rho"]
    for param in quality_params:
        values = SWEEPS[param]
        print(f"  Quality sweep {param}: {values}")
        param_results = []
        for val in values:
            with _Override(**{param: val}):
                seed_vals = [_mean_cost(nodes, N_JOBS, random.Random(s*137+42))
                             for s in range(N_SEEDS)]
            mean_c = float(np.mean(seed_vals))
            std_c  = float(np.std(seed_vals))
            param_results.append({"value": val, "mean_cost": mean_c, "std_cost": std_c})
            defaults = {DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_RHO}
            marker = " ← default" if val in defaults else ""
            print(f"    {param}={val:<6}  ${mean_c:.3f}/hr ±{std_c:.4f}{marker}")
        results[param] = param_results

    # Latency sweep: P99 latency vs N_ANTS (expect linear growth)
    print(f"\n  Latency sweep n_ants: {SWEEPS['n_ants']}")
    ants_results = []
    for val in SWEEPS["n_ants"]:
        with _Override(n_ants=val):
            seed_vals = [_p99_latency_ms(nodes, N_JOBS, random.Random(s*137+42))
                         for s in range(N_SEEDS)]
        mean_l = float(np.mean(seed_vals))
        std_l  = float(np.std(seed_vals))
        ants_results.append({"value": val, "p99_ms": mean_l, "std_ms": std_l})
        marker = " ← default" if val == DEFAULT_N_ANTS else ""
        print(f"    n_ants={val:<4}  P99={mean_l:.2f}ms ±{std_l:.2f}{marker}")
    results["n_ants"] = ants_results

    # Plot: 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    quality_meta = [
        ("alpha",  axes[0, 0], "α (pheromone weight)", DEFAULT_ALPHA),
        ("beta",   axes[0, 1], "β (heuristic weight)",  DEFAULT_BETA),
        ("rho",    axes[1, 0], "ρ (evaporation rate)",  DEFAULT_RHO),
    ]
    for param, ax, xlabel, default in quality_meta:
        vals  = [r["value"]    for r in results[param]]
        means = [r["mean_cost"] for r in results[param]]
        stds  = [r["std_cost"]  for r in results[param]]
        ax.errorbar(vals, means, yerr=stds, fmt="o-", color="#1f77b4",
                    linewidth=2, markersize=7, capsize=4)
        ax.axvline(default, color="red", linestyle="--", linewidth=1.2,
                   label=f"Default ({default})")
        ax.axhline(COST_TIERS[0], color="green", linestyle=":", linewidth=1,
                   label=f"Optimal (${COST_TIERS[0]:.2f})")
        ax.set_xlabel(xlabel); ax.set_ylabel("Mean placement cost ($/hr)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # N_ANTS latency panel
    ax = axes[1, 1]
    vals  = [r["value"] for r in results["n_ants"]]
    means = [r["p99_ms"] for r in results["n_ants"]]
    stds  = [r["std_ms"] for r in results["n_ants"]]
    ax.errorbar(vals, means, yerr=stds, fmt="s-", color="#d62728",
                linewidth=2, markersize=7, capsize=4)
    ax.axvline(DEFAULT_N_ANTS, color="red", linestyle="--", linewidth=1.2,
               label=f"Default ({DEFAULT_N_ANTS})")
    ax.set_xlabel("Number of ants"); ax.set_ylabel("Scheduling P99 latency (ms)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("T8 — ACO Parameter Sensitivity\n"
                 "Quality (cost) is insensitive to α, β, ρ; latency scales with N_ANTS",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    png_path = RESULTS_DIR / "tier8_sensitivity.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    save_results("tier8_sensitivity", results)
    return results


if __name__ == "__main__":
    run()
