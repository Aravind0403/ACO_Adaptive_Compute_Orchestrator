"""
T1.1 — ACO vs Naive Baseline with Real Azure VM Job Sizes
══════════════════════════════════════════════════════════
Addresses the core credibility gap: the "28% improvement" claim was built on
30 identical 1-core batch jobs. This benchmark uses a realistic Azure VM size
distribution (2 / 4 / 8 / 16 / 32 cores) and tests whether ACO still wins.

Run:
    python -m benchmarks.tier1_aco_vs_naive

Output:
    benchmarks/results/tier1_aco_vs_naive.png
    Printed: total costs, improvement %, PASS/FAIL
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List

# ── headless matplotlib ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobRequest,
    NodeArch,
    NodeCostProfile,
    NodeState,
    ResourceRequest,
    WorkloadType,
)
from orchestrator.control_plane.scheduler import aco_schedule, naive_schedule
from benchmarks._helpers import AZURE_JOB_SIZES, RESULTS_DIR


# ── Node factory (mirrors test_scheduler_comparison.py) ───────────────────────

def _node(node_id: str, cost: float, total_cpu: float = 64.0,
          instance_type: InstanceType = InstanceType.ON_DEMAND) -> ComputeNode:
    return ComputeNode(
        node_id=node_id,
        arch=NodeArch.X86_64,
        total_cpu_cores=total_cpu,
        total_memory_gb=256.0,
        cost_profile=NodeCostProfile(
            instance_type=instance_type,
            cost_per_hour_usd=cost,
            interruption_prob=0.15 if instance_type == InstanceType.SPOT else 0.0,
            region="us-east-1",
        ),
    )


def _job(job_id: str, cpu: float) -> JobRequest:
    return JobRequest(
        job_id=job_id,
        workload_type=WorkloadType.BATCH,
        resources=ResourceRequest(
            cpu_cores_min=cpu,
            memory_gb_min=cpu * 2.0,
            gpu_required=False,
            gpu_count=1,
        ),
        priority=50,
        preemptible=True,
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Run T1.1 benchmark. Returns summary dict with keys:
      naive_total, aco_total, improvement_pct, passed
    """
    # 3-node cluster: expensive → medium → cheap (naive picks left-to-right)
    nodes = [
        _node("n-exp",   cost=3.00),
        _node("n-mid",   cost=0.80),
        _node("n-cheap", cost=0.12, instance_type=InstanceType.SPOT),
    ]
    cost_map = {n.node_id: n.cost_profile.cost_per_hour_usd for n in nodes}

    n_calls = 30
    cpu_sizes = AZURE_JOB_SIZES[:n_calls]

    naive_costs: List[float] = []
    aco_costs:   List[float] = []
    naive_nodes: List[str]  = []
    aco_nodes:   List[str]  = []

    for i, cpu in enumerate(cpu_sizes):
        job = _job(f"j{i}", float(cpu))
        # Stateless calls: both algorithms see the same fresh node objects
        naive_nid = naive_schedule(job, nodes)
        aco_nid   = aco_schedule(job, nodes)

        naive_costs.append(cost_map[naive_nid])
        aco_costs.append(cost_map[aco_nid])
        naive_nodes.append(naive_nid)
        aco_nodes.append(aco_nid)

    naive_total = sum(naive_costs)
    aco_total   = sum(aco_costs)
    improvement = (naive_total - aco_total) / naive_total * 100.0
    passed = improvement >= 10.0

    # ── Print summary ─────────────────────────────────────────────────────────
    size_counter = Counter(cpu_sizes)
    mix_str = ", ".join(f"{k}c×{v}" for k, v in sorted(size_counter.items()))
    print(f"\n{'='*60}")
    print(f"T1.1 — ACO vs Naive (Azure VM job sizes)")
    print(f"{'='*60}")
    print(f"Azure job mix:  {mix_str}")
    print(f"Naive total:    ${naive_total:.2f}/hr")
    print(f"ACO total:      ${aco_total:.2f}/hr")
    print(f"Improvement:    {improvement:.1f}%")
    print(f"Result:         {'PASS ✓ (≥10%)' if passed else 'FAIL ✗ (<10%)'}")

    naive_node_dist = Counter(naive_nodes)
    aco_node_dist   = Counter(aco_nodes)
    print(f"\nNaive node distribution: {dict(naive_node_dist)}")
    print(f"ACO node distribution:   {dict(aco_node_dist)}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("T1.1 — ACO vs Naive: Azure VM Job Distribution", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: total cost bar chart
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(["Naive (First Fit)", "ACO"], [naive_total, aco_total],
                   color=["#e07b54", "#4c72b0"], width=0.5, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, [naive_total, aco_total]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f"${val:.2f}/hr", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Cost ($/hr across 30 placements)")
    ax1.set_title(f"Total Cost — {improvement:.1f}% improvement")
    ax1.set_ylim(0, max(naive_total, aco_total) * 1.18)
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)

    # Annotate improvement arrow
    ax1.annotate(
        f"−{improvement:.1f}%",
        xy=(1, aco_total), xytext=(0.5, (naive_total + aco_total) / 2),
        arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
        color="green", fontsize=12, fontweight="bold",
    )

    # Right: per-job cost scatter sorted by job cpu size
    ax2 = fig.add_subplot(gs[1])
    sorted_indices = sorted(range(n_calls), key=lambda i: cpu_sizes[i])
    xs = list(range(n_calls))
    naive_y = [naive_costs[i] for i in sorted_indices]
    aco_y   = [aco_costs[i]   for i in sorted_indices]
    cpu_y   = [cpu_sizes[i]   for i in sorted_indices]

    ax2.scatter(xs, naive_y, color="#e07b54", alpha=0.7, s=40, label="Naive", zorder=3)
    ax2.scatter(xs, aco_y,   color="#4c72b0", alpha=0.7, s=40, label="ACO",   zorder=3)

    # X-axis tick labels as cpu sizes
    tick_step = max(1, n_calls // 10)
    ax2.set_xticks(xs[::tick_step])
    ax2.set_xticklabels([f"{cpu_y[i]}c" for i in xs[::tick_step]], fontsize=8)
    ax2.set_xlabel("Job (sorted by CPU cores)")
    ax2.set_ylabel("Chosen node cost ($/hr)")
    ax2.set_title("Per-job placement cost")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)

    # Reference lines for the 3 price points
    for cost, label, color in [(3.00, "$3.00 n-exp", "#d62728"),
                                (0.80, "$0.80 n-mid", "#ff7f0e"),
                                (0.12, "$0.12 n-cheap", "#2ca02c")]:
        ax2.axhline(cost, linestyle="--", linewidth=0.8, color=color, alpha=0.6, label=label)

    ax2.legend(loc="upper left", fontsize=7, ncol=2)

    out = RESULTS_DIR / "tier1_aco_vs_naive.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved: {out}")

    return {
        "naive_total": naive_total,
        "aco_total": aco_total,
        "improvement_pct": improvement,
        "passed": passed,
    }


if __name__ == "__main__":
    run()
