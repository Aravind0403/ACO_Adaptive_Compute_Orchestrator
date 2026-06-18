"""
T4 — GPU-Aware Scheduling on Alibaba Production Cluster
════════════════════════════════════════════════════════
3-way comparison (Random / First-Fit / ACO) on real GPU workloads from
Alibaba's heterogeneous production GPU cluster.

Nodes:  openb_node_list_gpu_node.csv   (1213 real GPU nodes, 7 GPU types)
Jobs:   openb_pod_list_gpuspec33.csv   (8152 tasks, 33% with GPU type constraints)

Source: Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads
        with Fragmentation Gradient Descent", USENIX ATC 2023.

QoS mapping:
  LS  (Latency Sensitive)  → WorkloadType.LATENCY_CRITICAL  → ON_DEMAND preference
  BE  (Best Effort)        → WorkloadType.BATCH              → SPOT acceptable
  Burstable / Guaranteed   → WorkloadType.BATCH

GPU type → VRAM (GB):
  V100M32 → 32   V100M16 → 16   A10 → 24
  T4      → 16   P100    → 16   G2  → 32   G3 → 48 (Alibaba internal)

Node pricing (AWS EC2 equivalents as proxy — Alibaba internal prices not published):
  V100M32 → $3.20/hr ON_DEMAND    V100M16 → $2.00/hr ON_DEMAND
  A10     → $1.00/hr ON_DEMAND    T4      → $0.75/hr ON_DEMAND
  P100    → $0.60/hr SPOT         G2      → $0.45/hr SPOT
  G3      → $0.55/hr SPOT

Run:
    python -m benchmarks.tier4_gpu_scheduling

Output:
    benchmarks/results/tier4_gpu_scheduling.png
    benchmarks/results/tier4_gpu_scheduling.json
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

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
from benchmarks._helpers import RESULTS_DIR, save_results

# ── Paths ─────────────────────────────────────────────────────────────────────

_FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
NODE_CSV = _FIXTURES / "openb_node_list_gpu_node.csv"
POD_CSV  = _FIXTURES / "openb_pod_list_gpuspec33.csv"

# ── GPU type metadata ─────────────────────────────────────────────────────────

# VRAM per GPU model in GB
_VRAM_GB: Dict[str, float] = {
    "V100M32": 32.0,
    "V100M16": 16.0,
    "A10":     24.0,
    "T4":      16.0,
    "P100":    16.0,
    "G2":      32.0,   # Alibaba internal — estimated
    "G3":      48.0,   # Alibaba internal — estimated high-end
}

# Cost per hour (AWS EC2 equivalent proxy) and instance type
_GPU_COST: Dict[str, Tuple[float, InstanceType]] = {
    "V100M32": (3.20, InstanceType.ON_DEMAND),
    "V100M16": (2.00, InstanceType.ON_DEMAND),
    "A10":     (1.00, InstanceType.ON_DEMAND),
    "T4":      (0.75, InstanceType.ON_DEMAND),
    "P100":    (0.60, InstanceType.SPOT),
    "G2":      (0.45, InstanceType.SPOT),
    "G3":      (0.55, InstanceType.SPOT),
}

# gpu_spec string → minimum VRAM required (take the lowest-VRAM option in the list)
def _spec_to_vram(gpu_spec: str) -> Optional[float]:
    """Return minimum VRAM requirement from a pipe-separated gpu_spec string."""
    if not isinstance(gpu_spec, str) or gpu_spec.strip().lower() in ("nan", ""):
        return None
    models = [m.strip() for m in gpu_spec.split("|")]
    vrams = [_VRAM_GB[m] for m in models if m in _VRAM_GB]
    return min(vrams) if vrams else None


# ── Node loader ───────────────────────────────────────────────────────────────

def _load_nodes(n_per_type: int = 5) -> List[ComputeNode]:
    """
    Load a representative sample of GPU nodes from the Alibaba node list.
    Takes up to n_per_type nodes of each GPU model for diversity.
    Nodes with gpu=0 are excluded (CPU-only machines in the original file).
    """
    df = pd.read_csv(NODE_CSV)
    df = df[df["gpu"] > 0].reset_index(drop=True)

    sampled: List[ComputeNode] = []
    rng = random.Random(42)

    for model, group in df.groupby("model"):
        rows = group.sample(min(n_per_type, len(group)), random_state=42)
        cost_per_hour, instance_type = _GPU_COST.get(model, (0.50, InstanceType.SPOT))
        vram = _VRAM_GB.get(model, 16.0)

        for idx, row in enumerate(rows.itertuples()):
            cpu_cores = row.cpu_milli / 1000.0
            memory_gb = row.memory_mib / 1024.0
            gpu_count = int(row.gpu)

            # Scale cost by GPU count (more GPUs → proportionally pricier)
            node_cost = cost_per_hour * gpu_count

            sampled.append(ComputeNode(
                node_id=f"{model.lower()}-{idx:02d}",
                arch=NodeArch.X86_64,
                total_cpu_cores=cpu_cores,
                total_memory_gb=memory_gb,
                gpu_inventory={model: gpu_count},
                gpu_vram_gb={model: vram},
                cost_profile=NodeCostProfile(
                    instance_type=instance_type,
                    cost_per_hour_usd=node_cost,
                    interruption_prob=0.15 if instance_type == InstanceType.SPOT else 0.0,
                    region="cn-hangzhou",
                ),
            ))

    return sampled


# ── Job loader ────────────────────────────────────────────────────────────────

def _load_jobs(n_jobs: int = 100, seed: int = 42) -> List[JobRequest]:
    """
    Load a stratified sample of GPU jobs from openb_pod_list_gpuspec33.csv.
    Filters to: num_gpu >= 1, pod_phase in (Running, Succeeded).
    Stratifies by QoS to preserve the LS/BE ratio.
    """
    df = pd.read_csv(POD_CSV)
    df = df[(df["num_gpu"] >= 1) & (df["pod_phase"].isin(["Running", "Succeeded"]))]

    # Stratified sample: preserve LS/BE ratio
    sampled = (
        df.groupby("qos", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, int(n_jobs * len(g) / len(df)))),
                                   random_state=seed))
        .head(n_jobs)
    )

    jobs: List[JobRequest] = []
    for i, row in enumerate(sampled.itertuples()):
        qos = str(row.qos)
        workload_type = (
            WorkloadType.LATENCY_CRITICAL if qos == "LS"
            else WorkloadType.BATCH
        )
        cpu_cores = max(0.5, row.cpu_milli / 1000.0)
        memory_gb = max(0.5, row.memory_mib / 1024.0)
        gpu_count  = int(row.num_gpu)
        gpu_vram   = _spec_to_vram(str(row.gpu_spec)) if hasattr(row, "gpu_spec") else None

        jobs.append(JobRequest(
            job_id=f"gpu-job-{i:03d}",
            workload_type=workload_type,
            resources=ResourceRequest(
                cpu_cores_min=cpu_cores,
                memory_gb_min=memory_gb,
                gpu_required=True,
                gpu_count=gpu_count,
                gpu_memory_gb=gpu_vram,
            ),
            priority=80 if workload_type == WorkloadType.LATENCY_CRITICAL else 30,
            preemptible=(workload_type == WorkloadType.BATCH),
        ))

    return jobs


# ── Scenario runner ───────────────────────────────────────────────────────────

def _run_scenario(
    nodes: List[ComputeNode],
    jobs: List[JobRequest],
) -> Dict:
    """
    3-way comparison: Random / First-Fit (naive) / ACO on GPU jobs.
    Returns placement counts, costs, and QoS compliance per algorithm.
    """
    rng = random.Random(42)
    cost_map = {n.node_id: n.cost_profile.cost_per_hour_usd for n in nodes}
    instance_map = {n.node_id: n.cost_profile.instance_type for n in nodes}

    results: Dict[str, Dict] = {
        "random":    {"costs": [], "on_demand_ls": 0, "ls_total": 0},
        "naive":     {"costs": [], "on_demand_ls": 0, "ls_total": 0},
        "aco":       {"costs": [], "on_demand_ls": 0, "ls_total": 0},
    }

    for job in jobs:
        is_ls = (job.workload_type == WorkloadType.LATENCY_CRITICAL)

        naive_nid = naive_schedule(job, nodes)
        aco_nid   = aco_schedule(job, nodes)

        # Random: capable nodes (GPU count satisfied)
        capable = [
            n for n in nodes
            if sum(n.gpu_inventory.values()) >= job.resources.gpu_count
            and n.total_cpu_cores >= job.resources.cpu_cores_min
            and n.total_memory_gb >= job.resources.memory_gb_min
        ]
        rand_nid = rng.choice(capable or nodes).node_id

        for algo, nid in [("random", rand_nid), ("naive", naive_nid), ("aco", aco_nid)]:
            results[algo]["costs"].append(cost_map[nid])
            if is_ls:
                results[algo]["ls_total"] += 1
                if instance_map[nid] == InstanceType.ON_DEMAND:
                    results[algo]["on_demand_ls"] += 1

    # Aggregate
    out = {}
    for algo, d in results.items():
        total_cost = sum(d["costs"])
        ls = d["ls_total"]
        out[algo] = {
            "total_cost":        total_cost,
            "mean_cost":         total_cost / len(jobs) if jobs else 0.0,
            "ls_on_demand_pct":  (d["on_demand_ls"] / ls * 100.0) if ls > 0 else 0.0,
            "ls_jobs":           ls,
        }
    return out


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Load Alibaba GPU cluster topology and job trace.
    Run 3-way scheduling comparison and report cost efficiency + QoS compliance.
    """
    nodes = _load_nodes(n_per_type=5)
    jobs  = _load_jobs(n_jobs=100)

    n_nodes = len(nodes)
    n_jobs  = len(jobs)
    ls_jobs = sum(1 for j in jobs if j.workload_type == WorkloadType.LATENCY_CRITICAL)
    be_jobs = n_jobs - ls_jobs

    gpu_types = sorted({list(n.gpu_inventory.keys())[0] for n in nodes})
    node_cost_range = (
        min(n.cost_profile.cost_per_hour_usd for n in nodes),
        max(n.cost_profile.cost_per_hour_usd for n in nodes),
    )

    print(f"\n{'='*65}")
    print("T4 — GPU Scheduling: Alibaba Production Cluster (ATC'23)")
    print(f"{'='*65}")
    print(f"Nodes:      {n_nodes}  ({', '.join(gpu_types)})")
    print(f"Cost range: ${node_cost_range[0]:.2f}–${node_cost_range[1]:.2f}/hr")
    print(f"Jobs:       {n_jobs}  (LS={ls_jobs}, BE/other={be_jobs})")

    scenario = _run_scenario(nodes, jobs)

    aco   = scenario["aco"]
    naive = scenario["naive"]
    rand  = scenario["random"]

    imp_vs_naive  = (naive["total_cost"] - aco["total_cost"]) / naive["total_cost"] * 100
    imp_vs_random = (rand["total_cost"]  - aco["total_cost"]) / rand["total_cost"]  * 100
    ls_lift = aco["ls_on_demand_pct"] - naive["ls_on_demand_pct"]

    print(f"\n{'Algorithm':<12} {'Total cost':>12} {'Mean/job':>10} {'LS→ON_DEMAND':>14}")
    print("-" * 52)
    for label, d in [("Random", rand), ("First-Fit", naive), ("ACO", aco)]:
        print(f"{label:<12} ${d['total_cost']:>10.2f}/hr  "
              f"${d['mean_cost']:>7.2f}/hr  {d['ls_on_demand_pct']:>12.1f}%")

    print(f"\nACO vs First-Fit:  {imp_vs_naive:+.1f}% cost reduction")
    print(f"ACO vs Random:     {imp_vs_random:+.1f}% cost reduction")
    print(f"LS→ON_DEMAND lift: {ls_lift:+.1f}pp  (ACO vs First-Fit)")
    print(f"\nSource: Weng et al., Beware of Fragmentation, USENIX ATC 2023")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("T4 — GPU Scheduling on Alibaba Production Cluster (ATC'23)",
                 fontsize=13, fontweight="bold")
    gs_layout = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # Left: total cost bar chart
    ax1 = fig.add_subplot(gs_layout[0])
    labels  = ["Random", "First-Fit", "ACO"]
    totals  = [rand["total_cost"], naive["total_cost"], aco["total_cost"]]
    colors  = ["#9467bd", "#e07b54", "#4c72b0"]
    bars = ax1.bar(labels, totals, color=colors, width=0.5,
                   edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(totals) * 0.01,
                 f"${val:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax1.set_ylabel("Total placement cost $/hr (100 GPU jobs)")
    ax1.set_title(f"Cost efficiency — ACO {imp_vs_naive:+.1f}% vs First-Fit")
    ax1.set_ylim(0, max(totals) * 1.20)
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.annotate(
        f"−{imp_vs_naive:.1f}%",
        xy=(2, aco["total_cost"]),
        xytext=(1, (naive["total_cost"] + aco["total_cost"]) / 2),
        arrowprops=dict(arrowstyle="->", color="green", lw=1.3),
        color="green", fontsize=10, fontweight="bold",
    )

    # Right: LS → ON_DEMAND compliance
    ax2 = fig.add_subplot(gs_layout[1])
    ls_pcts = [rand["ls_on_demand_pct"], naive["ls_on_demand_pct"], aco["ls_on_demand_pct"]]
    bars2 = ax2.bar(labels, ls_pcts, color=colors, width=0.5,
                    edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars2, ls_pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_ylabel("LS jobs placed on ON_DEMAND nodes (%)")
    ax2.set_title(f"QoS compliance — LS→ON_DEMAND\n(lift: {ls_lift:+.1f}pp ACO vs First-Fit)")
    ax2.set_ylim(0, 115)
    ax2.axhline(100, linestyle="--", color="green", linewidth=0.9,
                alpha=0.6, label="100% target")
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "tier4_gpu_scheduling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out}")

    result = {
        "n_nodes": n_nodes,
        "n_jobs": n_jobs,
        "ls_jobs": ls_jobs,
        "be_jobs": be_jobs,
        "gpu_types": gpu_types,
        "random":  {k: v for k, v in rand.items()},
        "naive":   {k: v for k, v in naive.items()},
        "aco":     {k: v for k, v in aco.items()},
        "improvement_vs_naive_pct":  imp_vs_naive,
        "improvement_vs_random_pct": imp_vs_random,
        "ls_on_demand_lift_pp":      ls_lift,
    }
    save_results("tier4_gpu_scheduling", result)
    return result


if __name__ == "__main__":
    run()
