"""
T1.1 — ACO vs Naive vs Random + LSTM Ablation: Two Topology Scenarios
═══════════════════════════════════════════════════════════════════════
4-way comparison (Random / First-Fit / ACO-only / ACO+LSTM) across two
cluster topologies:

  Adversarial: 25× price spread ($0.12–$3.00) — extreme case.
  Balanced:    2× price spread ($0.50–$1.00) — paper primary claim.

LSTM ablation: trains one WorkloadPredictor per node on the Alibaba 2018
machine trace (200 warmup rows, staggered offset per node), then passes
the PredictionResult dict to aco_schedule() so the CostEngine can apply
the prediction_factor signal. Compares ACO+LSTM vs ACO-only to isolate
the LSTM contribution to cost routing.

Uses Azure VM size distribution (2/4/8/16/32 cores, seed=42).

Run:
    python -m benchmarks.tier1_aco_vs_naive

Output:
    benchmarks/results/tier1_aco_vs_naive.png
    benchmarks/results/tier1_aco_vs_naive.json
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# ── headless matplotlib ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobRequest,
    NodeArch,
    NodeCostProfile,
    NodeState,
    PredictionResult,
    ResourceRequest,
    WorkloadType,
)
from orchestrator.shared.telemetry import WorkloadProfile, ResourceSample
from orchestrator.control_plane.predictor import WorkloadPredictor
from orchestrator.control_plane.scheduler import aco_schedule, naive_schedule
from benchmarks._helpers import AZURE_JOB_SIZES, TRACE_CSV, RESULTS_DIR, save_results


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


# ── LSTM warmup ───────────────────────────────────────────────────────────────

def _build_lstm_predictions(
    nodes: List[ComputeNode],
    n_warmup_rows: int = 200,
) -> Dict[str, PredictionResult]:
    """
    Train one WorkloadPredictor per node on the Alibaba 2018 machine trace.
    Each node is given a staggered time offset (+20 rows) so their predictors
    see slightly different load patterns, simulating independent node histories.
    Returns a predictions dict ready for aco_schedule(predictors=...).
    """
    df = pd.read_csv(TRACE_CSV)
    cpu_pct = df["cpu_util_percent"].values.astype(float)
    mem_pct = df["mem_util_percent"].values.astype(float)

    predictions: Dict[str, PredictionResult] = {}
    for offset, node in enumerate(nodes):
        predictor = WorkloadPredictor(node_id=node.node_id)
        profile   = WorkloadProfile(workload_name=f"warmup-{node.node_id}")
        start = offset * 20
        end   = min(start + n_warmup_rows, len(cpu_pct) - 1)
        for i in range(start, end):
            profile.add_sample(ResourceSample(
                cpu_cores_used=cpu_pct[i] * 0.01 * node.total_cpu_cores,
                memory_gb_used=mem_pct[i] * 0.01 * node.total_memory_gb,
                duration_s=300.0,
                scheduling_latency_ms=0.0,
            ))
            if profile.has_enough_data:
                predictor.refit_if_needed(profile)
        if profile.has_enough_data:
            predictions[node.node_id] = predictor.predict(profile)
    return predictions


# ── Scenario runner ───────────────────────────────────────────────────────────

def _run_scenario(
    label: str,
    nodes: List[ComputeNode],
    cpu_sizes: List[int],
    predictions: Dict[str, PredictionResult],
) -> Dict:
    """
    4-way comparison: Random / First-Fit / ACO-only / ACO+LSTM.
    predictions: output of _build_lstm_predictions(); if empty, ACO+LSTM = ACO-only.
    """
    rng = random.Random(42)
    cost_map = {n.node_id: n.cost_profile.cost_per_hour_usd for n in nodes}

    naive_costs:    List[float] = []
    aco_costs:      List[float] = []
    aco_lstm_costs: List[float] = []
    random_costs:   List[float] = []

    for i, cpu in enumerate(cpu_sizes):
        job = _job(f"j{i}", float(cpu))

        naive_nid    = naive_schedule(job, nodes)
        aco_nid      = aco_schedule(job, nodes)
        aco_lstm_nid = aco_schedule(job, nodes, predictors=predictions or None)

        capable  = [n for n in nodes
                    if n.total_cpu_cores >= cpu and n.total_memory_gb >= cpu * 2.0]
        rand_nid = rng.choice(capable or nodes).node_id

        naive_costs.append(cost_map[naive_nid])
        aco_costs.append(cost_map[aco_nid])
        aco_lstm_costs.append(cost_map[aco_lstm_nid])
        random_costs.append(cost_map[rand_nid])

    naive_total    = sum(naive_costs)
    aco_total      = sum(aco_costs)
    aco_lstm_total = sum(aco_lstm_costs)
    random_total   = sum(random_costs)

    imp_vs_naive      = (naive_total - aco_total)      / naive_total * 100.0
    imp_lstm_vs_naive = (naive_total - aco_lstm_total) / naive_total * 100.0
    imp_lstm_vs_aco   = (aco_total   - aco_lstm_total) / aco_total   * 100.0 \
                        if aco_total > 0 else 0.0
    imp_vs_random     = (random_total - aco_total)     / random_total * 100.0

    # Summarise per-node confidence from predictions (for ablation table)
    conf_summary = {nid: round(p.confidence, 3) for nid, p in predictions.items()}

    print(f"\n{'='*65}")
    print(f"T1.1 — {label}")
    print(f"{'='*65}")
    mix_str = ", ".join(f"{k}c×{v}" for k, v in sorted(Counter(cpu_sizes).items()))
    print(f"Azure job mix:      {mix_str}")
    prices = sorted({n.cost_profile.cost_per_hour_usd for n in nodes})
    print(f"Node prices:        {['${:.2f}'.format(p) for p in prices]}")
    if conf_summary:
        print(f"LSTM confidence:    { {k: v for k, v in conf_summary.items()} }")
    print(f"\n{'Algorithm':<14} {'Total $/hr':>12} {'vs First-Fit':>13}")
    print("-" * 42)
    print(f"{'Random':<14} ${random_total:>10.2f}")
    print(f"{'First-Fit':<14} ${naive_total:>10.2f}   (baseline)")
    print(f"{'ACO-only':<14} ${aco_total:>10.2f}   {imp_vs_naive:+.1f}%")
    print(f"{'ACO+LSTM':<14} ${aco_lstm_total:>10.2f}   {imp_lstm_vs_naive:+.1f}%  "
          f"(LSTM delta vs ACO: {imp_lstm_vs_aco:+.2f}%)")
    print(f"\nResult: {'PASS ✓' if imp_vs_naive >= 10.0 else 'FAIL ✗'}  "
          f"(ACO {imp_vs_naive:.1f}% vs First-Fit)")

    return {
        "label": label,
        "random_total": random_total,
        "naive_total":  naive_total,
        "aco_total":    aco_total,
        "aco_lstm_total": aco_lstm_total,
        "improvement_vs_naive_pct":      imp_vs_naive,
        "improvement_vs_random_pct":     imp_vs_random,
        "lstm_improvement_vs_naive_pct": imp_lstm_vs_naive,
        "lstm_delta_vs_aco_pct":         imp_lstm_vs_aco,
        "lstm_confidence_per_node":      conf_summary,
        "passed": imp_vs_naive >= 10.0,
        "naive_costs":    naive_costs,
        "aco_costs":      aco_costs,
        "aco_lstm_costs": aco_lstm_costs,
        "random_costs":   random_costs,
        "cpu_sizes": list(cpu_sizes),
    }


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Run T1.1 across two cluster topologies, 4-way comparison (+ LSTM ablation).
    Returns dict with 'adversarial' and 'balanced' scenario results.
    """
    n_calls   = 30
    cpu_sizes = AZURE_JOB_SIZES[:n_calls]

    # Scenario A — adversarial: 25× price spread
    nodes_adversarial = [
        _node("n-exp",   cost=3.00),
        _node("n-mid",   cost=0.80),
        _node("n-cheap", cost=0.12, instance_type=InstanceType.SPOT),
    ]

    # Scenario B — balanced: 2× price spread; paper primary claim
    nodes_balanced = [
        _node("nb-mid",   cost=0.70),
        _node("nb-exp",   cost=1.00),
        _node("nb-cheap", cost=0.50, instance_type=InstanceType.SPOT),
    ]

    # ── LSTM warmup (shared across both scenarios) ─────────────────────────────
    print("\nWarming LSTM predictors on Alibaba 2018 trace (200 rows/node)...")
    preds_adv = _build_lstm_predictions(nodes_adversarial)
    preds_bal = _build_lstm_predictions(nodes_balanced)
    print(f"  Adversarial predictions: {list(preds_adv.keys())}")
    print(f"  Balanced predictions:    {list(preds_bal.keys())}")

    adv = _run_scenario("Adversarial topology (25× spread)", nodes_adversarial,
                        cpu_sizes, predictions=preds_adv)
    bal = _run_scenario("Balanced topology (2× spread — paper primary claim)",
                        nodes_balanced, cpu_sizes, predictions=preds_bal)

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "T1.1 — ACO vs First-Fit vs Random + LSTM Ablation: Two Cluster Topologies",
        fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.48)

    def _bar_subplot(ax, scenario: dict, title: str) -> None:
        labels = ["Random", "First-Fit", "ACO-only", "ACO+LSTM"]
        totals = [scenario["random_total"], scenario["naive_total"],
                  scenario["aco_total"],    scenario["aco_lstm_total"]]
        colors = ["#9467bd", "#e07b54", "#4c72b0", "#2ca02c"]
        bars = ax.bar(labels, totals, color=colors, width=0.55,
                      edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(totals) * 0.01,
                    f"${val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Total Cost $/hr (30 placements)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(totals) * 1.22)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        lstm_delta = scenario["lstm_delta_vs_aco_pct"]
        ax.annotate(
            f"LSTM Δ: {lstm_delta:+.2f}%",
            xy=(3, totals[3]), xytext=(2.5, (totals[2] + totals[3]) / 2 + max(totals) * 0.05),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
            color="#2ca02c", fontsize=9, fontweight="bold",
        )

    def _scatter_subplot(ax, scenario: dict, title: str) -> None:
        cpu_sizes_s  = scenario["cpu_sizes"]
        naive_c      = scenario["naive_costs"]
        aco_c        = scenario["aco_costs"]
        aco_lstm_c   = scenario["aco_lstm_costs"]
        random_c     = scenario["random_costs"]
        n = len(cpu_sizes_s)
        sorted_idx = sorted(range(n), key=lambda i: cpu_sizes_s[i])
        xs = list(range(n))
        ax.scatter(xs, [naive_c[i]    for i in sorted_idx], color="#e07b54",
                   alpha=0.6, s=25, label="First-Fit", zorder=3)
        ax.scatter(xs, [aco_c[i]      for i in sorted_idx], color="#4c72b0",
                   alpha=0.7, s=25, label="ACO-only",  zorder=4)
        ax.scatter(xs, [aco_lstm_c[i] for i in sorted_idx], color="#2ca02c",
                   alpha=0.7, s=15, marker="^", label="ACO+LSTM", zorder=5)
        ax.scatter(xs, [random_c[i]   for i in sorted_idx], color="#9467bd",
                   alpha=0.4, s=15, label="Random", zorder=2)
        tick_step = max(1, n // 10)
        ax.set_xticks(xs[::tick_step])
        ax.set_xticklabels(
            [f"{cpu_sizes_s[sorted_idx[i]]}c" for i in xs[::tick_step]], fontsize=8)
        ax.set_xlabel("Job (sorted by CPU cores)")
        ax.set_ylabel("Chosen node cost ($/hr)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

    _bar_subplot(
        fig.add_subplot(gs[0, 0]), adv,
        f"Adversarial: ACO {adv['improvement_vs_naive_pct']:.1f}% vs First-Fit",
    )
    _bar_subplot(
        fig.add_subplot(gs[0, 1]), bal,
        f"Balanced (paper claim): ACO {bal['improvement_vs_naive_pct']:.1f}% vs First-Fit",
    )
    _scatter_subplot(
        fig.add_subplot(gs[1, 0]), adv, "Per-job cost — adversarial topology",
    )
    _scatter_subplot(
        fig.add_subplot(gs[1, 1]), bal, "Per-job cost — balanced topology",
    )

    out = RESULTS_DIR / "tier1_aco_vs_naive.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved: {out}")

    results = {"adversarial": adv, "balanced": bal}
    strip_keys = {"naive_costs", "aco_costs", "aco_lstm_costs", "random_costs", "cpu_sizes"}
    results_json = {
        k: {kk: vv for kk, vv in v.items() if kk not in strip_keys}
        for k, v in results.items()
    }
    save_results("tier1_aco_vs_naive", results_json)
    return results


if __name__ == "__main__":
    run()
