"""
T10 — ACO Ablation: Does Pheromone Learning Add Value Beyond EMA Alone?

Tests whether the ACO colony's cross-call pheromone accumulation contributes
routing benefit over and above the EMA predictor signal alone.

Conditions (2×2 design):
  greedy_only   — Fresh uniform pheromone per job, no predictions (sanity ≈50%)
  aco_only      — Persistent pheromone, no predictions
  greedy_ema    — Fresh uniform pheromone per job, EMA(α=0.5) predictions
  aco_ema       — Persistent pheromone + EMA(α=0.5) predictions (full system)

"Greedy+EMA" = what T5.3 already measures for LC jobs (fast-path argmax).
"ACO+EMA"    = pheromone deposit after each placement, accumulated over 200 jobs.

If aco_ema ≈ greedy_ema → pheromone redundant given EMA (honest limitation).
If aco_ema > greedy_ema → pheromone adds stability/tail improvement.

Metric: % of LC jobs routed to stable nodes (10 seeds × 200 jobs per condition).
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

from benchmarks._helpers import TRACE_CSV, RESULTS_DIR, save_results
from orchestrator.control_plane.predictor import LOOKBACK
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.shared.models import (
    ComputeNode, InstanceType, JobRequest, NodeArch, NodeCostProfile,
    NodeState, PredictionResult, ResourceRequest, WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Constants (mirrors T5.3 / T9) ────────────────────────────────────────────

N_STABLE = 16; N_VOLATILE = 16; N_JOBS = 200; N_SEEDS = 10
WINDOW_SIZE = 100; SEARCH_STRIDE = 10
NODE_CPU_CORES = 32; NODE_MEM_GB = 64.0; NODE_PRICE = 0.80
EMA_ALPHA = 0.5        # optimal from T9
RHO = 0.1; Q = 1.0    # ACO pheromone deposit / evaporation
TAU_MIN = 0.01; TAU_MAX = 10.0


# ── Node / profile helpers ────────────────────────────────────────────────────

def _make_node(nid: str) -> ComputeNode:
    return ComputeNode(
        node_id=nid, arch=NodeArch.X86_64, state=NodeState.HEALTHY,
        total_cpu_cores=float(NODE_CPU_CORES), total_memory_gb=NODE_MEM_GB,
        cost_profile=NodeCostProfile(
            instance_type=InstanceType.ON_DEMAND,
            cost_per_hour_usd=NODE_PRICE, interruption_prob=0.0,
        ),
    )


def _build_profile(name: str, cpu_pcts: np.ndarray) -> WorkloadProfile:
    p = WorkloadProfile(workload_name=name)
    for pct in cpu_pcts:
        p.add_sample(ResourceSample(
            cpu_cores_used=(float(pct) / 100.0) * NODE_CPU_CORES,
            memory_gb_used=16.0, gpu_util_pct=None,
            duration_s=300.0, scheduling_latency_ms=1.0,
        ))
    return p


def _non_overlapping(cands: list, n: int) -> list:
    selected, used = [], []
    for w in cands:
        s, e = w["start"], w["end"]
        if any(not (e <= us or s >= ue) for us, ue in used):
            continue
        selected.append(w.copy()); used.append((s, e))
        if len(selected) == n:
            break
    return selected


def _select_windows(cpu_arr: np.ndarray):
    cands = []
    for start in range(0, len(cpu_arr) - WINDOW_SIZE + 1, SEARCH_STRIDE):
        w = cpu_arr[start:start + WINDOW_SIZE]
        score = float(w.std() + max(w[-10:].mean() - w[-30:-10].mean(), 0.0))
        cands.append({"start": start, "end": start + WINDOW_SIZE,
                      "score": score, "cpu_pcts": w})
    stable = _non_overlapping(
        sorted(cands, key=lambda x: x["score"]), N_STABLE)
    volatile = _non_overlapping(
        sorted(cands, key=lambda x: x["score"], reverse=True), N_VOLATILE)
    for i, w in enumerate(stable):   w["node_id"] = f"node-stable-{i:02d}"
    for i, w in enumerate(volatile): w["node_id"] = f"node-volatile-{i:02d}"
    return stable, volatile


def _lc_job() -> JobRequest:
    return JobRequest(
        job_id=str(uuid.uuid4()),
        workload_type=WorkloadType.LATENCY_CRITICAL,
        resources=ResourceRequest(cpu_cores_min=2.0, memory_gb_min=4.0,
                                  gpu_required=False, gpu_count=1),
        priority=9, preemptible=False,
    )


# ── Predictors ────────────────────────────────────────────────────────────────

def _ema_pred(nid: str, profile: WorkloadProfile) -> PredictionResult:
    h = profile.cpu_cores_history
    ema = float(h[0])
    for v in h[1:]:
        ema = EMA_ALPHA * float(v) + (1.0 - EMA_ALPHA) * ema
    recent_mean = max(float(np.mean(h[-LOOKBACK:])), 1e-3)
    gap = (ema - recent_mean) / recent_mean
    sp = float(np.clip(gap, 0.0, 1.0))
    if profile.burst_factor > 1.5:
        sp = min(sp + 0.2, 1.0)
    return PredictionResult(
        node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=ema, predicted_memory_util=50.0,
        predicted_gpu_util={}, spike_probability=sp, confidence=0.85,
        generated_at=datetime.now(timezone.utc),
    )


# ── Pheromone update (applied externally after each placement) ────────────────

def _update_pheromone(pheromone: dict, chosen: str) -> None:
    """Evaporate all, deposit on chosen node."""
    for nid in pheromone:
        pheromone[nid] = max(TAU_MIN, pheromone[nid] * (1.0 - RHO))
    pheromone[chosen] = min(TAU_MAX, pheromone.get(chosen, 1.0) + Q)


def _fresh_pheromone(nodes: List[ComputeNode]) -> dict:
    return {n.node_id: 1.0 for n in nodes}


# ── Single-seed run ───────────────────────────────────────────────────────────

def _run_seed(nodes, stable_ids, preds_ema, rng) -> dict:
    """
    Returns dict of condition → safe-node count (out of N_JOBS).
    greedy_* conditions reset pheromone for each seed (not each job).
    aco_*    conditions accumulate pheromone across N_JOBS.
    """
    results = {}

    for label, use_ema, use_pheromone in [
        ("greedy_only", False, False),
        ("aco_only",    False, True),
        ("greedy_ema",  True,  False),
        ("aco_ema",     True,  True),
    ]:
        pheromone = _fresh_pheromone(nodes)
        preds = preds_ema if use_ema else {}
        safe_count = 0

        for _ in range(N_JOBS):
            shuffled = nodes.copy(); rng.shuffle(shuffled)
            # Fresh pheromone per call for greedy conditions
            tau = pheromone if use_pheromone else _fresh_pheromone(nodes)
            chosen = aco_schedule(_lc_job(), shuffled,
                                  predictors=preds, node_pheromone=tau)
            if chosen in stable_ids:
                safe_count += 1
            if use_pheromone:
                _update_pheromone(pheromone, chosen)

        results[label] = safe_count / N_JOBS * 100.0

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 70)
    print("T10 — ACO Ablation: Pheromone vs EMA vs Full System")
    print(f"     {N_SEEDS} seeds × {N_JOBS} jobs  |  EMA α={EMA_ALPHA}")
    print("=" * 70)

    df  = pd.read_csv(TRACE_CSV)
    cpu = df["cpu_util_percent"].values
    stable_wins, volatile_wins = _select_windows(cpu)
    all_wins = stable_wins + volatile_wins

    profiles:   Dict[str, WorkloadProfile] = {}
    nodes:      List[ComputeNode] = []
    stable_ids: set = set()

    for w in all_wins:
        nid = w["node_id"]
        profiles[nid] = _build_profile(nid, w["cpu_pcts"])
        nodes.append(_make_node(nid))
        if nid.startswith("node-stable"):
            stable_ids.add(nid)

    preds_ema = {nid: _ema_pred(nid, profiles[nid]) for nid in profiles}

    conditions = ["greedy_only", "aco_only", "greedy_ema", "aco_ema"]
    all_vals   = {c: [] for c in conditions}

    for seed in range(N_SEEDS):
        rng = random.Random(seed * 1337 + 99)
        res = _run_seed(nodes, stable_ids, preds_ema, rng)
        for c in conditions:
            all_vals[c].append(res[c])
        print(f"  Seed {seed:2d}: " +
              "  ".join(f"{c}={res[c]:.0f}%" for c in conditions))

    means = {c: float(np.mean(all_vals[c])) for c in conditions}
    stds  = {c: float(np.std(all_vals[c]))  for c in conditions}

    print(f"\n{'─'*70}")
    print(f"  {'Condition':<16}  {'Mean':>8}  {'Std':>6}")
    print(f"{'─'*70}")
    for c in conditions:
        print(f"  {c:<16}  {means[c]:>7.1f}%  ±{stds[c]:.1f}")

    aco_lift   = means["aco_ema"] - means["greedy_ema"]
    ema_lift   = means["greedy_ema"] - means["greedy_only"]
    phero_lift = means["aco_only"]   - means["greedy_only"]
    print(f"\n  EMA contribution (greedy_ema − greedy_only):  {ema_lift:+.1f}pp")
    print(f"  ACO contribution (aco_only  − greedy_only):  {phero_lift:+.1f}pp")
    print(f"  ACO lift on top of EMA (aco_ema − greedy_ema): {aco_lift:+.1f}pp")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    labels  = ["Greedy\n(no pred)", "ACO\n(no pred)", "Greedy\n+EMA", "ACO\n+EMA"]
    xs      = np.arange(len(conditions))
    vals_m  = [means[c] for c in conditions]
    vals_s  = [stds[c]  for c in conditions]
    colors  = ["#aec7e8", "#1f77b4", "#ffbb78", "#ff7f0e"]

    bars = ax.bar(xs, vals_m, yerr=vals_s, capsize=5, color=colors,
                  edgecolor="white", linewidth=1.5)
    ax.axhline(50.0, color="black", lw=0.8, ls=":", alpha=0.5, label="Random (50%)")
    for bar, m in zip(bars, vals_m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Routing to stable nodes (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"T10 — ACO Ablation: Pheromone vs EMA vs Full System\n"
                 f"({N_SEEDS} seeds × {N_JOBS} LC jobs, 32-node cluster)",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Annotations for the two key deltas
    y_arrow = 95
    ax.annotate("", xy=(xs[2], y_arrow), xytext=(xs[0], y_arrow),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text((xs[0] + xs[2]) / 2, y_arrow + 1.5,
            f"EMA: {ema_lift:+.1f}pp", ha="center", fontsize=8, color="gray")
    ax.annotate("", xy=(xs[3], y_arrow), xytext=(xs[2], y_arrow),
                arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.2))
    ax.text((xs[2] + xs[3]) / 2, y_arrow + 1.5,
            f"ACO: {aco_lift:+.1f}pp", ha="center", fontsize=8, color="darkorange")

    fig.tight_layout()
    png_path = RESULTS_DIR / "tier10_aco_ablation.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    result = {
        "n_seeds": N_SEEDS, "n_jobs": N_JOBS, "ema_alpha": EMA_ALPHA,
        "means": means, "stds": stds, "raw": all_vals,
        "ema_lift_pp": ema_lift,
        "aco_phero_lift_pp": phero_lift,
        "aco_over_ema_lift_pp": aco_lift,
    }
    save_results("tier10_aco_ablation", result)
    return result


if __name__ == "__main__":
    run()
