"""
T7 — Workload Drift: Predictor Adaptation Under Non-Stationary Load

Hypothesis: LSTM should earn its place when workload characteristics shift mid-run.
Simple predictors (persistence, MA) react only to current values; LSTM, once
refitted on drift data, can extrapolate the trend ahead of the current reading.

Design:
  - 32 identical nodes: 16 "drifting" + 16 "safe"
  - Warmup (100 samples): both groups flat at ~25% CPU → predictors cannot distinguish
  - Drift phase (100 steps): drifting nodes rise 25%→75% linearly; safe stay flat
  - Every 10 steps: add telemetry, recompute predictions, schedule 10 LC jobs
  - Metric: % jobs to safe nodes per 10-job window + cumulative over drift phase

Predictors:
  No prediction  — p_i = 1.0 always
  Persistence    — spike_prob from last observed value vs recent mean
  MovingAvg(5)  — spike_prob from 5-step mean vs recent mean
  LSTM           — refit_if_needed every 10 new samples

N_SEEDS seeds for variance.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks._helpers import RESULTS_DIR, save_results
from orchestrator.control_plane.predictor import WorkloadPredictor, LOOKBACK
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.shared.models import (
    ComputeNode, InstanceType, JobRequest, NodeArch, NodeCostProfile,
    NodeState, PredictionResult, ResourceRequest, WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Constants ─────────────────────────────────────────────────────────────────

N_NODES      = 32
N_DRIFT      = 16   # nodes that will start rising after warmup
N_SAFE       = 16   # nodes that stay flat throughout
N_WARMUP     = 100  # telemetry samples per node before drift starts
N_DRIFT_STEPS = 100 # steps in the drift phase
BATCH_SIZE   = 10   # telemetry + scheduling batched every N steps
N_SEEDS      = 5
MA_WINDOW    = 5

NODE_CPU_CORES = 32
NODE_MEM_GB    = 64.0
NODE_PRICE     = 0.80

CPU_BASE   = 25.0   # starting CPU % for all nodes
CPU_PEAK   = 75.0   # peak CPU % for drifting nodes after N_DRIFT_STEPS
CPU_NOISE  = 2.0    # Gaussian noise std


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_node(nid: str) -> ComputeNode:
    return ComputeNode(
        node_id=nid, arch=NodeArch.X86_64, state=NodeState.HEALTHY,
        total_cpu_cores=float(NODE_CPU_CORES), total_memory_gb=NODE_MEM_GB,
        gpu_count=0,
        cost_profile=NodeCostProfile(
            instance_type=InstanceType.ON_DEMAND,
            on_demand_price_per_hour=NODE_PRICE,
            spot_price_per_hour=NODE_PRICE,
            preemption_probability=0.0,
        ),
    )


def _lc_job() -> JobRequest:
    return JobRequest(
        job_id=str(uuid.uuid4()),
        workload_type=WorkloadType.LATENCY_CRITICAL,
        resources=ResourceRequest(
            cpu_cores_min=2.0, memory_gb_min=4.0,
            gpu_required=False, gpu_count=1,
        ),
        priority=9, preemptible=False,
    )


def _add_sample(profile: WorkloadProfile, cpu_pct: float) -> None:
    profile.add_sample(ResourceSample(
        cpu_cores_used=(cpu_pct / 100.0) * NODE_CPU_CORES,
        memory_gb_used=16.0, gpu_util_pct=None,
        duration_s=300.0, scheduling_latency_ms=1.0,
    ))


def _persistence_pred(nid: str, profile: WorkloadProfile) -> PredictionResult:
    history = profile.cpu_cores_history
    last_val = history[-1]
    recent_mean = max(float(np.mean(history[-LOOKBACK:])), 1e-3)
    gap = (last_val - recent_mean) / recent_mean
    spike_prob = max(0.0, min(1.0, gap))
    if profile.burst_factor > 1.5:
        spike_prob = min(spike_prob + 0.2, 1.0)
    return PredictionResult(
        node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=last_val, predicted_memory_util=50.0,
        predicted_gpu_util={}, spike_probability=spike_prob,
        confidence=0.95, generated_at=datetime.now(timezone.utc),
    )


def _ma_pred(nid: str, profile: WorkloadProfile) -> PredictionResult:
    history = profile.cpu_cores_history
    ma_val = float(np.mean(history[-MA_WINDOW:]))
    recent_mean = max(float(np.mean(history[-LOOKBACK:])), 1e-3)
    gap = (ma_val - recent_mean) / recent_mean
    spike_prob = max(0.0, min(1.0, gap))
    if profile.burst_factor > 1.5:
        spike_prob = min(spike_prob + 0.2, 1.0)
    return PredictionResult(
        node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=ma_val, predicted_memory_util=50.0,
        predicted_gpu_util={}, spike_probability=spike_prob,
        confidence=0.85, generated_at=datetime.now(timezone.utc),
    )


# ── Single seed run ───────────────────────────────────────────────────────────

def _run_seed(seed: int) -> Dict[str, List[float]]:
    """
    Returns per-condition list of 10-batch routing percentages over the drift phase.
    Each list has N_DRIFT_STEPS // BATCH_SIZE = 10 values.
    """
    rng = random.Random(seed * 2053 + 7)
    rng_np = np.random.default_rng(seed * 2053 + 7)

    # Build nodes
    drift_ids = [f"drift-{i:02d}" for i in range(N_DRIFT)]
    safe_ids  = [f"safe-{i:02d}"  for i in range(N_SAFE)]
    all_ids   = drift_ids + safe_ids
    nodes     = [_make_node(nid) for nid in all_ids]
    safe_set  = set(safe_ids)

    # Build profiles and LSTM predictors
    profiles:  Dict[str, WorkloadProfile]  = {}
    lstms:     Dict[str, WorkloadPredictor] = {}
    for nid in all_ids:
        profiles[nid] = WorkloadProfile(workload_name=nid)
        lstms[nid]    = WorkloadPredictor(node_id=nid)

    # Warmup: both groups flat at CPU_BASE
    for _ in range(N_WARMUP):
        for nid in all_ids:
            cpu = float(np.clip(
                rng_np.normal(CPU_BASE, CPU_NOISE), 0, 100
            ))
            _add_sample(profiles[nid], cpu)

    # Fit LSTMs on warmup data
    for nid in all_ids:
        lstms[nid].fit(profiles[nid])

    # Drift phase — track routing per batch
    n_batches = N_DRIFT_STEPS // BATCH_SIZE
    batch_results: Dict[str, List[float]] = {
        "no_pred":    [],
        "persistence": [],
        "moving_avg": [],
        "lstm":       [],
    }

    for batch_idx in range(n_batches):
        # 1. Add BATCH_SIZE new telemetry samples
        drift_frac = (batch_idx * BATCH_SIZE) / N_DRIFT_STEPS   # 0→1 over drift phase
        cpu_drift_target = CPU_BASE + (CPU_PEAK - CPU_BASE) * drift_frac

        for step in range(BATCH_SIZE):
            frac = ((batch_idx * BATCH_SIZE) + step) / N_DRIFT_STEPS
            cpu_target = CPU_BASE + (CPU_PEAK - CPU_BASE) * frac
            for nid in drift_ids:
                cpu = float(np.clip(rng_np.normal(cpu_target, CPU_NOISE), 0, 100))
                _add_sample(profiles[nid], cpu)
            for nid in safe_ids:
                cpu = float(np.clip(rng_np.normal(CPU_BASE, CPU_NOISE), 0, 100))
                _add_sample(profiles[nid], cpu)

        # 2. Recompute predictions for each condition
        pers_preds: Dict[str, PredictionResult] = {}
        ma_preds:   Dict[str, PredictionResult] = {}
        lstm_preds: Dict[str, PredictionResult] = {}

        for nid in all_ids:
            lstms[nid].refit_if_needed(profiles[nid])
            pers_preds[nid] = _persistence_pred(nid, profiles[nid])
            ma_preds[nid]   = _ma_pred(nid, profiles[nid])
            lstm_preds[nid] = lstms[nid].predict(profiles[nid])

        # 3. Schedule BATCH_SIZE LC jobs under each condition
        for cond_label, preds in [
            ("no_pred",     {}),
            ("persistence", pers_preds),
            ("moving_avg",  ma_preds),
            ("lstm",        lstm_preds),
        ]:
            safe_count = 0
            for _ in range(BATCH_SIZE):
                shuffled = nodes.copy()
                rng.shuffle(shuffled)
                chosen = aco_schedule(_lc_job(), shuffled, predictors=preds)
                if chosen in safe_set:
                    safe_count += 1
            batch_results[cond_label].append(safe_count / BATCH_SIZE * 100.0)

    return batch_results


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    n_batches = N_DRIFT_STEPS // BATCH_SIZE
    print("T7 — Workload Drift: Predictor Adaptation")
    print(f"  {N_SEEDS} seeds, {N_DRIFT} drifting nodes, {N_SAFE} safe nodes")
    print(f"  Drift: CPU {CPU_BASE}%→{CPU_PEAK}% over {N_DRIFT_STEPS} steps ({n_batches} batches of {BATCH_SIZE})\n")

    n_batches = N_DRIFT_STEPS // BATCH_SIZE
    conditions = ["no_pred", "persistence", "moving_avg", "lstm"]
    all_seed_results = {c: [] for c in conditions}

    for seed in range(N_SEEDS):
        seed_res = _run_seed(seed)
        for c in conditions:
            all_seed_results[c].append(seed_res[c])
        print(f"  Seed {seed}: lstm final-batch={seed_res['lstm'][-1]:.0f}%  "
              f"ma final-batch={seed_res['moving_avg'][-1]:.0f}%  "
              f"pers final-batch={seed_res['persistence'][-1]:.0f}%")

    # Mean routing % per batch across seeds
    mean_by_batch: Dict[str, np.ndarray] = {
        c: np.mean(all_seed_results[c], axis=0) for c in conditions
    }

    # Summary: cumulative first-half vs second-half of drift
    half = n_batches // 2
    print(f"\n{'Condition':<14}  {'Early (b1-{half})':>14}  {'Late (b{half+1}-{n_batches})':>14}  {'Full':>8}")
    for c in conditions:
        vals = mean_by_batch[c]
        early = float(np.mean(vals[:half]))
        late  = float(np.mean(vals[half:]))
        full  = float(np.mean(vals))
        print(f"  {c:<14}  {early:>13.1f}%  {late:>13.1f}%  {full:>7.1f}%")

    # Plot
    xs = np.arange(1, n_batches + 1) * BATCH_SIZE
    colors = {"no_pred": "gray", "persistence": "#1f77b4",
              "moving_avg": "#ff7f0e", "lstm": "#2ca02c"}
    styles = {"no_pred": ":", "persistence": "--", "moving_avg": "-.", "lstm": "-"}
    labels = {"no_pred": "No prediction", "persistence": "Persistence",
              "moving_avg": f"Moving Avg (w={MA_WINDOW})", "lstm": "LSTM"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in conditions:
        ax.plot(xs, mean_by_batch[c], color=colors[c], linestyle=styles[c],
                linewidth=2, marker="o", markersize=5, label=labels[c])
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Drift start")
    ax.axhline(50.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5, label="Random baseline")
    ax.set_xlabel(f"Steps into drift phase (each point = {BATCH_SIZE} jobs)")
    ax.set_ylabel("% jobs routed to safe nodes")
    ax.set_title("T7 — Routing Adaptation Under Workload Drift (25%→75% CPU rise)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = RESULTS_DIR / "tier7_workload_drift.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    result = {
        "n_seeds": N_SEEDS, "n_batches": n_batches,
        "batch_size": BATCH_SIZE, "cpu_base": CPU_BASE, "cpu_peak": CPU_PEAK,
        "mean_by_batch": {c: mean_by_batch[c].tolist() for c in conditions},
    }
    save_results("tier7_workload_drift", result)
    return result


if __name__ == "__main__":
    run()
