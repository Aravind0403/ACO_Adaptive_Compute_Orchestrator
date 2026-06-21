"""
benchmarks/tier5_lstm_routing_impact.py
────────────────────────────────────────
T5.1 — LSTM Routing Benefit Under Heterogeneous Node Load

Motivation
───────────
T1.1 showed Δ=0.00% LSTM routing impact on cost, because:
  (a) ACO already finds the cost minimum — LSTM can't go lower
  (b) All nodes trained on the same Alibaba trace segment → near-identical
      spike_probability values → no routing differentiation

This benchmark isolates LSTM's actual contribution by creating the two
conditions T1.1 never had:
  1. Per-node heterogeneous utilisation histories (genuinely different
     spike_probability values per node)
  2. LC workload where spike avoidance matters (spike_penalty_weight=0.5)
  3. Same-cost nodes so the ONLY differentiator is prediction_factor

Setup
──────
Two equally-priced ON_DEMAND nodes, trained on real Alibaba 2018 trace
segments with divergent CPU patterns:

  node-stable   rows 1300–1380: CPU 30.3% ± 3.2%, flat
  node-volatile rows 1065–1145: CPU 36.7% ± 7.1%, rising +19.1pp into spike

Prediction results after LSTM training:
  node-stable   spike_prob ≈ 0.065  (low, flat history)
  node-volatile spike_prob ≈ 0.200  (elevated, rising trend detected)

Experiment
───────────
30 LC jobs × 2 conditions, node list shuffled per trial:

  Condition A — ACO-only (predictors={}):
    prediction_factor = 1.0 for both nodes → η equal → argmax picks
    randomly by list order → ~50% stable

  Condition B — ACO+LSTM (predictors with real spike estimates):
    η(stable) > η(volatile) → argmax always picks stable → 100%

Expected result: ~50pp routing improvement for stable node under LSTM.

Usage
──────
    python -m benchmarks.tier5_lstm_routing_impact
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks._helpers import TRACE_CSV, save_results
from orchestrator.control_plane.predictor import WorkloadPredictor
from orchestrator.control_plane.scheduler import aco_schedule
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
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED       = 42
N_TRIALS   = 30
TOTAL_CORES = 32
TOTAL_MEM   = 64.0   # GB

# Real Alibaba trace windows (identified via rolling-variance analysis)
STABLE_START   = 1300   # rows 1300–1380: CPU 30.3% ± 3.2%, flat
STABLE_END     = 1380
VOLATILE_START = 1065   # rows 1065–1145: CPU 36.7% ± 7.1%, rising +19pp
VOLATILE_END   = 1145


# ── Node factory ───────────────────────────────────────────────────────────────

def _make_node(node_id: str) -> ComputeNode:
    """
    Both nodes: same cost ($0.80/hr ON_DEMAND), same capacity.
    Identical across every CostEngine dimension except prediction_factor.
    This ensures any routing difference comes purely from LSTM predictions.
    """
    return ComputeNode(
        node_id=node_id,
        arch=NodeArch.X86_64,
        state=NodeState.HEALTHY,
        total_cpu_cores=TOTAL_CORES,
        total_memory_gb=TOTAL_MEM,
        gpu_count=0,
        cost_profile=NodeCostProfile(
            instance_type=InstanceType.ON_DEMAND,
            on_demand_price_per_hour=0.80,
            spot_price_per_hour=0.80,
            preemption_probability=0.0,
        ),
    )


# ── Trace → WorkloadProfile ────────────────────────────────────────────────────

def _build_profile(name: str, cpu_pcts) -> WorkloadProfile:
    """Convert a CPU% trace window into a WorkloadProfile for LSTM training."""
    profile = WorkloadProfile(workload_name=name)
    for pct in cpu_pcts:
        profile.add_sample(ResourceSample(
            cpu_cores_used=(pct / 100.0) * TOTAL_CORES,
            memory_gb_used=16.0,
            gpu_util_pct=None,
            duration_s=300.0,
            scheduling_latency_ms=1.0,
        ))
    return profile


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    rng = random.Random(SEED)

    # ── Load real trace windows ────────────────────────────────────────────────
    df  = pd.read_csv(TRACE_CSV)
    cpu = df["cpu_util_percent"].values

    stable_pcts   = cpu[STABLE_START:STABLE_END]
    volatile_pcts = cpu[VOLATILE_START:VOLATILE_END]

    print("=== T5.1 — LSTM Routing Under Heterogeneous Node Load ===\n")
    print(f"Stable   window (rows {STABLE_START}–{STABLE_END}): "
          f"mean={stable_pcts.mean():.1f}%  std={stable_pcts.std():.1f}%  "
          f"range=[{stable_pcts.min():.1f}–{stable_pcts.max():.1f}%]")
    print(f"Volatile window (rows {VOLATILE_START}–{VOLATILE_END}): "
          f"mean={volatile_pcts.mean():.1f}%  std={volatile_pcts.std():.1f}%  "
          f"range=[{volatile_pcts.min():.1f}–{volatile_pcts.max():.1f}%]")
    rise = volatile_pcts[-20:].mean() - volatile_pcts[:20].mean()
    print(f"  Rising trend (last 20 vs first 20): +{rise:.1f}pp\n")

    # ── Build profiles and train per-node LSTM ─────────────────────────────────
    stable_profile   = _build_profile("stable",   stable_pcts)
    volatile_profile = _build_profile("volatile", volatile_pcts)

    print("Training LSTM predictors on trace windows...")
    pred_stable   = WorkloadPredictor(node_id="node-stable")
    pred_volatile = WorkloadPredictor(node_id="node-volatile")

    pred_stable.refit_if_needed(stable_profile)
    pred_volatile.refit_if_needed(volatile_profile)

    result_stable   = pred_stable.predict(stable_profile)
    result_volatile = pred_volatile.predict(volatile_profile)

    print(f"\nPredictor results:")
    print(f"  node-stable   → spike_prob={result_stable.spike_probability:.3f}  "
          f"confidence={result_stable.confidence:.3f}")
    print(f"  node-volatile → spike_prob={result_volatile.spike_probability:.3f}  "
          f"confidence={result_volatile.confidence:.3f}")
    print(f"  spike_prob gap: {result_volatile.spike_probability - result_stable.spike_probability:+.3f}")

    # ── Build nodes ────────────────────────────────────────────────────────────
    node_stable   = _make_node("node-stable")
    node_volatile = _make_node("node-volatile")

    # ── LC job template (2 cores, 4GB RAM) ────────────────────────────────────
    def _lc_job(trial: int) -> JobRequest:
        return JobRequest(
            job_id=f"t5-lc-{trial:03d}",
            workload_type=WorkloadType.LATENCY_CRITICAL,
            resources=ResourceRequest(
                cpu_cores_min=2.0,
                memory_gb_min=4.0,
                gpu_required=False,
                gpu_count=1,
            ),
            priority=8,
            preemptible=False,
        )

    # ── Condition A: ACO-only, no predictions ──────────────────────────────────
    stable_no_pred   = 0
    volatile_no_pred = 0

    for trial in range(N_TRIALS):
        job   = _lc_job(trial)
        nodes = [node_stable, node_volatile]
        rng.shuffle(nodes)   # randomise order: ties go to index-0 so shuffle gives ~50/50
        chosen = aco_schedule(job, nodes, predictors={})
        if chosen == "node-stable":
            stable_no_pred += 1
        else:
            volatile_no_pred += 1

    # ── Condition B: ACO+LSTM, real predictions ────────────────────────────────
    predictions: Dict[str, PredictionResult] = {
        "node-stable":   result_stable,
        "node-volatile": result_volatile,
    }

    stable_with_pred   = 0
    volatile_with_pred = 0

    for trial in range(N_TRIALS):
        job   = _lc_job(trial + N_TRIALS)
        nodes = [node_stable, node_volatile]
        rng.shuffle(nodes)
        chosen = aco_schedule(job, nodes, predictors=predictions)
        if chosen == "node-stable":
            stable_with_pred += 1
        else:
            volatile_with_pred += 1

    # ── Results ────────────────────────────────────────────────────────────────
    pct_stable_no_pred   = stable_no_pred   / N_TRIALS * 100
    pct_stable_with_pred = stable_with_pred / N_TRIALS * 100
    routing_improvement  = pct_stable_with_pred - pct_stable_no_pred

    print(f"\n{'─'*55}")
    print(f"Routing results ({N_TRIALS} LC jobs per condition):")
    print(f"\n  Condition A — ACO-only (no predictions):")
    print(f"    node-stable   chosen: {stable_no_pred:2d}/{N_TRIALS}  ({pct_stable_no_pred:.1f}%)")
    print(f"    node-volatile chosen: {volatile_no_pred:2d}/{N_TRIALS}  ({100-pct_stable_no_pred:.1f}%)")

    print(f"\n  Condition B — ACO+LSTM (real trace predictions):")
    print(f"    node-stable   chosen: {stable_with_pred:2d}/{N_TRIALS}  ({pct_stable_with_pred:.1f}%)")
    print(f"    node-volatile chosen: {volatile_with_pred:2d}/{N_TRIALS}  ({100-pct_stable_with_pred:.1f}%)")

    print(f"\n  Routing improvement: "
          f"{pct_stable_no_pred:.1f}% → {pct_stable_with_pred:.1f}% "
          f"({routing_improvement:+.1f}pp)")
    routing_benefit_shown = routing_improvement >= 30.0
    print(f"  LSTM routing benefit: {'DEMONSTRATED ✓' if routing_benefit_shown else 'WEAK ✗'}")
    print(f"{'─'*55}")

    result = {
        "stable_window":          f"rows {STABLE_START}–{STABLE_END}",
        "volatile_window":        f"rows {VOLATILE_START}–{VOLATILE_END}",
        "stable_cpu_mean_pct":    round(float(stable_pcts.mean()), 2),
        "stable_cpu_std_pct":     round(float(stable_pcts.std()), 2),
        "volatile_cpu_mean_pct":  round(float(volatile_pcts.mean()), 2),
        "volatile_cpu_std_pct":   round(float(volatile_pcts.std()), 2),
        "volatile_rise_pp":       round(float(rise), 2),
        "spike_prob_stable":      round(result_stable.spike_probability, 3),
        "spike_prob_volatile":    round(result_volatile.spike_probability, 3),
        "spike_prob_gap":         round(result_volatile.spike_probability - result_stable.spike_probability, 3),
        "confidence_stable":      round(result_stable.confidence, 3),
        "confidence_volatile":    round(result_volatile.confidence, 3),
        "n_trials":               N_TRIALS,
        "no_pred_stable_pct":     round(pct_stable_no_pred, 1),
        "with_pred_stable_pct":   round(pct_stable_with_pred, 1),
        "routing_improvement_pp": round(routing_improvement, 1),
        "passed":                 routing_benefit_shown,
    }

    save_results("tier5_lstm_routing_impact", result)
    return result


if __name__ == "__main__":
    run()
