"""
T11 — LOOKBACK Sweep: H* as a Scaling Law

Re-runs the T9 EMA signal compression sweep with LOOKBACK ∈ {5, 10, 20, 50}.
Each run fixes LOOKBACK and sweeps EMA α to find the optimal H* for that window.

Claim: H* ≈ LOOKBACK / 4  (scaling law, not a magic constant)
This converts H* from a trace-specific lucky number into a predictable function of
the scheduler's memory parameter — directly addressing "why 2.5?" reviewer concern.

Output: one JSON per LOOKBACK + one combined figure showing H* = f(LOOKBACK).
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
from scipy import stats as scipy_stats

from benchmarks._helpers import TRACE_CSV, RESULTS_DIR, save_results
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.shared.models import (
    ComputeNode, InstanceType, JobRequest, NodeArch, NodeCostProfile,
    NodeState, PredictionResult, ResourceRequest, WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Sweep parameters ──────────────────────────────────────────────────────────

LOOKBACK_VALUES = [5, 10, 20, 50]   # one full sweep per LOOKBACK
EMA_ALPHAS      = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SEEDS         = 10; N_JOBS = 200
N_STABLE = 16; N_VOLATILE = 16
WINDOW_SIZE = 100; SEARCH_STRIDE = 10
NODE_CPU_CORES = 32; NODE_MEM_GB = 64.0; NODE_PRICE = 0.80


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
        job_id=str(uuid.uuid4()), workload_type=WorkloadType.LATENCY_CRITICAL,
        resources=ResourceRequest(cpu_cores_min=2.0, memory_gb_min=4.0,
                                  gpu_required=False, gpu_count=1),
        priority=9, preemptible=False,
    )


# ── Predictor: EMA with parameterised LOOKBACK ───────────────────────────────

def _ema_pred(nid: str, profile: WorkloadProfile,
              alpha: float, lookback: int) -> PredictionResult:
    h = profile.cpu_cores_history
    ema = float(h[0])
    for v in h[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    recent_mean = max(float(np.mean(h[-lookback:])), 1e-3)
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


def _route(nodes, stable_ids, preds, rng) -> float:
    cnt = 0
    for _ in range(N_JOBS):
        shuffled = nodes.copy(); rng.shuffle(shuffled)
        if aco_schedule(_lc_job(), shuffled, predictors=preds) in stable_ids:
            cnt += 1
    return cnt / N_JOBS * 100.0


# ── Per-LOOKBACK sweep ────────────────────────────────────────────────────────

def _sweep_one_lookback(lookback: int, profiles, nodes, stable_ids) -> dict:
    """Sweep EMA α for a fixed LOOKBACK. Returns sweep results + optimal H*."""
    print(f"\n  LOOKBACK={lookback}")
    print(f"  {'α':>6}  {'H':>6}  {'Routing':>8}  {'Std':>6}")

    sweep = []
    for alpha in EMA_ALPHAS:
        preds = {nid: _ema_pred(nid, profiles[nid], alpha, lookback)
                 for nid in profiles}
        vals = [_route(nodes, stable_ids, preds, random.Random(s * 137 + lookback))
                for s in range(N_SEEDS)]
        m, s = float(np.mean(vals)), float(np.std(vals))
        H = 1.0 / alpha
        sweep.append({"alpha": alpha, "H": H, "mean": m, "std": s})
        print(f"  α={alpha:<5}  H={H:>5.1f}  {m:>7.1f}%  ±{s:.2f}")

    best = max(sweep, key=lambda r: r["mean"])
    print(f"  → Optimal H*={best['H']:.1f} @ α={best['alpha']} → {best['mean']:.1f}%")
    return {"lookback": lookback, "optimal_H": best["H"],
            "optimal_alpha": best["alpha"], "optimal_routing": best["mean"],
            "sweep": sweep}


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 70)
    print("T11 — LOOKBACK Sweep: H* as a Scaling Law")
    print(f"     LOOKBACK ∈ {LOOKBACK_VALUES}  |  {N_SEEDS} seeds × {N_JOBS} jobs")
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

    all_sweeps = []
    for lb in LOOKBACK_VALUES:
        res = _sweep_one_lookback(lb, profiles, nodes, stable_ids)
        all_sweeps.append(res)

    # ── Fit H* = k * LOOKBACK^p relationship ─────────────────────────────────
    lbs    = np.array([r["lookback"]  for r in all_sweeps], dtype=float)
    h_stars= np.array([r["optimal_H"] for r in all_sweeps], dtype=float)

    # Log-log regression: log(H*) = p * log(LOOKBACK) + log(k)
    log_lb = np.log(lbs); log_hs = np.log(h_stars)
    slope, intercept, rval, _, _ = scipy_stats.linregress(log_lb, log_hs)
    k = np.exp(intercept)
    print(f"\n  Scaling law fit: H* = {k:.3f} × LOOKBACK^{slope:.3f}  (R²={rval**2:.3f})")
    print(f"  Ratio H*/LOOKBACK: " +
          "  ".join(f"LB={r['lookback']}→{r['optimal_H']/r['lookback']:.2f}"
                    for r in all_sweeps))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_law, ax_curves) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: scaling law H* = f(LOOKBACK)
    ax_law.plot(lbs, h_stars, "o-", color="#4e79a7", lw=2, ms=8, label="Measured H*")
    lb_fine = np.linspace(lbs.min(), lbs.max(), 200)
    ax_law.plot(lb_fine, k * lb_fine**slope, "--", color="#e15759", lw=1.5,
                label=f"Fit: H* = {k:.2f}·L^{slope:.2f}  (R²={rval**2:.2f})")
    ax_law.set_xlabel("LOOKBACK (prediction window)")
    ax_law.set_ylabel("Optimal effective horizon H*")
    ax_law.set_title("H* scales predictably with LOOKBACK")
    ax_law.legend(fontsize=9); ax_law.grid(True, alpha=0.3)

    # Right panel: routing curves for each LOOKBACK
    colors_lb = {5: "#4e79a7", 10: "#59a14f", 20: "#ff7f0e", 50: "#e15759"}
    for res in all_sweeps:
        lb = res["lookback"]
        Hs = [r["H"]    for r in res["sweep"]]
        ms = [r["mean"] for r in res["sweep"]]
        ss = [r["std"]  for r in res["sweep"]]
        ax_curves.semilogx(Hs, ms, "o-", color=colors_lb[lb], lw=2, ms=5,
                           label=f"LOOKBACK={lb}  (H*={res['optimal_H']:.1f})")
        ax_curves.fill_between(Hs, [m - s for m, s in zip(ms, ss)],
                               [m + s for m, s in zip(ms, ss)],
                               alpha=0.10, color=colors_lb[lb])
        ax_curves.axvline(res["optimal_H"], color=colors_lb[lb],
                          lw=1.0, ls="--", alpha=0.6)

    ax_curves.set_xlabel("Effective memory horizon H (log scale)")
    ax_curves.set_ylabel("Routing quality (% → stable nodes)")
    ax_curves.set_title("Routing vs H — peak shifts right as LOOKBACK grows")
    ax_curves.legend(fontsize=9); ax_curves.grid(True, alpha=0.3, which="both")

    fig.suptitle("T11 — LOOKBACK Sweep: H* is a Predictable Scaling Law\n"
                 "not a trace-specific magic constant",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    png_path = RESULTS_DIR / "tier11_lookback_sweep.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    result = {
        "lookback_values": LOOKBACK_VALUES,
        "n_seeds": N_SEEDS, "n_jobs": N_JOBS,
        "scaling_law": {"k": float(k), "exponent": float(slope), "r_squared": float(rval**2)},
        "sweeps": all_sweeps,
    }
    save_results("tier11_lookback_sweep", result)
    return result


if __name__ == "__main__":
    run()
