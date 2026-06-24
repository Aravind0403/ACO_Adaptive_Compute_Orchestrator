"""
T12 — Noise Robustness: EMA vs LSTM Under Measurement Noise

Adds additive Gaussian noise (σ = 0, 2, 5, 10% CPU) to stable/volatile profiles
and measures routing quality for EMA(α=0.5) vs LSTM.

Claim: EMA degrades gracefully under moderate noise; LSTM degrades faster or
erratically because its fixed training distribution doesn't cover the noise regime.

Motivation: reviewers may ask "does your trace happen to be clean? would EMA
still win with realistic sensor noise?" This answers it directly.

Design:
  - Same 32-node cluster as T5.3 / T9
  - Noise injected additively: cpu_noisy = cpu_clean + N(0, σ*μ), clipped to [0, 100]
    where μ = mean of the clean window (fraction of mean, not absolute %)
  - 5 noise levels: σ_rel ∈ {0.0, 0.01, 0.02, 0.05, 0.10}
  - For each noise level: 10 seeds × 200 LC jobs each for EMA and LSTM
  - LSTM re-trained on the noisy profile each seed (honest — it sees what EMA sees)
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
import torch
import torch.nn as nn

from benchmarks._helpers import TRACE_CSV, RESULTS_DIR, save_results
from orchestrator.control_plane.predictor import LOOKBACK, TRAIN_EPOCHS, LEARNING_RATE
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.shared.models import (
    ComputeNode, InstanceType, JobRequest, NodeArch, NodeCostProfile,
    NodeState, PredictionResult, ResourceRequest, WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Experiment constants ──────────────────────────────────────────────────────

NOISE_LEVELS   = [0.0, 0.01, 0.02, 0.05, 0.10]   # fraction of local mean
N_SEEDS        = 10; N_JOBS = 200
N_STABLE = 16; N_VOLATILE = 16
WINDOW_SIZE = 100; SEARCH_STRIDE = 10
NODE_CPU_CORES = 32; NODE_MEM_GB = 64.0; NODE_PRICE = 0.80
EMA_ALPHA = 0.5
HIDDEN    = 32


# ── Tiny LSTM (mirrors T5.3) ─────────────────────────────────────────────────

class _LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=HIDDEN,
                              num_layers=1, batch_first=True)
        self.linear = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def _train_lstm(history: np.ndarray) -> _LSTMModel:
    z = (history - history.mean()) / (history.std() + 1e-6)
    xs, ys = [], []
    for i in range(len(z) - LOOKBACK):
        xs.append(z[i:i + LOOKBACK])
        ys.append(z[i + LOOKBACK])
    if not xs:
        return _LSTMModel()
    X = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
    model = _LSTMModel()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    for _ in range(TRAIN_EPOCHS):
        opt.zero_grad()
        loss_fn(model(X), Y).backward()
        opt.step()
    return model


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


def _inject_noise(cpu_pcts: np.ndarray, sigma_rel: float, rng: np.random.Generator) -> np.ndarray:
    """Additive noise proportional to local mean. Clip to [0, 100]."""
    if sigma_rel == 0.0:
        return cpu_pcts.copy()
    mu  = cpu_pcts.mean()
    noise = rng.normal(0.0, sigma_rel * max(mu, 1.0), size=len(cpu_pcts))
    return np.clip(cpu_pcts + noise, 0.0, 100.0)


def _lc_job() -> JobRequest:
    return JobRequest(
        job_id=str(uuid.uuid4()), workload_type=WorkloadType.LATENCY_CRITICAL,
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


def _lstm_pred(nid: str, profile: WorkloadProfile, model: _LSTMModel) -> PredictionResult:
    h = np.array(profile.cpu_cores_history)
    mu, sigma = h.mean(), h.std() + 1e-6
    z = (h - mu) / sigma
    seq = torch.tensor(z[-LOOKBACK:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred_z = model(seq).item()
    pred_cpu = pred_z * sigma + mu
    recent_mean = max(float(np.mean(h[-LOOKBACK:])), 1e-3)
    gap = (pred_cpu - recent_mean) / recent_mean
    sp = float(np.clip(gap, 0.0, 1.0))
    return PredictionResult(
        node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=pred_cpu, predicted_memory_util=50.0,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 70)
    print("T12 — Noise Robustness: EMA vs LSTM Under Measurement Noise")
    print(f"     σ_rel ∈ {NOISE_LEVELS}  |  {N_SEEDS} seeds × {N_JOBS} jobs")
    print("=" * 70)

    df  = pd.read_csv(TRACE_CSV)
    cpu = df["cpu_util_percent"].values
    stable_wins, volatile_wins = _select_windows(cpu)
    all_wins = stable_wins + volatile_wins

    clean_cpu: Dict[str, np.ndarray] = {}
    nodes:     List[ComputeNode]     = []
    stable_ids: set = set()

    for w in all_wins:
        nid = w["node_id"]
        clean_cpu[nid] = w["cpu_pcts"].astype(float)
        nodes.append(_make_node(nid))
        if nid.startswith("node-stable"):
            stable_ids.add(nid)

    ema_results:  Dict[float, List[float]] = {σ: [] for σ in NOISE_LEVELS}
    lstm_results: Dict[float, List[float]] = {σ: [] for σ in NOISE_LEVELS}

    print(f"\n  {'σ_rel':>7}  {'EMA mean':>9}  {'EMA std':>7}  {'LSTM mean':>10}  {'LSTM std':>8}")
    print(f"  {'─'*60}")

    for sigma in NOISE_LEVELS:
        for seed in range(N_SEEDS):
            rng_np  = np.random.default_rng(seed * 7919 + int(sigma * 1000))
            rng_py  = random.Random(seed * 1337 + int(sigma * 1000))

            # Build noisy profiles
            noisy_profiles: Dict[str, WorkloadProfile] = {}
            for nid, cpct in clean_cpu.items():
                noisy = _inject_noise(cpct, sigma, rng_np)
                noisy_profiles[nid] = _build_profile(nid, noisy)

            # EMA predictions (stateless — no training needed)
            preds_ema = {nid: _ema_pred(nid, noisy_profiles[nid])
                         for nid in noisy_profiles}
            ema_results[sigma].append(
                _route(nodes, stable_ids, preds_ema, rng_py))

            # LSTM: train on noisy history, then predict
            lstm_models = {}
            for nid, prof in noisy_profiles.items():
                h = np.array(prof.cpu_cores_history)
                lstm_models[nid] = _train_lstm(h)
            preds_lstm = {nid: _lstm_pred(nid, noisy_profiles[nid], lstm_models[nid])
                          for nid in noisy_profiles}
            lstm_results[sigma].append(
                _route(nodes, stable_ids, preds_lstm, rng_py))

        em = float(np.mean(ema_results[sigma]))
        es = float(np.std(ema_results[sigma]))
        lm = float(np.mean(lstm_results[sigma]))
        ls = float(np.std(lstm_results[sigma]))
        print(f"  σ={sigma:.2f}   EMA={em:>6.1f}%±{es:.1f}  LSTM={lm:>6.1f}%±{ls:.1f}  "
              f"gap={em-lm:+.1f}pp")

    # ── Plot ──────────────────────────────────────────────────────────────────
    sigma_pcts = [s * 100 for s in NOISE_LEVELS]
    ema_means  = [float(np.mean(ema_results[s]))  for s in NOISE_LEVELS]
    ema_stds   = [float(np.std(ema_results[s]))   for s in NOISE_LEVELS]
    lstm_means = [float(np.mean(lstm_results[s])) for s in NOISE_LEVELS]
    lstm_stds  = [float(np.std(lstm_results[s]))  for s in NOISE_LEVELS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigma_pcts, ema_means,  yerr=ema_stds,  fmt="o-", color="#59a14f",
                lw=2, ms=7, capsize=5, label=f"EMA(α={EMA_ALPHA})")
    ax.errorbar(sigma_pcts, lstm_means, yerr=lstm_stds, fmt="s--", color="#9c755f",
                lw=2, ms=7, capsize=5, label="LSTM (retrained per seed)")
    ax.axhline(50.0, color="black", lw=0.8, ls=":", alpha=0.5, label="Random (50%)")

    ax.fill_between(sigma_pcts,
                    [m - s for m, s in zip(ema_means, ema_stds)],
                    [m + s for m, s in zip(ema_means, ema_stds)],
                    alpha=0.12, color="#59a14f")
    ax.fill_between(sigma_pcts,
                    [m - s for m, s in zip(lstm_means, lstm_stds)],
                    [m + s for m, s in zip(lstm_means, lstm_stds)],
                    alpha=0.12, color="#9c755f")

    ax.set_xlabel("Noise level σ_rel (% of local CPU mean)")
    ax.set_ylabel("Routing to stable nodes (%)")
    ax.set_title(f"T12 — Noise Robustness: EMA vs LSTM\n"
                 f"({N_SEEDS} seeds × {N_JOBS} LC jobs, 32-node cluster)",
                 fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 100)

    fig.tight_layout()
    png_path = RESULTS_DIR / "tier12_noise_robustness.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    result = {
        "noise_levels": NOISE_LEVELS, "n_seeds": N_SEEDS, "n_jobs": N_JOBS,
        "ema_alpha": EMA_ALPHA,
        "ema_means":  {str(s): float(np.mean(ema_results[s]))  for s in NOISE_LEVELS},
        "ema_stds":   {str(s): float(np.std(ema_results[s]))   for s in NOISE_LEVELS},
        "lstm_means": {str(s): float(np.mean(lstm_results[s])) for s in NOISE_LEVELS},
        "lstm_stds":  {str(s): float(np.std(lstm_results[s]))  for s in NOISE_LEVELS},
        "ema_raw":    {str(s): ema_results[s]  for s in NOISE_LEVELS},
        "lstm_raw":   {str(s): lstm_results[s] for s in NOISE_LEVELS},
    }
    save_results("tier12_noise_robustness", result)
    return result


if __name__ == "__main__":
    run()
