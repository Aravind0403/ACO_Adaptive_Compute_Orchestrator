"""
T5.3 — Full Predictor Ablation: Statistical vs ML Models

7-way routing comparison on the same 32-node stable/volatile cluster as T5.2.
Tests whether ML model class or statistical signal class drives routing quality.

Conditions:
  No prediction    — p_i = 1.0 (baseline)
  LSTM             — single-layer LSTM, hidden=32 (existing system)
  GRU              — single-layer GRU, hidden=32 (less gating than LSTM)
  MLP              — flat lag-window → 2-layer MLP, no recurrence
  TCN              — causal Conv1d, learnable smoothing kernel
  Persistence      — spike_prob from last observed value vs recent mean
  Moving Avg (5)   — spike_prob from 5-step mean vs recent mean

Metric: % of LC jobs routed to stable nodes (higher = better)
N=5 seeds × 200 jobs per condition.

Claim this answers: "Is ML the right class, or does any directional signal suffice?"
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
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression as _SKLinear
from statsmodels.tsa.arima.model import ARIMA as _ARIMA

from benchmarks._helpers import TRACE_CSV, RESULTS_DIR, save_results
from orchestrator.control_plane.predictor import LOOKBACK, TRAIN_EPOCHS, LEARNING_RATE
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.shared.models import (
    ComputeNode, InstanceType, JobRequest, NodeArch, NodeCostProfile,
    NodeState, PredictionResult, ResourceRequest, WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

# ── Experiment constants (mirror T5.2) ────────────────────────────────────────
N_STABLE = 16; N_VOLATILE = 16; N_JOBS = 200; N_SEEDS = 10
WINDOW_SIZE = 100; SEARCH_STRIDE = 10
NODE_CPU_CORES = 32; NODE_MEM_GB = 64.0; NODE_PRICE = 0.80
MA_WINDOW = 5
HIDDEN = 32


# ── Alternative PyTorch models ────────────────────────────────────────────────

class _GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru    = nn.GRU(input_size=1, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.linear = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])


class _MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LOOKBACK, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TransformerModel(nn.Module):
    """Single-layer Transformer encoder with linear readout."""
    def __init__(self, d_model: int = 32, nhead: int = 4, dim_ff: int = 64):
        super().__init__()
        self.proj    = nn.Linear(1, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.linear  = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          # (batch, LOOKBACK, d_model)
        x = self.encoder(x)       # (batch, LOOKBACK, d_model)
        return self.linear(x[:, -1, :])   # last timestep → (batch, 1)


class _TCNModel(nn.Module):
    """Causal 1-D convolution — acts like a learnable moving-average kernel."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.pad    = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.conv   = nn.Conv1d(1, HIDDEN, kernel_size=kernel_size)
        self.relu   = nn.ReLU()
        self.linear = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, LOOKBACK, 1) → (batch, 1, LOOKBACK) for Conv1d
        xt      = x.transpose(1, 2)
        padded  = self.pad(xt)
        conv_out = self.relu(self.conv(padded))   # (batch, HIDDEN, LOOKBACK)
        last    = conv_out[:, :, -1]              # (batch, HIDDEN) — causal last step
        return self.linear(last)


# ── Generic predictor wrapper ─────────────────────────────────────────────────

class _GenericPredictor:
    """
    Drop-in replacement for WorkloadPredictor using any PyTorch model class.
    Shares the same fit/predict logic (z-score normalisation, spike_prob formula).
    """

    def __init__(self, node_id: str, model_cls):
        self.node_id   = node_id
        self._model_cls = model_cls
        self._model: Optional[nn.Module] = None
        self._trained  = False
        self._cpu_mean = 0.0
        self._cpu_std  = 1.0

    def fit(self, profile: WorkloadProfile) -> None:
        history = profile.cpu_cores_history
        if len(history) <= LOOKBACK:
            return

        arr  = np.array(history, dtype=np.float64)
        mean = float(arr.mean())
        std  = float(max(arr.std(), 1e-6))
        self._cpu_mean = mean
        self._cpu_std  = std
        z = (arr - mean) / std

        X_np = np.lib.stride_tricks.sliding_window_view(z[:-1], LOOKBACK)
        y_np = z[LOOKBACK:]
        X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

        model = self._model_cls()
        model.train()
        opt = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        for _ in range(TRAIN_EPOCHS):
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()

        self._model   = model
        self._trained = True

    def predict(self, profile: WorkloadProfile) -> PredictionResult:
        if not self._trained or not profile.has_enough_data:
            return PredictionResult(
                node_id=self.node_id, forecast_horizon_min=5,
                predicted_cpu_util=50.0, predicted_memory_util=50.0,
                predicted_gpu_util={}, spike_probability=0.0, confidence=0.1,
                generated_at=datetime.now(timezone.utc),
            )

        history  = profile.cpu_cores_history
        last_seq = history[-LOOKBACK:]
        z_seq    = [(v - self._cpu_mean) / self._cpu_std for v in last_seq]
        x        = torch.tensor(z_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            z_pred = float(self._model(x).squeeze())

        pred_cpu = float(np.clip(z_pred * self._cpu_std + self._cpu_mean, 0, 100))

        recent_mean = max(float(np.mean(history[-LOOKBACK:])), 1e-3)
        gap         = (pred_cpu - recent_mean) / recent_mean
        spike_prob  = float(np.clip(gap, 0.0, 1.0))
        if profile.burst_factor > 1.5:
            spike_prob = min(spike_prob + 0.2, 1.0)

        return PredictionResult(
            node_id=self.node_id, forecast_horizon_min=5,
            predicted_cpu_util=pred_cpu, predicted_memory_util=50.0,
            predicted_gpu_util={}, spike_probability=spike_prob, confidence=0.8,
            generated_at=datetime.now(timezone.utc),
        )

    def refit_if_needed(self, profile: WorkloadProfile) -> None:
        self.fit(profile)


# ── Classical ML predictors ───────────────────────────────────────────────────

class _LinearRegressionPredictor:
    """Ordinary least squares on the last LOOKBACK CPU core values."""

    def __init__(self, node_id: str):
        self.node_id   = node_id
        self._model    = None
        self._trained  = False
        self._cpu_mean = 0.0
        self._cpu_std  = 1.0

    def fit(self, profile: WorkloadProfile) -> None:
        history = profile.cpu_cores_history
        if len(history) <= LOOKBACK:
            return
        arr  = np.array(history, dtype=np.float64)
        self._cpu_mean = float(arr.mean())
        self._cpu_std  = float(max(arr.std(), 1e-6))
        z = (arr - self._cpu_mean) / self._cpu_std
        X = np.lib.stride_tricks.sliding_window_view(z[:-1], LOOKBACK)
        y = z[LOOKBACK:]
        self._model   = _SKLinear().fit(X, y)
        self._trained = True

    def predict(self, profile: WorkloadProfile) -> PredictionResult:
        if not self._trained or not profile.has_enough_data:
            return _cold_pred(self.node_id)
        history  = profile.cpu_cores_history
        z_seq    = (np.array(history[-LOOKBACK:]) - self._cpu_mean) / self._cpu_std
        z_pred   = float(self._model.predict(z_seq.reshape(1, -1))[0])
        pred_cpu = float(np.clip(z_pred * self._cpu_std + self._cpu_mean, 0, 100))
        return _spike_result(self.node_id, pred_cpu, profile, confidence=0.85)

    def refit_if_needed(self, profile: WorkloadProfile) -> None:
        self.fit(profile)


class _ARIMAPredictor:
    """ARIMA(p,d,q) — order parameterised for hypothesis testing."""

    def __init__(self, node_id: str, order: tuple = (1, 0, 1)):
        self.node_id  = node_id
        self._order   = order
        self._result  = None
        self._trained = False

    def fit(self, profile: WorkloadProfile) -> None:
        history = profile.cpu_cores_history
        if len(history) <= LOOKBACK:
            return
        try:
            self._result  = _ARIMA(history, order=self._order).fit()
            self._trained = True
        except Exception:
            self._trained = False

    def predict(self, profile: WorkloadProfile) -> PredictionResult:
        if not self._trained or self._result is None:
            return _cold_pred(self.node_id)
        try:
            forecast = float(self._result.forecast(steps=1).iloc[0])
            pred_cpu = float(np.clip(forecast, 0, 100))
        except Exception:
            pred_cpu = float(np.mean(profile.cpu_cores_history[-LOOKBACK:]))
        return _spike_result(self.node_id, pred_cpu, profile, confidence=0.90)

    def refit_if_needed(self, profile: WorkloadProfile) -> None:
        self.fit(profile)


# ── Shared helpers for non-generic predictors ─────────────────────────────────

def _cold_pred(node_id: str) -> PredictionResult:
    return PredictionResult(
        node_id=node_id, forecast_horizon_min=5,
        predicted_cpu_util=50.0, predicted_memory_util=50.0,
        predicted_gpu_util={}, spike_probability=0.0, confidence=0.1,
        generated_at=datetime.now(timezone.utc),
    )


def _spike_result(node_id: str, pred_cpu: float,
                  profile: WorkloadProfile, confidence: float) -> PredictionResult:
    history     = profile.cpu_cores_history
    recent_mean = max(float(np.mean(history[-LOOKBACK:])), 1e-3)
    gap         = (pred_cpu - recent_mean) / recent_mean
    spike_prob  = float(np.clip(gap, 0.0, 1.0))
    if profile.burst_factor > 1.5:
        spike_prob = min(spike_prob + 0.2, 1.0)
    return PredictionResult(
        node_id=node_id, forecast_horizon_min=5,
        predicted_cpu_util=pred_cpu, predicted_memory_util=50.0,
        predicted_gpu_util={}, spike_probability=spike_prob, confidence=confidence,
        generated_at=datetime.now(timezone.utc),
    )


# ── Statistical predictors ────────────────────────────────────────────────────

def _persistence_result(nid: str, profile: WorkloadProfile) -> PredictionResult:
    h = profile.cpu_cores_history
    last = h[-1]
    mean = max(float(np.mean(h[-LOOKBACK:])), 1e-3)
    sp = float(np.clip((last - mean) / mean, 0.0, 1.0))
    if profile.burst_factor > 1.5: sp = min(sp + 0.2, 1.0)
    return PredictionResult(node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=last, predicted_memory_util=50.0, predicted_gpu_util={},
        spike_probability=sp, confidence=0.95, generated_at=datetime.now(timezone.utc))


def _ma_result(nid: str, profile: WorkloadProfile) -> PredictionResult:
    h  = profile.cpu_cores_history
    ma = float(np.mean(h[-MA_WINDOW:]))
    mean = max(float(np.mean(h[-LOOKBACK:])), 1e-3)
    sp = float(np.clip((ma - mean) / mean, 0.0, 1.0))
    if profile.burst_factor > 1.5: sp = min(sp + 0.2, 1.0)
    return PredictionResult(node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=ma, predicted_memory_util=50.0, predicted_gpu_util={},
        spike_probability=sp, confidence=0.85, generated_at=datetime.now(timezone.utc))


def _ema_result(nid: str, profile: WorkloadProfile, alpha: float) -> PredictionResult:
    """Exponential Moving Average — bridges MA (rigid window) and Persistence (lag-1)."""
    h = profile.cpu_cores_history
    ema = float(h[0])
    for v in h[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    mean = max(float(np.mean(h[-LOOKBACK:])), 1e-3)
    sp = float(np.clip((ema - mean) / mean, 0.0, 1.0))
    if profile.burst_factor > 1.5: sp = min(sp + 0.2, 1.0)
    return PredictionResult(node_id=nid, forecast_horizon_min=5,
        predicted_cpu_util=ema, predicted_memory_util=50.0, predicted_gpu_util={},
        spike_probability=sp, confidence=0.88, generated_at=datetime.now(timezone.utc))


# ── T5.2-compatible setup helpers ─────────────────────────────────────────────

def _make_node(nid: str) -> ComputeNode:
    return ComputeNode(
        node_id=nid, arch=NodeArch.X86_64, state=NodeState.HEALTHY,
        total_cpu_cores=float(NODE_CPU_CORES), total_memory_gb=NODE_MEM_GB,
        cost_profile=NodeCostProfile(
            instance_type=InstanceType.ON_DEMAND,
            cost_per_hour_usd=NODE_PRICE, interruption_prob=0.0,
        ),
    )


def _non_overlapping(cands: list, n: int) -> list:
    selected, used = [], []
    for w in cands:
        s, e = w["start"], w["end"]
        if any(not (e <= us or s >= ue) for us, ue in used): continue
        selected.append(w.copy()); used.append((s, e))
        if len(selected) == n: break
    return selected


def _select_windows(cpu_arr: np.ndarray):
    cands = []
    for start in range(0, len(cpu_arr) - WINDOW_SIZE + 1, SEARCH_STRIDE):
        w = cpu_arr[start:start + WINDOW_SIZE]
        score = float(w.std() + max(w[-10:].mean() - w[-30:-10].mean(), 0.0))
        cands.append({"start": start, "end": start + WINDOW_SIZE,
                      "score": score, "cpu_pcts": w})
    stable   = _non_overlapping(sorted(cands, key=lambda x: x["score"]),          N_STABLE)
    volatile = _non_overlapping(sorted(cands, key=lambda x: x["score"], reverse=True), N_VOLATILE)
    for i, w in enumerate(stable):   w["node_id"] = f"node-stable-{i:02d}"
    for i, w in enumerate(volatile): w["node_id"] = f"node-volatile-{i:02d}"
    return stable, volatile


def _build_profile(name: str, cpu_pcts: np.ndarray) -> WorkloadProfile:
    p = WorkloadProfile(workload_name=name)
    for pct in cpu_pcts:
        p.add_sample(ResourceSample(
            cpu_cores_used=(float(pct) / 100.0) * NODE_CPU_CORES,
            memory_gb_used=16.0, gpu_util_pct=None,
            duration_s=300.0, scheduling_latency_ms=1.0,
        ))
    return p


def _lc_job() -> JobRequest:
    return JobRequest(
        job_id=str(uuid.uuid4()), workload_type=WorkloadType.LATENCY_CRITICAL,
        resources=ResourceRequest(cpu_cores_min=2.0, memory_gb_min=4.0,
                                  gpu_required=False, gpu_count=1),
        priority=9, preemptible=False,
    )


def _run_condition(nodes, stable_ids, preds, n_jobs, rng) -> float:
    cnt = 0
    for _ in range(n_jobs):
        shuffled = nodes.copy(); rng.shuffle(shuffled)
        if stable_ids and aco_schedule(_lc_job(), shuffled, predictors=preds) in stable_ids:
            cnt += 1
    return cnt / n_jobs * 100.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 68)
    print("T5.3 — Full Predictor Ablation (14 conditions)")
    print("=" * 68)

    df  = pd.read_csv(TRACE_CSV)
    cpu = df["cpu_util_percent"].values
    stable_windows, volatile_windows = _select_windows(cpu)
    all_windows = stable_windows + volatile_windows

    print(f"\nBuilding 32 nodes, training models (incl. ARIMA fits)...")

    # Per-node predictors for each model
    from orchestrator.control_plane.predictor import WorkloadPredictor

    ml_predictors: Dict[str, Dict[str, object]] = {
        "lstm": {}, "gru": {}, "mlp": {}, "tcn": {},
        "transformer": {}, "linreg": {},
        "arima_101": {}, "arima_111": {},   # hypothesis A: d=0 vs d=1
    }
    stat_profiles: Dict[str, WorkloadProfile] = {}
    true_volatility: Dict[str, float] = {}   # ground truth from window selection
    nodes: List[ComputeNode] = []
    stable_ids: set = set()

    for w in all_windows:
        nid     = w["node_id"]
        profile = _build_profile(nid, w["cpu_pcts"])
        stat_profiles[nid]   = profile
        true_volatility[nid] = w["score"]   # std + tail_rise — actual volatility
        nodes.append(_make_node(nid))
        if nid.startswith("node-stable"): stable_ids.add(nid)

        # LSTM (existing)
        p = WorkloadPredictor(node_id=nid)
        p.refit_if_needed(profile)
        ml_predictors["lstm"][nid] = p.predict(profile)

        # GRU
        g = _GenericPredictor(nid, _GRUModel); g.fit(profile)
        ml_predictors["gru"][nid] = g.predict(profile)

        # MLP
        m = _GenericPredictor(nid, _MLPModel); m.fit(profile)
        ml_predictors["mlp"][nid] = m.predict(profile)

        # TCN
        t = _GenericPredictor(nid, _TCNModel); t.fit(profile)
        ml_predictors["tcn"][nid] = t.predict(profile)

        # Transformer encoder
        tr = _GenericPredictor(nid, _TransformerModel); tr.fit(profile)
        ml_predictors["transformer"][nid] = tr.predict(profile)

        # Linear regression
        lr = _LinearRegressionPredictor(nid); lr.fit(profile)
        ml_predictors["linreg"][nid] = lr.predict(profile)

        # ARIMA(1,0,1) — no differencing, assumes stationarity
        ar0 = _ARIMAPredictor(nid, order=(1, 0, 1)); ar0.fit(profile)
        ml_predictors["arima_101"][nid] = ar0.predict(profile)

        # ARIMA(1,1,1) — first-order differencing, non-stationarity robust
        ar1 = _ARIMAPredictor(nid, order=(1, 1, 1)); ar1.fit(profile)
        ml_predictors["arima_111"][nid] = ar1.predict(profile)

    # Statistical predictions (computed fresh from profiles)
    pers_preds   = {nid: _persistence_result(nid, stat_profiles[nid]) for nid in stat_profiles}
    ma_preds     = {nid: _ma_result(nid, stat_profiles[nid])          for nid in stat_profiles}
    ema01_preds  = {nid: _ema_result(nid, stat_profiles[nid], 0.1)   for nid in stat_profiles}
    ema03_preds  = {nid: _ema_result(nid, stat_profiles[nid], 0.3)   for nid in stat_profiles}
    ema05_preds  = {nid: _ema_result(nid, stat_profiles[nid], 0.5)   for nid in stat_profiles}

    volatile_ids = {w["node_id"] for w in volatile_windows}
    all_nids     = [w["node_id"] for w in all_windows]

    # ── All predictor map (label → pred dict) ─────────────────────────────────
    all_preds_map = [
        ("No prediction", {}),
        ("Lin Reg",        ml_predictors["linreg"]),
        ("MLP",            ml_predictors["mlp"]),
        ("GRU",            ml_predictors["gru"]),
        ("TCN",            ml_predictors["tcn"]),
        ("LSTM",           ml_predictors["lstm"]),
        ("Transformer",    ml_predictors["transformer"]),
        ("ARIMA(1,0,1)",   ml_predictors["arima_101"]),
        ("ARIMA(1,1,1)",   ml_predictors["arima_111"]),
        ("Persistence",    pers_preds),
        ("EMA α=0.1",      ema01_preds),
        ("EMA α=0.3",      ema03_preds),
        ("EMA α=0.5",      ema05_preds),
        ("Moving Avg",     ma_preds),
    ]

    # ── Spike prob gap (volatile vs stable) ───────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  Spike probability gap  (volatile mean − stable mean)")
    print(f"{'─'*68}")
    for label, preds in all_preds_map[1:]:   # skip no-prediction
        sp_fn = lambda n, p=preds: p[n].spike_probability if p else 0.0
        s = float(np.mean([sp_fn(n) for n in stable_ids]))
        v = float(np.mean([sp_fn(n) for n in volatile_ids]))
        print(f"  {label:<16}  stable={s:.3f}  volatile={v:.3f}  gap={v-s:+.3f}")

    # ── RANK METRICS: Spearman ρ and Top-K accuracy (K = 5, 10, 16) ─────────
    # True ground-truth = volatility score from window selection (std + tail_rise).
    # Top-K accuracy = fraction of the K most volatile nodes that the predictor
    # correctly places in its own top-K by spike_prob.
    # K=5 tests "can you find the worst offenders?" — critical for LC routing.
    print(f"\n{'─'*68}")
    print(f"  Rank quality — Spearman ρ + Top-K congestion detection (K=5,10,16)")
    print(f"{'─'*68}")
    true_vols  = [true_volatility[n] for n in all_nids]
    true_top5  = set(sorted(all_nids, key=lambda n: true_volatility[n], reverse=True)[:5])
    true_top10 = set(sorted(all_nids, key=lambda n: true_volatility[n], reverse=True)[:10])

    rank_spearman: Dict[str, float] = {}
    rank_top5:     Dict[str, float] = {}
    rank_top10:    Dict[str, float] = {}
    rank_top16:    Dict[str, float] = {}

    for label, preds in all_preds_map:
        if not preds:
            probs = [0.0] * len(all_nids)
        else:
            probs = [preds[n].spike_probability for n in all_nids]

        rho, rho_p = scipy_stats.spearmanr(true_vols, probs)
        sp_fn = (lambda n, p=preds: p[n].spike_probability) if preds else (lambda n: 0.0)
        pred_sorted = sorted(all_nids, key=sp_fn, reverse=True)

        acc5  = len(set(pred_sorted[:5])  & true_top5)  / 5.0
        acc10 = len(set(pred_sorted[:10]) & true_top10) / 10.0
        acc16 = len(set(pred_sorted[:16]) & volatile_ids) / 16.0

        rank_spearman[label] = float(rho)
        rank_top5[label]     = float(acc5)
        rank_top10[label]    = float(acc10)
        rank_top16[label]    = float(acc16)

        sig = "***" if rho_p < 0.001 else "**" if rho_p < 0.01 else "*" if rho_p < 0.05 else "n.s."
        print(f"  {label:<16}  ρ={rho:+.3f}({sig})  "
              f"Top-5={acc5:.2f}({int(acc5*5)}/5)  "
              f"Top-10={acc10:.2f}({int(acc10*10)}/10)  "
              f"Top-16={acc16:.2f}({int(acc16*16)}/16)")

    # ── Hypothesis A: ARIMA(1,0,1) vs ARIMA(1,1,1) in ranking quality ────────
    print(f"\n  Hypothesis A — does differencing (d=1) help ARIMA rank nodes better?")
    d0r = rank_spearman["ARIMA(1,0,1)"]; d1r = rank_spearman["ARIMA(1,1,1)"]
    if d1r > d0r + 0.02:
        print(f"  → d=1 HELPS: ρ {d0r:.3f} → {d1r:.3f}. Signal is non-stationary.")
    elif d0r > d1r + 0.02:
        print(f"  → d=1 HURTS: ρ {d0r:.3f} → {d1r:.3f}. Signal is short-memory stationary.")
        print(f"    LR > ARIMA(1,0,1) likely because ARIMA over-parameters a simple AR process.")
    else:
        print(f"  → d=1 neutral: ρ {d0r:.3f} ≈ {d1r:.3f}. Both ARIMA variants equally mis-ranked.")
        print(f"    Hypothesis C (ranking metric) is the real explanation: check LR vs ARIMA ρ.")

    print(f"\n  Hypothesis C — does LR outrank ARIMA on Spearman?")
    lr_r = rank_spearman["Lin Reg"]; ar_r = rank_spearman["ARIMA(1,0,1)"]
    if lr_r > ar_r + 0.02:
        print(f"  → CONFIRMED: LR ρ={lr_r:.3f} > ARIMA ρ={ar_r:.3f}")
        print(f"    LR > ARIMA because it preserves node ordering; ARIMA minimises MSE but distorts ranks.")
    else:
        print(f"  → NOT confirmed by Spearman: LR ρ={lr_r:.3f} ≈ ARIMA ρ={ar_r:.3f}")
        print(f"    Routing gap is likely a stochastic artefact — check if gap shrinks with more seeds.")

    # ── EMA sweep summary ─────────────────────────────────────────────────────
    print(f"\n  EMA α sweep — does decay rate matter more than MA window?")
    for lbl in ["EMA α=0.1", "EMA α=0.3", "EMA α=0.5", "Moving Avg"]:
        print(f"    {lbl:<14}  ρ={rank_spearman[lbl]:+.3f}  "
              f"Top-5={rank_top5[lbl]:.2f}  Top-10={rank_top10[lbl]:.2f}")

    # ── Run routing experiment ─────────────────────────────────────────────────
    conditions = all_preds_map

    print(f"\nRouting experiment: {N_SEEDS} seeds × {N_JOBS} LC jobs per condition")
    print(f"{'Condition':<16} {'Seeds':>45}  {'Mean':>6}  {'Std':>5}")

    results = {}
    for label, preds in conditions:
        vals = []
        for seed in range(N_SEEDS):
            rng = random.Random(seed * 137 + 42)
            vals.append(_run_condition(nodes, stable_ids, preds, N_JOBS, rng))
        results[label] = vals
        seed_str = "  ".join(f"{v:.1f}%" for v in vals)
        print(f"  {label:<16} {seed_str:>45}  {np.mean(vals):>6.1f}%  {np.std(vals):>5.2f}")

    # Statistical tests vs baseline
    baseline = np.array(results["No prediction"])
    print(f"\nΔ vs no-prediction baseline (paired t-test, df={N_SEEDS-1}):")
    for label, _ in conditions[1:]:
        arr = np.array(results[label])
        t, p = scipy_stats.ttest_rel(arr, baseline)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:<16}  Δ={np.mean(arr)-np.mean(baseline):+.1f}pp  "
              f"t={t:.2f}  p={p:.4f}  {sig}")

    # ── CHECK 2: Clean mean ± std summary table ───────────────────────────────
    ci_z = 1.96 / np.sqrt(N_SEEDS)  # approximate 95% CI half-width factor
    print(f"\n{'─'*68}")
    print(f"  {'Model':<16}  {'Mean':>6}  {'±Std':>6}  {'95% CI':>20}  "
          f"{'Top-5':>6}  {'Top-10':>7}  {'ρ':>7}")
    print(f"{'─'*68}")
    for label, _ in conditions:
        arr  = np.array(results[label])
        m, s = float(np.mean(arr)), float(np.std(arr))
        lo, hi = m - ci_z * s, m + ci_z * s
        t5  = rank_top5.get(label, float("nan"))
        t10 = rank_top10.get(label, float("nan"))
        rho = rank_spearman.get(label, float("nan"))
        ci_str = f"[{lo:.1f}%, {hi:.1f}%]"
        print(f"  {label:<16}  {m:>5.1f}%  {s:>5.2f}  {ci_str:>20}  "
              f"{t5:>5.2f}  {t10:>6.2f}  {rho:>+6.3f}")
    print(f"{'─'*68}")

    # ── CHECK 3: EMA α=0.1 stability — 25 seeds ──────────────────────────────
    N_STAB = 25
    print(f"\nEMA α=0.1 stability check ({N_STAB} seeds):")
    ema01_stab = []
    for seed in range(N_STAB):
        rng = random.Random(seed * 137 + 42)
        ema01_stab.append(_run_condition(nodes, stable_ids, ema01_preds, N_JOBS, rng))
    m01, s01 = float(np.mean(ema01_stab)), float(np.std(ema01_stab))
    below_50 = sum(1 for v in ema01_stab if v < 50.0)
    print(f"  mean={m01:.1f}%  std={s01:.2f}  min={min(ema01_stab):.1f}%  max={max(ema01_stab):.1f}%")
    print(f"  Seeds below random baseline (50%): {below_50}/{N_STAB} ({below_50/N_STAB:.0%})")
    seeds_str = "  ".join(f"{v:.0f}%" for v in ema01_stab)
    print(f"  All seeds: {seeds_str}")
    if m01 < 50.0 and below_50 >= N_STAB * 0.8:
        print(f"  → CONFIRMED: α=0.1 over-smooths across all seeds (not a seed artifact).")
    elif below_50 < N_STAB * 0.5:
        print(f"  → UNCERTAIN: α=0.1 below 50% only in {below_50}/{N_STAB} seeds — partially seed-dependent.")
    else:
        print(f"  → PARTIALLY STABLE: {below_50}/{N_STAB} seeds below 50% — mostly consistent.")

    # Plot
    labels  = [c[0] for c in conditions]
    means   = [float(np.mean(results[l])) for l in labels]
    stds    = [float(np.std(results[l]))  for l in labels]
    # 14 bars: baseline, LinReg, MLP, GRU, TCN, LSTM, Transformer,
    #          ARIMA(1,0,1), ARIMA(1,1,1), Persistence, EMA×3, MA
    colors = [
        "#aaaaaa",                                   # no prediction
        "#d4a0a0", "#e15759",                        # classical ML: LinReg, MLP
        "#f28e2b", "#76b7b2", "#9c755f", "#edc948", # neural: GRU, TCN, LSTM, Transformer
        "#b07aa1", "#c9a0dc",                        # classical TS: ARIMA 1,0,1 / 1,1,1
        "#4e79a7",                                   # Persistence
        "#a0c8e8", "#5ba3d0", "#1a6fa3",            # EMA 0.1, 0.3, 0.5
        "#59a14f",                                   # MA
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: routing % bar chart
    ax = axes[0]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(50.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6,
               label="Random baseline (50%)")
    ax.set_ylabel("% jobs routed to stable nodes")
    ax.set_ylim(0, 115)
    ax.set_title(f"Routing Quality: 14-Way Predictor Ablation\n"
                 f"({N_SEEDS} seeds × {N_JOBS} LC jobs, higher = better)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.5,
                f"{mean:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.axvspan(-0.5,  0.5,  alpha=0.05, color="gray")
    ax.axvspan(0.5,   2.5,  alpha=0.05, color="red")
    ax.axvspan(2.5,   6.5,  alpha=0.05, color="orange")
    ax.axvspan(6.5,   8.5,  alpha=0.05, color="purple")
    ax.axvspan(8.5,  13.5,  alpha=0.05, color="blue")
    ax.text(0,   112, "Base",       ha="center", fontsize=7, color="gray")
    ax.text(1.5, 112, "Classical",  ha="center", fontsize=7, color="#c0392b")
    ax.text(4.5, 112, "Neural",     ha="center", fontsize=7, color="#7d5c3a")
    ax.text(7.5, 112, "ARIMA",      ha="center", fontsize=7, color="#7d4e8a")
    ax.text(11,  112, "Statistical",ha="center", fontsize=7, color="#2471a3")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Right: Spearman ρ scatter (rank quality)
    ax2 = axes[1]
    rho_labels = [l for l, _ in all_preds_map if l != "No prediction"]
    rho_vals   = [rank_spearman[l] for l in rho_labels]
    rho_topk   = [rank_top5[l]     for l in rho_labels]   # use Top-5 (strictest) on scatter
    rho_colors = colors[1:]   # skip baseline color
    sc = ax2.scatter(rho_vals, rho_topk, c=rho_colors, s=80, zorder=5, edgecolors="gray", linewidth=0.5)
    for lbl, rx, ry in zip(rho_labels, rho_vals, rho_topk):
        ax2.annotate(lbl, (rx, ry), fontsize=6, xytext=(4, 2), textcoords="offset points")
    ax2.set_xlabel("Spearman ρ (predicted vs true volatility)")
    ax2.set_ylabel("Top-5 accuracy (5 worst nodes found)")
    ax2.set_title("Rank Quality — Top-5 Detection\n(upper-right = best node selector)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.05); ax2.set_ylim(-0.05, 1.1)

    fig.suptitle("T5.3 — Predictor Ablation: Routing Quality + Rank Fidelity", fontsize=11, fontweight="bold")
    fig.tight_layout()

    png_path = RESULTS_DIR / "tier5_predictor_ablation.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    result = {
        "n_seeds": N_SEEDS, "n_jobs": N_JOBS,
        "means":            {l: float(np.mean(results[l])) for l in labels},
        "stds":             {l: float(np.std(results[l]))  for l in labels},
        "raw":              {l: results[l] for l in labels},
        "rank_spearman":    rank_spearman,
        "rank_top5_acc":    rank_top5,
        "rank_top10_acc":   rank_top10,
        "rank_top16_acc":   rank_top16,
        "ema01_stability":  {
            "n_seeds": N_STAB, "mean": m01, "std": s01,
            "below_50_fraction": below_50 / N_STAB,
            "raw": ema01_stab,
        },
    }
    save_results("tier5_predictor_ablation", result)
    return result


if __name__ == "__main__":
    run()
