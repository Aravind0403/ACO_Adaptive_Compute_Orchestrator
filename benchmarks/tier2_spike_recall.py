"""
T2.1 — Spike Prediction Recall on Alibaba 2018 Trace
══════════════════════════════════════════════════════
Of all real CPU spikes in the Alibaba trace, what fraction did the LSTM
predictor catch BEFORE they happened?

Spike definition: cpu[i] > 65% AND rolling_5_mean(cpu[i-5:i]) < 50%
(a sudden jump from normal operation into high-load territory)

The predictor is fed samples one-by-one, refit every 10 new samples.
At each spike onset, we check spike_probability from the most recent predict().

Run:
    python -m benchmarks.tier2_spike_recall

Output:
    benchmarks/results/tier2_spike_recall.png
    Printed: spike count, recall, mean spike_prob at onset vs baseline
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.control_plane.predictor import WorkloadPredictor
from orchestrator.shared.telemetry import WorkloadProfile, ResourceSample
from benchmarks._helpers import TRACE_CSV, RESULTS_DIR


# ── Constants ─────────────────────────────────────────────────────────────────
SPIKE_THRESHOLD_CPU = 65.0   # CPU% above which a tick counts as a spike
BASELINE_MEAN_MAX   = 50.0   # rolling-5 mean must be < this before a spike onset
CATCH_THRESHOLD     = 0.4    # spike_probability above this = "caught"
NODE_TOTAL_CPU      = 32.0   # cores on the benchmark node (for unit conversion)
NODE_TOTAL_MEM      = 128.0  # GB


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    # ── Load trace ────────────────────────────────────────────────────────────
    df = pd.read_csv(TRACE_CSV)
    cpu_values = df["cpu_util_percent"].values.astype(float)
    mem_values = df["mem_util_percent"].values.astype(float)
    n_rows = len(cpu_values)

    # ── Identify spike onsets ─────────────────────────────────────────────────
    spike_onset_set: set[int] = set()
    for i in range(5, n_rows):
        rolling_mean = cpu_values[max(0, i - 5) : i].mean()
        if cpu_values[i] > SPIKE_THRESHOLD_CPU and rolling_mean < BASELINE_MEAN_MAX:
            spike_onset_set.add(i)

    # ── Replay trace through predictor ────────────────────────────────────────
    predictor = WorkloadPredictor("alibaba-bench")
    profile   = WorkloadProfile(workload_name="trace")

    spike_probs:  List[float] = []   # spike_probability at every tick
    confidences:  List[float] = []   # confidence at every tick

    onset_results: List[Tuple[int, float, bool]] = []
    # (tick, spike_probability, caught)

    last_spike_prob = 0.0
    last_confidence = 0.1

    for i in range(n_rows):
        # Add sample (core units, matching TelemetryCollector convention)
        sample = ResourceSample(
            cpu_cores_used=cpu_values[i] * 0.01 * NODE_TOTAL_CPU,
            memory_gb_used=mem_values[i] * 0.01 * NODE_TOTAL_MEM,
            duration_s=300.0,       # 5-minute intervals
            scheduling_latency_ms=0.0,
        )
        profile.add_sample(sample)

        # Refit + predict if enough data
        if profile.has_enough_data:
            predictor.refit_if_needed(profile)
            pred = predictor.predict(profile)
            last_spike_prob = pred.spike_probability
            last_confidence = pred.confidence

        spike_probs.append(last_spike_prob)
        confidences.append(last_confidence)

        # At spike onsets, record whether we caught it
        if i in spike_onset_set:
            caught = last_spike_prob > CATCH_THRESHOLD
            onset_results.append((i, last_spike_prob, caught))

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_spikes = len(onset_results)
    n_caught = sum(1 for _, _, c in onset_results if c)
    recall   = n_caught / n_spikes if n_spikes > 0 else 0.0

    onset_probs     = [p for _, p, _ in onset_results]
    non_onset_probs = [spike_probs[i] for i in range(n_rows) if i not in spike_onset_set]
    mean_at_onset   = np.mean(onset_probs) if onset_probs else 0.0
    mean_baseline   = np.mean(non_onset_probs) if non_onset_probs else 0.0

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("T2.1 — Spike Prediction Recall (Alibaba 2018 trace)")
    print(f"{'='*60}")
    print(f"Trace rows:              {n_rows}  (8 days @ 5-min intervals)")
    print(f"Spike onsets detected:   {n_spikes}")
    print(f"  Caught (prob > {CATCH_THRESHOLD}):  {n_caught}  →  Recall: {recall*100:.1f}%")
    print(f"  Missed:               {n_spikes - n_caught}")
    print(f"Mean spike_prob at onsets:    {mean_at_onset:.3f}")
    print(f"Mean spike_prob at baseline:  {mean_baseline:.3f}")
    if n_spikes > 0:
        print(f"Signal lift:                  {mean_at_onset / max(mean_baseline, 1e-6):.2f}x")
    print(f"\nNote: LSTM cold-start (first {10} ticks) produces spike_prob=0.0 by design")

    # ── Chart ─────────────────────────────────────────────────────────────────
    # Downsample for readability (show first 500 ticks max)
    show_n = min(n_rows, 500)
    ticks  = np.arange(show_n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("T2.1 — Spike Prediction Recall on Alibaba 2018 Trace",
                 fontsize=13, fontweight="bold")

    # Top: CPU timeseries + spike onset markers
    ax1.plot(ticks, cpu_values[:show_n], color="#4c72b0", linewidth=0.8,
             alpha=0.8, label="CPU utilisation %")
    ax1.axhline(SPIKE_THRESHOLD_CPU, linestyle="--", color="red",
                linewidth=0.9, alpha=0.7, label=f"Spike threshold ({SPIKE_THRESHOLD_CPU}%)")
    ax1.axhline(BASELINE_MEAN_MAX, linestyle=":", color="orange",
                linewidth=0.9, alpha=0.7, label=f"Baseline ceiling ({BASELINE_MEAN_MAX}%)")

    for tick, prob, caught in onset_results:
        if tick >= show_n:
            break
        color = "#2ca02c" if caught else "#d62728"
        ax1.axvline(tick, color=color, linewidth=0.9, alpha=0.6)
        ax1.plot(tick, cpu_values[tick], "v", color=color, markersize=7, zorder=5)

    # Legend entries for onset markers
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#2ca02c", marker="v", linestyle="-",
               markersize=7, label=f"Caught onset (prob > {CATCH_THRESHOLD})"),
        Line2D([0], [0], color="#d62728", marker="v", linestyle="-",
               markersize=7, label="Missed onset"),
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + legend_handles,
               loc="upper right", fontsize=8, ncol=2)
    ax1.set_ylabel("CPU utilisation (%)")
    ax1.set_title(f"Recall: {n_caught}/{n_spikes} spikes caught ({recall*100:.1f}%)")
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Bottom: spike_probability timeseries
    ax2.plot(ticks, spike_probs[:show_n], color="#9467bd", linewidth=0.9,
             alpha=0.8, label="spike_probability")
    ax2.axhline(CATCH_THRESHOLD, linestyle="--", color="gray",
                linewidth=0.9, alpha=0.7, label=f"Catch threshold ({CATCH_THRESHOLD})")

    for tick, prob, caught in onset_results:
        if tick >= show_n:
            break
        color = "#2ca02c" if caught else "#d62728"
        ax2.plot(tick, prob, "o", color=color, markersize=6, zorder=5)

    ax2.set_xlabel("Tick (5-min intervals)")
    ax2.set_ylabel("spike_probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    out = RESULTS_DIR / "tier2_spike_recall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out}")

    return {
        "n_spikes": n_spikes,
        "n_caught": n_caught,
        "recall": recall,
        "mean_at_onset": mean_at_onset,
        "mean_baseline": mean_baseline,
    }


if __name__ == "__main__":
    run()
