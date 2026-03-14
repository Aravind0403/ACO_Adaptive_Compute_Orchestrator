"""
T2.2 — LSTM Confidence Calibration
════════════════════════════════════
After the MAE-based confidence fix (Phase 6), does confidence actually
correlate with prediction accuracy? A well-calibrated model shows:
  high confidence → low error
  low confidence  → high error

This benchmark replays the Alibaba 2018 trace, collecting
(confidence, |predicted_cpu - actual_next_cpu|) pairs for every prediction.
Results are bucketed by confidence level and Pearson r is computed.

Expected structural reason this works: early in the trace, confidence is low
(cold start, few samples, high MAE) AND error is high. As samples accumulate,
confidence rises AND error falls. The 50/50 MAE-sample blend encodes this.

Run:
    python -m benchmarks.tier2_confidence_calibration

Output:
    benchmarks/results/tier2_confidence_calibration.png
    Printed: bucket table, Pearson r, GOOD/POOR calibration verdict
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
NODE_TOTAL_CPU = 32.0
NODE_TOTAL_MEM = 128.0
N_BUCKETS      = 10      # [0.1, 0.2), [0.2, 0.3), …, [0.9, 1.0]


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run() -> dict:
    # ── Load trace ────────────────────────────────────────────────────────────
    df = pd.read_csv(TRACE_CSV)
    cpu_values = df["cpu_util_percent"].values.astype(float)
    n_rows = len(cpu_values)
    mem_values = df["mem_util_percent"].values.astype(float)

    predictor = WorkloadPredictor("calib-bench")
    profile   = WorkloadProfile(workload_name="trace")

    records: List[Tuple[float, float]] = []   # (confidence, absolute_error_pct)

    for i in range(n_rows - 1):              # -1: need cpu[i+1] as ground truth
        sample = ResourceSample(
            cpu_cores_used=cpu_values[i] * 0.01 * NODE_TOTAL_CPU,
            memory_gb_used=mem_values[i] * 0.01 * NODE_TOTAL_MEM,
            duration_s=300.0,
            scheduling_latency_ms=0.0,
        )
        profile.add_sample(sample)

        if profile.has_enough_data:
            predictor.refit_if_needed(profile)
            pred = predictor.predict(profile)
            actual_next_cpu = cpu_values[i + 1]
            abs_error = abs(pred.predicted_cpu_util - actual_next_cpu)
            records.append((pred.confidence, abs_error))

    n_records = len(records)

    # ── Bucketing ─────────────────────────────────────────────────────────────
    bucket_edges = np.linspace(0.1, 1.0, N_BUCKETS + 1)
    bucket_means: List[float] = []
    bucket_stds:  List[float] = []
    bucket_counts: List[int]  = []
    bucket_labels: List[str]  = []

    confidences = np.array([r[0] for r in records])
    errors      = np.array([r[1] for r in records])

    for k in range(N_BUCKETS):
        lo, hi = bucket_edges[k], bucket_edges[k + 1]
        mask = (confidences >= lo) & (confidences < hi)
        bucket_errors = errors[mask]
        bucket_counts.append(int(mask.sum()))
        bucket_means.append(float(bucket_errors.mean()) if len(bucket_errors) > 0 else 0.0)
        bucket_stds.append(float(bucket_errors.std())  if len(bucket_errors) > 1 else 0.0)
        bucket_labels.append(f"[{lo:.1f}–{hi:.1f})")

    # Pearson r between bucket midpoints and bucket mean errors
    bucket_mids = (bucket_edges[:-1] + bucket_edges[1:]) / 2
    non_empty   = [k for k in range(N_BUCKETS) if bucket_counts[k] > 0]
    if len(non_empty) >= 2:
        pearson_r = float(np.corrcoef(
            [bucket_mids[k] for k in non_empty],
            [bucket_means[k] for k in non_empty],
        )[0, 1])
    else:
        pearson_r = float("nan")

    calibration_good = pearson_r < -0.3

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("T2.2 — LSTM Confidence Calibration")
    print(f"{'='*60}")
    print(f"Predictions recorded: {n_records}")
    print(f"\nConfidence buckets:")
    print(f"  {'Bucket':<14} {'Count':>7}  {'Mean Error':>11}  {'Std':>8}")
    for k in range(N_BUCKETS):
        print(f"  {bucket_labels[k]:<14} {bucket_counts[k]:>7}  "
              f"{bucket_means[k]:>10.2f}%  {bucket_stds[k]:>7.2f}%")
    print(f"\nPearson r (confidence vs mean_error): {pearson_r:+.3f}")
    print(f"Calibration: {'GOOD ✓ (r < -0.3)' if calibration_good else 'POOR ✗ (r ≥ -0.3)'}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("T2.2 — LSTM Confidence Calibration", fontsize=13, fontweight="bold")

    # Left: scatter (downsampled) + bucket means
    MAX_SCATTER = 2_000
    rng = np.random.default_rng(42)
    if n_records > MAX_SCATTER:
        idx = rng.choice(n_records, MAX_SCATTER, replace=False)
        sc_conf = confidences[idx]
        sc_err  = errors[idx]
    else:
        sc_conf = confidences
        sc_err  = errors

    ax1.scatter(sc_conf, sc_err, alpha=0.07, s=8, color="#4c72b0", label="Predictions (sample)")

    # Bucket means as big dots with error bars
    non_empty_mids   = [bucket_mids[k]  for k in non_empty]
    non_empty_means  = [bucket_means[k] for k in non_empty]
    non_empty_stds   = [bucket_stds[k]  for k in non_empty]
    ax1.errorbar(non_empty_mids, non_empty_means, yerr=non_empty_stds,
                 fmt="o", color="#d62728", markersize=9, linewidth=1.5,
                 label="Bucket mean ± std", zorder=5)

    # Best-fit line through bucket means
    if len(non_empty) >= 2:
        z = np.polyfit(non_empty_mids, non_empty_means, 1)
        xline = np.linspace(0.1, 1.0, 100)
        ax1.plot(xline, np.polyval(z, xline), "--", color="gray",
                 linewidth=1.2, alpha=0.8, label=f"Trend (r = {pearson_r:+.2f})")

    ax1.set_xlabel("Confidence score")
    ax1.set_ylabel("Absolute prediction error (%)")
    ax1.set_title("Confidence vs Prediction Error")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0.08, 1.02)

    # Right: bar chart of mean error per bucket
    bar_colors = [
        plt.cm.RdYlGn_r(k / N_BUCKETS) for k in range(N_BUCKETS)    # type: ignore[attr-defined]
    ]
    bars = ax2.bar(range(N_BUCKETS), bucket_means, color=bar_colors, edgecolor="white")
    ax2.set_xticks(range(N_BUCKETS))
    ax2.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Confidence bucket")
    ax2.set_ylabel("Mean absolute error (%)")
    ax2.set_title(f"Mean Error per Confidence Bucket\n(Pearson r = {pearson_r:+.2f})")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Count labels on bars
    for bar, count, mean in zip(bars, bucket_counts, bucket_means):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, mean + 0.2,
                     f"n={count}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = RESULTS_DIR / "tier2_confidence_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out}")

    return {
        "n_records": n_records,
        "pearson_r": pearson_r,
        "calibration_good": calibration_good,
        "bucket_means": bucket_means,
        "bucket_counts": bucket_counts,
    }


if __name__ == "__main__":
    run()
