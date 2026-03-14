"""
benchmarks/run_all.py
─────────────────────
Run all 7 benchmark scripts in tier order and print a final summary table.
Each script is imported and run() is called directly (same process).

Usage:
    python -m benchmarks.run_all

Output:
    PNGs in benchmarks/results/
    Final summary table printed to stdout
    Total runtime: ~3–8 minutes (LSTM refit is the bottleneck in T2.x)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Tier 1
from benchmarks import tier1_aco_vs_naive
from benchmarks import tier1_pheromone_curve
from benchmarks import tier1_p99_latency

# Tier 2
from benchmarks import tier2_spike_recall
from benchmarks import tier2_confidence_calibration

# Tier 3
from benchmarks import tier3_queue_drain
from benchmarks import tier3_cold_vs_warm_start


BENCHMARKS = [
    ("T1.1", "ACO vs Naive (Azure VM Distribution)",    tier1_aco_vs_naive),
    ("T1.2", "Pheromone Learning Curve",                tier1_pheromone_curve),
    ("T1.3", "P99 Latency Under Burst Load",            tier1_p99_latency),
    ("T2.1", "Spike Prediction Recall",                 tier2_spike_recall),
    ("T2.2", "Confidence Calibration",                  tier2_confidence_calibration),
    ("T3.1", "Queue Drain Under Saturation",            tier3_queue_drain),
    ("T3.2", "Cold vs Warm Pheromone Start",            tier3_cold_vs_warm_start),
]


def _status_from(tag: str, result: dict) -> str:
    """Extract a one-line status string from a benchmark result dict."""
    if tag == "T1.1":
        pct = result.get("improvement_pct", 0.0)
        ok  = result.get("passed", False)
        return f"{'PASS' if ok else 'FAIL'}  ACO {pct:.1f}% cheaper than Naive"
    if tag == "T1.2":
        ok = result.get("learning_detected", False)
        early = result.get("mean_early", 0.0)
        late  = result.get("mean_late", 0.0)
        return (f"{'DETECTED' if ok else 'NOT DETECTED'}  "
                f"early=${early:.4f}/hr → late=${late:.4f}/hr")
    if tag == "T1.3":
        # result is dict of burst→stats
        p99_at_100 = result.get(100, {}).get("p99", float("nan"))
        return f"P99@100={p99_at_100:.2f}ms"
    if tag == "T2.1":
        recall = result.get("recall", 0.0)
        n      = result.get("n_spikes", 0)
        caught = result.get("n_caught", 0)
        return f"Recall {recall*100:.1f}%  ({caught}/{n} spikes caught)"
    if tag == "T2.2":
        r      = result.get("pearson_r", float("nan"))
        ok     = result.get("calibration_good", False)
        return f"{'GOOD' if ok else 'POOR'}  Pearson r = {r:+.3f}"
    if tag == "T3.1":
        rate = result.get("drain_rate_per_s", 0.0)
        dep  = result.get("initial_depth", 0)
        return f"Queue {dep} jobs drained at {rate:.1f} jobs/s"
    if tag == "T3.2":
        pct = result.get("early_improvement_pct", 0.0)
        ok  = result.get("advantage_detected", False)
        return f"{'DETECTED' if ok else 'NOT DETECTED'}  early cost {pct:+.1f}% (warm vs cold)"
    return str(result)


def main() -> None:
    print("\n" + "=" * 70)
    print("ACO Adaptive Compute Scheduler — Empirical Validation Suite")
    print("=" * 70)

    summary: list[tuple[str, str, float, str]] = []
    # (tag, name, elapsed_s, status_str)

    for tag, name, module in BENCHMARKS:
        print(f"\n{'─'*70}")
        print(f"Running {tag}: {name}")
        print(f"{'─'*70}")
        t0 = time.perf_counter()
        try:
            result = module.run()
            elapsed = time.perf_counter() - t0
            status  = _status_from(tag, result)
        except Exception as exc:   # noqa: BLE001
            elapsed = time.perf_counter() - t0
            status  = f"ERROR: {exc}"
            result  = {}
        summary.append((tag, name, elapsed, status))

    # ── Final summary table ────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Tag':<6}  {'Time':>6}  {'Result'}")
    print("-" * 70)
    for tag, name, elapsed, status in summary:
        print(f"{tag:<6}  {elapsed:>5.1f}s  {status}")
    print("=" * 70)

    results_dir = Path(__file__).parent / "results"
    pngs = sorted(results_dir.glob("*.png"))
    if pngs:
        print(f"\nCharts saved to: {results_dir}/")
        for p in pngs:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
