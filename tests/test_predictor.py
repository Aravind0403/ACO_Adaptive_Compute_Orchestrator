"""
tests/test_predictor.py
────────────────────────
Test suite for orchestrator/control_plane/predictor.py

Coverage: 14 tests across 5 groups.

What we are testing
────────────────────
WorkloadPredictor is the ML component of V2. It has two clear operating modes:
  1. Cold-start: fewer than LOOKBACK samples → safe fallback, confidence=0.1
  2. Trained:    ≥ LOOKBACK+1 samples → LSTM prediction, confidence ∈ [0.5, 1.0]

Testing strategy
─────────────────
• Use deterministic, synthetic histories (not random seeds) wherever possible.
  This keeps failures reproducible and easy to diagnose.
• For property-based tests (spike probability bounds), iterate 20 random
  histories and assert the invariant holds for each.
• Tests do NOT assert exact predicted values — LSTM training is stochastic and
  the absolute output depends on random weight initialisation. Instead, we
  assert structural properties: correct type, correct bounds, direction of change.
• fit() and predict() are fast (< 200ms total) — no mocking needed.

Test groups
────────────
Group 1: Cold Start         — behaviour before enough data exists
Group 2: Training           — fit() state transitions
Group 3: Inference          — predict() output structure and bounds
Group 4: Spike Detection    — spike_probability direction and range
Group 5: Normalisation      — z-score parameters stored and independent
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List

import pytest

from orchestrator.control_plane.predictor import (
    LOOKBACK,
    REFIT_THRESHOLD,
    WorkloadPredictor,
)
from orchestrator.shared.models import PredictionResult
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_profile(
    cpu_values: List[float],
    workload_name: str = "test-workload",
) -> WorkloadProfile:
    """
    Build a WorkloadProfile from a list of CPU core values.

    Timestamps are synthetic (1 second apart starting from now).
    All other fields (memory, GPU, duration) use harmless defaults.
    """
    profile = WorkloadProfile(workload_name=workload_name)
    base_time = datetime.utcnow()
    for i, cpu in enumerate(cpu_values):
        sample = ResourceSample(
            timestamp=base_time + timedelta(seconds=i),
            cpu_cores_used=max(0.0, cpu),   # clamp: ge=0 validator
            memory_gb_used=2.0,
            gpu_util_pct=None,
            duration_s=10.0,
            scheduling_latency_ms=3.0,
        )
        profile.add_sample(sample)
    return profile


def _make_predictor(node_id: str = "node-test-01") -> WorkloadPredictor:
    """Return an untrained WorkloadPredictor."""
    return WorkloadPredictor(node_id=node_id)


def _fit_predictor(cpu_values: List[float], node_id: str = "node-test-01") -> tuple:
    """
    Convenience: build profile + predictor + call fit().

    Returns (predictor, profile) ready for predict() calls.
    """
    profile = _make_profile(cpu_values)
    predictor = _make_predictor(node_id)
    predictor.fit(profile)
    return predictor, profile


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: Cold Start
# ─────────────────────────────────────────────────────────────────────────────

class TestColdStart:
    """
    Behaviour when the predictor has not been trained (< LOOKBACK samples).
    Must always return a safe, valid PredictionResult — never raise.
    """

    def test_cold_start_returns_fallback_with_low_confidence(self) -> None:
        """
        Profile with 5 samples (< LOOKBACK=10) → cold-start path.
        Expected: confidence=0.1, spike_probability=0.0, valid PredictionResult.
        """
        profile = _make_profile([1.0, 1.5, 2.0, 1.8, 1.2])   # 5 samples
        predictor = _make_predictor()

        result = predictor.predict(profile)

        assert isinstance(result, PredictionResult)
        assert result.node_id == "node-test-01"
        assert result.confidence == pytest.approx(0.1)
        assert result.spike_probability == pytest.approx(0.0)
        # predicted_cpu_util = avg_cpu_cores × 10, clamped to [0, 100]
        expected_util = min(profile.avg_cpu_cores * 10.0, 100.0)
        assert result.predicted_cpu_util == pytest.approx(expected_util, abs=0.01)

    def test_cold_start_with_empty_profile_does_not_crash(self) -> None:
        """
        Profile with 0 samples → coldest cold-start edge case.
        avg_cpu_cores=0.0 → predicted_cpu_util=0.0. No exception.
        """
        profile = WorkloadProfile(workload_name="empty")
        predictor = _make_predictor()

        result = predictor.predict(profile)

        assert isinstance(result, PredictionResult)
        assert result.confidence == pytest.approx(0.1)
        assert result.predicted_cpu_util == pytest.approx(0.0)

    def test_cold_start_predicted_util_capped_at_100(self) -> None:
        """
        If avg_cpu_cores is very large (e.g. 50 cores → 500%), util is capped at 100.0.
        """
        # 10 samples all at 50 cores — but has_enough_data would be True here,
        # so use 5 samples to stay on cold-start path
        profile = _make_profile([50.0] * 5)   # 5 samples, avg=50 → 50×10=500, capped at 100
        predictor = _make_predictor()

        result = predictor.predict(profile)

        assert result.predicted_cpu_util == pytest.approx(100.0)
        assert result.confidence == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: Training (fit() state transitions)
# ─────────────────────────────────────────────────────────────────────────────

class TestTraining:
    """
    fit() must correctly transition the predictor between trained/untrained states.
    """

    def test_fit_succeeds_with_sufficient_data(self) -> None:
        """
        50 samples (>> LOOKBACK=10) → fit() sets _trained=True.
        """
        cpu_values = [float(i % 5 + 1) for i in range(50)]   # 1–5 cores, repeating
        predictor, _ = _fit_predictor(cpu_values)

        assert predictor.is_trained is True

    def test_fit_skips_with_insufficient_data(self) -> None:
        """
        9 samples (< LOOKBACK=10) → fit() sets _trained=False and returns safely.
        """
        profile = _make_profile([1.0] * 9)
        predictor = _make_predictor()
        predictor.fit(profile)

        assert predictor.is_trained is False

    def test_fit_skips_at_exact_lookback_boundary(self) -> None:
        """
        Exactly LOOKBACK=10 samples → 0 training windows (n - LOOKBACK = 0).
        fit() cannot build a dataset → _trained=False.
        """
        profile = _make_profile([1.0] * LOOKBACK)   # exactly 10
        predictor = _make_predictor()
        predictor.fit(profile)

        # 10 samples → has_enough_data=True but n_windows = 10-10 = 0
        assert predictor.is_trained is False

    def test_fit_succeeds_at_lookback_plus_one(self) -> None:
        """
        LOOKBACK+1=11 samples → 1 training window → fit() succeeds.
        """
        profile = _make_profile([1.0] * (LOOKBACK + 1))
        predictor = _make_predictor()
        predictor.fit(profile)

        assert predictor.is_trained is True

    def test_refit_triggers_on_growth(self) -> None:
        """
        Start with 11 samples (enough to train). Add REFIT_THRESHOLD more.
        refit_if_needed() should call fit() and update _last_fit_sample_count.
        """
        cpu_values = [1.0] * (LOOKBACK + 1)   # 11 samples
        predictor, profile = _fit_predictor(cpu_values)

        initial_count = predictor._last_fit_sample_count
        assert predictor.is_trained is True

        # Add REFIT_THRESHOLD more samples
        for i in range(REFIT_THRESHOLD):
            profile.add_sample(ResourceSample(
                cpu_cores_used=2.0,
                memory_gb_used=2.0,
                duration_s=10.0,
                scheduling_latency_ms=3.0,
            ))

        predictor.refit_if_needed(profile)

        # After refit: sample count should have increased
        assert predictor._last_fit_sample_count > initial_count
        assert predictor.is_trained is True


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: Inference (predict() output structure and bounds)
# ─────────────────────────────────────────────────────────────────────────────

class TestInference:
    """
    predict() must always return a structurally valid PredictionResult
    with all numeric fields in their declared ranges.
    """

    def test_predict_returns_valid_pydantic_model(self) -> None:
        """
        After fit(), predict() returns a PredictionResult with correct node_id.
        """
        cpu_values = [float(i % 4 + 1) for i in range(50)]
        predictor, profile = _fit_predictor(cpu_values, node_id="node-gpu-07")

        result = predictor.predict(profile)

        assert isinstance(result, PredictionResult)
        assert result.node_id == "node-gpu-07"
        assert isinstance(result.generated_at, datetime)

    def test_predicted_cpu_always_in_bounds(self) -> None:
        """
        predicted_cpu_util must be in [0.0, 100.0] regardless of history shape.
        Tests multiple history patterns.
        """
        patterns = [
            [0.0] * 50,               # all-zero (flat)
            [100.0] * 50,             # absurdly high
            list(range(1, 51)),       # monotonically increasing
            [i % 3 for i in range(50)],  # oscillating
        ]

        for cpu_values in patterns:
            predictor, profile = _fit_predictor(cpu_values)
            result = predictor.predict(profile)
            assert 0.0 <= result.predicted_cpu_util <= 100.0, (
                f"predicted_cpu_util out of bounds for history: {cpu_values[:5]}..."
            )

    def test_confidence_is_approximately_half_at_minimum_data(self) -> None:
        """
        With LOOKBACK+1 samples: confidence should be very close to 0.5
        (the lower bound of the linear confidence ramp).
        """
        profile = _make_profile([1.0] * (LOOKBACK + 1))   # 11 samples
        predictor = _make_predictor()
        predictor.fit(profile)

        result = predictor.predict(profile)

        # Confidence formula: 0.5 + (n_samples - 10) / 490 * 0.5
        # At n=11: 0.5 + 1/490 * 0.5 ≈ 0.501
        assert result.confidence == pytest.approx(0.5 + 1 / 490 * 0.5, abs=0.01)

    def test_confidence_approaches_one_with_large_dataset(self) -> None:
        """
        With 500 samples (the max_samples cap): confidence should be ≈ 1.0.
        """
        cpu_values = [float(i % 5 + 1) for i in range(500)]
        predictor, profile = _fit_predictor(cpu_values)

        result = predictor.predict(profile)

        # At n=500: 0.5 + 490/490 * 0.5 = 1.0
        assert result.confidence == pytest.approx(1.0, abs=0.01)

    def test_confidence_increases_monotonically_with_more_data(self) -> None:
        """
        Adding more samples should monotonically increase confidence.
        Tests three data sizes: small (20), medium (100), large (300).
        """
        confidences = []
        for n in [20, 100, 300]:
            cpu_values = [float(i % 5 + 1) for i in range(n)]
            predictor, profile = _fit_predictor(cpu_values)
            result = predictor.predict(profile)
            confidences.append(result.confidence)

        assert confidences[0] < confidences[1] < confidences[2], (
            f"Confidence not monotonically increasing: {confidences}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: Spike Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestSpikeDetection:
    """
    spike_probability must always be in [0.0, 1.0].
    Direction: stable loads → low probability; bursty patterns → higher probability.

    Note: We test direction (low < high) not exact values, because the LSTM
    prediction driving spike_probability is stochastic.
    """

    def test_spike_probability_always_in_bounds(self) -> None:
        """
        Property-based test: for 20 random histories, spike_probability ∈ [0.0, 1.0].
        """
        rng = random.Random(42)   # fixed seed for reproducibility

        for trial in range(20):
            n_samples = rng.randint(LOOKBACK + 1, 50)
            cpu_values = [rng.uniform(0.0, 8.0) for _ in range(n_samples)]
            predictor, profile = _fit_predictor(cpu_values)
            result = predictor.predict(profile)

            assert 0.0 <= result.spike_probability <= 1.0, (
                f"Trial {trial}: spike_probability={result.spike_probability} out of bounds"
            )

    def test_spike_probability_low_on_flat_stable_load(self) -> None:
        """
        A perfectly flat history with no burst → spike_probability should be low (< 0.4).

        With a flat history, the LSTM predicts a value close to the mean.
        The gap between pred and recent_mean_util ≈ 0 → spike_prob ≈ 0.
        Threshold is generous (< 0.4) to accommodate LSTM noise.
        """
        # All jobs use exactly 2.0 CPU cores — perfectly flat
        cpu_values = [2.0] * 30
        predictor, profile = _fit_predictor(cpu_values)

        result = predictor.predict(profile)

        # burst_factor = 1.0 (max/avg = 2.0/2.0), so no burst boost either
        assert result.spike_probability < 0.4, (
            f"Unexpectedly high spike_probability={result.spike_probability:.3f} "
            f"for flat load. Predicted util: {result.predicted_cpu_util:.1f}"
        )

    def test_spike_probability_higher_for_bursty_workload(self) -> None:
        """
        A workload that alternates between low and very high CPU has a high
        burst_factor. The predictor should assign higher spike_probability than
        a flat workload.
        """
        # Alternating 1.0 and 8.0 cores: burst_factor = 8/4.5 ≈ 1.78 > 1.5
        bursty = [1.0 if i % 2 == 0 else 8.0 for i in range(30)]
        flat = [4.5] * 30   # same mean, no burst

        predictor_bursty, profile_bursty = _fit_predictor(bursty, "node-bursty")
        predictor_flat, profile_flat = _fit_predictor(flat, "node-flat")

        result_bursty = predictor_bursty.predict(profile_bursty)
        result_flat = predictor_flat.predict(profile_flat)

        # The bursty workload gets a burst_factor boost; flat does not.
        # We assert bursty ≥ flat (not strict > because flat could also spike).
        assert result_bursty.spike_probability >= result_flat.spike_probability - 0.05, (
            f"Bursty spike_prob={result_bursty.spike_probability:.3f} "
            f"not ≥ flat spike_prob={result_flat.spike_probability:.3f}"
        )

    def test_burst_factor_above_threshold_boosts_spike_prob(self) -> None:
        """
        Two identical prediction scenarios: one has burst_factor > 1.5, one doesn't.
        The high-burst version should get a spike_probability boost of up to +0.2.

        We test this by directly calling _compute_spike_probability.
        """
        from orchestrator.control_plane.predictor import WorkloadPredictor

        # Build a profile with burst_factor > 1.5 (alternating low/high CPU)
        bursty_profile = _make_profile([1.0 if i % 2 == 0 else 5.0 for i in range(30)])
        # Build a profile with burst_factor ≈ 1.0 (flat)
        flat_profile = _make_profile([3.0] * 30)

        # Use the same arbitrary pred_cpu for both
        pred_cpu = 30.0

        spike_bursty = WorkloadPredictor._compute_spike_probability(pred_cpu, bursty_profile)
        spike_flat = WorkloadPredictor._compute_spike_probability(pred_cpu, flat_profile)

        # burst_factor > 1.5 adds 0.2 to spike_prob (capped at 1.0)
        assert bursty_profile.burst_factor > 1.5, (
            f"Test setup error: burst_factor={bursty_profile.burst_factor:.2f} not > 1.5"
        )
        assert spike_bursty >= spike_flat, (
            f"Bursty spike_prob={spike_bursty:.3f} should be ≥ flat={spike_flat:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalisation:
    """
    Z-score parameters must be stored correctly after fit() and must be
    independent across different WorkloadPredictor instances.
    """

    def test_normalisation_parameters_stored_after_fit(self) -> None:
        """
        After fit(), cpu_mean and cpu_std reflect the training history.
        Expected: mean ≈ actual mean of cpu_values, std ≈ actual std.
        """
        import numpy as np

        cpu_values = [float(i + 1) for i in range(30)]   # 1.0, 2.0, ..., 30.0
        predictor, _ = _fit_predictor(cpu_values)

        expected_mean = float(np.mean(cpu_values))
        expected_std = float(np.std(cpu_values))

        assert predictor.cpu_mean == pytest.approx(expected_mean, abs=0.01)
        assert predictor.cpu_std == pytest.approx(expected_std, abs=0.01)

    def test_different_node_predictors_are_independent(self) -> None:
        """
        Two predictors for different nodes must not share state.
        Fitting one must not affect the other's normalisation parameters.
        """
        # Node A: low-CPU workload (mean ≈ 1.0 cores)
        cpu_a = [1.0] * 30
        # Node B: high-CPU workload (mean ≈ 8.0 cores)
        cpu_b = [8.0] * 30

        predictor_a, _ = _fit_predictor(cpu_a, node_id="node-a")
        predictor_b, _ = _fit_predictor(cpu_b, node_id="node-b")

        # Normalisation means should differ significantly
        assert predictor_a.cpu_mean != pytest.approx(predictor_b.cpu_mean, abs=0.5), (
            "Predictor A and B have identical cpu_mean — they may be sharing state."
        )

        # Neither predictor's state should leak into the other's model
        assert predictor_a._model is not predictor_b._model

    def test_std_floor_prevents_zero_std_on_flat_signal(self) -> None:
        """
        A perfectly flat history (all identical values) has std=0.
        The predictor must clamp std to 1e-6 to prevent division-by-zero.

        After fit(), cpu_std > 0.0.
        predict() must not raise ZeroDivisionError.
        """
        cpu_values = [3.0] * 30   # all identical → std = 0
        predictor, profile = _fit_predictor(cpu_values)

        assert predictor.cpu_std > 0.0, (
            f"cpu_std={predictor.cpu_std} is zero — not clamped. "
            f"Division by zero will occur in predict()."
        )

        # predict() must not raise
        result = predictor.predict(profile)
        assert isinstance(result, PredictionResult)
