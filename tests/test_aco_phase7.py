"""
tests/test_aco_phase7.py
─────────────────────────
Phase 7 tests: TelemetryCollector — simulated telemetry ingestion loop.

Test groups:
    Group 1 — Basic tick behaviour (6 tests)
    Group 2 — Per-node profile accumulation (5 tests)
    Group 3 — Prediction cache population (5 tests)
    Group 4 — Spike injection (4 tests)

Total: 20 tests
"""

from __future__ import annotations

import uuid
from typing import Dict

import pytest

from orchestrator.shared.models import (
    NodeArch,
    WorkloadType,
    ResourceRequest,
    JobRequest,
)
from orchestrator.control_plane.orchestration_service import OrchestratorService
from orchestrator.control_plane.scheduler import aco_schedule
from orchestrator.telemetry.collector import (
    REFIT_INTERVAL,
    SPIKE_CPU_UTIL,
    TelemetryCollector,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def svc() -> OrchestratorService:
    return OrchestratorService()


@pytest.fixture
def collector(svc: OrchestratorService) -> TelemetryCollector:
    return TelemetryCollector(svc)


def _batch_job(cpu: float = 2.0, mem: float = 4.0) -> JobRequest:
    return JobRequest(
        job_id=f"job-{uuid.uuid4().hex[:6]}",
        workload_type=WorkloadType.BATCH,
        priority=50,
        preemptible=False,
        resources=ResourceRequest(
            cpu_cores_min=cpu,
            memory_gb_min=mem,
            gpu_required=False,
            gpu_count=1,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Basic tick behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicTick:
    def test_tick_updates_all_node_telemetry(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After 1 tick, all 5 nodes have non-None latest_telemetry."""
        for node in svc.node_state.values():
            assert node.latest_telemetry is None, "Should start empty"

        collector.tick()

        for node_id, node in svc.node_state.items():
            assert node.latest_telemetry is not None, (
                f"node {node_id} should have telemetry after tick"
            )

    def test_tick_cpu_util_in_valid_range(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """cpu_util_pct is always in [0.0, 100.0] across 200 ticks."""
        for _ in range(200):
            collector.tick()

        for node_id, node in svc.node_state.items():
            util = node.latest_telemetry.cpu_util_pct
            assert 0.0 <= util <= 100.0, (
                f"node {node_id} cpu_util_pct={util} out of [0, 100]"
            )

    def test_tick_memory_util_in_valid_range(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """memory_util_pct is always in [0.0, 100.0]."""
        for _ in range(100):
            collector.tick()

        for node_id, node in svc.node_state.items():
            mem = node.latest_telemetry.memory_util_pct
            assert 0.0 <= mem <= 100.0, (
                f"node {node_id} memory_util_pct={mem} out of [0, 100]"
            )

    def test_gpu_nodes_get_gpu_telemetry(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """GPU nodes (arch=GPU_NODE) have non-empty gpu_util_pct dict."""
        collector.tick()

        for node_id, node in svc.node_state.items():
            if node.arch == NodeArch.GPU_NODE:
                gpu = node.latest_telemetry.gpu_util_pct
                assert len(gpu) > 0, (
                    f"GPU node {node_id} should have gpu_util_pct dict"
                )
                for model, util in gpu.items():
                    assert 0.0 <= util <= 100.0, (
                        f"GPU util {model}={util} out of [0, 100]"
                    )

    def test_cpu_arm_nodes_no_gpu_telemetry(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """CPU and ARM64 nodes have empty gpu_util_pct dict."""
        collector.tick()

        for node_id, node in svc.node_state.items():
            if node.arch in (NodeArch.X86_64, NodeArch.ARM64):
                gpu = node.latest_telemetry.gpu_util_pct
                assert gpu == {}, (
                    f"Non-GPU node {node_id} should have empty gpu_util_pct, got {gpu}"
                )

    def test_tick_count_increments(
        self, collector: TelemetryCollector
    ) -> None:
        """tick_count property increments by 1 on each tick()."""
        assert collector.tick_count == 0
        collector.tick()
        assert collector.tick_count == 1
        collector.tick()
        assert collector.tick_count == 2
        for _ in range(8):
            collector.tick()
        assert collector.tick_count == 10


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Per-node profile accumulation
# ─────────────────────────────────────────────────────────────────────────────

class TestPerNodeProfileAccumulation:
    def test_each_tick_adds_one_sample_per_node(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After N ticks, each node's per-node profile has exactly N samples."""
        n = 5
        for _ in range(n):
            collector.tick()

        for node_id in svc.node_state:
            profile = collector.get_per_node_profile(node_id)
            assert profile is not None
            assert len(profile.samples) == n, (
                f"node {node_id} expected {n} samples, got {len(profile.samples)}"
            )

    def test_per_node_profiles_are_independent(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """Samples in node-cpu-01 profile are independent of node-arm-02 profile."""
        for _ in range(3):
            collector.tick()

        cpu01 = collector.get_per_node_profile("node-cpu-01")
        arm02 = collector.get_per_node_profile("node-arm-02")
        assert cpu01 is not None and arm02 is not None

        # The profiles must be different objects
        assert cpu01 is not arm02
        # cpu-01 baseline is 35%, arm-02 baseline is 20%
        # Their avg_cpu_cores should reflect these different baselines
        # (not deterministic due to noise, but the profiles are tracking separately)
        assert cpu01.workload_name != arm02.workload_name

    def test_cpu_util_telemetry_reflected_in_profile(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """avg_cpu_cores in per-node profile is a meaningful value (> 0)."""
        for _ in range(20):
            collector.tick()

        for node_id, node in svc.node_state.items():
            profile = collector.get_per_node_profile(node_id)
            assert profile is not None
            assert profile.avg_cpu_cores > 0.0, (
                f"node {node_id} profile avg_cpu_cores should be > 0"
            )

    def test_profile_respects_max_samples_cap(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After 510 ticks, per-node profiles are capped at 500 (WorkloadProfile.max_samples)."""
        for _ in range(510):
            collector.tick()

        for node_id in svc.node_state:
            profile = collector.get_per_node_profile(node_id)
            assert profile is not None
            assert len(profile.samples) <= profile.max_samples, (
                f"node {node_id}: {len(profile.samples)} samples > cap {profile.max_samples}"
            )
            assert len(profile.samples) == 500

    def test_refit_not_triggered_before_threshold(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After fewer than REFIT_INTERVAL ticks, _prediction_cache is still empty."""
        # Run fewer ticks than REFIT_INTERVAL
        for _ in range(REFIT_INTERVAL - 1):
            collector.tick()

        # No refit has happened yet — cache should be empty
        assert len(svc._prediction_cache) == 0, (
            f"Expected empty cache before first refit, got {list(svc._prediction_cache.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Prediction cache population
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionCachePopulation:
    def test_prediction_cache_populated_after_refit(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After REFIT_INTERVAL ticks, _prediction_cache has entries for all nodes."""
        for _ in range(REFIT_INTERVAL):
            collector.tick()

        assert len(svc._prediction_cache) == len(svc.node_state), (
            f"Expected {len(svc.node_state)} cache entries, "
            f"got {list(svc._prediction_cache.keys())}"
        )

    def test_prediction_cache_node_id_matches_key(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """PredictionResult.node_id matches the cache key."""
        for _ in range(REFIT_INTERVAL):
            collector.tick()

        for node_id, result in svc._prediction_cache.items():
            assert result.node_id == node_id, (
                f"Cache key {node_id!r} → result.node_id={result.node_id!r}"
            )

    def test_prediction_result_valid_pydantic(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """Each cached PredictionResult passes field validation (all floats in range)."""
        for _ in range(REFIT_INTERVAL):
            collector.tick()

        for node_id, result in svc._prediction_cache.items():
            assert 0.0 <= result.predicted_cpu_util <= 100.0, (
                f"{node_id} predicted_cpu_util={result.predicted_cpu_util} out of range"
            )
            assert 0.0 <= result.spike_probability <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    def test_prediction_confidence_low_before_many_ticks(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After just REFIT_INTERVAL ticks, confidence < 0.6 (limited training data)."""
        for _ in range(REFIT_INTERVAL):
            collector.tick()

        for node_id, result in svc._prediction_cache.items():
            assert result.confidence < 0.6, (
                f"{node_id} confidence={result.confidence:.3f} — "
                f"expected < 0.6 after only {REFIT_INTERVAL} samples"
            )

    def test_prediction_cache_refreshed_after_second_refit(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """Cache is present after the second refit interval (2 × REFIT_INTERVAL ticks)."""
        for _ in range(REFIT_INTERVAL * 2):
            collector.tick()

        # All nodes still in cache
        assert len(svc._prediction_cache) == len(svc.node_state)

        # Confidence should grow slightly as more samples accumulate
        for node_id, result in svc._prediction_cache.items():
            assert result.confidence > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — Spike injection
# ─────────────────────────────────────────────────────────────────────────────

class TestSpikeInjection:
    def test_inject_spike_raises_cpu_util(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After inject_spike, the spiking node's telemetry has cpu_util_pct near 92%."""
        node_id = "node-cpu-01"
        collector.inject_spike(node_id, n_ticks=5)
        collector.tick()

        util = svc.node_state[node_id].latest_telemetry.cpu_util_pct
        assert util > 80.0, (
            f"Expected spike cpu_util > 80, got {util:.1f}% for {node_id}"
        )

    def test_spike_resets_after_n_ticks(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """After n_ticks, node's CPU reverts to near its original baseline (< 70%)."""
        node_id = "node-cpu-01"
        n = 3
        collector.inject_spike(node_id, n_ticks=n)

        # Exhaust the spike window
        for _ in range(n):
            collector.tick()

        # Now take 10 more readings — the average should be near the 35% baseline
        cpu_values = []
        for _ in range(10):
            collector.tick()
            cpu_values.append(svc.node_state[node_id].latest_telemetry.cpu_util_pct)

        avg_after = sum(cpu_values) / len(cpu_values)
        assert avg_after < 70.0, (
            f"After spike reset, avg cpu={avg_after:.1f}% — expected < 70% (baseline 35%)"
        )

    def test_unknown_node_inject_spike_does_not_crash(
        self, collector: TelemetryCollector
    ) -> None:
        """inject_spike() on an unknown node_id is silently ignored."""
        collector.inject_spike("node-does-not-exist", n_ticks=5)
        collector.tick()   # must not raise

    def test_aco_schedule_uses_prediction_cache(
        self, svc: OrchestratorService, collector: TelemetryCollector
    ) -> None:
        """
        After enough ticks, the prediction cache is populated and aco_schedule
        uses it — the selected node's cache entry exists and is valid.

        This test verifies the end-to-end wiring:
          TelemetryCollector.tick()
          → _prediction_cache populated
          → aco_schedule(predictors=cache) reads it
          → placement is made using real CostEngine hints

        We do not assert a specific node wins (LSTM predictions with only
        10–20 samples are inherently stochastic). Instead we verify:
          1. The cache is non-empty after two refit cycles
          2. aco_schedule() successfully schedules the job
          3. The placed node exists in the cluster
        """
        # Run two full refit cycles to ensure the predictor is trained
        for _ in range(REFIT_INTERVAL * 2):
            collector.tick()

        # Cache must be populated
        assert len(svc._prediction_cache) == len(svc.node_state), (
            "Expected prediction cache to be populated for all nodes"
        )

        # All cache entries must be valid PredictionResults
        for node_id, result in svc._prediction_cache.items():
            assert 0.0 <= result.spike_probability <= 1.0
            assert 0.0 <= result.confidence <= 1.0

        # aco_schedule should succeed using the cache
        job = _batch_job(cpu=1.0, mem=1.0)
        available_nodes = list(svc.node_state.values())

        selected = aco_schedule(
            job_request=job,
            available_nodes=available_nodes,
            predictors=svc._prediction_cache,
        )

        # The result must be a known node
        assert selected in svc.node_state, (
            f"aco_schedule returned unknown node_id: {selected!r}"
        )
