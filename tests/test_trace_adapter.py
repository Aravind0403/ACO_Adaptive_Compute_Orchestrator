"""
tests/test_trace_adapter.py
────────────────────────────
Phase 7.5 — Alibaba Cluster Trace Replay Adapter

Tests for AlibabaMachineTraceAdapter and its integration with TelemetryCollector.

The Alibaba 2018 cluster trace CSV (tests/fixtures/alibaba_machine_usage_300s.csv)
is required. It is a 202 KB file downloaded from Zenodo (record 14564935) and
committed to the repository under tests/fixtures/.

Test groups:
  Group 1 — Adapter loading (4 tests)
  Group 2 — Trace replay semantics (4 tests)
  Group 3 — TelemetryCollector integration (4 tests)
"""

from __future__ import annotations

import math
import pytest
from pathlib import Path

from orchestrator.telemetry.trace_adapter import AlibabaMachineTraceAdapter
from orchestrator.telemetry.collector import TelemetryCollector, REFIT_INTERVAL, SPIKE_CPU_UTIL
from orchestrator.control_plane.orchestration_service import OrchestratorService

# ── Fixtures ────────────────────────────────────────────────────────────────────

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "alibaba_machine_usage_300s.csv"

KNOWN_NODES = [
    "node-cpu-01",
    "node-arm-02",
    "node-api-03",
    "node-gpu-04",
    "node-gpu-05",
]


@pytest.fixture(scope="module")
def adapter() -> AlibabaMachineTraceAdapter:
    """Load the Alibaba trace adapter once for the whole module (fast: 202 KB CSV)."""
    return AlibabaMachineTraceAdapter(FIXTURE_CSV)


@pytest.fixture
def svc() -> OrchestratorService:
    return OrchestratorService()


@pytest.fixture
def collector_trace(svc, adapter) -> TelemetryCollector:
    """TelemetryCollector wired with the real trace adapter."""
    return TelemetryCollector(svc, trace_adapter=adapter)


@pytest.fixture
def collector_gauss(svc) -> TelemetryCollector:
    """Default TelemetryCollector (Gaussian path, no adapter)."""
    return TelemetryCollector(svc)


# ── Group 1: Adapter loading ────────────────────────────────────────────────────


class TestAdapterLoading:

    def test_loads_csv_successfully(self, adapter):
        """Adapter initialises without exceptions and has a positive trace length."""
        assert adapter.trace_length > 0

    def test_trace_has_expected_length(self, adapter):
        """8 days at 300-second resolution = 2304 intervals; CSV has 2243 (missing ~61)."""
        # Allow a range — the exact row count depends on the dataset version
        assert 2000 < adapter.trace_length < 2500

    def test_all_five_nodes_mapped(self, adapter):
        """All 5 mock node IDs are recognised by the adapter."""
        returned_nodes = adapter.node_ids()
        for node_id in KNOWN_NODES:
            assert node_id in returned_nodes, f"{node_id} not in adapter node_ids()"

    def test_unknown_node_returns_fallback(self, adapter):
        """Unknown node_id returns the (40.0, 50.0) fallback without crashing."""
        cpu, mem = adapter.get_reading("unknown-node-xyz", tick_number=0)
        assert cpu == 40.0
        assert mem == 50.0


# ── Group 2: Trace replay semantics ────────────────────────────────────────────


class TestTraceReplaySemantics:

    def test_reading_in_valid_range(self, adapter):
        """Every reading for every known node at tick 0 is within [0, 100]."""
        for node_id in KNOWN_NODES:
            cpu, mem = adapter.get_reading(node_id, tick_number=0)
            assert 0.0 <= cpu <= 100.0, f"{node_id} cpu={cpu} out of range"
            assert 0.0 <= mem <= 100.0, f"{node_id} mem={mem} out of range"

    def test_trace_wraps_around(self, adapter):
        """Reading at tick == trace_length matches reading at tick == 0 (circular)."""
        for node_id in KNOWN_NODES:
            cpu_0, mem_0 = adapter.get_reading(node_id, tick_number=0)
            cpu_wrap, mem_wrap = adapter.get_reading(node_id, tick_number=adapter.trace_length)
            assert math.isclose(cpu_0, cpu_wrap, rel_tol=1e-9), (
                f"{node_id}: wrap mismatch cpu {cpu_0} vs {cpu_wrap}"
            )
            assert math.isclose(mem_0, mem_wrap, rel_tol=1e-9), (
                f"{node_id}: wrap mismatch mem {mem_0} vs {mem_wrap}"
            )

    def test_different_ticks_return_different_values(self, adapter):
        """Two different tick numbers produce different CPU values (real variation exists)."""
        cpus = {adapter.get_reading("node-cpu-01", tick_number=t)[0] for t in range(50)}
        # With real trace data there should be at least several distinct values in 50 ticks
        assert len(cpus) > 5, f"Too few distinct CPU readings in 50 ticks: {len(cpus)}"

    def test_different_nodes_get_different_readings(self, adapter):
        """Different nodes return different CPU readings at the same tick (due to offsets)."""
        cpus_at_tick_0 = {node: adapter.get_reading(node, 0)[0] for node in KNOWN_NODES}
        # All 5 nodes must be distinct at tick 0 (different offsets into the trace)
        unique_values = set(cpus_at_tick_0.values())
        assert len(unique_values) == len(KNOWN_NODES), (
            f"Nodes share CPU readings at tick 0: {cpus_at_tick_0}"
        )

    def test_readings_are_deterministic(self, adapter):
        """Same (node_id, tick_number) always returns the same values."""
        for node_id in KNOWN_NODES:
            for tick in (0, 100, 500, adapter.trace_length - 1):
                r1 = adapter.get_reading(node_id, tick)
                r2 = adapter.get_reading(node_id, tick)
                assert r1 == r2, f"{node_id} tick={tick}: {r1} != {r2}"


# ── Group 3: TelemetryCollector integration ─────────────────────────────────────


class TestCollectorIntegration:

    def test_collector_uses_trace_when_adapter_provided(self, svc, adapter):
        """After one tick, node telemetry CPU matches what the adapter would return for tick 0."""
        # Adapter returns reading for tick_number=self._tick_count which is 0 before tick()
        # After tick(), tick_count becomes 1, but _generate_telemetry is called when count=0
        # (tick_count is incremented BEFORE the node loop in tick())
        # So for tick 1, get_reading is called with tick_number=1
        collector = TelemetryCollector(svc, trace_adapter=adapter)
        collector.tick()  # tick_count goes 0→1; _generate_telemetry called with tick_count=1

        for node_id in KNOWN_NODES:
            node_state = svc.node_state.get(node_id)
            assert node_state is not None
            assert node_state.latest_telemetry is not None
            cpu_actual = node_state.latest_telemetry.cpu_util_pct
            cpu_expected, _ = adapter.get_reading(node_id, tick_number=1)
            assert math.isclose(cpu_actual, cpu_expected, rel_tol=1e-9), (
                f"{node_id}: telemetry cpu {cpu_actual} != adapter reading {cpu_expected}"
            )

    def test_collector_still_runs_without_adapter(self, collector_gauss):
        """Default TelemetryCollector (no adapter) runs tick() without errors."""
        for _ in range(3):
            collector_gauss.tick()
        for node_id in KNOWN_NODES:
            node_state = collector_gauss._service.node_state.get(node_id)
            assert node_state is not None
            assert node_state.latest_telemetry is not None
            cpu = node_state.latest_telemetry.cpu_util_pct
            assert 0.0 <= cpu <= 100.0

    def test_inject_spike_overrides_trace_value(self, svc, adapter):
        """inject_spike() raises CPU to SPIKE_CPU_UTIL even when trace adapter is active."""
        collector = TelemetryCollector(svc, trace_adapter=adapter)
        collector.inject_spike("node-cpu-01", n_ticks=3)
        collector.tick()
        telemetry = svc.node_state["node-cpu-01"].latest_telemetry
        assert telemetry.cpu_util_pct >= SPIKE_CPU_UTIL, (
            f"Spike override failed: cpu={telemetry.cpu_util_pct}, expected >={SPIKE_CPU_UTIL}"
        )

    def test_refit_works_with_trace_data(self, svc, adapter):
        """After REFIT_INTERVAL ticks with trace data, prediction cache is populated and valid."""
        collector = TelemetryCollector(svc, trace_adapter=adapter)
        for _ in range(REFIT_INTERVAL):
            collector.tick()

        # Prediction cache should be populated (refit triggered at tick == REFIT_INTERVAL)
        assert len(svc._prediction_cache) == len(svc.node_state), (
            f"Cache has {len(svc._prediction_cache)} entries, expected {len(svc.node_state)}"
        )
        for node_id, result in svc._prediction_cache.items():
            assert result.node_id == node_id
            assert 0.0 <= result.spike_probability <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.predicted_cpu_util <= 100.0
