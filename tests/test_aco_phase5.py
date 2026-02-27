"""
tests/test_aco_phase5.py
─────────────────────────
Test suite for Phase 5: Scheduler + Admission Controller + OrchestratorService.

Coverage: 28 tests across 4 groups.

What we are testing
────────────────────
Phase 5 wires together all previous phases:
  - Phase 2 (ACO Colony) ← Phase 4 (CostEngine η) ← Phase 3 (Predictor hints)
  into a coherent scheduling pipeline backed by OrchestratorService.

Test groups
────────────
Group 1: Admission Controller  — priority rules, GPU coherence, deadline checks
Group 2: aco_schedule()        — ACO placement correctness, fallback, hard gates
Group 3: naive_schedule()      — V1 First Fit (regression: unchanged behaviour)
Group 4: OrchestratorService   — end-to-end submit/complete lifecycle, metrics,
                                  telemetry ingestion, workload profile updates
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

import pytest

from orchestrator.control_plane.admission_controller import (
    AdmissionRejectedError,
    admit_job,
)
from orchestrator.control_plane.scheduler import (
    SchedulingFailedError,
    aco_schedule,
    naive_schedule,
)
from orchestrator.control_plane.orchestration_service import OrchestratorService
from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobExecution,
    JobRequest,
    JobState,
    NodeArch,
    NodeCostProfile,
    NodeState,
    NodeTelemetry,
    PredictionResult,
    ResourceRequest,
    WorkloadType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_node(
    node_id: str = "n-test",
    total_cpu: float = 16.0,
    allocated_cpu: float = 0.0,
    total_mem: float = 32.0,
    allocated_mem: float = 0.0,
    cost_per_hour: float = 0.10,
    instance_type: InstanceType = InstanceType.ON_DEMAND,
    interruption_prob: float = 0.0,
    arch: NodeArch = NodeArch.X86_64,
    state: NodeState = NodeState.HEALTHY,
) -> ComputeNode:
    return ComputeNode(
        node_id=node_id,
        state=state,
        arch=arch,
        total_cpu_cores=total_cpu,
        total_memory_gb=total_mem,
        allocated_cpu_cores=allocated_cpu,
        allocated_memory_gb=allocated_mem,
        cost_profile=NodeCostProfile(
            instance_type=instance_type,
            cost_per_hour_usd=cost_per_hour,
            interruption_prob=interruption_prob,
        ),
    )


def _make_job(
    job_id: str = "j-test",
    workload_type: WorkloadType = WorkloadType.BATCH,
    cpu_min: float = 2.0,
    mem_min: float = 4.0,
    priority: int = 50,
    gpu_required: bool = False,
    gpu_count: int = 1,
    arch_required: Optional[NodeArch] = None,
    cost_ceiling: Optional[float] = None,
    deadline_epoch: Optional[float] = None,
) -> JobRequest:
    return JobRequest(
        job_id=job_id,
        workload_type=workload_type,
        resources=ResourceRequest(
            cpu_cores_min=cpu_min,
            memory_gb_min=mem_min,
            gpu_required=gpu_required,
            gpu_count=gpu_count,
        ),
        priority=priority,
        arch_required=arch_required,
        cost_ceiling_usd=cost_ceiling,
        deadline_epoch=deadline_epoch,
    )


def _cluster(n: int = 3, cpu_per_node: float = 16.0) -> List[ComputeNode]:
    """Build a simple homogeneous cluster of n healthy nodes."""
    return [
        _make_node(node_id=f"node-{i:02d}", total_cpu=cpu_per_node)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: Admission Controller
# ─────────────────────────────────────────────────────────────────────────────

class TestAdmissionController:
    """
    admit_job() must enforce semantic rules before any scheduling occurs.
    """

    def test_valid_batch_job_passes(self) -> None:
        """A well-formed BATCH job with priority 50 passes admission."""
        job = _make_job(workload_type=WorkloadType.BATCH, priority=50)
        # Should not raise
        admit_job(job)

    def test_valid_lc_job_high_priority_passes(self) -> None:
        """LATENCY_CRITICAL job with priority ≥ 80 passes admission."""
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL, priority=95)
        admit_job(job)   # must not raise

    def test_lc_job_low_priority_rejected(self) -> None:
        """LATENCY_CRITICAL job with priority < 80 must be rejected."""
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL, priority=50)
        with pytest.raises(AdmissionRejectedError) as exc_info:
            admit_job(job)
        assert "priority" in str(exc_info.value).lower()

    def test_lc_job_exactly_priority_80_passes(self) -> None:
        """Boundary condition: priority=80 exactly passes for LC jobs."""
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL, priority=80)
        admit_job(job)   # must not raise

    def test_gpu_coherence_gpu_count_without_required_rejected(self) -> None:
        """gpu_required=False but gpu_count=4 is a misconfiguration → rejected."""
        job = _make_job(gpu_required=False, gpu_count=4)
        # Pydantic sets gpu_count min=1, so gpu_count=4 with gpu_required=False triggers our check
        with pytest.raises(AdmissionRejectedError) as exc_info:
            admit_job(job)
        assert "gpu" in str(exc_info.value).lower()

    def test_past_deadline_rejected(self) -> None:
        """A job with deadline already in the past must be rejected."""
        past_epoch = time.time() - 3600   # 1 hour ago
        job = _make_job(deadline_epoch=past_epoch)
        with pytest.raises(AdmissionRejectedError) as exc_info:
            admit_job(job)
        assert "deadline" in str(exc_info.value).lower()

    def test_future_deadline_passes(self) -> None:
        """A job with a deadline in the future passes admission."""
        future_epoch = time.time() + 3600   # 1 hour from now
        job = _make_job(deadline_epoch=future_epoch)
        admit_job(job)   # must not raise

    def test_stream_job_passes_without_constraints(self) -> None:
        """STREAM job with default settings passes all admission checks."""
        job = _make_job(workload_type=WorkloadType.STREAM, priority=60)
        admit_job(job)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: aco_schedule()
# ─────────────────────────────────────────────────────────────────────────────

class TestAcoSchedule:
    """
    aco_schedule() must return a valid node_id for feasible placements
    and raise SchedulingFailedError when no node can fit the job.
    """

    def test_returns_valid_node_id(self) -> None:
        """aco_schedule returns a node_id that exists in available_nodes."""
        nodes = _cluster(n=3)
        job = _make_job()

        node_id = aco_schedule(job, nodes)

        valid_ids = {n.node_id for n in nodes}
        assert node_id in valid_ids, f"Returned node_id={node_id!r} not in cluster"

    def test_raises_when_no_node_fits(self) -> None:
        """SchedulingFailedError raised when job requires more CPU than any node."""
        nodes = _cluster(n=3, cpu_per_node=4.0)
        job = _make_job(cpu_min=100.0)   # impossible

        with pytest.raises(SchedulingFailedError):
            aco_schedule(job, nodes)

    def test_raises_when_no_healthy_nodes(self) -> None:
        """SchedulingFailedError raised when all nodes are DEGRADED."""
        nodes = [
            _make_node(node_id=f"n{i}", state=NodeState.DEGRADED)
            for i in range(3)
        ]
        job = _make_job()

        with pytest.raises(SchedulingFailedError):
            aco_schedule(job, nodes)

    def test_lc_job_fast_path_deterministic(self) -> None:
        """
        Single LATENCY_CRITICAL job uses fast-path (argmax η).
        Same result on 20 consecutive calls → deterministic.
        """
        cheap = _make_node("cheap", total_cpu=32.0, cost_per_hour=0.05)
        expensive = _make_node("pricey", total_cpu=32.0, cost_per_hour=2.00)
        nodes = [cheap, expensive]

        lc_job = _make_job(
            workload_type=WorkloadType.LATENCY_CRITICAL,
            priority=90,
        )

        results = {aco_schedule(lc_job, nodes) for _ in range(20)}
        assert len(results) == 1, f"Fast path not deterministic: {results}"

    def test_cost_engine_prefers_cheaper_node(self) -> None:
        """
        With equal capacity, the CostEngine η should favour the cheaper node.
        Run 50 trials; cheaper node must win more often.
        """
        cheap = _make_node("cheap", total_cpu=32.0, cost_per_hour=0.05)
        expensive = _make_node("pricey", total_cpu=32.0, cost_per_hour=5.00)
        nodes = [cheap, expensive]
        job = _make_job(workload_type=WorkloadType.BATCH, priority=50)

        wins = {"cheap": 0, "pricey": 0}
        for _ in range(50):
            result = aco_schedule(job, nodes)
            wins[result] += 1

        assert wins["cheap"] > wins["pricey"], (
            f"Cheaper node won only {wins['cheap']}/50 times — CostEngine not working"
        )

    def test_spot_lc_above_threshold_hard_gated(self) -> None:
        """
        SPOT node with interruption_prob > 0.3 scores 0.0 for LC jobs (CostEngine
        reliability factor). aco_schedule must NOT place an LC job there.
        Provide one safe on-demand fallback.
        """
        from orchestrator.control_plane.cost_engine import SPOT_PENALTY_THRESHOLD
        risky_spot = _make_node(
            "risky",
            instance_type=InstanceType.SPOT,
            interruption_prob=SPOT_PENALTY_THRESHOLD + 0.1,
        )
        safe_od = _make_node("safe", instance_type=InstanceType.ON_DEMAND)
        nodes = [risky_spot, safe_od]

        lc_job = _make_job(
            workload_type=WorkloadType.LATENCY_CRITICAL, priority=90
        )

        result = aco_schedule(lc_job, nodes)
        assert result == "safe", (
            f"LC job should never land on risky SPOT node, got {result!r}"
        )

    def test_prediction_hints_used(self) -> None:
        """
        When a prediction for a node shows spike_probability=1.0,
        the CostEngine prediction_factor penalises it. The other node wins.
        """
        n_normal = _make_node("normal", total_cpu=32.0, cost_per_hour=0.10)
        n_spiking = _make_node("spiking", total_cpu=32.0, cost_per_hour=0.10)

        # Mark n_spiking as definitely going to spike (with full confidence)
        predictors = {
            "spiking": PredictionResult(
                node_id="spiking",
                predicted_cpu_util=99.0,
                predicted_memory_util=50.0,
                spike_probability=1.0,
                confidence=1.0,
            )
        }

        wins = {"normal": 0, "spiking": 0}
        job = _make_job(workload_type=WorkloadType.BATCH, priority=50)
        for _ in range(30):
            result = aco_schedule(job, [n_normal, n_spiking], predictors)
            wins[result] += 1

        assert wins["normal"] > wins["spiking"], (
            f"Node with spike prediction should win less: {wins}"
        )

    def test_performance_under_10ms_for_5_nodes(self) -> None:
        """
        aco_schedule must complete in < 10ms for 5 nodes (includes CostEngine overhead).
        Average over 10 runs.
        """
        import time
        nodes = [_make_node(f"n{i}", total_cpu=32.0) for i in range(5)]
        job = _make_job(workload_type=WorkloadType.BATCH)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            aco_schedule(job, nodes)
            times.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 10.0, f"aco_schedule avg={avg_ms:.2f}ms exceeds 10ms target"


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: naive_schedule() — V1 regression
# ─────────────────────────────────────────────────────────────────────────────

class TestNaiveSchedule:
    """
    naive_schedule() is kept verbatim from V1. These tests assert that
    V1 behaviour is preserved — no regression.
    """

    def test_returns_first_fit_node(self) -> None:
        """naive_schedule picks the first node that fits (First Fit semantics)."""
        nodes = [
            _make_node("n0", total_cpu=1.0),   # too small: cpu_min=2.0 won't fit
            _make_node("n1", total_cpu=16.0),  # fits
            _make_node("n2", total_cpu=16.0),  # also fits but comes second
        ]
        job = _make_job(cpu_min=2.0)

        result = naive_schedule(job, nodes)
        # n0 is too small, so First Fit returns n1
        assert result == "n1", f"Expected n1 (first fit), got {result!r}"

    def test_raises_when_no_node_fits(self) -> None:
        """naive_schedule raises SchedulingFailedError when no node can fit the job."""
        nodes = [_make_node("n0", total_cpu=1.0)]
        job = _make_job(cpu_min=100.0)

        with pytest.raises(SchedulingFailedError):
            naive_schedule(job, nodes)

    def test_skips_degraded_nodes(self) -> None:
        """naive_schedule only considers HEALTHY nodes."""
        nodes = [
            _make_node("n0", state=NodeState.DEGRADED),
            _make_node("n1", state=NodeState.HEALTHY),
        ]
        job = _make_job()

        result = naive_schedule(job, nodes)
        assert result == "n1"


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: OrchestratorService — end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorService:
    """
    End-to-end tests for the V2 OrchestratorService.
    Tests the full pipeline: submit → schedule → complete.
    """

    def _new_service(self) -> OrchestratorService:
        """Fresh service instance for each test."""
        return OrchestratorService()

    # ── Submit/Schedule ────────────────────────────────────────────────────────

    def test_submit_job_returns_scheduled_status(self) -> None:
        """Valid job submission returns {"status": "SCHEDULED", ...}."""
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 2.0, "memory_gb_min": 4.0},
            "priority": 50,
        })
        assert result["status"] == "SCHEDULED"
        assert result["node_id"] is not None
        assert result["job_id"].startswith("job-")

    def test_submit_lc_job_with_high_priority(self) -> None:
        """LATENCY_CRITICAL job with priority ≥ 80 schedules successfully."""
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "latency-critical",
            "resources": {"cpu_cores_min": 1.0, "memory_gb_min": 2.0},
            "priority": 95,
        })
        assert result["status"] == "SCHEDULED"

    def test_submit_lc_job_low_priority_rejected(self) -> None:
        """LATENCY_CRITICAL job with priority < 80 is rejected by admission."""
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "latency-critical",
            "resources": {"cpu_cores_min": 1.0, "memory_gb_min": 2.0},
            "priority": 30,
        })
        assert result["status"] == "REJECTED"
        assert "priority" in result["message"].lower()

    def test_submit_impossible_job_rejected(self) -> None:
        """Job requiring more CPU than any node has → REJECTED."""
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 9999.0, "memory_gb_min": 4.0},
            "priority": 50,
        })
        assert result["status"] == "REJECTED"

    def test_job_appears_in_active_jobs_after_submit(self) -> None:
        """After submit, the job appears in get_active_jobs()."""
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 2.0, "memory_gb_min": 4.0},
            "priority": 50,
        })
        job_id = result["job_id"]

        active = svc.get_active_jobs()
        assert any(j.job_id == job_id for j in active)

    # ── Complete ───────────────────────────────────────────────────────────────

    def test_complete_job_releases_resources(self) -> None:
        """
        After complete_job(), the node's allocated CPU drops back down.
        """
        svc = self._new_service()

        # Find the node with most space
        node_id_before = None
        cpu_before = None

        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 4.0, "memory_gb_min": 8.0},
            "priority": 50,
        })
        job_id = result["job_id"]
        assigned_node = result["node_id"]

        cpu_after_alloc = svc.node_state[assigned_node].allocated_cpu_cores

        svc.complete_job(job_id, success=True)

        cpu_after_release = svc.node_state[assigned_node].allocated_cpu_cores
        assert cpu_after_release < cpu_after_alloc, (
            f"CPU not released: before={cpu_after_alloc}, after={cpu_after_release}"
        )

    def test_complete_job_moves_to_completed_list(self) -> None:
        """After complete_job(), job is in completed_jobs, not active_jobs."""
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 2.0, "memory_gb_min": 4.0},
            "priority": 50,
        })
        job_id = result["job_id"]

        svc.complete_job(job_id, success=True)

        assert job_id not in svc.active_jobs
        completed_ids = [j.job_id for j in svc.completed_jobs]
        assert job_id in completed_ids

    def test_complete_nonexistent_job_returns_error(self) -> None:
        """Completing a job that doesn't exist returns error status."""
        svc = self._new_service()
        result = svc.complete_job("job-doesnotexist")
        assert result["status"] == "ERROR"

    def test_complete_job_updates_workload_profile(self) -> None:
        """
        When complete_job() is called with actual resource usage,
        the WorkloadProfile for that workload type gains a sample.
        """
        svc = self._new_service()
        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 2.0, "memory_gb_min": 4.0},
            "priority": 50,
        })
        job_id = result["job_id"]

        profile_before = len(svc.workload_profiles["batch"].samples)

        svc.complete_job(
            job_id,
            success=True,
            actual_cpu_used_cores=3.5,
            actual_memory_used_gb=5.0,
            actual_scheduling_latency_ms=4.2,
        )

        profile_after = len(svc.workload_profiles["batch"].samples)
        assert profile_after == profile_before + 1, (
            f"WorkloadProfile not updated: before={profile_before}, after={profile_after}"
        )

    # ── Telemetry ──────────────────────────────────────────────────────────────

    def test_update_node_telemetry_stored_on_node(self) -> None:
        """update_node_telemetry() updates node.latest_telemetry."""
        svc = self._new_service()
        target_node_id = list(svc.node_state.keys())[0]

        telemetry = NodeTelemetry(
            node_id=target_node_id,
            cpu_util_pct=72.5,
            memory_util_pct=45.0,
        )
        svc.update_node_telemetry(telemetry)

        node = svc.node_state[target_node_id]
        assert node.latest_telemetry is not None
        assert node.latest_telemetry.cpu_util_pct == pytest.approx(72.5)

    def test_unknown_node_telemetry_does_not_crash(self) -> None:
        """Telemetry for an unknown node_id is silently ignored."""
        svc = self._new_service()
        telemetry = NodeTelemetry(
            node_id="node-does-not-exist",
            cpu_util_pct=50.0,
            memory_util_pct=30.0,
        )
        # Should not raise
        svc.update_node_telemetry(telemetry)

    # ── Metrics ────────────────────────────────────────────────────────────────

    def test_get_scheduling_metrics_structure(self) -> None:
        """get_scheduling_metrics() returns all expected keys."""
        svc = self._new_service()
        metrics = svc.get_scheduling_metrics()

        assert "active_jobs" in metrics
        assert "completed_jobs" in metrics
        assert "scheduling_p99_ms" in metrics
        assert "avg_scheduling_ms" in metrics
        assert "node_utilisation" in metrics
        assert "profile_sample_counts" in metrics

    def test_metrics_active_count_after_submit(self) -> None:
        """active_jobs count in metrics increases after a successful submit."""
        svc = self._new_service()
        before = svc.get_scheduling_metrics()["active_jobs"]

        svc.submit_job({
            "workload_type": "batch",
            "resources": {"cpu_cores_min": 2.0, "memory_gb_min": 4.0},
            "priority": 50,
        })

        after = svc.get_scheduling_metrics()["active_jobs"]
        assert after == before + 1
