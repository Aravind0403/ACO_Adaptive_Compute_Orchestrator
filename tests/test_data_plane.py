"""
tests/test_data_plane.py
─────────────────────────
Phase 8 — NodeAgent (Data Plane Agent)

Tests for the async NodeAgent class that simulates per-node job execution and
sends telemetry heartbeats to OrchestratorService.

All tests are async (pytest-asyncio auto mode via pytest.ini: asyncio_mode = auto).

Test groups:
  Group 1 — Initialisation (4 tests)
  Group 2 — Job execution (6 tests)
  Group 3 — Heartbeat (5 tests)
  Group 4 — End-to-end (5 tests)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from orchestrator.control_plane.orchestration_service import OrchestratorService
from orchestrator.data_plane.agent import (
    NodeAgent,
    CPU_USAGE_RATIO,
    MEM_USAGE_RATIO,
    USAGE_BURST_CAP,
    HEARTBEAT_INTERVAL_S,
)
from orchestrator.shared.models import (
    JobExecution,
    JobRequest,
    JobState,
    ResourceRequest,
    WorkloadType,
)
from orchestrator.telemetry.trace_adapter import AlibabaMachineTraceAdapter

# ── Fixtures ────────────────────────────────────────────────────────────────────

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "alibaba_machine_usage_300s.csv"

# Node IDs from the mock cluster
CPU_NODE = "node-cpu-01"    # x86_64 ON_DEMAND, 32 cores
ARM_NODE = "node-arm-02"    # ARM64 SPOT, 32 cores
API_NODE = "node-api-03"    # x86_64 ON_DEMAND, 8 cores
GPU_NODE = "node-gpu-04"    # GPU_NODE ON_DEMAND, 16 cores + 4× A100


def _make_job_request(
    workload_type: WorkloadType = WorkloadType.BATCH,
    cpu_cores: float = 4.0,
    memory_gb: float = 8.0,
    gpu_required: bool = False,
) -> JobRequest:
    return JobRequest(
        job_id=f"j-{uuid.uuid4().hex[:8]}",
        workload_type=workload_type,
        resources=ResourceRequest(
            cpu_cores_min=cpu_cores,
            memory_gb_min=memory_gb,
            gpu_required=gpu_required,
            gpu_count=1,  # ge=1 constraint
        ),
        priority=50,
    )


def _make_job_execution(
    svc: OrchestratorService,
    node_id: str,
    **kwargs,
) -> JobExecution:
    """Create a JobExecution that is registered in svc.active_jobs."""
    job_request = _make_job_request(**kwargs)
    now = datetime.utcnow()
    ex = JobExecution(
        job_id=job_request.job_id,
        job_request=job_request,
        assigned_node_id=node_id,
        state=JobState.RUNNING,
        submitted_at=now,
        scheduled_at=now,
        started_at=now,
        scheduling_latency_ms=1.0,
    )
    # Register in service so complete_job() can find it
    svc.active_jobs[ex.job_id] = ex
    # Allocate resources so _release_resources works correctly
    svc._allocate_resources(node_id, job_request)
    return ex


@pytest.fixture
def svc() -> OrchestratorService:
    return OrchestratorService()


@pytest.fixture
def agent(svc) -> NodeAgent:
    return NodeAgent(CPU_NODE, svc)


@pytest.fixture(scope="module")
def adapter() -> AlibabaMachineTraceAdapter:
    return AlibabaMachineTraceAdapter(FIXTURE_CSV)


# ── Group 1: Initialisation ─────────────────────────────────────────────────────


class TestInitialisation:

    def test_agent_init_valid_node(self, svc):
        """NodeAgent initialises without error for a known node_id."""
        agent = NodeAgent(CPU_NODE, svc)
        assert agent.node_id == CPU_NODE
        assert len(agent.running_jobs) == 0

    def test_agent_init_unknown_node_raises(self, svc):
        """NodeAgent raises ValueError for an unknown node_id."""
        with pytest.raises(ValueError, match="not found in OrchestratorService"):
            NodeAgent("nonexistent-node-xyz", svc)

    def test_agent_repr_contains_node_id(self, agent):
        """repr() contains the node_id for easy debugging."""
        assert CPU_NODE in repr(agent)

    def test_heartbeat_task_none_on_init(self, agent):
        """Heartbeat task is None before start() is called."""
        assert agent._heartbeat_task is None


# ── Group 2: Job execution ──────────────────────────────────────────────────────


class TestJobExecution:

    async def test_execute_job_appears_in_completed_jobs(self, svc, agent):
        """After execute_job(), the job appears in svc.completed_jobs."""
        ex = _make_job_execution(svc, CPU_NODE)
        await agent.execute_job(ex)
        completed_ids = [j.job_id for j in svc.completed_jobs]
        assert ex.job_id in completed_ids

    async def test_execute_job_cleans_up_running_jobs(self, svc, agent):
        """_running_jobs is empty after execution finishes."""
        ex = _make_job_execution(svc, CPU_NODE)
        await agent.execute_job(ex)
        assert ex.job_id not in agent.running_jobs

    async def test_actual_cpu_within_bounds(self, svc, agent):
        """Actual CPU reported to service is > 0 and ≤ requested × USAGE_BURST_CAP.
        Verified via the ResourceSample added to the WorkloadProfile by complete_job()."""
        requested_cpu = 4.0
        ex = _make_job_execution(svc, CPU_NODE, cpu_cores=requested_cpu)
        samples_before = len(svc.workload_profiles[WorkloadType.BATCH.value].samples)
        await agent.execute_job(ex)
        samples = svc.workload_profiles[WorkloadType.BATCH.value].samples
        assert len(samples) > samples_before, "No sample was added to the workload profile"
        actual_cpu = samples[-1].cpu_cores_used
        assert 0.0 < actual_cpu <= requested_cpu * USAGE_BURST_CAP

    async def test_actual_memory_within_bounds(self, svc, agent):
        """Actual memory reported to service is > 0 and ≤ requested × USAGE_BURST_CAP.
        Verified via the ResourceSample added to the WorkloadProfile by complete_job()."""
        requested_mem = 8.0
        ex = _make_job_execution(svc, CPU_NODE, memory_gb=requested_mem)
        samples_before = len(svc.workload_profiles[WorkloadType.BATCH.value].samples)
        await agent.execute_job(ex)
        samples = svc.workload_profiles[WorkloadType.BATCH.value].samples
        assert len(samples) > samples_before, "No sample was added to the workload profile"
        actual_mem = samples[-1].memory_gb_used
        assert 0.0 < actual_mem <= requested_mem * USAGE_BURST_CAP

    async def test_workload_profile_updated_on_completion(self, svc, agent):
        """WorkloadProfile gains a sample when complete_job() receives real usage."""
        wt = WorkloadType.BATCH
        profile_before = svc.workload_profiles[wt.value].sample_count
        ex = _make_job_execution(svc, CPU_NODE, workload_type=wt, cpu_cores=4.0)
        await agent.execute_job(ex)
        profile_after = svc.workload_profiles[wt.value].sample_count
        assert profile_after > profile_before

    async def test_concurrent_jobs_all_complete(self, svc):
        """Three concurrent jobs on the same node all complete independently."""
        agent = NodeAgent(CPU_NODE, svc)
        executions = [
            _make_job_execution(svc, CPU_NODE, cpu_cores=2.0)
            for _ in range(3)
        ]
        await asyncio.gather(*(agent.execute_job(ex) for ex in executions))
        completed_ids = {j.job_id for j in svc.completed_jobs}
        for ex in executions:
            assert ex.job_id in completed_ids


# ── Group 3: Heartbeat ──────────────────────────────────────────────────────────


class TestHeartbeat:

    async def test_heartbeat_updates_node_telemetry(self, svc, agent):
        """After start() + a short wait, node.latest_telemetry is populated."""
        assert svc.node_state[CPU_NODE].latest_telemetry is None
        await agent.start()
        # Wait slightly more than one heartbeat interval
        await asyncio.sleep(HEARTBEAT_INTERVAL_S + 0.1)
        await agent.stop()
        assert svc.node_state[CPU_NODE].latest_telemetry is not None

    async def test_heartbeat_cpu_in_valid_range(self, svc, agent):
        """Heartbeat telemetry has cpu_util_pct in [0, 100]."""
        await agent.start()
        await asyncio.sleep(HEARTBEAT_INTERVAL_S + 0.1)
        await agent.stop()
        cpu = svc.node_state[CPU_NODE].latest_telemetry.cpu_util_pct
        assert 0.0 <= cpu <= 100.0

    async def test_heartbeat_stop_cancels_task(self, svc, agent):
        """After stop(), the heartbeat task is done."""
        await agent.start()
        await asyncio.sleep(0.05)
        await agent.stop()
        assert agent._heartbeat_task.done()

    async def test_heartbeat_is_idempotent(self, svc, agent):
        """Calling start() twice creates only one background task."""
        await agent.start()
        task_before = agent._heartbeat_task
        await agent.start()  # second call — should be a no-op
        assert agent._heartbeat_task is task_before
        await agent.stop()

    async def test_stop_before_start_does_not_crash(self, svc, agent):
        """stop() without a prior start() is a safe no-op."""
        await agent.stop()  # should not raise


# ── Group 4: End-to-end ─────────────────────────────────────────────────────────


class TestEndToEnd:

    async def test_submit_execute_complete_pipeline(self, svc):
        """Full pipeline: submit_job → create JobExecution → execute_job → resources released."""
        result = svc.submit_job({
            "workload_type": "batch",
            "resources": {
                "cpu_cores_min": 4.0,
                "memory_gb_min": 8.0,
                "gpu_required": False,
                "gpu_count": 1,
            },
            "priority": 50,
        })
        assert result["status"] == "SCHEDULED"
        job_id = result["job_id"]
        node_id = result["node_id"]

        # Get the JobExecution created by submit_job
        job_ex = svc.active_jobs[job_id]

        # Run via agent
        agent = NodeAgent(node_id, svc)
        cpu_before = svc.node_state[node_id].allocated_cpu_cores
        await agent.execute_job(job_ex)

        # Resources must be released
        cpu_after = svc.node_state[node_id].allocated_cpu_cores
        assert cpu_after < cpu_before

        # Job must be in completed_jobs
        completed_ids = [j.job_id for j in svc.completed_jobs]
        assert job_id in completed_ids

    async def test_multiple_agents_independent(self, svc):
        """Two agents on different nodes complete their own jobs, no cross-contamination."""
        agent_cpu = NodeAgent(CPU_NODE, svc)
        agent_arm = NodeAgent(ARM_NODE, svc)

        ex_cpu = _make_job_execution(svc, CPU_NODE, cpu_cores=4.0)
        ex_arm = _make_job_execution(svc, ARM_NODE, cpu_cores=4.0)

        await asyncio.gather(
            agent_cpu.execute_job(ex_cpu),
            agent_arm.execute_job(ex_arm),
        )

        completed_ids = {j.job_id for j in svc.completed_jobs}
        assert ex_cpu.job_id in completed_ids
        assert ex_arm.job_id in completed_ids

    async def test_completed_job_releases_cpu(self, svc):
        """Node's allocated_cpu_cores decreases after agent completes the job."""
        agent = NodeAgent(CPU_NODE, svc)
        cpu_cores = 8.0
        ex = _make_job_execution(svc, CPU_NODE, cpu_cores=cpu_cores)
        allocated_before = svc.node_state[CPU_NODE].allocated_cpu_cores
        await agent.execute_job(ex)
        allocated_after = svc.node_state[CPU_NODE].allocated_cpu_cores
        assert allocated_after == allocated_before - cpu_cores

    async def test_agent_with_trace_adapter(self, svc, adapter):
        """NodeAgent with trace_adapter sends real Alibaba telemetry in heartbeat."""
        agent = NodeAgent(CPU_NODE, svc, trace_adapter=adapter)
        await agent.start()
        await asyncio.sleep(HEARTBEAT_INTERVAL_S + 0.1)
        await agent.stop()
        telemetry = svc.node_state[CPU_NODE].latest_telemetry
        assert telemetry is not None
        # Trace adapter returns deterministic values — run again to verify determinism
        agent2 = NodeAgent(CPU_NODE, OrchestratorService(), trace_adapter=adapter)
        await agent2.start()
        await asyncio.sleep(HEARTBEAT_INTERVAL_S + 0.1)
        await agent2.stop()
        telemetry2 = agent2._service.node_state[CPU_NODE].latest_telemetry
        # Both should have used tick=0 for first heartbeat → same CPU value
        assert abs(telemetry.cpu_util_pct - telemetry2.cpu_util_pct) < 1e-6

    async def test_gpu_node_agent_sends_gpu_telemetry(self, svc):
        """NodeAgent on a GPU node includes gpu_util_pct in its heartbeat."""
        agent = NodeAgent(GPU_NODE, svc)
        await agent.start()
        await asyncio.sleep(HEARTBEAT_INTERVAL_S + 0.1)
        await agent.stop()
        telemetry = svc.node_state[GPU_NODE].latest_telemetry
        assert telemetry is not None
        assert "A100" in telemetry.gpu_util_pct
        assert 0.0 <= telemetry.gpu_util_pct["A100"] <= 100.0
