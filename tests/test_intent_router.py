"""
tests/test_intent_router.py
────────────────────────────
Phase 6 tests: WorkloadIntentRouter + SchedulingStrategy integration.

Test groups:
    Group 1 — Strategy classification (6 tests)
    Group 2 — Deadline-urgent override (3 tests)
    Group 3 — Strategy-aware node filtering via aco_schedule (3 tests)
    Group 4 — CostEngine threshold overrides (3 tests)
    Group 5 — Colocation policy (4 tests)

Total: 19 tests
"""

from __future__ import annotations

import time
import uuid
from typing import List

import pytest

from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobRequest,
    NodeArch,
    NodeCostProfile,
    NodeState,
    ResourceRequest,
    WorkloadType,
    PredictionResult,
)
from orchestrator.control_plane.intent_router import (
    DEADLINE_URGENT_WINDOW_S,
    SLA_DEADLINE_BOOST,
    SchedulingStrategy,
    WorkloadIntentRouter,
)
from orchestrator.control_plane.scheduler import aco_schedule, SchedulingFailedError
from orchestrator.control_plane.cost_engine import CostEngine, SLA_STRICT_THRESHOLD
from orchestrator.control_plane.orchestration_service import OrchestratorService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _job(
    workload_type: WorkloadType = WorkloadType.BATCH,
    gpu_required: bool = False,
    gpu_count: int = 1,
    preemptible: bool = False,
    priority: int = 50,
    deadline_epoch: float | None = None,
    cpu_cores_min: float = 2.0,
    memory_gb_min: float = 4.0,
) -> JobRequest:
    """Build a minimal valid JobRequest for testing."""
    return JobRequest(
        job_id=f"job-{uuid.uuid4().hex[:6]}",
        workload_type=workload_type,
        priority=priority,
        preemptible=preemptible,
        deadline_epoch=deadline_epoch,
        resources=ResourceRequest(
            cpu_cores_min=cpu_cores_min,
            memory_gb_min=memory_gb_min,
            gpu_required=gpu_required,
            gpu_count=gpu_count,  # model default=1, ge=1 — always valid
        ),
    )


def _lc_job(**kwargs) -> JobRequest:
    """Shorthand: LATENCY_CRITICAL job with priority ≥ 80."""
    return _job(workload_type=WorkloadType.LATENCY_CRITICAL, priority=90, **kwargs)


def _batch_job(**kwargs) -> JobRequest:
    return _job(workload_type=WorkloadType.BATCH, **kwargs)


def _stream_job(**kwargs) -> JobRequest:
    return _job(workload_type=WorkloadType.STREAM, **kwargs)


def _node(
    node_id: str = "node-x",
    arch: NodeArch = NodeArch.X86_64,
    instance_type: InstanceType = InstanceType.ON_DEMAND,
    interruption_prob: float = 0.0,
    cpu: float = 16.0,
    mem: float = 32.0,
    cost: float = 0.20,
    gpu_inventory: dict | None = None,
) -> ComputeNode:
    return ComputeNode(
        node_id=node_id,
        arch=arch,
        total_cpu_cores=cpu,
        total_memory_gb=mem,
        gpu_inventory=gpu_inventory or {},
        cost_profile=NodeCostProfile(
            instance_type=instance_type,
            cost_per_hour_usd=cost,
            interruption_prob=interruption_prob,
            region="us-east-1",
        ),
    )


@pytest.fixture
def router() -> WorkloadIntentRouter:
    return WorkloadIntentRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Strategy classification
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyClassification:
    def test_gpu_inference_classified_correctly(self, router: WorkloadIntentRouter):
        """LC + gpu_required=True → GPU_INFERENCE strategy."""
        job = _lc_job(gpu_required=True, gpu_count=1)
        strategy = router.classify(job)
        assert strategy.name == "GPU_INFERENCE"

    def test_cpu_serving_classified_correctly(self, router: WorkloadIntentRouter):
        """LC + gpu_required=False → CPU_SERVING strategy."""
        job = _lc_job(gpu_required=False)
        strategy = router.classify(job)
        assert strategy.name == "CPU_SERVING"

    def test_gpu_training_non_preemptible(self, router: WorkloadIntentRouter):
        """BATCH + gpu=True + preemptible=False → GPU_TRAINING."""
        job = _batch_job(gpu_required=True, gpu_count=1, preemptible=False)
        strategy = router.classify(job)
        assert strategy.name == "GPU_TRAINING"

    def test_preemptible_actor_batch(self, router: WorkloadIntentRouter):
        """BATCH + preemptible=True → PREEMPTIBLE_ACTOR."""
        job = _batch_job(preemptible=True)
        strategy = router.classify(job)
        assert strategy.name == "PREEMPTIBLE_ACTOR"

    def test_preemptible_actor_stream(self, router: WorkloadIntentRouter):
        """STREAM + preemptible=True → PREEMPTIBLE_ACTOR."""
        job = _stream_job(preemptible=True)
        strategy = router.classify(job)
        assert strategy.name == "PREEMPTIBLE_ACTOR"

    def test_stateful_stream_non_preemptible(self, router: WorkloadIntentRouter):
        """STREAM + preemptible=False → STATEFUL_STREAM."""
        job = _stream_job(preemptible=False)
        strategy = router.classify(job)
        assert strategy.name == "STATEFUL_STREAM"


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Deadline-urgent override
# ─────────────────────────────────────────────────────────────────────────────

class TestDeadlineOverride:
    def test_deadline_imminent_forces_fast_path(self, router: WorkloadIntentRouter):
        """Deadline < 60s → use_fast_path forced True even for colony strategy."""
        job = _batch_job(
            deadline_epoch=time.time() + 30.0,  # 30s away — under the 60s window
        )
        strategy = router.classify(job)
        assert strategy.use_fast_path is True

    def test_deadline_non_imminent_no_override(self, router: WorkloadIntentRouter):
        """Deadline > 60s → base strategy unchanged (GENERIC has use_fast_path=False)."""
        job = _batch_job(
            deadline_epoch=time.time() + 120.0,  # 2 min away — over the window
        )
        strategy = router.classify(job)
        # GENERIC base strategy has use_fast_path=False; override shouldn't apply
        assert strategy.use_fast_path is False

    def test_deadline_tightens_sla_threshold(self, router: WorkloadIntentRouter):
        """Imminent deadline → sla_strict_threshold boosted by SLA_DEADLINE_BOOST."""
        job = _batch_job(
            deadline_epoch=time.time() + 10.0,  # 10s away
        )
        strategy = router.classify(job)
        # GENERIC base: sla_threshold=0.20 + SLA_DEADLINE_BOOST=0.10 → 0.30
        expected_min = 0.20 + SLA_DEADLINE_BOOST - 0.001
        assert strategy.sla_strict_threshold >= expected_min

    def test_deadline_adds_noisy_neighbour_avoidance(self, router: WorkloadIntentRouter):
        """Imminent deadline → BATCH and STREAM added to avoid_workload_types."""
        job = _lc_job(
            gpu_required=False,
            deadline_epoch=time.time() + 5.0,  # 5s away
        )
        strategy = router.classify(job)
        assert WorkloadType.BATCH in strategy.avoid_workload_types
        assert WorkloadType.STREAM in strategy.avoid_workload_types


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Strategy-aware node filtering (integration with aco_schedule)
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyNodeFiltering:
    def test_cpu_serving_never_lands_on_gpu_node(self, router: WorkloadIntentRouter):
        """CPU_SERVING strategy requires X86_64 or ARM64 → GPU_NODE excluded."""
        job = _lc_job(gpu_required=False, cpu_cores_min=1.0, memory_gb_min=1.0)
        strategy = router.classify(job)
        assert strategy.name == "CPU_SERVING"

        cpu_node = _node("n-cpu", arch=NodeArch.X86_64)
        gpu_node = _node("n-gpu", arch=NodeArch.GPU_NODE,
                         gpu_inventory={"A100": 4})
        result = aco_schedule(
            job_request=job,
            available_nodes=[cpu_node, gpu_node],
            strategy=strategy,
        )
        assert result == "n-cpu"

    def test_gpu_inference_never_on_spot(self, router: WorkloadIntentRouter):
        """GPU_INFERENCE requires ON_DEMAND → SPOT GPU node excluded."""
        job = _lc_job(gpu_required=True, gpu_count=1, cpu_cores_min=1.0, memory_gb_min=1.0)
        strategy = router.classify(job)
        assert strategy.name == "GPU_INFERENCE"

        on_demand_gpu = _node(
            "gpu-on-demand",
            arch=NodeArch.GPU_NODE,
            instance_type=InstanceType.ON_DEMAND,
            gpu_inventory={"A100": 4},
        )
        spot_gpu = _node(
            "gpu-spot",
            arch=NodeArch.GPU_NODE,
            instance_type=InstanceType.SPOT,
            interruption_prob=0.20,
            gpu_inventory={"A100": 4},
        )
        result = aco_schedule(
            job_request=job,
            available_nodes=[on_demand_gpu, spot_gpu],
            strategy=strategy,
        )
        assert result == "gpu-on-demand"

    def test_preemptible_actor_can_land_on_spot(self, router: WorkloadIntentRouter):
        """PREEMPTIBLE_ACTOR allows any instance type → SPOT eligible."""
        job = _batch_job(preemptible=True, cpu_cores_min=1.0, memory_gb_min=1.0)
        strategy = router.classify(job)
        assert strategy.name == "PREEMPTIBLE_ACTOR"
        assert strategy.required_instance is None  # no instance filter

        spot_node = _node(
            "arm-spot",
            arch=NodeArch.ARM64,
            instance_type=InstanceType.SPOT,
            interruption_prob=0.30,
        )
        result = aco_schedule(
            job_request=job,
            available_nodes=[spot_node],
            strategy=strategy,
        )
        assert result == "arm-spot"


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — CostEngine threshold overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestCostEngineThresholdOverrides:
    def test_gpu_inference_higher_sla_threshold_gates_busy_node(self):
        """GPU_INFERENCE sla_threshold=0.30 gates a node at 75% CPU (headroom=0.25)."""
        engine = CostEngine()
        job = _lc_job(gpu_required=True, gpu_count=1, cpu_cores_min=1.0, memory_gb_min=1.0)

        # 75% utilised → headroom = 0.25
        # Default threshold: 0.20 → would pass
        # GPU_INFERENCE threshold: 0.30 → fails (0.25 < 0.30)
        node = _node("gpu-busy", arch=NodeArch.GPU_NODE,
                     gpu_inventory={"A100": 4})
        # Simulate 75% allocation
        node.allocated_cpu_cores = node.total_cpu_cores * 0.75

        default_score = engine.score_node(job, node)
        assert default_score > 0.0, "With default threshold, 75% CPU should pass"

        strategy_score = engine.score_node(job, node, sla_threshold=0.30)
        assert strategy_score == 0.0, "With GPU_INFERENCE threshold, 75% CPU should be gated"

    def test_training_tolerates_busy_node(self):
        """GPU_TRAINING sla_threshold=0.05: node at 90% CPU still scores > 0."""
        engine = CostEngine()
        job = _batch_job(gpu_required=True, gpu_count=1, cpu_cores_min=1.0, memory_gb_min=1.0)

        node = _node("gpu-heavy", arch=NodeArch.GPU_NODE,
                     gpu_inventory={"A100": 4})
        # 90% allocated → headroom = 0.10
        node.allocated_cpu_cores = node.total_cpu_cores * 0.90

        score = engine.score_node(job, node, sla_threshold=0.05)
        # BATCH workload uses max(headroom, 0.1) = 0.1 → should be > 0
        assert score > 0.0

    def test_spike_penalty_stronger_for_inference(self):
        """Same spike prediction: GPU_INFERENCE penalty > GPU_TRAINING penalty."""
        engine = CostEngine()
        prediction = PredictionResult(
            node_id="n1",
            predicted_cpu_util=90.0,
            predicted_memory_util=50.0,
            spike_probability=0.8,
            confidence=1.0,
        )

        # GPU_INFERENCE: spike_weight=1.0
        inference_factor = engine.prediction_factor(prediction, spike_weight=1.0)
        # GPU_TRAINING: spike_weight=0.20
        training_factor = engine.prediction_factor(prediction, spike_weight=0.20)

        assert inference_factor < training_factor, (
            "GPU_INFERENCE should be penalised more heavily for spike predictions"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 5 — Colocation policy
# ─────────────────────────────────────────────────────────────────────────────

class TestColocationPolicy:
    def test_gpu_inference_avoids_training_node(self, router: WorkloadIntentRouter):
        """GPU_INFERENCE avoids BATCH: node running BATCH → aco gives it η=0.0."""
        job = _lc_job(gpu_required=True, gpu_count=1, cpu_cores_min=1.0, memory_gb_min=1.0)
        strategy = router.classify(job)
        assert WorkloadType.BATCH in strategy.avoid_workload_types

        on_demand_gpu_clear = _node(
            "gpu-clear",
            arch=NodeArch.GPU_NODE,
            instance_type=InstanceType.ON_DEMAND,
            gpu_inventory={"A100": 4},
        )
        on_demand_gpu_busy = _node(
            "gpu-batch-running",
            arch=NodeArch.GPU_NODE,
            instance_type=InstanceType.ON_DEMAND,
            gpu_inventory={"A100": 4},
        )

        # Simulate BATCH running on gpu-batch-running
        node_workload_map = {"gpu-batch-running": [WorkloadType.BATCH]}

        result = aco_schedule(
            job_request=job,
            available_nodes=[on_demand_gpu_clear, on_demand_gpu_busy],
            strategy=strategy,
            node_workload_map=node_workload_map,
        )
        assert result == "gpu-clear", "Should avoid node running BATCH"

    def test_training_allows_training_colocation(self, router: WorkloadIntentRouter):
        """GPU_TRAINING avoid_workload_types=[] → can colocate with another BATCH."""
        job = _batch_job(gpu_required=True, gpu_count=1, preemptible=False,
                         cpu_cores_min=1.0, memory_gb_min=1.0)
        strategy = router.classify(job)
        assert strategy.name == "GPU_TRAINING"
        assert WorkloadType.BATCH not in strategy.avoid_workload_types

        gpu_node = _node(
            "gpu-with-training",
            arch=NodeArch.GPU_NODE,
            gpu_inventory={"A100": 4},
        )

        # Node is already running a BATCH job — GPU_TRAINING should still accept it
        node_workload_map = {"gpu-with-training": [WorkloadType.BATCH]}
        result = aco_schedule(
            job_request=job,
            available_nodes=[gpu_node],
            strategy=strategy,
            node_workload_map=node_workload_map,
        )
        assert result == "gpu-with-training"

    def test_colocation_map_updated_on_alloc(self):
        """_allocate_resources adds workload type to _node_workload_map."""
        svc = OrchestratorService()
        result = svc.submit_job({
            "workload_type": "batch",
            "priority": 50,
            "preemptible": False,
            "resources": {
                "cpu_cores_min": 2.0,
                "memory_gb_min": 4.0,
                "gpu_required": False,
                "gpu_count": 1,
            },
        })
        assert result["status"] == "SCHEDULED"
        node_id = result["node_id"]
        # Workload type should be recorded in the map
        assert node_id in svc._node_workload_map
        assert WorkloadType.BATCH in svc._node_workload_map[node_id]

    def test_colocation_map_cleared_on_release(self):
        """_release_resources removes workload type from _node_workload_map."""
        svc = OrchestratorService()
        result = svc.submit_job({
            "workload_type": "batch",
            "priority": 50,
            "preemptible": False,
            "resources": {
                "cpu_cores_min": 2.0,
                "memory_gb_min": 4.0,
                "gpu_required": False,
                "gpu_count": 1,
            },
        })
        assert result["status"] == "SCHEDULED"
        job_id = result["job_id"]
        node_id = result["node_id"]
        assert WorkloadType.BATCH in svc._node_workload_map[node_id]

        # Complete the job → resources released → map entry cleared
        svc.complete_job(job_id)
        remaining = svc._node_workload_map.get(node_id, [])
        assert WorkloadType.BATCH not in remaining


# ─────────────────────────────────────────────────────────────────────────────
# Group 6 — Strategy fields correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyFields:
    def test_gpu_inference_strategy_fields(self, router: WorkloadIntentRouter):
        """GPU_INFERENCE fields match the plan specification."""
        strategy = router.classify(_lc_job(gpu_required=True, gpu_count=1))
        assert strategy.required_arch == [NodeArch.GPU_NODE]
        assert strategy.required_instance == [InstanceType.ON_DEMAND]
        assert strategy.use_fast_path is True
        assert strategy.allow_spot is False
        assert strategy.sla_strict_threshold == 0.30
        assert strategy.spike_penalty_weight == 1.0
        assert WorkloadType.BATCH in strategy.avoid_workload_types

    def test_preemptible_actor_high_spot_tolerance(self, router: WorkloadIntentRouter):
        """PREEMPTIBLE_ACTOR tolerates 70% spot interruption probability."""
        strategy = router.classify(_batch_job(preemptible=True))
        assert strategy.spot_penalty_threshold == 0.70

    def test_stateful_stream_on_demand_only(self, router: WorkloadIntentRouter):
        """STATEFUL_STREAM requires ON_DEMAND — can't afford data loss."""
        strategy = router.classify(_stream_job(preemptible=False))
        assert strategy.required_instance == [InstanceType.ON_DEMAND]
        assert strategy.allow_spot is False

    def test_gpu_training_any_instance_type(self, router: WorkloadIntentRouter):
        """GPU_TRAINING allows any instance type (spot OK for training)."""
        strategy = router.classify(_batch_job(gpu_required=True, gpu_count=1, preemptible=False))
        assert strategy.required_instance is None
        assert strategy.allow_spot is True
