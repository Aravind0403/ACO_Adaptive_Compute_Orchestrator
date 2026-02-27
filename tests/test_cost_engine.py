"""
tests/test_cost_engine.py
──────────────────────────
Test suite for orchestrator/control_plane/cost_engine.py

Coverage: 22 tests across 5 groups.

What we are testing
────────────────────
CostEngine.score_node() produces a composite score [0.0, 1.0] that is fed
into the ACO η heuristic. Every sub-score must:
  • Be deterministic given the same inputs
  • Respect hard gates (return 0.0 on constraint violations)
  • Respect soft gradients (cheaper/healthier nodes score higher)

Test groups
────────────
Group 1: reliability_factor  — spot/on-demand + LC/batch interactions
Group 2: cost_efficiency_factor — inverse cost function properties
Group 3: sla_headroom_factor  — utilisation-based headroom gating
Group 4: prediction_factor    — spike penalty and confidence dampening
Group 5: composite score_node — end-to-end integration and score_breakdown
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pytest

from orchestrator.control_plane.cost_engine import (
    SPIKE_PENALTY_WEIGHT,
    SPOT_PENALTY_THRESHOLD,
    SLA_STRICT_THRESHOLD,
    CostEngine,
)
from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobRequest,
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
    cost_per_hour: float = 0.10,
    instance_type: InstanceType = InstanceType.ON_DEMAND,
    interruption_prob: float = 0.0,
    cpu_util_pct: Optional[float] = None,   # if set, adds real telemetry
    total_cpu: float = 16.0,
    allocated_cpu: float = 4.0,
    arch: NodeArch = NodeArch.X86_64,
) -> ComputeNode:
    """Build a minimal ComputeNode for testing."""
    telemetry = None
    if cpu_util_pct is not None:
        telemetry = NodeTelemetry(
            node_id=node_id,
            cpu_util_pct=cpu_util_pct,
            memory_util_pct=20.0,
        )

    return ComputeNode(
        node_id=node_id,
        state=NodeState.HEALTHY,
        arch=arch,
        total_cpu_cores=total_cpu,
        total_memory_gb=64.0,
        allocated_cpu_cores=allocated_cpu,
        allocated_memory_gb=8.0,
        cost_profile=NodeCostProfile(
            instance_type=instance_type,
            cost_per_hour_usd=cost_per_hour,
            interruption_prob=interruption_prob,
        ),
        latest_telemetry=telemetry,
    )


def _make_job(
    job_id: str = "j-test",
    workload_type: WorkloadType = WorkloadType.BATCH,
    cpu_min: float = 2.0,
    priority: int = 50,
) -> JobRequest:
    """Build a minimal JobRequest for testing."""
    return JobRequest(
        job_id=job_id,
        workload_type=workload_type,
        resources=ResourceRequest(
            cpu_cores_min=cpu_min,
            memory_gb_min=4.0,
        ),
        priority=priority,
    )


def _make_prediction(
    spike_probability: float = 0.0,
    confidence: float = 1.0,
    node_id: str = "n-test",
) -> PredictionResult:
    """Build a minimal PredictionResult for testing."""
    return PredictionResult(
        node_id=node_id,
        predicted_cpu_util=50.0,
        predicted_memory_util=30.0,
        spike_probability=spike_probability,
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: reliability_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestReliabilityFactor:
    """
    ON_DEMAND nodes are always reliable (1.0).
    SPOT nodes are penalised based on interruption_prob and job type.
    """

    engine = CostEngine()

    def test_on_demand_always_scores_one(self) -> None:
        """ON_DEMAND node must always return reliability=1.0, regardless of job type."""
        node = _make_node(instance_type=InstanceType.ON_DEMAND, interruption_prob=0.0)

        for wt in [WorkloadType.LATENCY_CRITICAL, WorkloadType.BATCH, WorkloadType.STREAM]:
            job = _make_job(workload_type=wt)
            score = self.engine.reliability_factor(job, node)
            assert score == pytest.approx(1.0), f"ON_DEMAND + {wt}: expected 1.0, got {score}"

    def test_spot_lc_hard_gate_above_threshold(self) -> None:
        """SPOT node with interruption_prob > SPOT_PENALTY_THRESHOLD gates LC jobs (→ 0.0)."""
        high_risk_prob = SPOT_PENALTY_THRESHOLD + 0.01   # just over the threshold
        node = _make_node(instance_type=InstanceType.SPOT, interruption_prob=high_risk_prob)
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.reliability_factor(job, node)
        assert score == pytest.approx(0.0), (
            f"SPOT + LC + interruption_prob={high_risk_prob} should be 0.0, got {score}"
        )

    def test_spot_lc_soft_penalty_below_threshold(self) -> None:
        """SPOT node below threshold: LC job gets soft penalty = 1 - interruption_prob."""
        prob = 0.2   # below threshold of 0.3
        node = _make_node(instance_type=InstanceType.SPOT, interruption_prob=prob)
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.reliability_factor(job, node)
        assert score == pytest.approx(1.0 - prob)

    def test_spot_batch_soft_penalty(self) -> None:
        """SPOT + BATCH: soft penalty even at high interruption_prob (no hard gate)."""
        prob = 0.8   # above LC threshold, but BATCH is tolerant
        node = _make_node(instance_type=InstanceType.SPOT, interruption_prob=prob)
        job = _make_job(workload_type=WorkloadType.BATCH)

        score = self.engine.reliability_factor(job, node)
        expected = 1.0 - 0.3 * prob   # = 1.0 - 0.24 = 0.76
        assert score == pytest.approx(expected)

    def test_spot_stream_soft_penalty(self) -> None:
        """SPOT + STREAM: same soft penalty logic as BATCH."""
        prob = 0.5
        node = _make_node(instance_type=InstanceType.SPOT, interruption_prob=prob)
        job = _make_job(workload_type=WorkloadType.STREAM)

        score = self.engine.reliability_factor(job, node)
        expected = 1.0 - 0.3 * prob   # = 0.85
        assert score == pytest.approx(expected)

    def test_spot_lc_zero_interruption_prob_scores_one(self) -> None:
        """SPOT node with 0% interruption risk → reliability = 1.0 even for LC."""
        node = _make_node(instance_type=InstanceType.SPOT, interruption_prob=0.0)
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.reliability_factor(job, node)
        assert score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: cost_efficiency_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestCostEfficiencyFactor:
    """
    Inverse cost function: cheaper nodes score higher.
    Must always be in (0.0, 1.0] — never zero, even for expensive nodes.
    """

    engine = CostEngine()

    def test_free_node_scores_one(self) -> None:
        """A $0/hr node should score 1.0 (maximum cost efficiency)."""
        node = _make_node(cost_per_hour=0.0)
        score = self.engine.cost_efficiency_factor(node)
        assert score == pytest.approx(1.0)

    def test_reference_cost_scores_half(self) -> None:
        """At MAX_COST_REFERENCE ($1.00/hr): score = 1/(1+1) = 0.5."""
        from orchestrator.control_plane.cost_engine import MAX_COST_REFERENCE
        node = _make_node(cost_per_hour=MAX_COST_REFERENCE)
        score = self.engine.cost_efficiency_factor(node)
        assert score == pytest.approx(0.5)

    def test_cheaper_node_scores_higher(self) -> None:
        """A cheaper node must always score higher than a more expensive one."""
        cheap = _make_node(cost_per_hour=0.10)
        expensive = _make_node(cost_per_hour=0.80)

        score_cheap = self.engine.cost_efficiency_factor(cheap)
        score_expensive = self.engine.cost_efficiency_factor(expensive)

        assert score_cheap > score_expensive

    def test_very_expensive_node_never_zero(self) -> None:
        """Even an absurdly expensive node must score > 0.0 (never hard-blocked by cost alone)."""
        very_expensive = _make_node(cost_per_hour=100.0)
        score = self.engine.cost_efficiency_factor(very_expensive)
        assert score > 0.0

    def test_cost_score_in_bounds(self) -> None:
        """score_efficiency must always be in (0.0, 1.0] for various cost values."""
        for cost in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 32.0, 100.0]:
            node = _make_node(cost_per_hour=cost)
            score = self.engine.cost_efficiency_factor(node)
            assert 0.0 < score <= 1.0, f"cost={cost} → score={score} out of bounds"


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: sla_headroom_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestSlaHeadroomFactor:
    """
    LATENCY_CRITICAL jobs need real breathing room (strict).
    BATCH/STREAM jobs can tolerate busier nodes (forgiving, floor at 0.1).
    """

    engine = CostEngine()

    def test_lc_low_utilisation_scores_high(self) -> None:
        """LC job on a nearly empty node → headroom ≈ 1.0."""
        node = _make_node(cpu_util_pct=10.0)   # 10% util → 90% headroom
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.sla_headroom_factor(job, node)
        assert score == pytest.approx(0.9)

    def test_lc_high_utilisation_hard_gate(self) -> None:
        """LC job on a nearly full node (>80% util) → hard gate → 0.0."""
        node = _make_node(cpu_util_pct=85.0)   # 85% util → 15% headroom < SLA_STRICT_THRESHOLD=20%
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.sla_headroom_factor(job, node)
        assert score == pytest.approx(0.0), (
            f"LC on 85%-loaded node should be 0.0, got {score}"
        )

    def test_lc_exactly_at_threshold_boundary(self) -> None:
        """
        headroom = SLA_STRICT_THRESHOLD exactly (boundary condition).
        Must NOT be hard-gated: returns SLA_STRICT_THRESHOLD, not 0.0.
        """
        # SLA_STRICT_THRESHOLD=0.2 → util must be exactly 80%
        util_at_threshold = (1.0 - SLA_STRICT_THRESHOLD) * 100.0   # 80.0%
        node = _make_node(cpu_util_pct=util_at_threshold)
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.sla_headroom_factor(job, node)
        assert score == pytest.approx(SLA_STRICT_THRESHOLD)

    def test_batch_high_utilisation_not_gated(self) -> None:
        """BATCH job on a 95%-loaded node → not hard-gated → floored at 0.1."""
        node = _make_node(cpu_util_pct=95.0)   # 5% headroom
        job = _make_job(workload_type=WorkloadType.BATCH)

        score = self.engine.sla_headroom_factor(job, node)
        assert score == pytest.approx(0.1), (
            f"BATCH on 95%-loaded node should be floored at 0.1, got {score}"
        )

    def test_stream_uses_real_telemetry_when_available(self) -> None:
        """
        When latest_telemetry is present, sla_headroom_factor uses the real CPU util,
        not the allocation-based estimate.
        """
        # Allocation-based would give: (16-4)/16 = 75% util → 25% headroom
        # But real telemetry says 50% util → 50% headroom
        node = _make_node(
            total_cpu=16.0, allocated_cpu=12.0,   # allocation says 75%
            cpu_util_pct=50.0,                    # real telemetry says 50%
        )
        job = _make_job(workload_type=WorkloadType.STREAM)

        score = self.engine.sla_headroom_factor(job, node)
        # Uses real telemetry: headroom = 0.5, max(0.5, 0.1) = 0.5
        assert score == pytest.approx(0.5)

    def test_falls_back_to_allocation_without_telemetry(self) -> None:
        """Without latest_telemetry, uses allocation-based CPU utilisation."""
        # No telemetry: allocation = 4/16 = 25% → headroom = 75%
        node = _make_node(total_cpu=16.0, allocated_cpu=4.0, cpu_util_pct=None)
        job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.sla_headroom_factor(job, node)
        assert score == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: prediction_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionFactor:
    """
    prediction_factor penalises nodes expected to spike.
    Penalty is dampened by confidence so uncertain predictions barely affect score.
    """

    engine = CostEngine()

    def test_no_prediction_returns_one(self) -> None:
        """Without a prediction, factor is 1.0 (neutral)."""
        score = self.engine.prediction_factor(None)
        assert score == pytest.approx(1.0)

    def test_zero_spike_probability_returns_one(self) -> None:
        """spike_probability=0.0 → no penalty → 1.0."""
        pred = _make_prediction(spike_probability=0.0, confidence=1.0)
        score = self.engine.prediction_factor(pred)
        assert score == pytest.approx(1.0)

    def test_full_spike_full_confidence_returns_half(self) -> None:
        """spike_probability=1.0, confidence=1.0 → 1 − (1.0 × 0.5 × 1.0) = 0.5."""
        pred = _make_prediction(spike_probability=1.0, confidence=1.0)
        score = self.engine.prediction_factor(pred)
        assert score == pytest.approx(1.0 - SPIKE_PENALTY_WEIGHT)

    def test_cold_start_confidence_dampens_penalty(self) -> None:
        """
        confidence=0.1 (cold-start predictor): even 100% spike prediction
        barely affects score — the predictor's uncertainty dampens the penalty.
        1 - (1.0 × 0.5 × 0.1) = 0.95
        """
        pred = _make_prediction(spike_probability=1.0, confidence=0.1)
        score = self.engine.prediction_factor(pred)
        assert score == pytest.approx(1.0 - 1.0 * SPIKE_PENALTY_WEIGHT * 0.1, abs=0.001)

    def test_prediction_factor_never_below_half(self) -> None:
        """
        The minimum prediction_factor is 0.5 (hard floor).
        Spike predictions should not be able to completely zero out a node.
        """
        pred = _make_prediction(spike_probability=1.0, confidence=1.0)
        score = self.engine.prediction_factor(pred)
        assert score >= 0.5

    def test_higher_spike_prob_gives_lower_score(self) -> None:
        """Monotonicity: higher spike_probability → lower prediction_factor."""
        scores = [
            self.engine.prediction_factor(_make_prediction(spike_probability=p))
            for p in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Monotonicity violated: score[{i}]={scores[i]:.3f} < score[{i+1}]={scores[i+1]:.3f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: composite score_node + score_breakdown
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScore:
    """
    Integration tests: end-to-end score_node() and score_breakdown() correctness.
    """

    engine = CostEngine()

    def test_ideal_placement_scores_high(self) -> None:
        """
        Cheap ON_DEMAND node + low CPU util + no spike prediction
        → composite score should be high (> 0.7).
        """
        node = _make_node(
            cost_per_hour=0.10,
            instance_type=InstanceType.ON_DEMAND,
            cpu_util_pct=20.0,   # 80% headroom
        )
        job = _make_job(workload_type=WorkloadType.BATCH)

        score = self.engine.score_node(job, node)
        assert score > 0.7, f"Ideal placement scored too low: {score:.3f}"

    def test_hard_gate_propagates_to_zero(self) -> None:
        """
        If reliability returns 0.0 (SPOT + LC + high interruption_prob),
        the composite score must be 0.0 regardless of other sub-scores.
        """
        # SPOT + interruption_prob > threshold → reliability = 0.0
        node = _make_node(
            instance_type=InstanceType.SPOT,
            interruption_prob=SPOT_PENALTY_THRESHOLD + 0.1,
            cost_per_hour=0.01,   # very cheap
            cpu_util_pct=5.0,     # very idle
        )
        lc_job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.score_node(lc_job, node)
        assert score == pytest.approx(0.0), (
            f"Hard gate should produce 0.0, got {score:.3f}"
        )

    def test_sla_gate_propagates_to_zero(self) -> None:
        """
        If sla_headroom returns 0.0 (LC + node > 80% util),
        the composite score must be 0.0.
        """
        node = _make_node(
            cost_per_hour=0.01,
            instance_type=InstanceType.ON_DEMAND,
            cpu_util_pct=90.0,   # over-loaded
        )
        lc_job = _make_job(workload_type=WorkloadType.LATENCY_CRITICAL)

        score = self.engine.score_node(lc_job, node)
        assert score == pytest.approx(0.0)

    def test_score_breakdown_matches_score_node(self) -> None:
        """
        score_breakdown()'s 'composite' key must equal score_node().
        """
        node = _make_node(cost_per_hour=0.50, cpu_util_pct=40.0)
        job = _make_job(workload_type=WorkloadType.BATCH)
        pred = _make_prediction(spike_probability=0.3, confidence=0.8)

        score = self.engine.score_node(job, node, pred)
        breakdown = self.engine.score_breakdown(job, node, pred)

        assert breakdown["composite"] == pytest.approx(score, abs=1e-9)
        assert set(breakdown.keys()) == {"reliability", "cost_efficiency", "sla_headroom", "prediction", "composite"}

    def test_score_breakdown_sub_scores_multiply_to_composite(self) -> None:
        """Product of sub-scores equals composite."""
        node = _make_node(cost_per_hour=0.20, cpu_util_pct=30.0)
        job = _make_job(workload_type=WorkloadType.STREAM)
        pred = _make_prediction(spike_probability=0.1, confidence=0.9)

        breakdown = self.engine.score_breakdown(job, node, pred)
        manual_product = (
            breakdown["reliability"]
            * breakdown["cost_efficiency"]
            * breakdown["sla_headroom"]
            * breakdown["prediction"]
        )
        assert manual_product == pytest.approx(breakdown["composite"], abs=1e-9)

    def test_cheaper_node_wins_ceteris_paribus(self) -> None:
        """
        With identical utilisation and reliability, the cheaper node scores higher.
        """
        cheap = _make_node(node_id="cheap", cost_per_hour=0.05, cpu_util_pct=30.0)
        pricey = _make_node(node_id="pricey", cost_per_hour=0.90, cpu_util_pct=30.0)
        job = _make_job(workload_type=WorkloadType.BATCH)

        assert self.engine.score_node(job, cheap) > self.engine.score_node(job, pricey)

    def test_all_sub_scores_in_zero_to_one(self) -> None:
        """
        Property: all sub-scores and composite must be in [0.0, 1.0]
        across a variety of node/job combinations.
        """
        scenarios = [
            (_make_node(cpu_util_pct=10.0), _make_job(WorkloadType.LATENCY_CRITICAL)),
            (_make_node(cpu_util_pct=90.0), _make_job(WorkloadType.LATENCY_CRITICAL)),
            (_make_node(cpu_util_pct=50.0, instance_type=InstanceType.SPOT, interruption_prob=0.5), _make_job(WorkloadType.BATCH)),
            (_make_node(cpu_util_pct=0.0, cost_per_hour=0.0), _make_job(WorkloadType.STREAM)),
        ]
        for node, job in scenarios:
            breakdown = self.engine.score_breakdown(job, node)
            for key, val in breakdown.items():
                assert 0.0 <= val <= 1.0, (
                    f"Sub-score '{key}' = {val:.4f} out of [0,1] for {node.node_id}/{job.workload_type}"
                )
