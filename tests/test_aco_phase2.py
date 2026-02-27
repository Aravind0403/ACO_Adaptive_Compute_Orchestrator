"""
tests/test_aco_phase2.py
────────────────────────
Phase 2 test suite — 4 groups covering every layer of the ACO engine.

Reading guide
─────────────
Group 1 — PheromoneMatrix unit tests
    Verifies the shared memory layer in isolation.
    Tests initial state, evaporation math, deposit math.

Group 2 — Heuristic (η) unit tests
    Verifies the domain knowledge encoding.
    Tests infeasibility gates, affinity multipliers, hard constraints.

Group 3 — Ant construction tests
    Verifies one ant builds a valid solution.
    Tests feasibility, priority ordering.

Group 4 — Colony integration tests
    Verifies the full ACO loop end-to-end.
    Tests output type, performance benchmark, fast path, failure mode,
    and pheromone convergence.

Helpers
───────
_make_node() and _make_job() are fixture factories that produce
valid model instances with sensible defaults. Tests override only the
fields relevant to what they are testing — everything else stays default.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from aco_core import Colony, ColonyFailedError
from aco_core.ant import (
    Ant,
    ALPHA,
    BETA,
    AFFINITY_GPU_BATCH,
    AFFINITY_GPU_LATENCY_CRITICAL,
    INFEASIBLE_PENALTY,
)
from aco_core.pheromone import (
    PheromoneMatrix,
    TAU_INITIAL,
    TAU_MIN,
    TAU_MAX,
    EVAPORATION_RATE,
    Q,
)
from orchestrator.shared.models import (
    ComputeNode,
    JobRequest,
    NodeArch,
    NodeCostProfile,
    NodeState,
    ResourceRequest,
    WorkloadType,
    InstanceType,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — fixture factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_node(
    node_id: str = "node-01",
    arch: NodeArch = NodeArch.X86_64,
    total_cpu: float = 16.0,
    total_mem: float = 64.0,
    allocated_cpu: float = 0.0,
    allocated_mem: float = 0.0,
    gpu_inventory: dict = None,
    gpu_vram_gb: dict = None,
    cost_per_hour: float = 0.50,
    interruption_prob: float = 0.0,
    instance_type: InstanceType = InstanceType.ON_DEMAND,
    state: NodeState = NodeState.HEALTHY,
) -> ComputeNode:
    """Create a ComputeNode with sensible defaults."""
    return ComputeNode(
        node_id=node_id,
        arch=arch,
        state=state,
        total_cpu_cores=total_cpu,
        total_memory_gb=total_mem,
        allocated_cpu_cores=allocated_cpu,
        allocated_memory_gb=allocated_mem,
        gpu_inventory=gpu_inventory or {},
        gpu_vram_gb=gpu_vram_gb or {},
        cost_profile=NodeCostProfile(
            instance_type=instance_type,
            cost_per_hour_usd=cost_per_hour,
            interruption_prob=interruption_prob,
        ),
    )


def _make_job(
    job_id: str = "job-01",
    workload_type: WorkloadType = WorkloadType.BATCH,
    cpu_min: float = 2.0,
    mem_min: float = 4.0,
    gpu_required: bool = False,
    gpu_count: int = 1,
    gpu_mem_gb: float = None,
    priority: int = 50,
    cost_ceiling: float = None,
    arch_required: NodeArch = None,
    gpu_model_preferred: str = None,
    latency_p99_ms: int = None,
    preemptible: bool = False,
) -> JobRequest:
    """Create a JobRequest with sensible defaults."""
    return JobRequest(
        job_id=job_id,
        workload_type=workload_type,
        resources=ResourceRequest(
            cpu_cores_min=cpu_min,
            memory_gb_min=mem_min,
            gpu_required=gpu_required,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_mem_gb,
        ),
        priority=priority,
        cost_ceiling_usd=cost_ceiling,
        arch_required=arch_required,
        gpu_model_preferred=gpu_model_preferred,
        latency_p99_ms=latency_p99_ms,
        preemptible=preemptible,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — PheromoneMatrix unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPheromoneMatrix:
    """Verify the shared memory layer in complete isolation."""

    def test_initial_state_shape(self):
        """Matrix has the correct shape after construction."""
        m = PheromoneMatrix(3, 5)
        assert m.shape == (3, 5)
        assert m.n_jobs == 3
        assert m.n_nodes == 5

    def test_initial_state_all_tau_initial(self):
        """Every cell equals TAU_INITIAL (1.0) at construction."""
        m = PheromoneMatrix(4, 4)
        snap = m.snapshot()
        # np.allclose is safer than == for float comparisons
        assert np.allclose(snap, TAU_INITIAL), (
            f"Expected all cells = {TAU_INITIAL}, got min={snap.min()}, max={snap.max()}"
        )

    def test_snapshot_is_deep_copy(self):
        """Mutating the snapshot does NOT affect the live matrix."""
        m = PheromoneMatrix(2, 2)
        snap = m.snapshot()
        snap[0, 0] = 999.0          # Mutate the copy
        # Live matrix must be unchanged
        assert np.isclose(m.snapshot()[0, 0], TAU_INITIAL)

    def test_evaporation_reduces_values(self):
        """After one evaporate(), all cells equal TAU_INITIAL × (1 − ρ), clipped."""
        m = PheromoneMatrix(3, 3)
        m.evaporate()
        snap = m.snapshot()
        expected = np.clip(
            np.full((3, 3), TAU_INITIAL * (1.0 - EVAPORATION_RATE)),
            TAU_MIN, TAU_MAX,
        )
        # With TAU_INITIAL=1.0 and EVAPORATION_RATE=0.1: expected = 0.9
        assert np.allclose(snap, expected), (
            f"After evaporation: expected {expected[0, 0]:.4f}, got {snap[0, 0]:.4f}"
        )

    def test_evaporation_respects_tau_min(self):
        """Repeated evaporation never goes below TAU_MIN."""
        m = PheromoneMatrix(2, 2)
        for _ in range(1000):           # 1000 × 10% decay — would reach ~0 without floor
            m.evaporate()
        snap = m.snapshot()
        assert np.all(snap >= TAU_MIN), (
            f"Found cells below TAU_MIN={TAU_MIN}: min={snap.min()}"
        )

    def test_deposit_increases_target_cell(self):
        """
        deposit(job_idx=1, node_idx=2, cost=0.5) adds Q/cost = 2.0 to cell [1][2].
        """
        m = PheromoneMatrix(3, 4)
        before = float(m.snapshot()[1, 2])
        m.deposit(1, 2, solution_cost=0.5)
        after = float(m.snapshot()[1, 2])
        expected_increase = Q / 0.5          # = 2.0
        assert np.isclose(after, before + expected_increase), (
            f"Expected cell [1,2] = {before + expected_increase:.4f}, got {after:.4f}"
        )

    def test_deposit_leaves_other_cells_unchanged(self):
        """Depositing on [1,2] must not alter any other cell."""
        m = PheromoneMatrix(3, 4)
        before = m.snapshot().copy()
        m.deposit(1, 2, solution_cost=1.0)
        after = m.snapshot()
        # All cells except [1,2] should be unchanged
        mask = np.ones((3, 4), dtype=bool)
        mask[1, 2] = False
        assert np.allclose(before[mask], after[mask]), (
            "deposit() modified cells other than [1,2]"
        )

    def test_deposit_zero_cost_skipped(self):
        """deposit() with cost=0.0 must be a no-op (no division by zero)."""
        m = PheromoneMatrix(2, 2)
        before = m.snapshot().copy()
        m.deposit(0, 0, solution_cost=0.0)      # Should not raise, should not change
        assert np.allclose(m.snapshot(), before), (
            "deposit(cost=0) modified the matrix — should be a no-op"
        )

    def test_deposit_negative_cost_skipped(self):
        """deposit() with cost < 0 is also a no-op."""
        m = PheromoneMatrix(2, 2)
        before = m.snapshot().copy()
        m.deposit(0, 0, solution_cost=-5.0)
        assert np.allclose(m.snapshot(), before)

    def test_deposit_respects_tau_max(self):
        """Depositing many times should never exceed TAU_MAX."""
        m = PheromoneMatrix(2, 2)
        for _ in range(1000):
            m.deposit(0, 0, solution_cost=0.001)   # Each adds Q/0.001 = 1000
        assert m.snapshot()[0, 0] <= TAU_MAX + 1e-9, (
            f"Cell [0,0] exceeded TAU_MAX: {m.snapshot()[0, 0]:.2f}"
        )

    def test_get_row_returns_view(self):
        """get_row() returns a view; its values match the matrix row."""
        m = PheromoneMatrix(3, 4)
        row = m.get_row(1)
        assert row.shape == (4,)
        snap_row = m.snapshot()[1]
        assert np.allclose(row, snap_row)

    def test_invalid_dimensions_raise(self):
        """PheromoneMatrix with n_jobs=0 or n_nodes=0 raises ValueError."""
        with pytest.raises(ValueError):
            PheromoneMatrix(0, 5)
        with pytest.raises(ValueError):
            PheromoneMatrix(3, 0)

    def test_repr_contains_shape(self):
        """__repr__ includes shape info (smoke test for debugging)."""
        m = PheromoneMatrix(2, 3)
        r = repr(m)
        assert "2" in r and "3" in r


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — Heuristic (η) unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEtaHeuristic:
    """
    Verify the domain knowledge layer encoded in the Ant's _compute_eta().

    Strategy: create an Ant with known job-node combinations, then inspect
    self._eta[i][j] directly.
    """

    def _make_ant(self, jobs, nodes) -> Ant:
        """Shortcut: create an Ant with a uniform pheromone matrix."""
        matrix = PheromoneMatrix(len(jobs), len(nodes))
        return Ant(jobs, nodes, matrix)

    # ── Infeasibility gates ────────────────────────────────────────────────

    def test_infeasible_node_eta_is_zero(self):
        """
        A node that cannot fit the job (can_fit=False) must have η=0.0.
        Zero η means zero selection probability — the ant never picks this node.
        """
        small_node = _make_node("n1", total_cpu=1.0)   # Only 1 core free
        hungry_job = _make_job("j1", cpu_min=4.0)       # Needs 4 cores

        assert not small_node.can_fit(hungry_job.resources), (
            "Test setup error: expected can_fit=False"
        )

        ant = self._make_ant([hungry_job], [small_node])
        assert ant._eta[0, 0] == 0.0, (
            f"η for infeasible node should be 0.0, got {ant._eta[0, 0]}"
        )

    def test_infeasible_node_never_selected(self):
        """
        If all nodes are infeasible for a job, _select_node() returns None.
        This guards against calling np.random.choice with all-zero probabilities.
        """
        node = _make_node("n1", total_cpu=1.0)
        job  = _make_job("j1", cpu_min=100.0)

        matrix = PheromoneMatrix(1, 1)
        ant = Ant([job], [node], matrix)
        result = ant._select_node(0)
        assert result is None, "Expected None when no feasible node exists"

    def test_cost_ceiling_hard_gate(self):
        """
        A node whose cost exceeds the job's ceiling gets η=0.0.
        Even high pheromone on that arc cannot override budget compliance.
        """
        expensive_node = _make_node("n1", cost_per_hour=5.00)
        cheap_job      = _make_job("j1", cost_ceiling=1.00)

        ant = self._make_ant([cheap_job], [expensive_node])
        assert ant._eta[0, 0] == 0.0, (
            f"Over-budget node η should be 0.0, got {ant._eta[0, 0]}"
        )

    def test_within_cost_ceiling_has_nonzero_eta(self):
        """A node within budget gets a nonzero η (given other checks pass)."""
        affordable_node = _make_node("n1", cost_per_hour=0.50)
        job             = _make_job("j1", cost_ceiling=1.00)

        ant = self._make_ant([job], [affordable_node])
        assert ant._eta[0, 0] > 0.0, (
            f"Within-budget node should have η > 0, got {ant._eta[0, 0]}"
        )

    def test_arch_mismatch_is_zero(self):
        """
        Architecture mismatch → η=0.0 regardless of other scores.
        An x86-only job cannot run on ARM64, full stop.
        """
        arm_node = _make_node("n1", arch=NodeArch.ARM64)
        x86_job  = _make_job("j1", arch_required=NodeArch.X86_64)

        ant = self._make_ant([x86_job], [arm_node])
        assert ant._eta[0, 0] == 0.0, (
            "Arch-incompatible node should have η=0.0"
        )

    def test_arch_match_has_nonzero_eta(self):
        """Matching architecture → affinity gate passes → η > 0."""
        x86_node = _make_node("n1", arch=NodeArch.X86_64)
        x86_job  = _make_job("j1", arch_required=NodeArch.X86_64)

        ant = self._make_ant([x86_job], [x86_node])
        assert ant._eta[0, 0] > 0.0

    # ── Workload affinity multipliers ──────────────────────────────────────

    def test_gpu_batch_vs_gpu_latency_critical_ratio(self):
        """
        GPU_NODE + BATCH should have 3× higher η than GPU_NODE + LATENCY_CRITICAL
        (ratio = AFFINITY_GPU_BATCH / AFFINITY_GPU_LATENCY_CRITICAL = 1.5 / 0.5 = 3.0).
        All other η factors must be equal so only affinity differs.
        """
        gpu_node = _make_node(
            "n1",
            arch=NodeArch.GPU_NODE,
            total_cpu=32.0,
            total_mem=128.0,
            gpu_inventory={"A100": 4},
            gpu_vram_gb={"A100": 80.0},
            cost_per_hour=2.40,
        )
        # Same resources and priority — only workload_type differs
        batch_job = _make_job(
            "j-batch", workload_type=WorkloadType.BATCH,
            cpu_min=2.0, mem_min=4.0, priority=50
        )
        lc_job = _make_job(
            "j-lc", workload_type=WorkloadType.LATENCY_CRITICAL,
            cpu_min=2.0, mem_min=4.0, priority=50, latency_p99_ms=10
        )

        ant = self._make_ant([batch_job, lc_job], [gpu_node])
        eta_batch = ant._eta[0, 0]
        eta_lc    = ant._eta[1, 0]

        assert eta_batch > 0, "BATCH job on GPU node should have η > 0"
        assert eta_lc > 0,    "LATENCY_CRITICAL job on GPU node should have η > 0"

        ratio = eta_batch / eta_lc
        expected_ratio = AFFINITY_GPU_BATCH / AFFINITY_GPU_LATENCY_CRITICAL  # = 3.0
        assert np.isclose(ratio, expected_ratio, rtol=1e-5), (
            f"η ratio BATCH/LC on GPU node should be {expected_ratio:.1f}, got {ratio:.4f}"
        )

    def test_cpu_node_neutral_affinity(self):
        """CPU nodes have AFFINITY_DEFAULT=1.0 for all workload types."""
        cpu_node = _make_node("n1", arch=NodeArch.X86_64)
        batch_job = _make_job("j1", workload_type=WorkloadType.BATCH)
        lc_job    = _make_job("j2", workload_type=WorkloadType.LATENCY_CRITICAL,
                              latency_p99_ms=10)

        ant = self._make_ant([batch_job, lc_job], [cpu_node])
        # Affinity score should be 1.0 for both on a CPU node.
        # Both jobs have the same resources/priority → η should be equal.
        assert np.isclose(ant._eta[0, 0], ant._eta[1, 0], rtol=1e-5), (
            "CPU node should be equally attractive for BATCH and LC jobs"
        )

    # ── Urgency score ──────────────────────────────────────────────────────

    def test_higher_priority_has_higher_eta(self):
        """Priority 90 job should have higher η than priority 10 job (same node)."""
        node       = _make_node("n1")
        high_prio  = _make_job("j1", priority=90)
        low_prio   = _make_job("j2", priority=10)

        ant = self._make_ant([high_prio, low_prio], [node])
        assert ant._eta[0, 0] > ant._eta[1, 0], (
            "High-priority job should have higher η"
        )

    def test_urgency_score_formula(self):
        """urgency_score = 1.0 + priority/100. Verify specific values."""
        assert np.isclose(Ant._urgency_score(_make_job(priority=100)), 2.0)
        assert np.isclose(Ant._urgency_score(_make_job(priority=50)),  1.5)
        assert np.isclose(Ant._urgency_score(_make_job(priority=1)),   1.01)

    # ── Resource headroom ──────────────────────────────────────────────────

    def test_headroom_capped_at_one(self):
        """A very roomy node gets headroom score = 1.0 (not > 1)."""
        roomy_node = _make_node("n1", total_cpu=1000.0, total_mem=1000.0)
        tiny_job   = _make_job("j1", cpu_min=0.1, mem_min=0.1)

        score = Ant._resource_headroom_score(tiny_job, roomy_node)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_headroom_uses_bottleneck(self):
        """
        When CPU headroom > mem headroom, the mem ratio (bottleneck) determines score.
        """
        # Node: 16 CPU (8 free), 8 GB (1.6 free)
        node = _make_node("n1", total_cpu=16.0, total_mem=8.0,
                          allocated_cpu=8.0, allocated_mem=6.4)
        # Job: needs 2 CPU, 1.5 GB
        # cpu_ratio = 8/2 = 4.0 → capped 1.0
        # mem_ratio = 1.6/1.5 = 1.066 → capped 1.0
        # min(1.0, 1.0) = 1.0
        job   = _make_job("j1", cpu_min=2.0, mem_min=1.5)
        score = Ant._resource_headroom_score(job, node)
        assert score == 1.0

        # Now make memory the actual bottleneck (nearly full)
        node2 = _make_node("n2", total_cpu=16.0, total_mem=8.0,
                           allocated_cpu=0.0, allocated_mem=7.5)
        # cpu_ratio = 16/2 = 8.0 → capped 1.0
        # mem_ratio = 0.5/1.5 = 0.333 → not capped
        # min(1.0, 0.333) = 0.333
        score2 = Ant._resource_headroom_score(job, node2)
        assert np.isclose(score2, 0.5 / 1.5, rtol=1e-5), (
            f"Expected mem-bottleneck score ~{0.5/1.5:.3f}, got {score2:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3 — Ant construction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAntConstruct:
    """Verify one ant builds a correct, complete solution."""

    def test_all_jobs_placed_on_healthy_cluster(self):
        """
        3 jobs on 5 healthy, well-resourced nodes → is_feasible=True,
        all 3 jobs mapped.
        """
        nodes = [_make_node(f"n{i}", total_cpu=64.0, total_mem=256.0)
                 for i in range(5)]
        jobs  = [_make_job(f"j{i}", cpu_min=2.0, mem_min=4.0) for i in range(3)]

        matrix = PheromoneMatrix(3, 5)
        ant    = Ant(jobs, nodes, matrix)
        result = ant.construct()

        assert result is True,          "Expected is_feasible=True"
        assert ant.is_feasible is True
        assert len(ant.solution) == 3,  "All 3 jobs must be in solution"
        assert ant.total_cost > 0.0,    "total_cost must be positive"

        for job_idx in range(3):
            assert job_idx in ant.solution,  f"Job {job_idx} missing from solution"
            node_idx = ant.solution[job_idx]
            assert 0 <= node_idx < 5,        f"Node idx {node_idx} out of range"

    def test_infeasible_cluster_sets_is_feasible_false(self):
        """
        1 job needing 1000 cores on 1 node with 16 cores → is_feasible=False.
        """
        node = _make_node("n1", total_cpu=16.0)
        job  = _make_job("j1", cpu_min=1000.0)

        matrix = PheromoneMatrix(1, 1)
        ant    = Ant([job], [node], matrix)
        result = ant.construct()

        assert result is False
        assert ant.is_feasible is False
        assert ant.total_cost >= INFEASIBLE_PENALTY, (
            "Infeasible solution should include INFEASIBLE_PENALTY in cost"
        )

    def test_solution_indices_are_valid(self):
        """solution dict values must be valid node indices."""
        nodes  = [_make_node(f"n{i}") for i in range(4)]
        jobs   = [_make_job(f"j{i}") for i in range(3)]
        matrix = PheromoneMatrix(3, 4)
        ant    = Ant(jobs, nodes, matrix)
        ant.construct()

        for job_idx, node_idx in ant.solution.items():
            assert 0 <= job_idx < 3,  f"job_idx={job_idx} out of range"
            assert 0 <= node_idx < 4, f"node_idx={node_idx} out of range"

    def test_priority_ordering_gpu_contention(self):
        """
        1 GPU node, 1 CPU fallback node.
        High-priority GPU job (priority=90) vs low-priority GPU job (priority=10).
        High-priority job should dominate GPU node placement.

        Strategy: run 50 trials. Because high-priority job gets first pick
        of the GPU node (processed first in construct()), it should always
        win the GPU node when both compete.
        """
        gpu_node = _make_node(
            "n-gpu",
            arch=NodeArch.GPU_NODE,
            total_cpu=16.0, total_mem=64.0,
            gpu_inventory={"A100": 1},
            gpu_vram_gb={"A100": 80.0},
        )
        cpu_node = _make_node(
            "n-cpu",
            arch=NodeArch.X86_64,
            total_cpu=16.0, total_mem=64.0,
        )
        nodes = [gpu_node, cpu_node]

        high_job = _make_job(
            "j-high",
            workload_type=WorkloadType.BATCH,
            cpu_min=2.0, mem_min=4.0,
            gpu_required=True, gpu_count=1,
            priority=90,
        )
        low_job = _make_job(
            "j-low",
            workload_type=WorkloadType.BATCH,
            cpu_min=2.0, mem_min=4.0,
            gpu_required=True, gpu_count=1,
            priority=10,
        )

        high_job_idx = 0  # First in list → row 0
        gpu_node_idx = 0  # First in nodes list → col 0

        # In a healthy scenario, we don't guarantee 100% because the heuristic
        # is stochastic — but priority ordering ensures high-priority processes first.
        # Count how many times high_job gets the GPU node.
        gpu_wins = 0
        trials = 50
        for _ in range(trials):
            matrix = PheromoneMatrix(2, 2)
            ant    = Ant([high_job, low_job], nodes, matrix)
            ant.construct()
            if ant.solution.get(0) == 0:  # high_job → gpu_node
                gpu_wins += 1

        # High-priority job should win the GPU node in the vast majority of trials
        assert gpu_wins >= 35, (
            f"High-priority job won GPU node only {gpu_wins}/{trials} times. "
            f"Priority ordering may not be working."
        )

    def test_repr_after_construct(self):
        """__repr__ shows placed count and feasibility (smoke test)."""
        nodes  = [_make_node("n1")]
        jobs   = [_make_job("j1")]
        matrix = PheromoneMatrix(1, 1)
        ant    = Ant(jobs, nodes, matrix)
        ant.construct()
        r = repr(ant)
        assert "Ant(" in r
        assert "feasible=" in r


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4 — Colony integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestColony:
    """End-to-end colony tests — the full ACO loop."""

    # ── Basic correctness ──────────────────────────────────────────────────

    def test_returns_valid_placement_plan(self):
        """
        colony.run() must return a dict of str→str where:
        • Every job_id from the input is a key.
        • Every node_id in values is from the input node list.
        """
        nodes = [_make_node(f"n{i}", total_cpu=32.0, total_mem=128.0)
                 for i in range(4)]
        jobs  = [_make_job(f"j{i}") for i in range(3)]

        colony = Colony(jobs=jobs, nodes=nodes)
        plan   = colony.run()

        assert isinstance(plan, dict), "Plan must be a dict"
        assert len(plan) == 3, f"Expected 3 entries, got {len(plan)}"

        node_ids = {n.node_id for n in nodes}
        for job_id, node_id in plan.items():
            assert isinstance(job_id, str),  "Keys must be strings (job_id)"
            assert isinstance(node_id, str), "Values must be strings (node_id)"
            assert node_id in node_ids, f"node_id {node_id!r} not in cluster"

        job_ids = {j.job_id for j in jobs}
        for jid in job_ids:
            assert jid in plan, f"job_id {jid!r} missing from plan"

    def test_single_job_single_node(self):
        """Edge case: 1 job, 1 node — must succeed and return correct mapping."""
        node = _make_node("n1")
        job  = _make_job("j1", workload_type=WorkloadType.BATCH)

        colony = Colony(jobs=[job], nodes=[node])
        plan   = colony.run()

        assert plan == {"j1": "n1"}, f"Expected {{'j1': 'n1'}}, got {plan}"

    def test_empty_jobs_raises_value_error(self):
        """Colony must reject an empty job list with ValueError."""
        with pytest.raises(ValueError, match="job"):
            Colony(jobs=[], nodes=[_make_node("n1")])

    def test_empty_nodes_raises_value_error(self):
        """Colony must reject an empty node list with ValueError."""
        with pytest.raises(ValueError, match="node"):
            Colony(jobs=[_make_job("j1")], nodes=[])

    def test_raises_colony_failed_error_when_infeasible(self):
        """
        If no node can fit any job, ColonyFailedError must be raised.
        The caller is responsible for catching and falling back to naive scheduling.
        """
        nodes = [_make_node(f"n{i}", total_cpu=4.0) for i in range(3)]
        impossible_job = _make_job("j1", cpu_min=1000.0)

        with pytest.raises(ColonyFailedError) as exc_info:
            Colony(jobs=[impossible_job], nodes=nodes).run()

        err = exc_info.value
        assert err.n_jobs == 1
        assert err.n_nodes == 3

    # ── Fast path ──────────────────────────────────────────────────────────

    def test_fast_path_latency_critical_returns_correct_job(self):
        """
        Single LATENCY_CRITICAL job must be placed and the plan contains the job.
        """
        nodes = [_make_node(f"n{i}", total_cpu=32.0, total_mem=128.0)
                 for i in range(5)]
        lc_job = _make_job(
            "j-lc",
            workload_type=WorkloadType.LATENCY_CRITICAL,
            latency_p99_ms=10,
        )

        colony = Colony(jobs=[lc_job], nodes=nodes)
        plan   = colony.run()

        assert "j-lc" in plan, "LATENCY_CRITICAL job must appear in plan"
        assert plan["j-lc"] in {n.node_id for n in nodes}

    def test_fast_path_is_deterministic(self):
        """
        Fast path uses argmax (not roulette wheel) → same node chosen every time.
        Run 100 trials to verify.
        """
        nodes = [_make_node(f"n{i}", total_cpu=32.0, total_mem=128.0,
                            cost_per_hour=float(i + 1) * 0.5)
                 for i in range(5)]
        lc_job = _make_job(
            "j-lc",
            workload_type=WorkloadType.LATENCY_CRITICAL,
            latency_p99_ms=5,
        )

        chosen_nodes = set()
        for _ in range(100):
            plan = Colony(jobs=[lc_job], nodes=nodes).run()
            chosen_nodes.add(plan["j-lc"])

        assert len(chosen_nodes) == 1, (
            f"Fast path should always pick the same node. "
            f"Got {len(chosen_nodes)} different nodes: {chosen_nodes}"
        )

    def test_fast_path_picks_best_node(self):
        """
        Fast path (argmax η) should select the most suitable node.
        Set up nodes so one is clearly better (higher headroom, lower cost).
        """
        # n0: expensive AND nearly full (allocated_cpu=3.8, total=4.0 → available=0.2 < cpu_min=1.0)
        #     can_fit() returns False → η=0.0 → never selected
        n0 = _make_node("n0", total_cpu=4.0, allocated_cpu=3.8, cost_per_hour=3.0)
        # n1: cheap, very roomy — only feasible node
        n1 = _make_node("n1", total_cpu=32.0, allocated_cpu=0.0, cost_per_hour=0.20)

        lc_job = _make_job(
            "j-lc",
            workload_type=WorkloadType.LATENCY_CRITICAL,
            cpu_min=1.0, mem_min=1.0, latency_p99_ms=5,   # cpu_min=1.0 > n0's 0.2 available
        )

        plan = Colony(jobs=[lc_job], nodes=[n0, n1]).run()
        assert plan["j-lc"] == "n1", (
            f"Fast path should pick the feasible, cheap node n1. Got: {plan['j-lc']}"
        )

    def test_fast_path_latency_raises_if_no_node_fits(self):
        """Fast path must raise ColonyFailedError if no node can fit the LC job."""
        tiny_node = _make_node("n1", total_cpu=0.1, total_mem=0.1)
        lc_job    = _make_job(
            "j-lc",
            workload_type=WorkloadType.LATENCY_CRITICAL,
            cpu_min=100.0, mem_min=100.0, latency_p99_ms=10,
        )
        with pytest.raises(ColonyFailedError):
            Colony(jobs=[lc_job], nodes=[tiny_node]).run()

    # ── Performance benchmark ──────────────────────────────────────────────

    def test_performance_benchmark_8ms(self):
        """
        Colony (20 ants × 5 iterations, 10 nodes, 5 jobs) must complete in ≤8ms
        on average over 10 runs.

        This validates the sub-10ms scheduling latency target.
        If this test fails, the first optimisation is reusing Ant objects
        between iterations (eliminating re-instantiation overhead).
        """
        nodes = [
            _make_node(f"n{i}", total_cpu=32.0, total_mem=128.0,
                       cost_per_hour=float(i + 1) * 0.3)
            for i in range(10)
        ]
        jobs = [
            _make_job(f"j{i}",
                      workload_type=WorkloadType.BATCH if i % 2 == 0 else WorkloadType.STREAM,
                      cpu_min=2.0, mem_min=4.0, priority=10 + i * 10)
            for i in range(5)
        ]

        elapsed_times: List[float] = []
        for _ in range(10):
            colony = Colony(jobs=jobs, nodes=nodes)
            t0 = time.perf_counter()
            colony.run()
            elapsed_times.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = sum(elapsed_times) / len(elapsed_times)
        max_ms = max(elapsed_times)

        assert avg_ms <= 8.0, (
            f"Average colony latency {avg_ms:.2f}ms exceeds 8ms budget.\n"
            f"Individual runs: {[f'{t:.2f}' for t in elapsed_times]}"
        )
        # Log the result even on success (visible with pytest -s)
        print(
            f"\n[PERF] Colony avg={avg_ms:.2f}ms, max={max_ms:.2f}ms "
            f"over {len(elapsed_times)} runs"
        )

    def test_last_run_ms_is_set(self):
        """colony.last_run_ms is populated after run() (useful for monitoring)."""
        nodes  = [_make_node("n1")]
        jobs   = [_make_job("j1", workload_type=WorkloadType.BATCH)]
        colony = Colony(jobs=jobs, nodes=nodes)
        assert colony.last_run_ms == 0.0, "Should start at 0.0 before run()"
        colony.run()
        assert colony.last_run_ms > 0.0, "Should be > 0 after run()"

    # ── Convergence (pheromone learning) ──────────────────────────────────

    def test_pheromone_converges_to_best_node(self):
        """
        With one clearly superior node (more headroom, lower cost),
        the colony should select it in the majority of independent runs.

        This tests that the ACO pheromone feedback mechanism is working:
        good placements accumulate pheromone, attracting more ants,
        further reinforcing those placements.

        We run the colony 50 times independently (fresh Colony each time,
        fresh PheromoneMatrix each time). If convergence works, the best
        node should appear in ≥70% of runs.

        Note: Since each run is fresh (no shared state between Colony instances),
        this test validates that the heuristic consistently guides ants to the
        best node, and that within each run's iterations, pheromone reinforces
        the correct arc.
        """
        # n-best: very cheap, very roomy
        n_best  = _make_node("n-best",  total_cpu=128.0, total_mem=512.0, cost_per_hour=0.10)
        # n-ok:   moderate cost, moderate resources
        n_ok    = _make_node("n-ok",    total_cpu=16.0,  total_mem=64.0,  cost_per_hour=1.00)
        # n-poor: expensive, nearly full
        n_poor  = _make_node("n-poor",  total_cpu=4.0,   total_mem=8.0,   cost_per_hour=5.00,
                             allocated_cpu=3.0, allocated_mem=6.0)
        nodes = [n_poor, n_ok, n_best]   # Deliberately put best last

        job = _make_job("j1", workload_type=WorkloadType.BATCH,
                        cpu_min=1.0, mem_min=2.0, priority=50)

        best_node_wins = 0
        trials = 50
        for _ in range(trials):
            plan = Colony(jobs=[job], nodes=nodes).run()
            if plan["j1"] == "n-best":
                best_node_wins += 1

        win_rate = best_node_wins / trials
        assert win_rate >= 0.70, (
            f"Colony chose best node only {best_node_wins}/{trials} times "
            f"({win_rate:.0%}). Expected ≥70%. "
            f"Heuristic or pheromone convergence may not be working."
        )
        print(f"\n[CONVERGENCE] Best node selected {best_node_wins}/{trials} times ({win_rate:.0%})")

    # ── repr ──────────────────────────────────────────────────────────────

    def test_repr_after_run(self):
        """Colony repr includes job/node counts and last_run_ms."""
        colony = Colony(jobs=[_make_job("j1")], nodes=[_make_node("n1")])
        colony.run()
        r = repr(colony)
        assert "Colony(" in r
        assert "last_run_ms" in r
