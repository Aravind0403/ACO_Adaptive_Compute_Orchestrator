# ACO Adaptive Compute Scheduler — Technical Documentation

> Version 2.0.0 · Python 3.11+ · FastAPI · PyTorch · NumPy

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Mock Cluster](#mock-cluster)
4. [Phase-by-Phase Components](#phase-by-phase-components)
   - [Phase 1 — Data Models](#phase-1--data-models)
   - [Phase 2 — ACO Core Engine](#phase-2--aco-core-engine)
   - [Phase 3 — LSTM Predictor](#phase-3--lstm-predictor)
   - [Phase 4 — Cost Engine](#phase-4--cost-engine)
   - [Phase 5 — ACO Scheduler & Orchestration](#phase-5--aco-scheduler--orchestration)
   - [Phase 6 — Workload Intent Router](#phase-6--workload-intent-router)
   - [Phase 7 — Telemetry Collector](#phase-7--telemetry-collector)
   - [Phase 7.5 — Trace Replay Adapters](#phase-75--trace-replay-adapters)
   - [Phase 8 — Data Plane Agent](#phase-8--data-plane-agent)
   - [Phase 9 — REST API & Dashboard](#phase-9--rest-api--dashboard)
5. [Data Flow](#data-flow)
6. [API Reference](#api-reference)
7. [Running the System](#running-the-system)
8. [Testing](#testing)
9. [Configuration Reference](#configuration-reference)
10. [V3 Upgrade Path](#v3-upgrade-path)

---

## System Overview

The ACO Adaptive Compute Scheduler is a **predictive, intent-aware, cost-conscious job placement engine** for heterogeneous compute clusters. It replaces naive first-fit scheduling with three overlapping signals:

| Signal | Source | Used by |
|--------|--------|---------|
| **Pheromone** (learned history) | ACO colony convergence | Ant selection probability |
| **Heuristic** (cost + headroom + spike risk) | CostEngine | η value per (job, node) pair |
| **Intent** (workload type + constraints) | WorkloadIntentRouter | Node pre-filtering, strategy selection |

All three are combined in under **10 ms** for a typical workload mix.

### Design Goals

| Goal | Mechanism |
|------|-----------|
| <10 ms scheduling latency | NumPy ACO + fast path for latency-critical jobs |
| +28% utilisation vs first-fit | Predictive placement avoids pre-spike nodes |
| 95%+ SLA adherence | Hard gates in CostEngine + intent-based node filtering |
| >80% spike recall | LSTM predictor trained on real cluster trace data |
| Zero external dependencies at runtime | All state in-memory; no database, no broker |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI (api/main.py)                    │
│  POST /jobs   GET /metrics   GET /predict/:id   POST /upload-trace│
│  POST /simulation/start     GET /nodes   GET /jobs/history       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ submit_job()
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    OrchestratorService                            │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
│  │ Admission    │  │ IntentRouter    │  │ aco_schedule()    │   │
│  │ Controller   │→ │ (classify job)  │→ │ (ACO colony or    │   │
│  └──────────────┘  └─────────────────┘  │  fast path)       │   │
│                                          └───────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  node_state  │  active_jobs  │  prediction_cache         │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌───────────┐ ┌──────────────┐
│ ACO Colony   │ │ CostEngine│ │ Predictor    │
│ PheromoneMatrix│ (η heuristic│ (LSTM, per   │
│ 20 Ants      │ │ per pair)  │ │ node)        │
│ 5 Iterations │ └───────────┘ └──────────────┘
└──────────────┘
        │
        ▼ (background, every 5s)
┌──────────────────────────────────────────────────────────────────┐
│                     TelemetryCollector                            │
│  tick() → generate NodeTelemetry → update_node_telemetry()       │
│  every 10 ticks: _refit() → LSTM retrains → prediction_cache    │
│                                                                   │
│  Telemetry source (hot-swappable):                               │
│    synthetic Gaussian  OR  Alibaba 2018 trace  OR  Borg 2019     │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼ (per node, background heartbeat)
┌──────────────────────────────────────────────────────────────────┐
│                       NodeAgent (×5)                              │
│  execute_job() → asyncio.sleep(duration) → complete_job()        │
│  _heartbeat_loop() → update_node_telemetry() every 5s            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Mock Cluster

Five nodes spanning three hardware tiers, mimicking a real heterogeneous cluster:

| Node ID | Arch | Instance | CPU cores | Memory GB | GPU | Cost/hr | Baseline CPU% |
|---------|------|----------|-----------|-----------|-----|---------|---------------|
| `node-cpu-01` | x86_64 | ON_DEMAND | 32 | 128 | — | $0.48 | 35% |
| `node-arm-02` | ARM64 | SPOT | 16 | 64 | — | $0.12 | 20% |
| `node-api-03` | x86_64 | ON_DEMAND | 8 | 32 | — | $0.18 | 60% |
| `node-gpu-04` | GPU_NODE | ON_DEMAND | 16 | 128 | A100 | $3.20 | 45% |
| `node-gpu-05` | GPU_NODE | SPOT | 16 | 128 | A100 | $1.10 | 30% |

**Routing heuristic:** The intent router naturally directs:
- GPU inference → `node-gpu-04` (ON_DEMAND GPU, never spot)
- CPU API serving → `node-api-03` (small x86, low latency)
- MCTS actors → `node-arm-02` (cheap ARM64 spot)
- Replay buffers → `node-cpu-01` (large x86, memory-rich, ON_DEMAND)
- GPU training → `node-gpu-04` or `node-gpu-05` (spot allowed)

---

## Phase-by-Phase Components

### Phase 1 — Data Models

**Files:** `orchestrator/shared/models.py`, `orchestrator/shared/telemetry.py`

All Pydantic v2 models. Key types:

```
JobRequest          — a job submission (workload_type, resources, priority, constraints)
ResourceRequest     — CPU/memory/GPU minimums
ComputeNode         — node capacity + live state + telemetry
NodeTelemetry       — real-time cpu_util_pct, memory_util_pct, gpu_util_pct
JobExecution        — tracks a job from PENDING → RUNNING → COMPLETED
PredictionResult    — LSTM output: predicted_cpu_util, spike_probability, confidence
WorkloadProfile     — per-node sample history, avg/burst metrics for LSTM training
```

**Key constraints to know:**
- `ResourceRequest.gpu_count` has `ge=1` — always pass `gpu_count=1` even for non-GPU jobs
- `WorkloadType` values: `'batch'`, `'latency-critical'`, `'stream-processing'`
- `PredictionResult` requires both `predicted_cpu_util` AND `predicted_memory_util`

---

### Phase 2 — ACO Core Engine

**Files:** `aco_core/pheromone.py`, `aco_core/ant.py`, `aco_core/colony.py`

#### PheromoneMatrix (`pheromone.py`)

A 2D NumPy array `τ[n_jobs][n_nodes]` — the colony's shared memory.

| Constant | Value | Purpose |
|----------|-------|---------|
| `TAU_INITIAL` | 1.0 | All cells start equal — no prior bias |
| `TAU_MIN` | 0.01 | Floor: keeps all options alive (prevents stagnation) |
| `TAU_MAX` | 10.0 | Ceiling: no one solution dominates too fast |
| `RHO` | 0.1 | Evaporation rate: 10% decay per iteration |
| `Q` | 1.0 | Deposit numerator |

Operations:
- `evaporate()` — in-place: `matrix *= (1 - RHO)`, then clip to `[TAU_MIN, TAU_MAX]`
- `deposit(job_idx, node_idx, cost)` — adds `Q / cost` to one cell; skips if `cost <= 0`
- `get_row(job_idx)` — returns a **view** (not copy) of one row

#### Ant (`ant.py`)

One ant builds one complete `PlacementPlan`. Uses probabilistic roulette-wheel selection.

**η (heuristic) formula:**
```
η[job][node] = resource_headroom × cost_gate × workload_affinity × urgency
```

- `resource_headroom` — `min(cpu_ratio, mem_ratio)`, capped at 1.0 (bottleneck resource)
- `cost_gate` — 1.0 if within cost ceiling, 0.0 if over (hard gate)
- `workload_affinity` — GPU×BATCH = 1.5, GPU×LATENCY = 0.5, arch mismatch = 0.0
- `urgency` — `1.0 + priority / 100.0` (range 1.01–2.0)

If `node.can_fit(job.resources)` is False → `η = 0.0` (never selected).

**Selection (roulette wheel):**
```python
numerators = (τ_row ** ALPHA) * (η_row ** BETA)   # ALPHA=1.0, BETA=2.0
if numerators.sum() == 0.0:
    return None   # no feasible node
probabilities = numerators / numerators.sum()
return int(np.random.choice(len(nodes), p=probabilities))
```

#### Colony (`colony.py`)

Outer loop: `N_ANTS=20` ants × `N_ITERATIONS=5` iterations. Early stop after `STAGNATION_LIMIT=3` iterations without improvement.

**Two scheduling paths:**

| Path | Trigger | Latency | Method |
|------|---------|---------|--------|
| Fast path | Single LATENCY_CRITICAL job | <1 ms | Deterministic `argmax(η)` — no randomness |
| Normal path | All other cases | ≤8 ms | Full colony (20×5 iterations) |

Why deterministic for LC? Variance in placement is unacceptable for strict P99 SLA targets.

**Performance estimate:**
```
100 ant constructions × (10-node vector ops × 20 jobs) ≈ 100,000 NumPy ops
At 100M+ NumPy ops/sec → ~0.001ms NumPy time
2,000 np.random.choice calls → ~0.5ms
Python loop overhead → ~3–5ms
Total: 4–6ms ✓ (well within 8ms budget)
```

---

### Phase 3 — LSTM Predictor

**File:** `orchestrator/control_plane/predictor.py`

#### Architecture

```
Input:  (1, 10, 1)   — batch=1, seq_len=10 (lookback), features=1 (CPU util)
LSTM:   hidden_size=32, num_layers=1, batch_first=True
Linear: 32 → 1
Output: scalar → denormalise → clamp [0.0, 100.0]
```

Why 32 hidden units: minimum to capture short-term autocorrelation; 64+ overfits on ≤500 samples; single layer avoids vanishing gradients at seq_len=10.

#### Training

- **Full-batch**: ≤490 samples fits in one tensor; mini-batching overhead > benefit
- **50 epochs** with Adam (lr=0.01) + MSELoss
- **Z-score normalisation** per node: `mean`, `std = max(np.std(history), 1e-6)` — prevents scale drift across nodes with different base loads
- **Refit trigger**: `refit_if_needed()` refits when sample count grows by ≥10 since last fit

#### Inference

**Cold-start** (not yet trained, <10 samples):
```
predicted_cpu_util = min(avg_cpu_cores × 10, 100.0)
confidence = 0.1
spike_probability = 0.0
```

**Trained path:**
```
spike_probability = clamp((pred_cpu - rolling_mean) / max(rolling_mean, 1.0), 0.0, 1.0)
if burst_factor > 1.5: spike_probability = min(spike_probability + 0.2, 1.0)

confidence = min(0.5 + (n_samples - 10) / (500 - 10) × 0.5, 1.0)
             # 10 samples → 0.5;  500 samples → 1.0
```

---

### Phase 4 — Cost Engine

**File:** `orchestrator/control_plane/cost_engine.py`

Translates raw economics + risk into a single scalar score used as the ACO η heuristic.

#### Composite Score

```
score(job, node) = reliability_factor
                × cost_efficiency_factor
                × sla_headroom_factor
                × prediction_factor
```

All four factors are in `(0.0, 1.0]`. A score of `0.0` means "never place here."

| Factor | Formula | Notes |
|--------|---------|-------|
| `reliability_factor` | ON_DEMAND → 1.0; SPOT+LC → `(1 − interruption_prob)`; SPOT+BATCH → `1 − 0.3×prob` | Hard gate for LC on high-risk SPOT |
| `cost_efficiency_factor` | `1 / (1 + cost_per_hour / MAX_COST_REFERENCE)` | Asymptotic — cheap nodes favoured but expensive never zeroed |
| `sla_headroom_factor` | `headroom = (100 − cpu_util) / 100`; LC uses strict headroom; BATCH uses `max(headroom, 0.1)` | Uses real telemetry if available, allocation estimate otherwise |
| `prediction_factor` | `1 − spike_weight × spike_probability` | Reduces score on nodes predicted to spike soon |

**Strategy overrides (Phase 6):** All thresholds accept keyword overrides from `SchedulingStrategy`, making the CostEngine fully backwards-compatible while enabling per-workload-type tuning.

---

### Phase 5 — ACO Scheduler & Orchestration

**Files:** `orchestrator/control_plane/scheduler.py`, `orchestrator/control_plane/orchestration_service.py`

#### `aco_schedule()`

```python
def aco_schedule(
    job_request: JobRequest,
    available_nodes: List[ComputeNode],
    predictors: Optional[Dict[str, PredictionResult]] = None,
    strategy: Optional[SchedulingStrategy] = None,
    node_workload_map: Optional[Dict[str, List[WorkloadType]]] = None,
) -> str:   # returns node_id
```

Steps:
1. Filter nodes by `can_fit()` + `strategy.required_arch` + `strategy.required_instance`
2. Apply colocation filter: nodes running `strategy.avoid_workload_types` get `η = 0.0`
3. Score each (job, node) pair with CostEngine (using strategy threshold overrides)
4. If `strategy.use_fast_path` or single LC job → deterministic `argmax(η)`
5. Else → full colony run
6. Falls back to `naive_schedule()` if colony raises `ColonyFailedError`

#### `OrchestratorService`

Central state machine. Public API:

| Method | Description |
|--------|-------------|
| `submit_job(request_data)` | Admit → classify intent → schedule → allocate → return placement |
| `complete_job(job_id, success, actual_cpu, actual_mem)` | Release resources, update WorkloadProfile, record latency |
| `update_node_telemetry(telemetry)` | Store latest NodeTelemetry on the node (used by CostEngine) |
| `get_prediction(node_id)` | Return cached LSTM result for a node |
| `refit_all_predictors()` | Refit all per-node predictors (called by TelemetryCollector) |
| `get_scheduling_metrics()` | P99 latency, avg utilisation, SLA adherence counts |

---

### Phase 6 — Workload Intent Router

**File:** `orchestrator/control_plane/intent_router.py`

Reads `JobRequest` intent fields and returns a `SchedulingStrategy` — a named configuration struct that pre-filters nodes and overrides CostEngine thresholds.

#### Routing Table (first match wins)

| Rule | Trigger | Strategy | Key behaviour |
|------|---------|----------|---------------|
| GPU Inference | LC + `gpu_required=True` | `GPU_INFERENCE` | Fast path; ON_DEMAND only; avoids BATCH colocates |
| CPU Serving | LC + `gpu_required=False` | `CPU_SERVING` | Fast path; ON_DEMAND x86/ARM; spike_weight=0.8 |
| GPU Training | BATCH + `gpu_required=True` + non-preemptible | `GPU_TRAINING` | Full colony; SPOT allowed; spike_weight=0.2 |
| Preemptible Actor | BATCH or STREAM + `preemptible=True` | `PREEMPTIBLE_ACTOR` | Full colony; SPOT preferred; very forgiving SLA |
| Stateful Stream | STREAM + `preemptible=False` | `STATEFUL_STREAM` | Full colony; ON_DEMAND only (can't be interrupted) |
| Deadline override | Any + `deadline_epoch < now() + 60s` | *(modifies above)* | Forces fast path; tightens SLA threshold +0.10 |

#### `SchedulingStrategy` fields

```python
@dataclass
class SchedulingStrategy:
    name: str
    required_arch: Optional[List[NodeArch]]         # hard filter
    required_instance: Optional[List[InstanceType]] # hard filter
    use_fast_path: bool                             # True = argmax η
    allow_spot: bool
    sla_strict_threshold: float                     # headroom requirement
    spike_penalty_weight: float                     # how much to penalise spiky nodes
    spot_penalty_threshold: float                   # max tolerable interruption risk
    avoid_workload_types: List[WorkloadType]        # colocation policy
```

---

### Phase 7 — Telemetry Collector

**File:** `orchestrator/telemetry/collector.py`

Drives the full prediction loop. Called by the background `_telemetry_loop()` in `api/main.py` every `_TELEMETRY_INTERVAL_S = 5.0` seconds.

#### `TelemetryCollector`

| Constant | Value | Purpose |
|----------|-------|---------|
| `REFIT_INTERVAL` | 10 | Refit predictors every N ticks (LSTM training is ~50ms) |
| `CPU_NOISE_STD` | 5.0% | Gaussian noise std dev for synthetic mode |
| `SPIKE_CPU_UTIL` | 92.0% | Injected CPU level during spike simulation |
| `MEMORY_BASE_UTIL` | 50.0% | Memory baseline for all nodes |

**`tick()` — one collection cycle:**
```
For each node:
  1. _generate_telemetry(node_id) → NodeTelemetry
  2. service.update_node_telemetry(telemetry)
  3. Add ResourceSample to _per_node_profiles[node_id]
Increment _tick_count
If _tick_count % REFIT_INTERVAL == 0: _refit()
```

**`_refit()`:**
```
For each node:
  predictor.refit_if_needed(_per_node_profiles[node_id])
  service._prediction_cache[node_id] = predictor.predict(profile)
```

**`inject_spike(node_id, n_ticks=5)`:**
Raises baseline to `SPIKE_CPU_UTIL = 92%` for N ticks, then reverts. Works transparently on top of both synthetic and trace-replay telemetry.

**Two telemetry modes:**

| Mode | Trigger | CPU source |
|------|---------|-----------|
| Synthetic | `trace_adapter=None` (default) | `N(base_cpu, 5.0)` Gaussian |
| Trace replay | `trace_adapter=AlibabaMachineTraceAdapter(...)` | Real cluster trace |

---

### Phase 7.5 — Trace Replay Adapters

**File:** `orchestrator/telemetry/trace_adapter.py`

Two adapters that replace synthetic Gaussian noise with real cluster data. Both implement the same interface:

```python
def get_reading(self, node_id: str, tick_number: int) -> Tuple[float, float]:
    """Returns (cpu_util_pct, mem_util_pct), both in [0.0, 100.0]."""
```

#### `AlibabaMachineTraceAdapter`

Source: Alibaba 2018 cluster trace (Zenodo record 14564935), pre-processed to 300-second intervals over 8 days (~2243 rows).

CSV schema: `cpu_util_percent`, `mem_util_percent` — already in 0–100 scale.

**Why this trace is realistic:**
- CPU range 16–79% (mean ≈ 40%) — similar to real production clusters
- Temporal autocorrelation (avg tick-to-tick change: 4.6%)
- Diurnal cycles (load rises/falls through each 24-hour period)
- Burst events (sudden spikes, not gradual drifts)

**Per-node mapping:** Each of the 5 mock nodes is assigned a unique time offset into the 8-day trace, so nodes see different sections simultaneously (independent temporal patterns):

| Node | Offset | CPU scale | CPU bias | Target baseline |
|------|--------|-----------|----------|-----------------|
| `node-cpu-01` | 0 | 0.85 | 0.0 | ~35% |
| `node-arm-02` | 448 | 0.50 | 0.0 | ~20% |
| `node-api-03` | 896 | 1.00 | 19.8 | ~60% |
| `node-gpu-04` | 1344 | 1.00 | 4.8 | ~45% |
| `node-gpu-05` | 1791 | 0.70 | 2.0 | ~30% |

Memory: scaled ×0.5 (raw trace is 78–95% — high, near-constant cluster allocation).

**Circular buffer:** after 2243 ticks (8 days), wraps back to tick 0 — tests can run indefinitely.

#### `BorgTraceAdapter`

Source: Google Cluster Trace 2019 (Borg/GCE) — Kaggle format.

Required columns: `time` (nanoseconds), `average_usage` (dict string: `{'cpus': 0.021, 'memory': 0.014}`).

**Processing pipeline:**
1. Parse `average_usage` via regex (`_BORG_USAGE_RE`) → `(cpu_frac, mem_frac)`
2. Filter sentinel timestamps (`Long.MAX_VALUE`) and zero-usage rows
3. Sort by event time to preserve real temporal ordering
4. Bucket into ~2000 row-count-based buckets and average each
5. Auto-scale: P75 of bucket averages → target baseline (40% CPU / 50% mem)
6. Apply per-node offset + scale/bias (same pattern as Alibaba adapter)

**Hot-swap at runtime:** `POST /upload-trace` auto-detects format (Alibaba vs Borg by column names) and installs the new adapter immediately — no restart required.

---

### Phase 8 — Data Plane Agent

**File:** `orchestrator/data_plane/agent.py`

Simulates a lightweight per-node daemon. In V2, runs in-process. In V3, becomes a real agent connecting over HTTP with `httpx`.

#### `NodeAgent`

| Constant | Value | Purpose |
|----------|-------|---------|
| `HEARTBEAT_INTERVAL_S` | 5.0 | Telemetry push frequency |
| `CPU_USAGE_RATIO` | 0.85 | Jobs use ~85% of requested CPU on average |
| `MEM_USAGE_RATIO` | 0.90 | Jobs use ~90% of requested memory |
| `USAGE_NOISE_STD` | 0.05 | ±5% usage variance |
| `DURATION_SCALE_S` | 10.0 | Base job duration multiplier |
| `MIN_DURATION_S` | 0.05 | Floor: prevents zero-duration in fast tests |
| `MAX_DURATION_S` | 30.0 | Ceiling: keeps tests tractable |

**`async execute_job(job_execution)`:**

```
1. Register job as in-flight
2. Compute duration: cpu_fraction × DURATION_SCALE_S, ±20% noise
3. await asyncio.sleep(duration)   ← cooperative, non-blocking
4. Sample actual CPU/mem: ~85%/90% of requested, ±5%, capped at requested × 1.1
5. service.complete_job(job_id, success=True, actual_cpu=..., actual_mem=...)
6. Remove from _running_jobs
```

**`_heartbeat_loop()`:** Every `HEARTBEAT_INTERVAL_S` seconds, calls `service.update_node_telemetry()` with current node load. Cancelled cleanly on `stop()`.

**Why 85%/90% usage ratios?** Real cluster data shows jobs typically use 80–95% of reserved resources. Feeding actual usage (not just requested) to `WorkloadProfile` gives the LSTM more realistic training signals.

**Why 10% burst cap?** Matches Kubernetes `resource.limits` behaviour — jobs may briefly burst 10% above reservation but no more.

---

### Phase 9 — REST API & Dashboard

**File:** `api/main.py`

FastAPI application with a built-in real-time dashboard at `GET /`.

#### Background Tasks (lifespan)

On startup:
- One `NodeAgent` per cluster node → `agent.start()` (heartbeat running)
- One `asyncio.create_task(_telemetry_loop())` → calls `collector.tick()` every 5s

On shutdown:
- Cancel simulation task (if running)
- Cancel telemetry loop task
- `agent.stop()` for each node

#### Simulation Loop

Auto-submits random jobs from `_SIM_WORKLOADS` every `_sim_interval_s` seconds (default 8s). Workload mix:

| Workload type | CPU | Mem | GPU | Priority |
|---------------|-----|-----|-----|----------|
| batch | 4.0 | 8.0 | No | 30 |
| batch | 8.0 | 16.0 | No | 50 |
| batch | 2.0 | 4.0 | No | 20 |
| latency-critical | 2.0 | 4.0 | No | 90 |
| latency-critical | 1.0 | 2.0 | No | 95 |
| stream-processing | 2.0 | 8.0 | No | 60 |
| batch (GPU) | 4.0 | 16.0 | Yes | 40 |
| stream-processing | 1.0 | 2.0 | No | 70 |

Toggle via the "▶ Start Simulation" button on the dashboard, or via `POST /simulation/start`.

---

## Data Flow

### Job Submission (happy path)

```
POST /jobs
  → FastAPI: parse JobSubmitRequest
  → OrchestratorService.submit_job()
      → admit_job() — semantic checks (resources > 0, valid workload type)
      → WorkloadIntentRouter.classify(job) → SchedulingStrategy
      → aco_schedule(job, available_nodes, prediction_cache, strategy)
          → filter nodes (capacity + arch + instance type + colocation)
          → CostEngine.score_node() × n_nodes → η array
          → fast path or colony → node_id
      → _allocate_resources(node_id, job) — deduct from ComputeNode
      → JobExecution(state=RUNNING) stored in active_jobs
  → FastAPI: asyncio.create_task(agent.execute_job(job_ex))
  → Return 202 {"status": "SCHEDULED", "node_id": "node-cpu-01", ...}
```

### Job Completion (background)

```
NodeAgent.execute_job()
  → asyncio.sleep(simulated_duration)
  → OrchestratorService.complete_job(job_id, actual_cpu, actual_mem)
      → _release_resources(node_id, job) — return CPU/mem to ComputeNode
      → WorkloadProfile.add_sample(ResourceSample)  ← training signal
      → JobExecution(state=COMPLETED) moved to completed_jobs
      → scheduling_latencies.append(latency_ms)
```

### Telemetry → Prediction Pipeline (background, every 5s)

```
_telemetry_loop() calls collector.tick()
  → For each node:
      _generate_telemetry(node_id)  ← trace or synthetic
      service.update_node_telemetry(telemetry)  ← node.latest_telemetry updated
      _per_node_profiles[node_id].add_sample(ResourceSample)
  → Every 10 ticks: _refit()
      → For each node:
          predictor.refit_if_needed(profile)  ← LSTM trains if ≥10 new samples
          service._prediction_cache[node_id] = predictor.predict(profile)
```

Next call to `aco_schedule()` reads `_prediction_cache` → CostEngine penalises nodes with high `spike_probability`.

---

## API Reference

### Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit a job. Returns 202 (SCHEDULED) or 422 (REJECTED). |
| `GET` | `/jobs` | List all active (running) jobs. |
| `GET` | `/jobs/history?limit=20` | List recently completed jobs. |
| `GET` | `/jobs/{job_id}` | Get status of a specific job. |

**`POST /jobs` request body:**

```json
{
  "workload_type": "batch",
  "cpu_cores_min": 4.0,
  "memory_gb_min": 8.0,
  "gpu_required": false,
  "gpu_count": 1,
  "priority": 50,
  "preemptible": false,
  "arch_required": null,
  "cost_ceiling_usd": null,
  "deadline_epoch": null,
  "latency_p99_ms": null
}
```

**`POST /jobs` response (202):**

```json
{
  "status": "SCHEDULED",
  "job_id": "job-abc123",
  "node_id": "node-cpu-01",
  "scheduling_latency_ms": 4.2
}
```

### Nodes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/nodes` | All cluster nodes with live utilisation + telemetry. |

### Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/metrics` | P99 scheduling latency, avg utilisation, prediction confidence per node. |
| `GET` | `/predict/{node_id}` | Latest LSTM prediction for a specific node. |

**`GET /metrics` response:**

```json
{
  "active_jobs": 3,
  "completed_jobs": 27,
  "p99_latency_ms": 6.4,
  "avg_cpu_utilisation_pct": 42.1,
  "telemetry_source": "alibaba-2018",
  "predictions": {
    "node-cpu-01": {
      "predicted_cpu_util": 38.5,
      "spike_probability": 0.12,
      "confidence": 0.73
    }
  }
}
```

### Simulation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/simulation/start?interval_s=8` | Start auto-submitting random jobs. |
| `POST` | `/simulation/stop` | Stop the simulation loop. |
| `GET` | `/simulation/status` | Is the simulation running? |

### Trace Upload

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-trace` | Upload a cluster trace CSV (hot-swap, no restart). |

Format auto-detected: Alibaba (columns `cpu_util_percent`, `mem_util_percent`) or Borg (columns `time`, `average_usage`).

---

## Running the System

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `fastapi`, `uvicorn[standard]`, `torch>=2.2.0`, `numpy>=1.26.0`, `httpx`, `pytest-asyncio`.

### Start the server

```bash
# From project root:
python run.py

# Or directly:
uvicorn api.main:app --reload --port 8000
```

Open the dashboard: `http://localhost:8000`

### Load real cluster trace data

```bash
# Alibaba 2018 trace (202 KB, free, no account required)
curl -L "https://zenodo.org/records/14564935/files/machine_usage_days_1_to_8_grouped_300_seconds.csv" \
     -o tests/fixtures/alibaba_machine_usage_300s.csv
```

Then upload via the dashboard (drag-and-drop) or:
```bash
curl -X POST http://localhost:8000/upload-trace \
     -F "file=@tests/fixtures/alibaba_machine_usage_300s.csv"
```

### Submit a job manually

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"workload_type": "latency-critical", "cpu_cores_min": 2.0, "memory_gb_min": 4.0, "priority": 90}'
```

---

## Testing

### Run all tests

```bash
python -m pytest tests/ -v
```

Expected: **202 tests passing** (as of Phase 8 completion).

### Test suite breakdown

| File | Phase | Tests | Focus |
|------|-------|-------|-------|
| `test_aco_phase2.py` | 2 | 43 | PheromoneMatrix, Ant η, Colony convergence, performance |
| `test_predictor.py` | 3 | 20 | LSTM training, cold-start, spike detection, confidence |
| `test_cost_engine.py` | 4 | 30 | All four sub-scores, edge cases, threshold overrides |
| `test_aco_phase5.py` | 5 | 32 | aco_schedule integration, OrchestratorService lifecycle |
| `test_intent_router.py` | 6 | 24 | Strategy classification, deadline override, colocation |
| `test_aco_phase7.py` | 7 | 20 | TelemetryCollector, per-node profiles, prediction cache |
| `test_trace_adapter.py` | 7.5 | 13 | Adapter loading, trace replay, integration with collector |
| `test_data_plane.py` | 8 | 20 | NodeAgent init, execute_job, heartbeat, end-to-end |

### Run a specific phase

```bash
python -m pytest tests/test_aco_phase2.py -v     # ACO core
python -m pytest tests/test_predictor.py -v      # LSTM predictor
python -m pytest tests/test_data_plane.py -v     # NodeAgent (async)
```

### Performance benchmark

```bash
python -m pytest tests/test_aco_phase2.py -k "benchmark" -v
```

Target: ACO colony for 5 jobs × 10 nodes averages ≤8ms over 10 runs.

---

## Configuration Reference

### ACO Core (`aco_core/`)

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `N_ANTS` | `colony.py` | 20 | Ants per iteration |
| `N_ITERATIONS` | `colony.py` | 5 | Max iterations |
| `STAGNATION_LIMIT` | `colony.py` | 3 | Early-stop after N iterations without improvement |
| `ALPHA` | `ant.py` | 1.0 | Pheromone exponent (exploitation weight) |
| `BETA` | `ant.py` | 2.0 | Heuristic exponent (exploration weight) |
| `TAU_INITIAL` | `pheromone.py` | 1.0 | Initial pheromone level |
| `RHO` | `pheromone.py` | 0.1 | Evaporation rate per iteration |

### LSTM Predictor (`orchestrator/control_plane/predictor.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `LOOKBACK` | 10 | Sequence length for LSTM input |
| `HIDDEN_SIZE` | 32 | LSTM hidden units |
| `TRAIN_EPOCHS` | 50 | Training epochs per refit |
| `LEARNING_RATE` | 0.01 | Adam optimizer learning rate |

### Telemetry Collector (`orchestrator/telemetry/collector.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `REFIT_INTERVAL` | 10 | Ticks between LSTM refits |
| `CPU_NOISE_STD` | 5.0% | Gaussian noise std dev (synthetic mode) |
| `SPIKE_CPU_UTIL` | 92.0% | CPU level during injected spike |
| `MEMORY_BASE_UTIL` | 50.0% | Memory baseline for synthetic mode |

### API (`api/main.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `_TELEMETRY_INTERVAL_S` | 5.0 | Seconds between collector ticks |
| `_sim_interval_s` | 8.0 | Seconds between simulation job submissions |

### NodeAgent (`orchestrator/data_plane/agent.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `HEARTBEAT_INTERVAL_S` | 5.0 | Seconds between telemetry pushes |
| `CPU_USAGE_RATIO` | 0.85 | Fraction of requested CPU actually consumed |
| `MEM_USAGE_RATIO` | 0.90 | Fraction of requested memory actually consumed |
| `DURATION_SCALE_S` | 10.0 | Base job duration multiplier |

---

## V3 Upgrade Path

V2 is intentionally in-process and in-memory to validate the scheduling algorithms without infrastructure complexity. The upgrade path to production (V3):

| V2 (current) | V3 (production) |
|--------------|-----------------|
| In-memory `node_state` dict | Redis / etcd cluster state |
| In-process `NodeAgent` | Real agents on each node, HTTP via `httpx` |
| Synthetic / trace telemetry | Real `node_exporter` / Prometheus metrics |
| `asyncio.sleep()` job simulation | Actual job execution (K8s pods, containers) |
| Single-process FastAPI | Distributed FastAPI + load balancer |
| No authentication | OAuth2 / mTLS middleware |
| `uvicorn` (already async) | `uvicorn` + `uvloop` event loop (in requirements) |
| No persistence | Postgres for job history, S3 for pheromone snapshots |

**Key V3 changes by file:**
- `api/main.py` → add auth middleware, use `uvloop` explicitly
- `orchestrator/control_plane/orchestration_service.py` → swap dict for DB-backed store; add `asyncio.Lock` for concurrent requests
- `orchestrator/data_plane/agent.py` → replace direct `service.complete_job()` with `httpx.AsyncClient.post("/jobs/{id}/complete")`
- `orchestrator/telemetry/collector.py` → replace `_generate_telemetry()` with Prometheus scrape via `httpx`
