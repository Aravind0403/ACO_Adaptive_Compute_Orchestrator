# ACO — Adaptive Compute Orchestrator

> **Traditional schedulers react to load. ACO predicts it.**

Most job schedulers are reactive: they see a spike after it starts. By then, SLAs are already at risk.  
ACO combines **ant colony optimization**, **LSTM-based spike prediction**, and **workload-intent routing** to place jobs on the right node *before* contention hits.

---

## Performance

| Metric | Result | Source |
|--------|--------|--------|
| **Scheduling latency (P99)** | **< 0.75 ms** at 200 concurrent jobs | T1.3 |
| **Cost reduction vs First-Fit** | **28.6%** (CPU workloads, Azure distribution) | T1.1 |
| **Cost reduction vs Random** | **79.6%** (GPU workloads, 32-node cluster) | T4.1 |
| **GPU QoS compliance** | **97.4%** LS → ON_DEMAND (ACO+QoS mode) | T4.1 |
| **LSTM calibration signal** | **Pearson r = −0.64** on 8-day Alibaba trace | T2.2 |
| **LSTM routing improvement** | **+31.5 pp** safe-node selection (32-node cluster, t=22.56, p<0.001) | T5.2 |
| **Test coverage** | **210 tests passing** | pytest |
| **External dependencies at runtime** | **Zero** — fully in-memory | — |

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-210%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Standard schedulers treat workloads as opaque units and nodes as interchangeable bins.  
They don't know:

- A GPU inference job cannot tolerate spot interruption
- A batch actor on a node about to spike will miss its deadline
- Placing a replay buffer and a latency-critical job on the same node will destroy P99

ACO treats workloads as **profiles with intent** and nodes as **typed resources with predictable futures**.

---

## How It Works

Three overlapping signals, combined in **< 10ms**:

| Signal | Source | Role |
|--------|--------|------|
| **Pheromone** (learned history) | ACO colony convergence | Which placements have worked before |
| **Heuristic** (cost + headroom + spike risk) | CostEngine | How good is this node right now |
| **Intent** (workload type + constraints) | WorkloadIntentRouter | Which nodes are even eligible |

### Scheduling Decision Flow

```
POST /jobs
  ↓
OrchestratorService
  ├── Admit job (resource sanity checks)
  ├── WorkloadIntentRouter → SchedulingStrategy
  │     GPU Inference?   → ON_DEMAND only, fast path
  │     Batch + GPU?     → SPOT allowed, full colony
  │     LC + deadline?   → tighten SLA threshold, fast path
  │     Preemptible?     → SPOT preferred, forgiving SLA
  ↓
aco_schedule()
  ├── Filter nodes: capacity + arch + colocation policy
  ├── CostEngine.score_node() × n_nodes → η array
  │     reliability_factor × cost_efficiency × sla_headroom × prediction_factor
  ├── Fast path (latency-critical): deterministic argmax(η) → < 1ms
  └── Full colony (20 ants × 5 iterations): probabilistic → ≤ 8ms
  ↓
Allocate → NodeAgent.execute_job() (async, non-blocking)
Return 202: { node_id, scheduling_latency_ms }
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI :8000                            │
│  POST /jobs   GET /metrics   GET /predict/:id               │
│  POST /simulation/start      POST /upload-trace             │
└──────────────────────┬──────────────────────────────────────┘
                       │ submit_job()
                       ▼
┌────────────────────────────────────────────────────────────┐
│                  OrchestratorService                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Admission   │→ │ IntentRouter │→ │ aco_schedule()   │  │
│  │ Controller  │  │ classify job │  │ ACO or fast path │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────┬─────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐  ┌───────────┐  ┌─────────────┐
│  ACO Colony │  │ CostEngine│  │ LSTM        │
│  20 Ants    │  │ η per pair│  │ Predictor   │
│  5 Iters    │  │           │  │ per node    │
└─────────────┘  └───────────┘  └─────────────┘
        │
        ▼ (background, every 5s)
┌────────────────────────────────────────────────────────────┐
│                  TelemetryCollector                         │
│  tick() → NodeTelemetry → update state                     │
│  every 10 ticks: refit LSTM per node                       │
│  Source: synthetic Gaussian OR Alibaba 2018 OR Borg 2019   │
└────────────────────────────────────────────────────────────┘
        │
        ▼ (per node, background heartbeat)
┌────────────────────────────────────────────────────────────┐
│                   NodeAgent (×5)                            │
│  execute_job() → asyncio.sleep → complete_job()            │
│  _heartbeat_loop() → telemetry push every 5s              │
└────────────────────────────────────────────────────────────┘
```

---

## Mock Cluster — 5-Node Heterogeneous Fleet

| Node | Arch | Instance | CPU | Mem | GPU | Cost/hr |
|------|------|----------|-----|-----|-----|---------|
| `node-cpu-01` | x86_64 | ON_DEMAND | 32c | 128GB | — | $0.48 |
| `node-arm-02` | ARM64 | SPOT | 16c | 64GB | — | $0.12 |
| `node-api-03` | x86_64 | ON_DEMAND | 8c | 32GB | — | $0.18 |
| `node-gpu-04` | GPU_NODE | ON_DEMAND | 16c | 128GB | A100 | $3.20 |
| `node-gpu-05` | GPU_NODE | SPOT | 16c | 128GB | A100 | $1.10 |

The intent router naturally directs GPU inference to `node-gpu-04` (ON_DEMAND, never spot), batch actors to `node-arm-02` (cheap ARM64), and latency-critical API traffic to `node-api-03` — without explicit configuration.

---

## Key Components

### ACO Colony (`aco_core/`)

- **PheromoneMatrix**: 2D NumPy `τ[n_jobs][n_nodes]`, evaporation rate 10%/iter, floor 0.01 (prevents stagnation)
- **Ant**: Roulette-wheel selection on `(τ^α × η^β)` with `α=1.0, β=2.0`
- **Colony**: 20 ants × 5 iterations, early stop after 3 stagnant iterations
- **Fast path**: Single latency-critical job → deterministic `argmax(η)` in < 1ms (no variance acceptable for P99)

### LSTM Predictor (`orchestrator/control_plane/predictor.py`)

- Architecture: `(1, 10, 1)` input → LSTM(hidden=32) → Linear(32→1)
- Per-node model, refits every 10 telemetry ticks (~50ms refit time)
- Cold-start handled: uses CPU average with `confidence=0.1` until ≥10 samples
- Confidence = 0.5×sample_score + 0.5×mae_score; cold-start floor 0.1, peaks at ~0.95 (500 samples, low MAE)
- **Routing benefit (T5.2):** at 32-node cluster scale with independent per-node trace histories, LSTM predictions shift safe-node selection from 48.8% → 80.3% for latency-critical jobs (+31.5 pp, t=22.56, p<0.001, N=5 seeds). Calibration on 8-day Alibaba trace: Pearson r = −0.64 (correct directional signal at burst onsets).

### CostEngine — Composite Score

```
score(job, node) = reliability_factor
                × cost_efficiency_factor
                × sla_headroom_factor
                × prediction_factor    ← penalises nodes about to spike
```

All factors in `(0.0, 1.0]`. A score of `0.0` means never place here. Hard gate: latency-critical jobs never land on high-risk SPOT nodes.

### Trace Replay Adapters (`orchestrator/telemetry/trace_adapter.py`)

Real cluster data instead of synthetic noise:

- **Alibaba 2018**: 8-day trace, 300s intervals, CPU range 16–79%, diurnal cycles, burst events
- **Borg 2019**: Google cluster trace, auto-detected by column names, hot-swap at runtime via `POST /upload-trace`

---

## Running It

### Start the server

```bash
pip install -r requirements.txt
python run.py
# Open: http://localhost:8000
```

### Load real trace data

```bash
curl -L "https://zenodo.org/records/14564935/files/machine_usage_days_1_to_8_grouped_300_seconds.csv" \
     -o alibaba_trace.csv

curl -X POST http://localhost:8000/upload-trace \
     -F "file=@alibaba_trace.csv"
```

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"workload_type": "latency-critical", "cpu_cores_min": 2.0, "memory_gb_min": 4.0, "priority": 90}'

# Response: {"status": "SCHEDULED", "node_id": "node-api-03", "scheduling_latency_ms": 4.2}
```

### Run all tests

```bash
python -m pytest tests/ -v
# 210 passed
```

### Run the benchmark suite

```bash
python -m benchmarks.run_all
# ~3–8 minutes; results saved to benchmarks/results/*.json
```

Individual tiers can be run independently:

```bash
python -m benchmarks.tier1_aco_vs_naive       # T1.1 — cost vs First-Fit
python -m benchmarks.tier1_p99_latency        # T1.3 — latency CDF
python -m benchmarks.tier2_confidence_calibration  # T2.2 — LSTM calibration
python -m benchmarks.tier4_gpu_scheduling     # T4.1 — GPU QoS
python -m benchmarks.tier5_lstm_routing_impact    # T5.1 — routing benefit (2-node baseline)
python -m benchmarks.tier5_lstm_routing_32nodes  # T5.2 — routing at 32-node cluster scale
```

---

## Benchmarks

Nine scripts across five tiers. All results reproducible from the Alibaba 2018 trace included in `tests/fixtures/`.

| ID | Name | Key result |
|----|------|------------|
| T1.1 | ACO vs Naive (Azure VM distribution) | **28.6%** cost reduction vs First-Fit (balanced topology) |
| T1.2 | Pheromone learning curve | **316×** pheromone spread after 200 sequential jobs — convergence detected |
| T1.3 | P99 latency under burst | **0.75 ms P99** at 200 concurrent jobs |
| T2.1 | Spike prediction recall | Calibration signal present; recall low at threshold 0.4 (threshold-dependent) |
| T2.2 | Confidence calibration | **Pearson r = −0.64** — correct directional signal at burst onsets |
| T3.1 | Queue drain under saturation | Steady drain rate across burst load |
| T3.2 | Cold vs warm pheromone start | Warm pheromone starts lower — advantage detectable within 50 jobs |
| T4.1 | GPU scheduling (Alibaba ATC'23, 32 nodes) | **79.6%** cost reduction vs Random; **97.4%** QoS compliance |
| T5.1 | LSTM routing benefit (2-node baseline) | **+70 pp** safe-node selection for LC jobs (2-node control experiment) |
| T5.2 | LSTM routing at 32-node cluster scale | **+31.5 pp** safe-node selection (t=22.56, p<0.001, N=5 seeds) |

---

## Test Coverage

| Test file | Phase | Tests | Focus |
|-----------|-------|-------|-------|
| `test_aco_phase2.py` | ACO Core | 43 | PheromoneMatrix, Ant η, Colony convergence, performance |
| `test_predictor.py` | LSTM | 20 | Training, cold-start, spike detection, confidence |
| `test_cost_engine.py` | CostEngine | 30 | All 4 sub-scores, edge cases, threshold overrides |
| `test_aco_phase5.py` | Orchestration | 32 | aco_schedule integration, OrchestratorService lifecycle |
| `test_intent_router.py` | Intent Router | 24 | Strategy classification, deadline override, colocation |
| `test_aco_phase7.py` | Telemetry | 20 | TelemetryCollector, per-node profiles, prediction cache |
| `test_trace_adapter.py` | Trace Replay | 13 | Adapter loading, trace replay, collector integration |
| `test_data_plane.py` | NodeAgent | 20 | execute_job, heartbeat, end-to-end async |

---

## V3 Upgrade Path

V2 is intentionally in-memory to validate scheduling algorithms without infrastructure complexity.

| V2 (current) | V3 (production target) |
|---|---|
| In-memory `node_state` dict | Redis / etcd cluster state |
| In-process `NodeAgent` | Real agents over HTTP via `httpx` |
| Synthetic / trace telemetry | Real `node_exporter` / Prometheus |
| `asyncio.sleep()` simulation | Actual K8s pod execution |
| Single-process FastAPI | FastAPI + load balancer |
| No persistence | Postgres job history + S3 pheromone snapshots |

---

## Project Structure

```
aco_core/
├── pheromone.py              PheromoneMatrix (NumPy, evaporation, deposit)
├── ant.py                    Single ant: η computation + roulette-wheel selection
└── colony.py                 Colony: 20 ants × 5 iters, fast path, fallback

orchestrator/
├── control_plane/
│   ├── scheduler.py          aco_schedule() — main entry point
│   ├── orchestration_service.py  OrchestratorService — state machine + RLock
│   ├── intent_router.py      WorkloadIntentRouter — 6 strategies
│   ├── cost_engine.py        CostEngine — 4-factor composite score
│   └── predictor.py          LSTM predictor — per-node, refits every 10 ticks
├── data_plane/
│   └── agent.py              NodeAgent — async job execution + heartbeat
└── telemetry/
    ├── collector.py           TelemetryCollector — drives prediction loop
    └── trace_adapter.py       Alibaba 2018 trace replay (hot-swappable)

api/
└── main.py                   FastAPI app, dashboard, simulation loop, lifespan

benchmarks/
├── tier1_*.py                Latency + cost vs baselines (T1.1–T1.3)
├── tier2_*.py                LSTM calibration + spike recall (T2.1–T2.2)
├── tier3_*.py                Queue drain + warm-start (T3.1–T3.2)
├── tier4_gpu_scheduling.py   GPU QoS + cost on Alibaba ATC'23 data (T4.1)
├── tier5_lstm_routing_impact.py    LSTM routing benefit — 2-node baseline (T5.1)
├── tier5_lstm_routing_32nodes.py  LSTM routing at 32-node cluster scale (T5.2)
├── run_all.py                Run all 9 benchmarks, print summary table
└── results/                  JSON result files per benchmark

paper/
└── aco_scheduler_ftc2026.tex HiPC 2026 paper (IEEE, Track 2: AI Systems and Applications)

tests/                        210 tests (pytest-asyncio)
tests/fixtures/               Alibaba 2018 trace CSV (alibaba_machine_usage_300s.csv)
```

---

## Paper

This system is the subject of a paper submitted to HiPC 2026 (IEEE International Conference on High Performance Computing), Track 2: AI Systems and Applications:

> **"ACO Adaptive Compute Scheduler: Predictive, Intent-Aware Job Placement for Heterogeneous Compute Clusters"**  
> Aravind Sundaresan. *HiPC 2026, Track 2: AI Systems and Applications.*  
> Source: `paper/aco_scheduler_ftc2026.tex`

For deep technical detail on system design, algorithms, and evaluation methodology, see the paper. The `TECHNICAL.md` in this repo is an earlier supplementary reference.

---

## Author

**Aravind Sundaresan** — Infrastructure & Distributed Systems Engineer  
Microsoft (distributed validation platform, 17K+ microservices) · Ex-Amazon (Alexa device infrastructure)

- 🌐 [aravindsundaresan.netlify.app](https://aravindsundaresan.netlify.app)
- 💼 [LinkedIn](https://linkedin.com/in/aravind-sundaresan)
- ✍️ [Substack](https://aravindsundaresan.substack.com)

---
