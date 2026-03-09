"""
api/ — FastAPI REST layer for the ACO Adaptive Compute Scheduler.

Phase 9: exposes the OrchestratorService over HTTP with a built-in dashboard.

Endpoints:
    POST /jobs              — submit a job
    GET  /jobs              — list active jobs
    GET  /jobs/{job_id}     — get job status
    GET  /nodes             — list nodes with live utilisation
    GET  /metrics           — scheduling performance metrics
    GET  /predict/{node_id} — latest LSTM prediction for a node
    POST /upload-trace      — upload a custom cluster trace CSV
    GET  /                  — HTML dashboard (auto-refreshing)
"""
