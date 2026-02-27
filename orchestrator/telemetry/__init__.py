"""
orchestrator/telemetry — background telemetry ingestion and prediction pipeline.

Phase 7: TelemetryCollector drives the prediction loop end-to-end.

Public API:
    TelemetryCollector  — simulated node telemetry loop + LSTM refit trigger
"""

from orchestrator.telemetry.collector import TelemetryCollector

__all__ = ["TelemetryCollector"]
