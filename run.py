"""
run.py — convenience entry point for the ACO Scheduler API.

Usage (from any directory):
    python /path/to/ACO_Adaptive_Compute_Scheduler/run.py

Or from the project root:
    python run.py

Starts uvicorn on http://localhost:8000 with hot-reload enabled.
"""

import os
import sys

# Ensure project root is on sys.path regardless of cwd
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Also change cwd so uvicorn's --reload watches the right directory
os.chdir(_PROJECT_ROOT)

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[_PROJECT_ROOT],
    )
