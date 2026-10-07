"""Exact classifier assignment under a queue-movement budget."""

from importlib.metadata import PackageNotFoundError, version

from queue_shift.assignment import (
    AssignmentResult,
    case_flip_weights,
    flip_load,
    queue_shift,
    solve_assignment,
    workload_shift,
)
from queue_shift.evaluation import evaluate_matched_budget

try:
    __version__ = version("queue-shift")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0"

__all__ = [
    "AssignmentResult",
    "__version__",
    "case_flip_weights",
    "evaluate_matched_budget",
    "flip_load",
    "queue_shift",
    "solve_assignment",
    "workload_shift",
]
