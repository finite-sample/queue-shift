"""Exact classifier assignment under a queue-movement budget."""

from importlib.metadata import PackageNotFoundError, version

from queue_shift.assignment import AssignmentResult, queue_shift, solve_assignment
from queue_shift.evaluation import evaluate_matched_budget

try:
    __version__ = version("queue-shift")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0"

__all__ = [
    "AssignmentResult",
    "__version__",
    "evaluate_matched_budget",
    "queue_shift",
    "solve_assignment",
]
