"""Exact batch prediction under a queue-movement budget.

The optimizer treats a classifier's per-case, per-queue costs as inputs. It assigns
every case to one queue while limiting the half-L1 distance between resulting and
incumbent queue loads. The linear program is a minimum-cost flow, so integer
inputs admit an integral optimum.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class AssignmentResult:
    """A globally optimal batch assignment and its operational summary."""

    labels: np.ndarray
    queue_loads: np.ndarray
    moved_load: int
    objective: float


def queue_shift(queue_loads: np.ndarray, incumbent_loads: np.ndarray) -> int:
    """Return the workload moved between two integer queue-load vectors."""
    current = np.asarray(queue_loads)
    incumbent = np.asarray(incumbent_loads)
    if current.ndim != 1 or incumbent.ndim != 1 or current.shape != incumbent.shape:
        raise ValueError(
            "queue-load vectors must be one-dimensional and have equal length"
        )
    if not np.issubdtype(current.dtype, np.integer):
        raise ValueError("queue loads must be integers")
    if not np.issubdtype(incumbent.dtype, np.integer):
        raise ValueError("incumbent loads must be integers")
    if np.any(current < 0) or np.any(incumbent < 0):
        raise ValueError("queue loads must be nonnegative")
    if int(current.sum()) != int(incumbent.sum()):
        raise ValueError("queue-load vectors must have the same total")
    return int(np.abs(current - incumbent).sum() // 2)


def solve_assignment(
    costs: np.ndarray,
    incumbent_loads: np.ndarray,
    move_budget: int,
) -> AssignmentResult:
    """Minimize total prediction cost subject to an exact queue-movement budget.

    ``costs[i, k]`` is the cost of assigning case ``i`` to queue ``k``.  For calibrated
    class probabilities, ``1 - probability`` is expected 0-1 loss.  ``incumbent_loads``
    gives the number of cases assigned to each queue by the incumbent.  ``move_budget``
    bounds half the L1 distance from that vector.
    """
    cost = np.asarray(costs, dtype=float)
    incumbent = np.asarray(incumbent_loads)
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] < 2:
        raise ValueError(
            "costs must have shape (n_cases, n_queues) with at least two queues"
        )
    if not np.all(np.isfinite(cost)):
        raise ValueError("costs must be finite")
    if incumbent.ndim != 1 or incumbent.shape[0] != cost.shape[1]:
        raise ValueError("incumbent_loads must contain one value per queue")
    if not np.issubdtype(incumbent.dtype, np.integer):
        raise ValueError("incumbent_loads must be integers")
    if np.any(incumbent < 0):
        raise ValueError("incumbent_loads must be nonnegative")

    n_cases, n_queues = cost.shape
    if int(incumbent.sum()) != n_cases:
        raise ValueError("incumbent_loads must sum to the number of cases")
    if not isinstance(move_budget, (int, np.integer)):
        raise ValueError("move_budget must be an integer")
    if move_budget < 0 or move_budget > n_cases:
        raise ValueError("move_budget must lie between zero and the number of cases")

    source = 0
    item_start = 1
    queue_start = item_start + n_cases
    overflow = queue_start + n_queues
    sink = overflow + 1
    n_nodes = sink + 1

    tails: list[int] = []
    heads: list[int] = []
    capacities: list[float] = []
    edge_costs: list[float] = []

    def add_edge(tail: int, head: int, capacity: int, edge_cost: float) -> None:
        tails.append(tail)
        heads.append(head)
        capacities.append(float(capacity))
        edge_costs.append(float(edge_cost))

    for i in range(n_cases):
        add_edge(source, item_start + i, 1, 0.0)

    assignment_start = len(tails)
    for i in range(n_cases):
        for k in range(n_queues):
            add_edge(item_start + i, queue_start + k, 1, cost[i, k])
    assignment_stop = len(tails)

    for k, baseline_capacity in enumerate(incumbent):
        add_edge(queue_start + k, sink, int(baseline_capacity), 0.0)
        add_edge(queue_start + k, overflow, n_cases - int(baseline_capacity), 0.0)
    add_edge(overflow, sink, move_budget, 0.0)

    columns = np.arange(len(tails))
    incidence = coo_matrix(
        (
            np.concatenate([np.ones(len(tails)), -np.ones(len(tails))]),
            (
                np.concatenate([np.asarray(tails), np.asarray(heads)]),
                np.concatenate([columns, columns]),
            ),
        ),
        shape=(n_nodes, len(tails)),
    ).tocsr()
    supply = np.zeros(n_nodes)
    supply[source] = n_cases
    supply[sink] = -n_cases

    result = linprog(
        np.asarray(edge_costs),
        A_eq=incidence[:-1],
        b_eq=supply[:-1],
        bounds=np.column_stack([np.zeros(len(tails)), np.asarray(capacities)]),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"queue-shift optimization failed: {result.message}")

    assignment = result.x[assignment_start:assignment_stop].reshape(n_cases, n_queues)
    rounded = np.rint(assignment)
    if not np.allclose(assignment, rounded, atol=1e-7):
        raise RuntimeError(
            "minimum-cost flow solver returned a non-integral assignment"
        )
    if not np.all(rounded.sum(axis=1) == 1):
        raise RuntimeError("minimum-cost flow solver returned an invalid assignment")

    labels = rounded.argmax(axis=1)
    loads = np.bincount(labels, minlength=n_queues)
    moved = queue_shift(loads, incumbent.astype(int))
    if moved > move_budget:
        raise RuntimeError("minimum-cost flow solver violated the queue-shift budget")

    return AssignmentResult(
        labels=labels,
        queue_loads=loads,
        moved_load=moved,
        objective=float(cost[np.arange(n_cases), labels].sum()),
    )
