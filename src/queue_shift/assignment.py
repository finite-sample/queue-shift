"""Exact batch prediction under queue-movement and case-churn budgets.

The optimizer treats a classifier's per-case, per-queue costs as inputs. It assigns
every case to one queue while limiting the half-L1 distance between resulting and
incumbent queue loads. With only that limit, or with a linear penalty on cases that
leave their incumbent queue, the linear program is a minimum-cost flow and integer
inputs admit an integral optimum. A hard budget on weighted case churn adds one side
constraint that can break integrality, so that mode is solved as an exact MILP.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, linprog, milp
from scipy.sparse import coo_array, csr_array


@dataclass(frozen=True)
class AssignmentResult:
    """A globally optimal batch assignment and its operational summary."""

    labels: np.ndarray
    queue_loads: np.ndarray
    moved_load: int
    workload_shift: float
    churn: int
    flip_load: float
    objective: float
    queue_prices: np.ndarray | None


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


def workload_shift(
    queue_loads: np.ndarray, incumbent_loads: np.ndarray, queue_costs: np.ndarray
) -> float:
    """Return the cost of work added to queues beyond their incumbent loads.

    ``queue_costs[k]`` converts one case in queue ``k`` into the budget's currency:
    one for cases, average handle hours for hours, average cost per case for money.
    Work removed from one queue earns no credit against work added to another,
    because staff are not interchangeable across queues. With unit costs this equals
    :func:`queue_shift`.
    """
    queue_shift(queue_loads, incumbent_loads)
    unit_costs = np.asarray(queue_costs, dtype=float)
    if unit_costs.shape != np.shape(queue_loads):
        raise ValueError("queue_costs must contain one cost per queue")
    added = np.maximum(np.asarray(queue_loads) - np.asarray(incumbent_loads), 0)
    return float(unit_costs @ added)


def case_flip_weights(
    probability: np.ndarray,
    incumbent_labels: np.ndarray,
    kind: Literal["expected_negative", "churn"],
) -> np.ndarray:
    """Return the per-case cost of moving a case away from its incumbent queue.

    ``"churn"`` counts every changed label. ``"expected_negative"`` weights a change
    by ``probability[i, a_i]``: if case ``i`` leaves incumbent queue ``a_i``, it is a
    negative flip exactly when its true class is ``a_i``.
    """
    labels = np.asarray(incumbent_labels)
    if kind == "churn":
        return np.ones(len(labels))
    if kind == "expected_negative":
        prob = np.asarray(probability, dtype=float)
        if prob.ndim != 2 or prob.shape[0] != len(labels):
            raise ValueError("probability must have one row per incumbent label")
        return prob[np.arange(len(labels)), labels]
    raise ValueError("kind must be 'expected_negative' or 'churn'")


def flip_tolerance(weights: np.ndarray) -> float:
    """Return the numerical slack allowed on a weighted flip budget."""
    return 2e-6 * (1.0 + float(np.asarray(weights, dtype=float).sum()))


def flip_load(
    labels: np.ndarray, incumbent_labels: np.ndarray, weights: np.ndarray
) -> float:
    """Return total weight of cases whose label differs from the incumbent."""
    changed = np.asarray(labels) != np.asarray(incumbent_labels)
    return float(np.asarray(weights, dtype=float)[changed].sum())


def _flow_problem(
    n_cases: int, incumbent_loads: np.ndarray, move_budget: int
) -> tuple[csr_array, np.ndarray, np.ndarray, slice, slice]:
    """Build incidence, supplies, capacities, and assignment and overflow arcs."""
    n_queues = len(incumbent_loads)
    source = 0
    item_start = 1
    queue_start = item_start + n_cases
    overflow = queue_start + n_queues
    sink = overflow + 1
    n_nodes = sink + 1

    item = np.arange(n_cases)
    queue = np.arange(n_queues)
    tails = np.concatenate(
        [
            np.full(n_cases, source),
            np.repeat(item_start + item, n_queues),
            queue_start + queue,
            queue_start + queue,
            [overflow],
        ]
    )
    heads = np.concatenate(
        [
            item_start + item,
            np.tile(queue_start + queue, n_cases),
            np.full(n_queues, sink),
            np.full(n_queues, overflow),
            [sink],
        ]
    )
    capacities = np.concatenate(
        [
            np.ones(n_cases),
            np.ones(n_cases * n_queues),
            incumbent_loads,
            n_cases - incumbent_loads,
            [move_budget],
        ]
    ).astype(float)
    columns = np.arange(len(tails))
    incidence = coo_array(
        (
            np.concatenate([np.ones(len(tails)), -np.ones(len(tails))]),
            (np.concatenate([tails, heads]), np.concatenate([columns, columns])),
        ),
        shape=(n_nodes, len(tails)),
    ).tocsr()
    supply = np.zeros(n_nodes)
    supply[source] = n_cases
    supply[sink] = -n_cases
    # One conservation row is redundant; dropping the sink keeps the system full rank.
    assignment = slice(n_cases, n_cases + n_cases * n_queues)
    overflow_arcs = slice(
        n_cases + n_cases * n_queues + n_queues,
        n_cases + n_cases * n_queues + 2 * n_queues,
    )
    return csr_array(incidence[:-1]), supply[:-1], capacities, assignment, overflow_arcs


def solve_assignment(
    costs: np.ndarray,
    incumbent_labels: np.ndarray,
    move_budget: float,
    flip_weights: np.ndarray | None = None,
    flip_budget: float | None = None,
    flip_penalty: float = 0.0,
    queue_costs: np.ndarray | None = None,
) -> AssignmentResult:
    """Minimize total prediction cost subject to queue-movement and churn limits.

    ``costs[i, k]`` is the cost of assigning case ``i`` to queue ``k``. For calibrated
    class probabilities, ``1 - probability`` is expected 0-1 loss.
    ``incumbent_labels[i]`` is the queue the incumbent sends case ``i`` to, and
    ``move_budget`` bounds the work added to queues beyond their incumbent loads
    (:func:`workload_shift`). Without ``queue_costs`` work is counted in cases, the
    budget must be an integer, and the bound equals half the L1 distance between
    load vectors. ``queue_costs[k]`` prices one case in queue ``k``, for example in
    handle hours or money; unequal prices make the budget a side constraint solved
    as an exact MILP.

    ``flip_weights[i]`` is the cost of moving case ``i`` off its incumbent queue
    (see :func:`case_flip_weights`; default one per case). Either cap total weight at
    ``flip_budget`` (exact MILP) or subtract ``flip_penalty`` per unit of weight from
    the objective (pure minimum-cost flow). ``objective`` reports prediction cost
    only, never the penalty. In flow mode, ``queue_prices`` holds dual prices
    ``u`` such that every case's label minimizes ``cost[i, k] + penalty + u[k]``;
    the MILP mode has no such certificate and returns ``None``.
    """
    cost = np.asarray(costs, dtype=float)
    incumbent = np.asarray(incumbent_labels)
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] < 2:
        raise ValueError(
            "costs must have shape (n_cases, n_queues) with at least two queues"
        )
    if not np.all(np.isfinite(cost)):
        raise ValueError("costs must be finite")
    n_cases, n_queues = cost.shape
    if incumbent.shape != (n_cases,):
        raise ValueError("incumbent_labels must contain one label per case")
    if not np.issubdtype(incumbent.dtype, np.integer):
        raise ValueError("incumbent_labels must be integers")
    if np.any(incumbent < 0) or np.any(incumbent >= n_queues):
        raise ValueError("incumbent_labels must index the cost columns")
    if queue_costs is None:
        unit_costs = np.ones(n_queues)
        if not isinstance(move_budget, (int, np.integer)):
            raise ValueError("move_budget must be an integer when counting cases")
        if move_budget > n_cases:
            raise ValueError("move_budget must not exceed the number of cases")
    else:
        unit_costs = np.asarray(queue_costs, dtype=float)
        if unit_costs.shape != (n_queues,):
            raise ValueError("queue_costs must contain one cost per queue")
        if not np.all(np.isfinite(unit_costs)) or np.any(unit_costs < 0):
            raise ValueError("queue_costs must be finite and nonnegative")
    if not np.isfinite(move_budget) or move_budget < 0:
        raise ValueError("move_budget must be finite and nonnegative")
    weighted_budget = bool(np.ptp(unit_costs) > 0)
    weights = np.ones(n_cases) if flip_weights is None else np.asarray(flip_weights)
    weights = weights.astype(float)
    if weights.shape != (n_cases,):
        raise ValueError("flip_weights must contain one weight per case")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("flip_weights must be finite and nonnegative")
    if not np.isfinite(flip_penalty) or flip_penalty < 0:
        raise ValueError("flip_penalty must be finite and nonnegative")
    if flip_budget is not None:
        if flip_penalty > 0:
            raise ValueError("flip_budget and flip_penalty must not both be set")
        if not np.isfinite(flip_budget) or flip_budget < 0:
            raise ValueError("flip_budget must be finite and nonnegative")

    incumbent_loads = np.bincount(incumbent, minlength=n_queues)
    # A common cost per case turns the budget into a case count on the overflow
    # edge. Unequal costs cannot be one capacity, so the edge is left open and a
    # weighted row bounds the added work instead.
    if weighted_budget or unit_costs[0] == 0:
        overflow_capacity = n_cases
    else:
        overflow_capacity = min(int(move_budget / unit_costs[0] + 1e-9), n_cases)
    incidence, supply, capacities, assignment, overflow_arcs = _flow_problem(
        n_cases, incumbent_loads, overflow_capacity
    )
    moved_off = np.ones((n_cases, n_queues))
    moved_off[np.arange(n_cases), incumbent] = 0
    flip_coefficients = (weights[:, None] * moved_off).ravel()

    n_arcs = len(capacities)
    edge_costs = np.zeros(n_arcs)
    edge_costs[assignment] = cost.ravel() + flip_penalty * flip_coefficients

    side_rows = []
    if flip_budget is not None:
        flip_row = np.zeros((1, n_arcs))
        flip_row[0, assignment] = flip_coefficients
        side_rows.append((flip_row, -np.inf, float(flip_budget)))
    if weighted_budget:
        work_row = np.zeros((1, n_arcs))
        work_row[0, overflow_arcs] = unit_costs
        side_rows.append((work_row, -np.inf, float(move_budget)))

    if not side_rows:
        result = linprog(
            edge_costs,
            A_eq=incidence,
            b_eq=supply,
            bounds=np.column_stack([np.zeros_like(capacities), capacities]),
            method="highs",
        )
    else:
        integrality = np.zeros(n_arcs)
        integrality[assignment] = 1
        result = milp(
            edge_costs,
            integrality=integrality,
            # SciPy documents array bounds but annotates Bounds.ub as a float.
            bounds=Bounds(0, capacities),  # pyright: ignore[reportArgumentType]
            constraints=[(incidence, supply, supply), *side_rows],
            # The default relative gap of 1e-4 would forfeit the exactness the
            # matched-budget guarantee depends on.
            options={"mip_rel_gap": 0.0},
        )
    if not result.success:
        raise RuntimeError(f"queue-shift optimization failed: {result.message}")
    queue_prices = None
    if not side_rows:
        # Queue-node potentials: the solution assigns each case to a queue that
        # minimizes its arc cost plus that queue's potential.
        queue_start = 1 + n_cases
        queue_prices = np.asarray(result.eqlin.marginals)[
            queue_start : queue_start + n_queues
        ]

    chosen = np.asarray(result.x)[assignment].reshape(n_cases, n_queues)
    rounded = np.rint(chosen)
    # HiGHS accepts MILP integer values within 1e-6 of an integer; the flow LP is
    # integral at a vertex. Rounding then shifts the flip load by at most
    # 2e-6 per unit of weight, which bounds the flip-budget check below.
    if not np.allclose(chosen, rounded, atol=1e-6):
        raise RuntimeError("solver returned a non-integral assignment")
    if not np.all(rounded.sum(axis=1) == 1):
        raise RuntimeError("solver returned an invalid assignment")

    labels = rounded.argmax(axis=1)
    loads = np.bincount(labels, minlength=n_queues)
    moved = queue_shift(loads, incumbent_loads)
    added_work = workload_shift(loads, incumbent_loads, unit_costs)
    # Near-integral MILP values can misstate loads by up to 1e-6 per case.
    if added_work > move_budget + 2e-6 * (1.0 + n_cases * float(unit_costs.max())):
        raise RuntimeError("solver violated the movement budget")
    weighted_flips = flip_load(labels, incumbent, weights)
    if flip_budget is not None and weighted_flips > flip_budget + flip_tolerance(
        weights
    ):
        raise RuntimeError("solver violated the flip budget")

    return AssignmentResult(
        labels=labels,
        queue_loads=loads,
        moved_load=moved,
        workload_shift=added_work,
        churn=int((labels != incumbent).sum()),
        flip_load=weighted_flips,
        objective=float(cost[np.arange(n_cases), labels].sum()),
        queue_prices=queue_prices,
    )
