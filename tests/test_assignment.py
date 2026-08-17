"""Adversarial tests for exact prediction under a queue-movement budget."""

import itertools

import numpy as np
import pytest

from queue_shift.assignment import queue_shift, solve_assignment


def brute_force(
    costs: np.ndarray, incumbent: np.ndarray, budget: int
) -> tuple[float, set]:
    """Enumerate every labeling for small instances, retaining all optimal labelings."""
    n_cases, n_queues = costs.shape
    best = np.inf
    optimizers = set()
    for labels_tuple in itertools.product(range(n_queues), repeat=n_cases):
        labels = np.asarray(labels_tuple)
        loads = np.bincount(labels, minlength=n_queues)
        if queue_shift(loads, incumbent) > budget:
            continue
        value = float(costs[np.arange(n_cases), labels].sum())
        if value < best - 1e-10:
            best = value
            optimizers = {labels_tuple}
        elif abs(value - best) <= 1e-10:
            optimizers.add(labels_tuple)
    return best, optimizers


def test_matches_exhaustive_search_across_random_instances_and_budgets() -> None:
    rng = np.random.default_rng(90210)
    for _ in range(30):
        n_cases = int(rng.integers(3, 7))
        n_queues = int(rng.integers(2, 4))
        incumbent_labels = rng.integers(0, n_queues, size=n_cases)
        incumbent = np.bincount(incumbent_labels, minlength=n_queues)
        costs = rng.normal(size=(n_cases, n_queues))
        for budget in range(n_cases + 1):
            expected, optimizers = brute_force(costs, incumbent, budget)
            result = solve_assignment(costs, incumbent, budget)
            assert result.objective == pytest.approx(expected, abs=1e-8)
            assert tuple(result.labels) in optimizers
            assert result.moved_load <= budget


def test_zero_budget_preserves_loads_but_not_individual_predictions() -> None:
    incumbent_labels = np.array([0, 0, 1, 1])
    incumbent = np.bincount(incumbent_labels, minlength=2)
    probabilities = np.array(
        [
            [0.10, 0.90],
            [0.90, 0.10],
            [0.80, 0.20],
            [0.20, 0.80],
        ]
    )
    result = solve_assignment(1 - probabilities, incumbent, move_budget=0)
    assert np.array_equal(result.queue_loads, incumbent)
    assert result.moved_load == 0
    assert not np.array_equal(result.labels, incumbent_labels)
    assert result.objective < float(
        (1 - probabilities)[np.arange(4), incumbent_labels].sum()
    )


def test_unconstrained_solution_is_casewise_best() -> None:
    rng = np.random.default_rng(88)
    costs = rng.normal(size=(20, 4))
    incumbent = np.array([4, 5, 6, 5])
    result = solve_assignment(costs, incumbent, move_budget=20)
    assert np.array_equal(result.labels, costs.argmin(axis=1))


def test_accuracy_cost_frontier_is_monotone() -> None:
    rng = np.random.default_rng(777)
    costs = rng.uniform(size=(16, 4))
    incumbent = np.array([4, 4, 4, 4])
    frontier = [solve_assignment(costs, incumbent, budget) for budget in range(17)]
    objectives = np.array([point.objective for point in frontier])
    assert np.all(np.diff(objectives) <= 1e-9)
    assert all(point.moved_load <= budget for budget, point in enumerate(frontier))


def test_plugin_regret_bound_holds_against_true_probabilities() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(100):
        n_cases, n_queues = 12, 3
        true_probabilities = rng.dirichlet(np.ones(n_queues), size=n_cases)
        estimated_probabilities = np.clip(
            true_probabilities + rng.uniform(-0.08, 0.08, size=(n_cases, n_queues)),
            0,
            1,
        )
        incumbent_labels = rng.integers(0, n_queues, size=n_cases)
        incumbent = np.bincount(incumbent_labels, minlength=n_queues)
        budget = int(rng.integers(0, n_cases + 1))

        oracle = solve_assignment(1 - true_probabilities, incumbent, budget)
        plugin = solve_assignment(1 - estimated_probabilities, incumbent, budget)
        oracle_value = true_probabilities[np.arange(n_cases), oracle.labels].sum()
        plugin_value = true_probabilities[np.arange(n_cases), plugin.labels].sum()
        estimation_error = (
            np.abs(true_probabilities - estimated_probabilities).max(axis=1).sum()
        )

        assert oracle_value - plugin_value <= 2 * estimation_error + 1e-8


@pytest.mark.parametrize(
    ("costs", "incumbent", "budget"),
    [
        (np.ones((3, 2)), np.array([1, 1]), 0),
        (np.ones((3, 2)), np.array([2.0, 1.0]), 0),
        (np.ones((3, 2)), np.array([2, 1]), -1),
        (np.ones((3, 2)), np.array([2, 1]), 4),
        (np.array([[0.0, np.inf], [1.0, 0.0]]), np.array([1, 1]), 0),
    ],
)
def test_invalid_inputs_fail_loudly(
    costs: np.ndarray, incumbent: np.ndarray, budget: int
) -> None:
    with pytest.raises(ValueError, match=r"must|finite"):
        solve_assignment(costs, incumbent, budget)
