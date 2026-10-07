"""Adversarial tests for exact prediction under a queue-movement budget."""

import itertools

import numpy as np
import pytest

from queue_shift.assignment import (
    case_flip_weights,
    flip_load,
    queue_shift,
    solve_assignment,
    workload_shift,
)


def brute_force(
    costs: np.ndarray,
    incumbent_labels: np.ndarray,
    budget: int,
    weights: np.ndarray | None = None,
    flip_budget: float = np.inf,
) -> tuple[float, set]:
    """Enumerate every labeling for small instances, retaining all optimal labelings."""
    n_cases, n_queues = costs.shape
    incumbent = np.bincount(incumbent_labels, minlength=n_queues)
    case_weights = np.ones(n_cases) if weights is None else weights
    best = np.inf
    optimizers = set()
    for labels_tuple in itertools.product(range(n_queues), repeat=n_cases):
        labels = np.asarray(labels_tuple)
        loads = np.bincount(labels, minlength=n_queues)
        if queue_shift(loads, incumbent) > budget:
            continue
        if flip_load(labels, incumbent_labels, case_weights) > flip_budget + 1e-12:
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
        costs = rng.normal(size=(n_cases, n_queues))
        for budget in range(n_cases + 1):
            expected, optimizers = brute_force(costs, incumbent_labels, budget)
            result = solve_assignment(costs, incumbent_labels, budget)
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
    result = solve_assignment(1 - probabilities, incumbent_labels, move_budget=0)
    assert np.array_equal(result.queue_loads, incumbent)
    assert result.moved_load == 0
    assert not np.array_equal(result.labels, incumbent_labels)
    assert result.objective < float(
        (1 - probabilities)[np.arange(4), incumbent_labels].sum()
    )


def test_unconstrained_solution_is_casewise_best() -> None:
    rng = np.random.default_rng(88)
    costs = rng.normal(size=(20, 4))
    incumbent_labels = np.repeat(np.arange(4), [4, 5, 6, 5])
    result = solve_assignment(costs, incumbent_labels, move_budget=20)
    assert np.array_equal(result.labels, costs.argmin(axis=1))


def test_accuracy_cost_frontier_is_monotone() -> None:
    rng = np.random.default_rng(777)
    costs = rng.uniform(size=(16, 4))
    incumbent_labels = np.repeat(np.arange(4), 4)
    frontier = [
        solve_assignment(costs, incumbent_labels, budget) for budget in range(17)
    ]
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
        budget = int(rng.integers(0, n_cases + 1))

        oracle = solve_assignment(1 - true_probabilities, incumbent_labels, budget)
        plugin = solve_assignment(1 - estimated_probabilities, incumbent_labels, budget)
        oracle_value = true_probabilities[np.arange(n_cases), oracle.labels].sum()
        plugin_value = true_probabilities[np.arange(n_cases), plugin.labels].sum()
        estimation_error = (
            np.abs(true_probabilities - estimated_probabilities).max(axis=1).sum()
        )

        assert oracle_value - plugin_value <= 2 * estimation_error + 1e-8


def test_flip_budget_matches_exhaustive_search_for_both_weight_kinds() -> None:
    rng = np.random.default_rng(4242)
    for _ in range(25):
        n_cases = int(rng.integers(3, 7))
        n_queues = int(rng.integers(2, 4))
        incumbent_labels = rng.integers(0, n_queues, size=n_cases)
        probability = rng.dirichlet(np.ones(n_queues), size=n_cases)
        for kind in ("churn", "expected_negative"):
            weights = case_flip_weights(probability, incumbent_labels, kind)
            for budget in range(n_cases + 1):
                for flip_budget in np.linspace(0, weights.sum(), 5):
                    expected, optimizers = brute_force(
                        1 - probability,
                        incumbent_labels,
                        budget,
                        weights,
                        float(flip_budget),
                    )
                    result = solve_assignment(
                        1 - probability,
                        incumbent_labels,
                        budget,
                        flip_weights=weights,
                        flip_budget=float(flip_budget),
                    )
                    assert result.objective == pytest.approx(expected, abs=1e-8)
                    assert tuple(result.labels) in optimizers
                    assert result.flip_load <= flip_budget + 1e-6


def test_penalty_solutions_lie_on_the_hard_budget_frontier() -> None:
    rng = np.random.default_rng(515)
    for _ in range(20):
        n_cases, n_queues = 40, 4
        probability = rng.dirichlet(np.ones(n_queues), size=n_cases)
        incumbent_labels = rng.integers(0, n_queues, size=n_cases)
        weights = case_flip_weights(probability, incumbent_labels, "expected_negative")
        budget = int(rng.integers(0, n_cases // 2))
        for penalty in (0.1, 0.5, 2.0):
            penalized = solve_assignment(
                1 - probability,
                incumbent_labels,
                budget,
                flip_weights=weights,
                flip_penalty=penalty,
            )
            hard = solve_assignment(
                1 - probability,
                incumbent_labels,
                budget,
                flip_weights=weights,
                flip_budget=penalized.flip_load,
            )
            assert hard.objective == pytest.approx(penalized.objective, abs=1e-7)


def test_zero_penalty_and_slack_flip_budget_reduce_to_shift_only() -> None:
    rng = np.random.default_rng(616)
    probability = rng.dirichlet(np.ones(3), size=30)
    incumbent_labels = rng.integers(0, 3, size=30)
    shift_only = solve_assignment(1 - probability, incumbent_labels, 3)
    zero_penalty = solve_assignment(
        1 - probability, incumbent_labels, 3, flip_penalty=0.0
    )
    slack = solve_assignment(1 - probability, incumbent_labels, 3, flip_budget=30.0)
    assert zero_penalty.objective == pytest.approx(shift_only.objective, abs=1e-9)
    assert slack.objective == pytest.approx(shift_only.objective, abs=1e-9)


def test_zero_flip_budget_returns_incumbent() -> None:
    rng = np.random.default_rng(717)
    probability = rng.dirichlet(np.ones(3), size=12)
    incumbent_labels = rng.integers(0, 3, size=12)
    result = solve_assignment(1 - probability, incumbent_labels, 12, flip_budget=0.0)
    assert np.array_equal(result.labels, incumbent_labels)
    assert result.churn == 0


def test_expected_negative_weights_are_incumbent_class_probabilities() -> None:
    probability = np.array([[0.7, 0.3], [0.2, 0.8]])
    weights = case_flip_weights(probability, np.array([1, 1]), "expected_negative")
    assert np.allclose(weights, [0.3, 0.8])


def test_queue_cost_budget_matches_exhaustive_search() -> None:
    rng = np.random.default_rng(828)
    for _ in range(25):
        n_cases = int(rng.integers(3, 7))
        n_queues = int(rng.integers(2, 4))
        incumbent_labels = rng.integers(0, n_queues, size=n_cases)
        incumbent = np.bincount(incumbent_labels, minlength=n_queues)
        probability = rng.dirichlet(np.ones(n_queues), size=n_cases)
        queue_costs = rng.uniform(0.2, 3.0, size=n_queues)
        for budget in np.linspace(0, 3.0 * n_cases, 7):
            best = np.inf
            for labels in itertools.product(range(n_queues), repeat=n_cases):
                loads = np.bincount(np.asarray(labels), minlength=n_queues)
                if workload_shift(loads, incumbent, queue_costs) > budget + 1e-12:
                    continue
                best = min(
                    best, float((1 - probability)[np.arange(n_cases), labels].sum())
                )
            result = solve_assignment(
                1 - probability,
                incumbent_labels,
                float(budget),
                queue_costs=queue_costs,
            )
            assert result.objective == pytest.approx(best, abs=1e-8)
            assert result.workload_shift <= budget + 1e-6


def test_unit_queue_costs_reproduce_case_counting() -> None:
    rng = np.random.default_rng(929)
    probability = rng.dirichlet(np.ones(4), size=40)
    incumbent_labels = rng.integers(0, 4, size=40)
    for budget in (0, 3, 10):
        counted = solve_assignment(1 - probability, incumbent_labels, budget)
        priced = solve_assignment(
            1 - probability,
            incumbent_labels,
            float(budget),
            queue_costs=np.ones(4),
        )
        assert priced.objective == pytest.approx(counted.objective, abs=1e-9)
        assert counted.workload_shift == counted.moved_load


def test_case_budget_can_overload_an_expensive_queue() -> None:
    # One case belongs in the expensive queue; counting cases lets it move there,
    # pricing work in hours does not.
    probability = np.array([[0.1, 0.9], [0.9, 0.1]])
    incumbent_labels = np.array([0, 0])
    hours = np.array([1.0, 5.0])
    by_cases = solve_assignment(1 - probability, incumbent_labels, 1)
    by_hours = solve_assignment(
        1 - probability, incumbent_labels, 1.0, queue_costs=hours
    )
    assert workload_shift(by_cases.queue_loads, np.array([2, 0]), hours) == 5.0
    assert by_hours.workload_shift <= 1.0
    assert by_hours.objective > by_cases.objective


@pytest.mark.parametrize(
    ("costs", "incumbent", "budget", "options"),
    [
        (np.ones((3, 2)), np.array([0, 1]), 0, {}),
        (np.ones((3, 2)), np.array([0.0, 1.0, 1.0]), 0, {}),
        (np.ones((3, 2)), np.array([0, 1, 2]), 0, {}),
        (np.ones((3, 2)), np.array([0, 1, 1]), -1, {}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 4, {}),
        (np.array([[0.0, np.inf], [1.0, 0.0]]), np.array([0, 1]), 0, {}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 0, {"flip_weights": np.ones(2)}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 0, {"flip_weights": -np.ones(3)}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 0, {"flip_penalty": -1.0}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 0, {"flip_budget": -1.0}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 0.5, {}),
        (np.ones((3, 2)), np.array([0, 1, 1]), 1.0, {"queue_costs": np.ones(3)}),
        (
            np.ones((3, 2)),
            np.array([0, 1, 1]),
            1.0,
            {"queue_costs": np.array([1.0, -1.0])},
        ),
        (
            np.ones((3, 2)),
            np.array([0, 1, 1]),
            0,
            {"flip_budget": 1.0, "flip_penalty": 1.0},
        ),
    ],
)
def test_invalid_inputs_fail_loudly(
    costs: np.ndarray, incumbent: np.ndarray, budget: int, options: dict
) -> None:
    with pytest.raises(ValueError, match=r"must|finite"):
        solve_assignment(costs, incumbent, budget, **options)
