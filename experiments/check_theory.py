"""Adversarial checks for the Queue Shift formal results."""

from __future__ import annotations

import itertools

import numpy as np

from queue_shift.assignment import queue_shift, solve_assignment


def check_net_queue_shift() -> None:
    """Verify the half-L1 identity and its churn endpoints."""
    rng = np.random.default_rng(301)
    for _ in range(500):
        n_queues = int(rng.integers(2, 15))
        flow = rng.integers(0, 50, size=(n_queues, n_queues))
        old_load = flow.sum(axis=1)
        new_load = flow.sum(axis=0)
        positive_change = np.maximum(new_load - old_load, 0).sum()
        half_l1 = np.abs(new_load - old_load).sum() / 2
        churn = flow.sum() - np.trace(flow)
        if positive_change != half_l1 or half_l1 > churn:
            raise AssertionError("net queue-shift identity failed")

    balanced = np.array([[0, 20], [20, 0]])
    directed = np.array([[0, 40], [0, 0]])
    balanced_shift = np.abs(balanced.sum(axis=0) - balanced.sum(axis=1)).sum() / 2
    directed_shift = np.abs(directed.sum(axis=0) - directed.sum(axis=1)).sum() / 2
    if balanced_shift != 0 or directed_shift != 40:
        raise AssertionError("queue-shift endpoints failed")
    if balanced.sum() - np.trace(balanced) <= balanced_shift:
        raise AssertionError(
            "positive control failed to separate churn from queue shift"
        )


def check_same_metrics_different_staffing() -> None:
    """Verify the fixed three-case counterexample to NFR identification."""
    truth = np.array([1, 2, 1])
    incumbent = np.array([1, 1, 2])
    directed = np.array([2, 2, 2])
    balanced = np.array([2, 1, 1])

    def metrics(candidate: np.ndarray) -> tuple[int, int, int, int, int]:
        incumbent_correct = incumbent == truth
        candidate_correct = candidate == truth
        negative = int((incumbent_correct & ~candidate_correct).sum())
        positive = int((~incumbent_correct & candidate_correct).sum())
        old_load = np.bincount(incumbent, minlength=3)[1:]
        new_load = np.bincount(candidate, minlength=3)[1:]
        movement = int(np.abs(new_load - old_load).sum() / 2)
        churn = int((incumbent != candidate).sum())
        return int(candidate_correct.sum()), negative, positive, churn, movement

    directed_metrics = metrics(directed)
    balanced_metrics = metrics(balanced)
    if directed_metrics[:4] != balanced_metrics[:4]:
        raise AssertionError("counterexample does not match accuracy, NFR, and churn")
    if directed_metrics[4] != 2 or balanced_metrics[4] != 0:
        raise AssertionError("counterexample does not separate queue movement")


def check_matched_movement_dominance() -> None:
    """Compare flow with exhaustive search and verify the error bound."""
    rng = np.random.default_rng(351)
    for _ in range(200):
        n_cases = int(rng.integers(2, 9))
        n_queues = int(rng.integers(2, 5))
        incumbent = rng.integers(0, n_queues, size=n_cases)
        baseline = rng.integers(0, n_queues, size=n_cases)
        incumbent_loads = np.bincount(incumbent, minlength=n_queues)
        baseline_loads = np.bincount(baseline, minlength=n_queues)
        budget = queue_shift(baseline_loads, incumbent_loads)
        score = rng.dirichlet(np.ones(n_queues), size=n_cases)
        truth = rng.dirichlet(np.ones(n_queues), size=n_cases)
        exact = solve_assignment(1 - score, incumbent_loads, budget)

        baseline_value = score[np.arange(n_cases), baseline].mean()
        exact_value = score[np.arange(n_cases), exact.labels].mean()
        true_gain = (
            truth[np.arange(n_cases), exact.labels]
            - truth[np.arange(n_cases), baseline]
        ).mean()
        error = truth - score
        penalty = (
            np.abs(error[np.arange(n_cases), exact.labels])
            + np.abs(error[np.arange(n_cases), baseline])
        ).mean()

        feasible_values = []
        for labels in itertools.product(range(n_queues), repeat=n_cases):
            loads = np.bincount(labels, minlength=n_queues)
            if queue_shift(loads, incumbent_loads) <= budget:
                feasible_values.append(score[np.arange(n_cases), labels].mean())
        if abs(exact_value - max(feasible_values)) > 1e-8:
            raise AssertionError("flow solver missed the exhaustive optimum")
        if exact_value < baseline_value - 1e-9:
            raise AssertionError("exact assignment lost to a feasible baseline")
        if true_gain < exact_value - baseline_value - penalty - 1e-9:
            raise AssertionError("probability-error bound failed")

    score = np.array([[0.6, 0.4]])
    truth = np.array([[0.4, 0.6]])
    predicted_gain = score[0, 0] - score[0, 1]
    true_gain = truth[0, 0] - truth[0, 1]
    one_error = np.abs(truth - score).max()
    if true_gain >= predicted_gain - one_error:
        raise AssertionError("positive control did not break the false one-error bound")


def main() -> None:
    """Run the adversarial checks and their positive controls."""
    check_net_queue_shift()
    check_same_metrics_different_staffing()
    check_matched_movement_dominance()
    print("all Queue Shift formal checks passed; the positive controls fired")


if __name__ == "__main__":
    main()
