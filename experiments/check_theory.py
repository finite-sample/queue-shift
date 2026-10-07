"""Adversarial checks for the Queue Shift formal results."""

from __future__ import annotations

import itertools

import numpy as np

from queue_shift.assignment import (
    case_flip_weights,
    flip_load,
    queue_shift,
    solve_assignment,
)


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
        exact = solve_assignment(1 - score, incumbent, budget)

        baseline_value = score[np.arange(n_cases), baseline].mean()
        exact_value = score[np.arange(n_cases), exact.labels].mean()
        true_gain = (
            truth[np.arange(n_cases), exact.labels]
            - truth[np.arange(n_cases), baseline]
        ).mean()
        error = truth - score
        differ = exact.labels != baseline
        penalty = (
            np.abs(error[np.arange(n_cases), exact.labels])
            + np.abs(error[np.arange(n_cases), baseline])
        )[differ].sum() / n_cases

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


def check_joint_dominance() -> None:
    """Verify dominance at matched queue shift and matched expected flips."""
    rng = np.random.default_rng(461)
    for _ in range(150):
        n_cases = int(rng.integers(2, 8))
        n_queues = int(rng.integers(2, 4))
        incumbent = rng.integers(0, n_queues, size=n_cases)
        baseline = rng.integers(0, n_queues, size=n_cases)
        score = rng.dirichlet(np.ones(n_queues), size=n_cases)
        incumbent_loads = np.bincount(incumbent, minlength=n_queues)
        budget = queue_shift(np.bincount(baseline, minlength=n_queues), incumbent_loads)
        weights = case_flip_weights(score, incumbent, "expected_negative")
        flip_budget = flip_load(baseline, incumbent, weights)
        exact = solve_assignment(
            1 - score,
            incumbent,
            budget,
            flip_weights=weights,
            flip_budget=flip_budget,
        )
        best = max(
            score[np.arange(n_cases), labels].sum()
            for labels in map(
                np.asarray, itertools.product(range(n_queues), repeat=n_cases)
            )
            if queue_shift(np.bincount(labels, minlength=n_queues), incumbent_loads)
            <= budget
            and flip_load(labels, incumbent, weights) <= flip_budget + 1e-12
        )
        exact_value = score[np.arange(n_cases), exact.labels].sum()
        if abs(exact_value - best) > 1e-8:
            raise AssertionError("joint solver missed the exhaustive optimum")
        if exact_value < score[np.arange(n_cases), baseline].sum() - 1e-9:
            raise AssertionError("joint assignment lost to a feasible baseline")
        if exact.moved_load > budget or exact.flip_load > flip_budget + 1e-6:
            raise AssertionError("joint assignment exceeded a matched budget")

    # Positive control: matching raw churn instead of expected flips does not bound
    # expected flips, so a churn-matched optimum can exceed the baseline's.
    # With two queues the gain from moving, 1 - 2 p_a, already ranks cases by
    # expected flips, so the control needs a third queue.
    score = np.array([[0.45, 0.55, 0.00], [0.30, 0.35, 0.35]])
    incumbent = np.array([0, 0])
    baseline = np.array([0, 1])
    weights = case_flip_weights(score, incumbent, "expected_negative")
    churn_matched = solve_assignment(
        1 - score,
        incumbent,
        1,
        flip_weights=case_flip_weights(score, incumbent, "churn"),
        flip_budget=1.0,
    )
    if flip_load(churn_matched.labels, incumbent, weights) <= flip_load(
        baseline, incumbent, weights
    ):
        raise AssertionError("positive control did not separate churn from flips")


def check_dual_prices() -> None:
    """Verify that queue prices certify the flow optimum, penalty included."""
    rng = np.random.default_rng(571)
    for _ in range(200):
        n_cases = int(rng.integers(10, 120))
        n_queues = int(rng.integers(2, 7))
        score = rng.dirichlet(np.ones(n_queues), size=n_cases)
        incumbent = rng.integers(0, n_queues, size=n_cases)
        budget = int(rng.integers(0, n_cases // 2))
        penalty = float(rng.choice([0.0, 0.2, 1.0]))
        weights = case_flip_weights(score, incumbent, "expected_negative")
        result = solve_assignment(
            1 - score,
            incumbent,
            budget,
            flip_weights=weights,
            flip_penalty=penalty,
        )
        if result.queue_prices is None:
            raise AssertionError("flow mode did not return queue prices")
        moved_off = np.arange(n_queues)[None, :] != incumbent[:, None]
        priced = 1 - score + penalty * weights[:, None] * moved_off
        priced = priced + result.queue_prices[None, :]
        chosen = priced[np.arange(n_cases), result.labels]
        if np.any(chosen > priced.min(axis=1) + 1e-8):
            raise AssertionError("queue prices do not certify the assignment")
        tied = (priced <= priced.min(axis=1, keepdims=True) + 1e-8).sum(axis=1) > 1
        if tied.sum() > n_queues:
            raise AssertionError("more tied cases than queues")


def main() -> None:
    """Run the adversarial checks and their positive controls."""
    check_net_queue_shift()
    check_same_metrics_different_staffing()
    check_matched_movement_dominance()
    check_joint_dominance()
    check_dual_prices()
    print("all Queue Shift formal checks passed; the positive controls fired")


if __name__ == "__main__":
    main()
