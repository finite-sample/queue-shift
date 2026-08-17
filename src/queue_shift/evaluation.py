"""Matched-budget evaluation for operationally constrained model updates."""

from __future__ import annotations

import numpy as np

from queue_shift.assignment import queue_shift, solve_assignment


def probability_matrix(probability: np.ndarray) -> np.ndarray:
    """Return an ``n x k`` probability matrix for binary or multiclass output."""
    prob = np.asarray(probability, dtype=float)
    if prob.ndim == 1:
        prob = np.column_stack([1 - prob, prob])
    if prob.ndim != 2 or prob.shape[1] < 2:
        raise ValueError("probability must be a binary vector or an n-by-k matrix")
    if not np.all(np.isfinite(prob)) or np.any(prob < 0) or np.any(prob > 1):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    return prob


def predictions(probability: np.ndarray) -> np.ndarray:
    """Convert binary or multiclass probabilities to hard labels."""
    return probability_matrix(probability).argmax(axis=1)


def negative_flip_rate(
    incumbent_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, int, float]:
    """Return negative-flip count, denominator, and rate."""
    incumbent = np.asarray(incumbent_prediction)
    candidate = np.asarray(candidate_prediction)
    y = np.asarray(labels)
    if incumbent.shape != candidate.shape or incumbent.shape != y.shape:
        raise ValueError("incumbent, candidate, and labels must have the same shape")
    incumbent_correct = incumbent == y
    count = int((incumbent_correct & (candidate != y)).sum())
    denominator = int(incumbent_correct.sum())
    return count, denominator, count / denominator if denominator else 0.0


def select_nfr_alpha(
    flip_counts: dict[float, tuple[int, int]],
    accuracies: dict[float, float],
    epsilon: float,
) -> float:
    """Select the most accurate interpolation point within a calibration NFR target."""
    if flip_counts.keys() != accuracies.keys() or not flip_counts:
        raise ValueError(
            "flip_counts and accuracies must have the same nonempty alpha grid"
        )
    feasible = []
    for alpha, (count, denominator) in flip_counts.items():
        nfr = count / denominator if denominator else 0.0
        if nfr <= epsilon + 1e-12:
            feasible.append(alpha)
    if not feasible:
        return max(flip_counts)
    return min(feasible, key=lambda alpha: (-accuracies[alpha], alpha))


def evaluate_matched_budget(
    incumbent_prediction: np.ndarray,
    baseline_probability: np.ndarray,
    value_probability: np.ndarray,
    labels: np.ndarray,
    true_probability: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compare a baseline assignment with the operational optimum at its queue shift.

    ``value_probability`` is the common value signal, normally the unconstrained
    improved model. The exact assignment uses no labels. Labels enter only after
    both assignments have been fixed and measure realized performance.
    """
    incumbent = np.asarray(incumbent_prediction, dtype=int)
    baseline_prob = probability_matrix(baseline_probability)
    value_prob = probability_matrix(value_probability)
    y = np.asarray(labels, dtype=int)
    if baseline_prob.shape != value_prob.shape:
        raise ValueError("baseline and value probabilities must have the same shape")
    if incumbent.shape != y.shape or len(y) != value_prob.shape[0]:
        raise ValueError(
            "predictions, probabilities, and labels must cover the same cases"
        )

    n_cases, n_queues = value_prob.shape
    if np.any(incumbent < 0) or np.any(incumbent >= n_queues):
        raise ValueError("incumbent predictions contain an unknown queue")
    if np.any(y < 0) or np.any(y >= n_queues):
        raise ValueError("labels contain an unknown class")

    baseline = baseline_prob.argmax(axis=1)
    incumbent_loads = np.bincount(incumbent, minlength=n_queues)
    baseline_loads = np.bincount(baseline, minlength=n_queues)
    budget = queue_shift(baseline_loads, incumbent_loads)
    operational = solve_assignment(1 - value_prob, incumbent_loads, budget)

    baseline_accuracy = float((baseline == y).mean())
    operational_accuracy = float((operational.labels == y).mean())
    baseline_value = float(value_prob[np.arange(n_cases), baseline].mean())
    operational_value = float(value_prob[np.arange(n_cases), operational.labels].mean())
    baseline_nfr_count, incumbent_correct, baseline_nfr = negative_flip_rate(
        incumbent, baseline, y
    )
    operational_nfr_count, _, operational_nfr = negative_flip_rate(
        incumbent, operational.labels, y
    )

    predicted_gain = 100 * (operational_value - baseline_value)
    if predicted_gain < -1e-7:
        raise RuntimeError(
            "operational assignment is worse under the common value signal"
        )

    output = {
        "n_cases": n_cases,
        "move_budget": budget,
        "baseline_moved_load": budget,
        "operational_moved_load": operational.moved_load,
        "baseline_move_share": budget / n_cases,
        "operational_move_share": operational.moved_load / n_cases,
        "baseline_churn": int((baseline != incumbent).sum()),
        "operational_churn": int((operational.labels != incumbent).sum()),
        "baseline_accuracy": baseline_accuracy,
        "operational_accuracy": operational_accuracy,
        "realized_gain_pp": 100 * (operational_accuracy - baseline_accuracy),
        "baseline_common_value": baseline_value,
        "operational_common_value": operational_value,
        "predicted_gain_pp": max(0.0, predicted_gain),
        "incumbent_correct": incumbent_correct,
        "baseline_nfr_count": baseline_nfr_count,
        "operational_nfr_count": operational_nfr_count,
        "baseline_nfr": baseline_nfr,
        "operational_nfr": operational_nfr,
    }
    if true_probability is not None:
        truth = probability_matrix(true_probability)
        if truth.shape != value_prob.shape:
            raise ValueError("true and value probabilities must have the same shape")
        baseline_true_value = float(truth[np.arange(n_cases), baseline].mean())
        operational_true_value = float(
            truth[np.arange(n_cases), operational.labels].mean()
        )
        error = truth - value_prob
        error_penalty = float(
            (
                np.abs(error[np.arange(n_cases), operational.labels])
                + np.abs(error[np.arange(n_cases), baseline])
            ).mean()
        )
        true_gain = 100 * (operational_true_value - baseline_true_value)
        robust_lower_bound = predicted_gain - 100 * error_penalty
        if true_gain < robust_lower_bound - 1e-8:
            raise RuntimeError("probability-error robustness bound was violated")
        output.update(
            {
                "baseline_true_value": baseline_true_value,
                "operational_true_value": operational_true_value,
                "conditional_gain_pp": true_gain,
                "score_error_penalty_pp": 100 * error_penalty,
                "robust_lower_bound_pp": robust_lower_bound,
            }
        )
    return output
