"""Matched-budget evaluation for operationally constrained model updates."""

from __future__ import annotations

import numpy as np

from queue_shift.assignment import (
    case_flip_weights,
    flip_load,
    queue_shift,
    solve_assignment,
    workload_shift,
)


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


def assignment_metrics(
    assigned: np.ndarray,
    incumbent: np.ndarray,
    labels: np.ndarray,
    value_probability: np.ndarray,
    true_probability: np.ndarray | None = None,
    queue_costs: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return movement, churn, flip, and accuracy measures for one assignment.

    ``expected_flips`` uses ``value_probability``, the signal available when the
    assignment is chosen; ``true_expected_flips`` uses the true probabilities.
    """
    n_cases, n_queues = value_probability.shape
    rows = np.arange(n_cases)
    moved = queue_shift(
        np.bincount(assigned, minlength=n_queues),
        np.bincount(incumbent, minlength=n_queues),
    )
    nfr_count, _, nfr = negative_flip_rate(incumbent, assigned, labels)
    metrics: dict[str, float | int] = {
        "moved_load": moved,
        "move_share": moved / n_cases,
        "churn": int((assigned != incumbent).sum()),
        "expected_flips": flip_load(
            assigned,
            incumbent,
            case_flip_weights(value_probability, incumbent, "expected_negative"),
        ),
        "accuracy": float((assigned == labels).mean()),
        "common_value": float(value_probability[rows, assigned].mean()),
        "nfr_count": nfr_count,
        "nfr": nfr,
    }
    if queue_costs is not None:
        metrics["workload_shift"] = workload_shift(
            np.bincount(assigned, minlength=n_queues),
            np.bincount(incumbent, minlength=n_queues),
            queue_costs,
        )
    if true_probability is not None:
        metrics["true_value"] = float(true_probability[rows, assigned].mean())
        metrics["true_expected_flips"] = flip_load(
            assigned,
            incumbent,
            case_flip_weights(true_probability, incumbent, "expected_negative"),
        )
    return metrics


def evaluate_matched_budget(
    incumbent_prediction: np.ndarray,
    baseline_probability: np.ndarray,
    value_probability: np.ndarray,
    labels: np.ndarray,
    true_probability: np.ndarray | None = None,
    include_joint: bool = True,
    extra_assignments: dict[str, np.ndarray] | None = None,
    queue_costs: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compare a baseline assignment with exact assignments at its constraints.

    ``operational`` matches the baseline's queue shift only. ``joint`` also matches
    its expected negative flips under ``value_probability``; it needs a MILP, so
    ``include_joint=False`` skips it for large batches. ``value_probability`` is
    the common value signal, normally the unconstrained improved model. The exact
    assignments use no labels. Labels enter only after every assignment is fixed and
    measure realized performance. ``extra_assignments`` are summarized with the same
    measures but carry no dominance guarantee. ``queue_costs`` prices one case in
    each queue (for example handle hours); the matched movement budget is then the
    baseline's :func:`~queue_shift.assignment.workload_shift` in that currency.
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

    rows = np.arange(n_cases)
    baseline = baseline_prob.argmax(axis=1)
    baseline_loads = np.bincount(baseline, minlength=n_queues)
    incumbent_loads = np.bincount(incumbent, minlength=n_queues)
    budget: float = queue_shift(baseline_loads, incumbent_loads)
    if queue_costs is not None:
        budget = workload_shift(baseline_loads, incumbent_loads, queue_costs)
    weights = case_flip_weights(value_prob, incumbent, "expected_negative")
    flip_budget = flip_load(baseline, incumbent, weights)
    assignments = {
        "baseline": baseline,
        "operational": solve_assignment(
            1 - value_prob, incumbent, budget, queue_costs=queue_costs
        ).labels,
    }
    if include_joint:
        assignments["joint"] = solve_assignment(
            1 - value_prob,
            incumbent,
            budget,
            flip_weights=weights,
            flip_budget=flip_budget,
            queue_costs=queue_costs,
        ).labels
    for name, assigned in (extra_assignments or {}).items():
        if name in assignments:
            raise ValueError(f"extra assignment name {name!r} is reserved")
        if np.shape(assigned) != (n_cases,):
            raise ValueError("extra assignments must contain one label per case")
        assignments[name] = np.asarray(assigned, dtype=int)

    truth = None if true_probability is None else probability_matrix(true_probability)
    if truth is not None and truth.shape != value_prob.shape:
        raise ValueError("true and value probabilities must have the same shape")
    _, incumbent_correct, _ = negative_flip_rate(incumbent, incumbent, y)
    output: dict[str, float | int] = {
        "n_cases": n_cases,
        "move_budget": budget,
        "flip_budget": flip_budget,
        "incumbent_correct": incumbent_correct,
    }
    for name, assigned in assignments.items():
        metrics = assignment_metrics(
            assigned, incumbent, y, value_prob, truth, queue_costs
        )
        output.update({f"{name}_{key}": value for key, value in metrics.items()})

    for name in ("operational", "joint")[: 2 if include_joint else 1]:
        assigned = assignments[name]
        predicted_gain = 100 * (
            output[f"{name}_common_value"] - output["baseline_common_value"]
        )
        if predicted_gain < -1e-7:
            raise RuntimeError(f"{name} assignment is worse under the common score")
        prefix = "" if name == "operational" else "joint_"
        output[f"{prefix}predicted_gain_pp"] = max(0.0, predicted_gain)
        output[f"{prefix}realized_gain_pp"] = 100 * (
            output[f"{name}_accuracy"] - output["baseline_accuracy"]
        )
        if truth is None:
            continue
        # Cases with equal labels contribute identical errors to both assignments,
        # so only the cases where the two assignments differ enter the bound.
        differ = assigned != baseline
        error = truth - value_prob
        error_penalty = float(
            (np.abs(error[rows, assigned]) + np.abs(error[rows, baseline]))[
                differ
            ].sum()
            / n_cases
        )
        true_gain = 100 * (output[f"{name}_true_value"] - output["baseline_true_value"])
        robust_lower_bound = predicted_gain - 100 * error_penalty
        if true_gain < robust_lower_bound - 1e-8:
            raise RuntimeError("probability-error robustness bound was violated")
        output.update(
            {
                f"{prefix}conditional_gain_pp": true_gain,
                f"{prefix}score_error_penalty_pp": 100 * error_penalty,
                f"{prefix}robust_lower_bound_pp": robust_lower_bound,
            }
        )
    return output
