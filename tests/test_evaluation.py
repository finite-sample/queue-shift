"""Tests for matched-budget comparisons against NFR-controlled predictions."""

import numpy as np
import pytest

from queue_shift.evaluation import (
    evaluate_matched_budget,
    probability_matrix,
    select_nfr_alpha,
)


def test_binary_probabilities_become_two_class_matrix() -> None:
    got = probability_matrix(np.array([0.2, 0.8]))
    assert np.allclose(got, [[0.8, 0.2], [0.2, 0.8]])


def test_operational_assignment_weakly_dominates_under_common_value() -> None:
    incumbent = np.array([0, 0, 1, 1])
    baseline_probability = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7]])
    value_probability = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
    labels = np.array([1, 0, 0, 1])
    result = evaluate_matched_budget(
        incumbent, baseline_probability, value_probability, labels
    )
    assert result["baseline_moved_load"] == 0
    assert result["operational_moved_load"] == 0
    assert result["predicted_gain_pp"] > 0
    assert result["realized_gain_pp"] > 0


def test_raw_argmax_is_null_at_its_own_budget() -> None:
    rng = np.random.default_rng(19)
    value_probability = rng.dirichlet(np.ones(3), size=30)
    incumbent = rng.integers(0, 3, size=30)
    labels = rng.integers(0, 3, size=30)
    result = evaluate_matched_budget(
        incumbent, value_probability, value_probability, labels
    )
    assert result["predicted_gain_pp"] == pytest.approx(0.0, abs=1e-8)
    assert result["realized_gain_pp"] == pytest.approx(0.0, abs=1e-8)


def test_realized_gain_is_not_falsely_guaranteed() -> None:
    incumbent = np.array([0, 1])
    baseline_probability = np.array([[0.9, 0.1], [0.1, 0.9]])
    value_probability = np.array([[0.4, 0.6], [0.6, 0.4]])
    labels = np.array([0, 1])
    result = evaluate_matched_budget(
        incumbent, baseline_probability, value_probability, labels
    )
    assert result["predicted_gain_pp"] > 0
    assert result["realized_gain_pp"] < 0


def test_nfr_selector_maximizes_calibration_accuracy_within_target() -> None:
    counts = {0.0: (4, 10), 0.5: (2, 10), 1.0: (0, 10)}
    accuracies = {0.0: 0.9, 0.5: 0.8, 1.0: 0.6}
    assert select_nfr_alpha(counts, accuracies, epsilon=0.25) == 0.5
    assert select_nfr_alpha(counts, accuracies, epsilon=0.0) == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.2, 0.2]]),
        np.array([[1.1, -0.1]]),
        np.array([np.nan]),
    ],
)
def test_invalid_probabilities_fail_loudly(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"sum|lie|finite"):
        probability_matrix(bad)
