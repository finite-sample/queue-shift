"""Tests for the stable-distribution model-update simulation."""

import numpy as np
import pytest

from queue_shift.simulation import (
    accuracy,
    draw_stable_batch,
    draw_stable_feature_batch,
    evaluate_interpolation_path,
    release_decision,
    temperature_scale,
    validate_priors,
)

PRIORS = np.array([0.5, 0.3, 0.2])


def test_draw_is_reproducible_and_probabilities_are_valid() -> None:
    first = draw_stable_batch(np.random.default_rng(7), 100, PRIORS, 0.8, 0.7)
    second = draw_stable_batch(np.random.default_rng(7), 100, PRIORS, 0.8, 0.7)
    assert np.array_equal(first.labels, second.labels)
    assert np.allclose(first.incumbent_probability, second.incumbent_probability)
    assert np.allclose(first.candidate_probability, second.candidate_probability)
    assert np.allclose(first.incumbent_probability.sum(axis=1), 1)
    assert np.allclose(first.candidate_probability.sum(axis=1), 1)


def test_feature_draw_matches_compact_draw() -> None:
    compact = draw_stable_batch(np.random.default_rng(8), 100, PRIORS, 0.8, 0.7)
    features = draw_stable_feature_batch(
        np.random.default_rng(8), 100, PRIORS, 0.8, 0.7
    )
    assert np.array_equal(compact.labels, features.labels)
    assert np.allclose(compact.incumbent_probability, features.incumbent_probability)
    assert np.allclose(compact.candidate_probability, features.candidate_probability)


def test_temperature_scaling_preserves_predictions() -> None:
    batch = draw_stable_batch(np.random.default_rng(11), 100, PRIORS, 0.8, 0.7)
    scaled = temperature_scale(batch.candidate_probability, 0.5)
    assert np.array_equal(
        scaled.argmax(axis=1), batch.candidate_probability.argmax(axis=1)
    )
    assert np.allclose(scaled.sum(axis=1), 1)
    assert not np.allclose(scaled, batch.candidate_probability)


def test_candidate_additional_signal_improves_large_sample_accuracy() -> None:
    batch = draw_stable_batch(np.random.default_rng(19), 100_000, PRIORS, 0.7, 1.0)
    incumbent = accuracy(batch.incumbent_probability, batch.labels)
    candidate = accuracy(batch.candidate_probability, batch.labels)
    assert candidate > incumbent + 0.05


def test_release_gate_requires_strict_oos_improvement() -> None:
    batch = draw_stable_batch(np.random.default_rng(23), 10_000, PRIORS, 0.7, 1.0)
    decision = release_decision(batch)
    assert decision["accepted"]
    assert decision["candidate_accuracy"] > decision["incumbent_accuracy"]


def test_matched_path_dominates_in_expected_value_and_respects_movement() -> None:
    batch = draw_stable_batch(np.random.default_rng(29), 300, PRIORS, 0.8, 0.8)
    rows = evaluate_interpolation_path(batch, np.array([0.0, 0.5, 1.0]))
    for row in rows:
        assert row["predicted_gain_pp"] >= -1e-9
        assert row["operational_moved_load"] <= row["baseline_moved_load"]
    assert rows[-1]["predicted_gain_pp"] == pytest.approx(0.0, abs=1e-8)
    assert rows[-1]["operational_accuracy"] == rows[-1]["baseline_accuracy"]


def test_probability_error_bound_holds_for_misscaled_scores() -> None:
    batch = draw_stable_batch(np.random.default_rng(31), 300, PRIORS, 0.8, 0.8)
    misscaled = type(batch)(
        labels=batch.labels,
        incumbent_probability=batch.incumbent_probability,
        candidate_probability=temperature_scale(batch.candidate_probability, 0.4),
    )
    rows = evaluate_interpolation_path(
        misscaled,
        np.array([0.0, 0.5, 1.0]),
        true_probability=batch.candidate_probability,
    )
    for row in rows:
        assert row["conditional_gain_pp"] >= row["robust_lower_bound_pp"] - 1e-9


@pytest.mark.parametrize(
    "priors",
    [np.array([0.5, 0.4]), np.array([0.0, 1.0]), np.array([np.nan, np.nan])],
)
def test_invalid_priors_are_rejected(priors: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"sum|positive|finite"):
        validate_priors(priors)
