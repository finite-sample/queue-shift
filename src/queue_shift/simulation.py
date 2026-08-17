"""Stable-distribution simulation for operationally constrained model updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from queue_shift.evaluation import evaluate_matched_budget, probability_matrix


@dataclass(frozen=True)
class StableBatch:
    """Labels and exact posteriors before and after a model innovation."""

    labels: np.ndarray
    incumbent_probability: np.ndarray
    candidate_probability: np.ndarray


@dataclass(frozen=True)
class StableFeatureBatch:
    """Features, labels, and true posteriors from the stable classification problem."""

    labels: np.ndarray
    incumbent_features: np.ndarray
    innovation_features: np.ndarray
    incumbent_probability: np.ndarray
    candidate_probability: np.ndarray


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def validate_priors(priors: np.ndarray) -> np.ndarray:
    """Validate and normalize a vector of class probabilities."""
    probability = np.asarray(priors, dtype=float)
    if probability.ndim != 1 or len(probability) < 2:
        raise ValueError("priors must contain at least two classes")
    if not np.all(np.isfinite(probability)) or np.any(probability <= 0):
        raise ValueError("priors must be finite and strictly positive")
    if not np.isclose(probability.sum(), 1.0):
        raise ValueError("priors must sum to one")
    return probability


def draw_stable_batch(
    rng: np.random.Generator,
    n_cases: int,
    priors: np.ndarray,
    incumbent_signal: float,
    innovation_signal: float,
) -> StableBatch:
    """Draw one batch from a fixed Gaussian classification problem.

    The incumbent observes one signal block. The candidate observes that block plus
    an independent innovation block. Both outputs are exact Bayes posteriors under
    the unchanged data-generating process.
    """
    feature_batch = draw_stable_feature_batch(
        rng,
        n_cases,
        priors,
        incumbent_signal,
        innovation_signal,
    )
    return StableBatch(
        labels=feature_batch.labels,
        incumbent_probability=feature_batch.incumbent_probability,
        candidate_probability=feature_batch.candidate_probability,
    )


def draw_stable_feature_batch(
    rng: np.random.Generator,
    n_cases: int,
    priors: np.ndarray,
    incumbent_signal: float,
    innovation_signal: float,
) -> StableFeatureBatch:
    """Draw features and true posteriors from the unchanged Gaussian problem."""
    class_priors = validate_priors(priors)
    if not isinstance(n_cases, (int, np.integer)) or n_cases <= 0:
        raise ValueError("n_cases must be a positive integer")
    if incumbent_signal < 0 or innovation_signal < 0:
        raise ValueError("signal strengths must be nonnegative")

    n_queues = len(class_priors)
    labels = rng.choice(n_queues, size=n_cases, p=class_priors)
    class_indicator = np.eye(n_queues)[labels]
    incumbent_features = rng.normal(size=(n_cases, n_queues))
    incumbent_features += incumbent_signal * class_indicator
    innovation_features = rng.normal(size=(n_cases, n_queues))
    innovation_features += innovation_signal * class_indicator

    log_prior = np.log(class_priors)
    incumbent_logits = (
        log_prior + incumbent_signal * incumbent_features - 0.5 * incumbent_signal**2
    )
    candidate_logits = (
        incumbent_logits
        + innovation_signal * innovation_features
        - 0.5 * innovation_signal**2
    )
    return StableFeatureBatch(
        labels=labels,
        incumbent_features=incumbent_features,
        innovation_features=innovation_features,
        incumbent_probability=probability_matrix(_softmax(incumbent_logits)),
        candidate_probability=probability_matrix(_softmax(candidate_logits)),
    )


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Apply multiclass temperature scaling without changing hard predictions."""
    prob = probability_matrix(probability)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(prob, np.finfo(float).tiny, 1.0)) / temperature
    return probability_matrix(_softmax(logits))


def accuracy(probability: np.ndarray, labels: np.ndarray) -> float:
    """Return hard-prediction accuracy."""
    prediction = probability_matrix(probability).argmax(axis=1)
    y = np.asarray(labels)
    if prediction.shape != y.shape:
        raise ValueError("probabilities and labels must cover the same cases")
    return float((prediction == y).mean())


def release_decision(batch: StableBatch) -> dict[str, float | int | bool]:
    """Accept an update only when it is more accurate on an independent release set."""
    incumbent_accuracy = accuracy(batch.incumbent_probability, batch.labels)
    candidate_accuracy = accuracy(batch.candidate_probability, batch.labels)
    return {
        "release_n": len(batch.labels),
        "incumbent_accuracy": incumbent_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_gain_pp": 100 * (candidate_accuracy - incumbent_accuracy),
        "accepted": candidate_accuracy > incumbent_accuracy,
    }


def evaluate_interpolation_path(
    batch: StableBatch,
    alphas: np.ndarray,
    true_probability: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Match each NFR interpolation point to an equal-movement exact solution."""
    alpha_grid = np.asarray(alphas, dtype=float)
    if alpha_grid.ndim != 1 or len(alpha_grid) == 0:
        raise ValueError("alphas must be a nonempty vector")
    if np.any(alpha_grid < 0) or np.any(alpha_grid > 1):
        raise ValueError("alphas must lie in [0, 1]")

    incumbent_prediction = batch.incumbent_probability.argmax(axis=1)
    rows = []
    for alpha in alpha_grid:
        baseline_probability = (
            1 - alpha
        ) * batch.incumbent_probability + alpha * batch.candidate_probability
        row = evaluate_matched_budget(
            incumbent_prediction,
            baseline_probability,
            batch.candidate_probability,
            batch.labels,
            true_probability=true_probability,
        )
        row["alpha"] = float(alpha)
        rows.append(row)
    return rows
