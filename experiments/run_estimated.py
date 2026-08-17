"""Validate matched-movement assignment with estimated multiclass models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from queue_shift.simulation import (
    StableBatch,
    draw_stable_feature_batch,
    evaluate_interpolation_path,
    release_decision,
    temperature_scale,
)

PRIORS = np.array([0.40, 0.25, 0.15, 0.12, 0.08])


@dataclass(frozen=True)
class Scenario:
    """One model-estimation stress scenario."""

    name: str
    training_size: int
    innovation_signal: float
    candidate_temperature: float


SCENARIOS = (
    Scenario("well_estimated", 5_000, 0.8, 1.0),
    Scenario("small_history", 500, 0.8, 1.0),
    Scenario("weak_innovation", 2_000, 0.3, 1.0),
    Scenario("overconfident", 2_000, 0.8, 0.5),
    Scenario("underconfident", 2_000, 0.8, 2.0),
)


def _fit_model(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
    model.fit(features, labels)
    expected_classes = np.arange(len(PRIORS))
    if not np.array_equal(model.classes_, expected_classes):
        raise RuntimeError("training sample did not contain every queue")
    return model


def _candidate_features(incumbent: np.ndarray, innovation: np.ndarray) -> np.ndarray:
    return np.column_stack([incumbent, innovation])


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all scenarios and return release and planning rows."""
    selected = {scenario.name: scenario for scenario in SCENARIOS}
    unknown = set(args.scenarios) - set(selected)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")

    release_rows: list[dict] = []
    result_rows: list[dict] = []
    alphas = np.linspace(0, 1, args.alpha_points)
    seed_sequence = np.random.SeedSequence(args.seed)
    scenario_seeds = seed_sequence.spawn(len(args.scenarios))

    for scenario_name, scenario_seed in zip(
        args.scenarios, scenario_seeds, strict=True
    ):
        scenario = selected[scenario_name]
        rng = np.random.default_rng(scenario_seed)
        for repetition in range(args.repetitions):
            training = draw_stable_feature_batch(
                rng,
                scenario.training_size,
                PRIORS,
                args.incumbent_signal,
                scenario.innovation_signal,
            )
            incumbent_model = _fit_model(training.incumbent_features, training.labels)
            candidate_model = _fit_model(
                _candidate_features(
                    training.incumbent_features,
                    training.innovation_features,
                ),
                training.labels,
            )

            release = draw_stable_feature_batch(
                rng,
                args.release_size,
                PRIORS,
                args.incumbent_signal,
                scenario.innovation_signal,
            )
            release_batch = StableBatch(
                labels=release.labels,
                incumbent_probability=incumbent_model.predict_proba(
                    release.incumbent_features
                ),
                candidate_probability=temperature_scale(
                    candidate_model.predict_proba(
                        _candidate_features(
                            release.incumbent_features,
                            release.innovation_features,
                        )
                    ),
                    scenario.candidate_temperature,
                ),
            )
            decision = release_decision(release_batch)
            decision.update(
                {
                    "scenario": scenario.name,
                    "repetition": repetition,
                    "training_size": scenario.training_size,
                    "innovation_signal": scenario.innovation_signal,
                    "candidate_temperature": scenario.candidate_temperature,
                }
            )
            release_rows.append(decision)
            if not decision["accepted"]:
                continue

            for batch_index in range(args.planning_batches):
                planning = draw_stable_feature_batch(
                    rng,
                    args.batch_size,
                    PRIORS,
                    args.incumbent_signal,
                    scenario.innovation_signal,
                )
                estimated_batch = StableBatch(
                    labels=planning.labels,
                    incumbent_probability=incumbent_model.predict_proba(
                        planning.incumbent_features
                    ),
                    candidate_probability=temperature_scale(
                        candidate_model.predict_proba(
                            _candidate_features(
                                planning.incumbent_features,
                                planning.innovation_features,
                            )
                        ),
                        scenario.candidate_temperature,
                    ),
                )
                rows = evaluate_interpolation_path(
                    estimated_batch,
                    alphas,
                    true_probability=planning.candidate_probability,
                )
                for row in rows:
                    row.update(
                        {
                            "scenario": scenario.name,
                            "repetition": repetition,
                            "batch": batch_index,
                            "release_gain_pp": decision["accuracy_gain_pp"],
                        }
                    )
                    result_rows.append(row)

    return pd.DataFrame(release_rows), pd.DataFrame(result_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--release-size", type=int, default=2_000)
    parser.add_argument("--planning-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--alpha-points", type=int, default=5)
    parser.add_argument("--incumbent-signal", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[scenario.name for scenario in SCENARIOS],
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/estimated"))
    return parser.parse_args()


def main() -> None:
    """Run the estimated-model validation and save its raw outputs."""
    args = parse_args()
    if (
        min(
            args.repetitions,
            args.release_size,
            args.planning_batches,
            args.batch_size,
        )
        <= 0
    ):
        raise ValueError("sample counts must be positive")
    if args.alpha_points < 2:
        raise ValueError("alpha-points must be at least two")
    release, results = run(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    release.to_csv(args.outdir / "release.csv", index=False)
    results.to_csv(args.outdir / "matched_budget.csv", index=False)
    print(release.groupby("scenario")["accepted"].agg(["sum", "count", "mean"]))
    print(f"wrote {len(results):,} matched-budget rows to {args.outdir}")


if __name__ == "__main__":
    main()
