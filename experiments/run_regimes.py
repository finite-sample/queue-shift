"""Map where exact constrained assignment beats simpler update rules.

Each scenario is a mechanism that makes a released candidate's predicted queue mix
differ from the incumbent's. Within a scenario, every trained model pair is
evaluated under each batch regime, so regimes differ only in their planning batches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from queue_shift.assignment import queue_shift, solve_assignment
from queue_shift.evaluation import evaluate_matched_budget
from queue_shift.simulation import (
    StableBatch,
    draw_stable_feature_batch,
    release_decision,
)

FIVE_QUEUES = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
ZIPF_FIFTEEN = 1 / np.arange(1, 16)
ZIPF_FIFTEEN = ZIPF_FIFTEEN / ZIPF_FIFTEEN.sum()
# The joint arm needs a MILP whose time grows quickly with batch size.
MAX_JOINT_BATCH = 300


@dataclass(frozen=True)
class Scenario:
    """A mechanism for a difference between incumbent and candidate queue mixes."""

    name: str
    priors: np.ndarray
    incumbent_priors: np.ndarray
    incumbent_signal: float | np.ndarray
    innovation_signal: float | np.ndarray


@dataclass(frozen=True)
class BatchRegime:
    """Planning-batch size and batch-to-batch variability of the class mix."""

    name: str
    size: int
    concentration: float | None
    batches: int


SCENARIOS = (
    Scenario("stable", FIVE_QUEUES, FIVE_QUEUES, 0.9, 0.8),
    # The incumbent barely separates the two largest queues; the innovation does.
    Scenario(
        "confused_pair",
        FIVE_QUEUES,
        FIVE_QUEUES,
        np.array([0.3, 0.3, 0.9, 0.9, 0.9]),
        np.array([1.2, 1.2, 0.3, 0.3, 0.3]),
    ),
    # Demand moved after the incumbent was trained; the candidate saw recent data.
    Scenario(
        "stale_priors",
        np.array([0.25, 0.25, 0.20, 0.15, 0.15]),
        FIVE_QUEUES,
        0.9,
        0.8,
    ),
    Scenario("specialist_queues", ZIPF_FIFTEEN, ZIPF_FIFTEEN, 1.2, 1.0),
)

REGIMES = (
    BatchRegime("small_volatile", 100, 20.0, 4),
    BatchRegime("medium_stable", 300, None, 4),
    BatchRegime("medium_volatile", 300, 20.0, 4),
    BatchRegime("large_stable", 2_000, None, 2),
)


def _fit(features: np.ndarray, labels: np.ndarray, n_queues: int) -> LogisticRegression:
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1_000)
    model.fit(features, labels)
    if not np.array_equal(model.classes_, np.arange(n_queues)):
        raise RuntimeError("training sample did not contain every queue")
    return model


def _score(
    incumbent: LogisticRegression,
    candidate: LogisticRegression,
    rng: np.random.Generator,
    size: int,
    priors: np.ndarray,
    scenario: Scenario,
) -> tuple[StableBatch, np.ndarray]:
    """Draw a sample and return model scores plus the true candidate posterior."""
    sample = draw_stable_feature_batch(
        rng, size, priors, scenario.incumbent_signal, scenario.innovation_signal
    )
    batch = StableBatch(
        labels=sample.labels,
        incumbent_probability=incumbent.predict_proba(sample.incumbent_features),
        candidate_probability=candidate.predict_proba(
            np.column_stack([sample.incumbent_features, sample.innovation_features])
        ),
    )
    return batch, sample.candidate_probability


def _interpolate(batch: StableBatch, alpha: float) -> np.ndarray:
    return (
        (1 - alpha) * batch.incumbent_probability + alpha * batch.candidate_probability
    ).argmax(axis=1)


def _historical_prices(history: StableBatch, alpha: float) -> np.ndarray:
    """Learn queue prices on history at interpolation's movement share there."""
    incumbent = history.incumbent_probability.argmax(axis=1)
    n_queues = history.candidate_probability.shape[1]
    budget = queue_shift(
        np.bincount(_interpolate(history, alpha), minlength=n_queues),
        np.bincount(incumbent, minlength=n_queues),
    )
    result = solve_assignment(1 - history.candidate_probability, incumbent, budget)
    if result.queue_prices is None:
        raise RuntimeError("flow solve did not return queue prices")
    return result.queue_prices


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every scenario and regime; return release and planning rows."""
    scenarios = [s for s in SCENARIOS if s.name in args.scenarios]
    regimes = [r for r in REGIMES if r.name in args.regimes]
    if len(scenarios) != len(args.scenarios) or len(regimes) != len(args.regimes):
        raise ValueError("unknown scenario or regime name")
    alphas = np.linspace(0, 1, args.alpha_points)
    seeds = np.random.SeedSequence(args.seed).spawn(len(scenarios))
    release_rows: list[dict] = []
    result_rows: list[dict] = []

    for scenario, seed in zip(scenarios, seeds, strict=True):
        rng = np.random.default_rng(seed)
        n_queues = len(scenario.priors)
        for repetition in range(args.repetitions):
            old = draw_stable_feature_batch(
                rng,
                args.training_size,
                scenario.incumbent_priors,
                scenario.incumbent_signal,
                scenario.innovation_signal,
            )
            new = draw_stable_feature_batch(
                rng,
                args.training_size,
                scenario.priors,
                scenario.incumbent_signal,
                scenario.innovation_signal,
            )
            incumbent_model = _fit(old.incumbent_features, old.labels, n_queues)
            candidate_model = _fit(
                np.column_stack([new.incumbent_features, new.innovation_features]),
                new.labels,
                n_queues,
            )
            history, _ = _score(
                incumbent_model,
                candidate_model,
                rng,
                args.release_size,
                scenario.priors,
                scenario,
            )
            decision = release_decision(history)
            release_rows.append(
                {**decision, "scenario": scenario.name, "repetition": repetition}
            )
            if not decision["accepted"]:
                continue
            prices = {alpha: _historical_prices(history, alpha) for alpha in alphas}

            for regime in regimes:
                for batch_index in range(regime.batches):
                    priors = scenario.priors
                    if regime.concentration is not None:
                        priors = rng.dirichlet(regime.concentration * scenario.priors)
                        priors = np.clip(priors, 1e-6, None)
                        priors = priors / priors.sum()
                    batch, truth = _score(
                        incumbent_model,
                        candidate_model,
                        rng,
                        regime.size,
                        priors,
                        scenario,
                    )
                    incumbent = batch.incumbent_probability.argmax(axis=1)
                    raw_shift = queue_shift(
                        np.bincount(
                            batch.candidate_probability.argmax(axis=1),
                            minlength=n_queues,
                        ),
                        np.bincount(incumbent, minlength=n_queues),
                    )
                    for alpha in alphas:
                        offsets = (
                            1 - batch.candidate_probability + prices[alpha][None, :]
                        ).argmin(axis=1)
                        metrics = evaluate_matched_budget(
                            incumbent,
                            (1 - alpha) * batch.incumbent_probability
                            + alpha * batch.candidate_probability,
                            batch.candidate_probability,
                            batch.labels,
                            true_probability=truth,
                            include_joint=regime.size <= MAX_JOINT_BATCH,
                            extra_assignments={"offsets": offsets},
                        )
                        # Prices carry no budget guarantee, so also solve at the
                        # movement they actually used to measure their suboptimality.
                        at_offsets_shift = solve_assignment(
                            1 - batch.candidate_probability,
                            incumbent,
                            int(metrics["offsets_moved_load"]),
                        )
                        result_rows.append(
                            {
                                **metrics,
                                "offsets_shift_optimum_true_value": float(
                                    truth[
                                        np.arange(regime.size),
                                        at_offsets_shift.labels,
                                    ].mean()
                                ),
                                "scenario": scenario.name,
                                "regime": regime.name,
                                "repetition": repetition,
                                "batch": batch_index,
                                "alpha": float(alpha),
                                "raw_move_share": raw_shift / regime.size,
                            }
                        )
    return pd.DataFrame(release_rows), pd.DataFrame(result_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--training-size", type=int, default=3_000)
    parser.add_argument("--release-size", type=int, default=2_000)
    parser.add_argument("--alpha-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--scenarios", nargs="+", default=[s.name for s in SCENARIOS])
    parser.add_argument("--regimes", nargs="+", default=[r.name for r in REGIMES])
    parser.add_argument("--outdir", type=Path, default=Path("results/regimes"))
    return parser.parse_args()


def main() -> None:
    """Run the regime map and save raw outputs before any analysis."""
    args = parse_args()
    if min(args.repetitions, args.training_size, args.release_size) <= 0:
        raise ValueError("sample counts must be positive")
    if args.alpha_points < 2:
        raise ValueError("alpha-points must be at least two")
    release, results = run(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    release.to_csv(args.outdir / "release.csv", index=False)
    # Gzip keeps the checked-in table under the repository's large-file limit;
    # a fixed mtime keeps the compressed bytes reproducible.
    results.to_csv(
        args.outdir / "matched_budget.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(release.groupby("scenario")["accepted"].agg(["sum", "count", "mean"]))
    print(f"wrote {len(results):,} rows to {args.outdir}")


if __name__ == "__main__":
    main()
