"""Run the stable-distribution model-update experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from queue_shift.simulation import (
    draw_stable_batch,
    evaluate_interpolation_path,
    release_decision,
)

DEFAULT_PRIORS = [0.40, 0.25, 0.15, 0.12, 0.08]


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run release gates and matched-budget planning batches."""
    rng = np.random.default_rng(args.seed)
    alphas = np.linspace(0, 1, args.alpha_points)
    priors = np.asarray(args.priors, dtype=float)
    release_rows = []
    result_rows = []

    for repetition in range(args.repetitions):
        release_batch = draw_stable_batch(
            rng,
            args.release_size,
            priors,
            args.incumbent_signal,
            args.innovation_signal,
        )
        decision = release_decision(release_batch)
        decision["repetition"] = repetition
        release_rows.append(decision)
        if not decision["accepted"]:
            continue

        for batch_index in range(args.planning_batches):
            planning_batch = draw_stable_batch(
                rng,
                args.batch_size,
                priors,
                args.incumbent_signal,
                args.innovation_signal,
            )
            for row in evaluate_interpolation_path(planning_batch, alphas):
                row.update(
                    {
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
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--release-size", type=int, default=2000)
    parser.add_argument("--planning-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--alpha-points", type=int, default=11)
    parser.add_argument("--incumbent-signal", type=float, default=0.9)
    parser.add_argument("--innovation-signal", type=float, default=0.8)
    parser.add_argument("--priors", type=float, nargs="+", default=DEFAULT_PRIORS)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--outdir", type=Path, default=Path("results/oracle"))
    return parser.parse_args()


def main() -> None:
    """Run the oracle validation and save its raw outputs."""
    args = parse_args()
    if args.repetitions <= 0 or args.release_size <= 0:
        raise ValueError("repetitions and release-size must be positive")
    if args.planning_batches <= 0 or args.batch_size <= 0:
        raise ValueError("planning-batches and batch-size must be positive")
    if args.alpha_points < 2:
        raise ValueError("alpha-points must be at least two")

    release, results = run(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    release.to_csv(args.outdir / "release.csv", index=False)
    results.to_csv(args.outdir / "matched_budget.csv", index=False)
    acceptance = release["accepted"].mean()
    print(
        f"accepted {int(release['accepted'].sum())}/{len(release)} updates "
        f"({acceptance:.1%})"
    )
    print(f"wrote {len(results):,} matched-budget rows to {args.outdir}")


if __name__ == "__main__":
    main()
