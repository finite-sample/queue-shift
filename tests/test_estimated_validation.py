"""Tests for the estimated-model validation design."""

import argparse

import numpy as np

from experiments.analyze_estimated import summarize
from experiments.run_estimated import run


def test_small_validation_run_satisfies_invariants() -> None:
    args = argparse.Namespace(
        repetitions=2,
        release_size=1_000,
        planning_batches=1,
        batch_size=100,
        alpha_points=3,
        incumbent_signal=0.9,
        seed=71,
        scenarios=["well_estimated"],
    )
    release, results = run(args)
    assert len(release) == 2
    assert not results.empty
    summary = summarize(release, results)
    assert set(summary["alpha"]) == {0.0, 0.5, 1.0}
    assert np.all(summary["predicted_gain_pp"] >= -1e-9)
    endpoint = summary[summary["alpha"] == 1.0]
    assert endpoint["predicted_gain_pp"].iloc[0] == 0.0
