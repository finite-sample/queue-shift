"""Tests for the regime-map simulation."""

import argparse

from experiments.analyze_regimes import load_lock_cost, summarize
from experiments.run_regimes import MAX_JOINT_BATCH, run


def test_small_regime_run_satisfies_invariants() -> None:
    args = argparse.Namespace(
        repetitions=1,
        training_size=1_500,
        release_size=500,
        alpha_points=3,
        seed=11,
        scenarios=["stable", "confused_pair"],
        regimes=["small_volatile", "large_stable"],
    )
    release, results = run(args)
    assert len(release) == 2
    summary = summarize(release, results)
    assert set(summary["alpha"]) == {0.0, 0.5, 1.0}
    large = results[results["regime"] == "large_stable"]
    assert (large["n_cases"] > MAX_JOINT_BATCH).all()
    assert large["joint_moved_load"].isna().all()
    small = results[results["regime"] == "small_volatile"]
    assert (small["joint_expected_flips"] <= small["flip_budget"] + 1e-3).all()
    assert (small["joint_moved_load"] <= small["baseline_moved_load"]).all()
    lock = load_lock_cost(results)
    assert len(lock) == 4
    assert lock["cost_pp"].notna().all()
