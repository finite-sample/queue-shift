"""Roundoff in tied repetitions must not count as a realized improvement."""

from pathlib import Path

import pandas as pd
import pytest

from experiments import analyze_estimated, analyze_oracle


@pytest.mark.parametrize(
    ("analyzer", "experiment", "share"),
    [
        (analyze_oracle, "oracle", "positive_repetition_share"),
        (analyze_estimated, "estimated", "positive_realized_share"),
    ],
)
def test_positive_share_excludes_roundoff(analyzer, experiment, share) -> None:
    directory = Path(__file__).parents[2] / "results" / experiment
    template = pd.read_csv(directory / "matched_budget.csv", nrows=1).iloc[0]
    rows = []
    for repetition, gain in enumerate([1e-14, 0.1, -0.1]):
        for alpha in [0.0, 1.0]:
            row = template.copy()
            row["repetition"] = repetition
            row["alpha"] = alpha
            row["realized_gain_pp"] = gain if alpha == 0 else 0.0
            row["predicted_gain_pp"] = 0.0
            rows.append(row)
    release = pd.read_csv(directory / "release.csv", nrows=1)
    summary = analyzer.summarize(release, pd.DataFrame(rows))
    assert summary.loc[summary["alpha"] == 0, share].iloc[0] == pytest.approx(1 / 3)
    assert summary.loc[summary["alpha"] == 1, share].iloc[0] == 0
