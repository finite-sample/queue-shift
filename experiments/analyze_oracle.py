"""Summarize the stable-distribution model-update experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import pandas as pd

from experiments.plot_style import BLUE, GRAY, LIGHT_GRAY, apply_plot_style

mpl.use("Agg")
import matplotlib.pyplot as plt

apply_plot_style(mpl)


def summarize(release: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Collapse matched comparisons by interpolation point."""
    if results.empty:
        raise ValueError("no update passed the release gate")
    repetition_means = results.groupby(["repetition", "alpha"], as_index=False).agg(
        mean_move_share=("baseline_move_share", "mean"),
        mean_nfr=("baseline_nfr", "mean"),
        baseline_accuracy=("baseline_accuracy", "mean"),
        operational_accuracy=("operational_accuracy", "mean"),
        expected_gain_pp=("predicted_gain_pp", "mean"),
        realized_gain_pp=("realized_gain_pp", "mean"),
    )
    summary = (
        repetition_means.groupby("alpha", as_index=False)
        .agg(
            repetitions=("realized_gain_pp", "size"),
            mean_move_share=("mean_move_share", "mean"),
            mean_nfr=("mean_nfr", "mean"),
            baseline_accuracy=("baseline_accuracy", "mean"),
            baseline_accuracy_se=("baseline_accuracy", "sem"),
            operational_accuracy=("operational_accuracy", "mean"),
            operational_accuracy_se=("operational_accuracy", "sem"),
            expected_gain_pp=("expected_gain_pp", "mean"),
            realized_gain_pp=("realized_gain_pp", "mean"),
            realized_gain_se=("realized_gain_pp", "sem"),
            positive_repetition_share=("realized_gain_pp", lambda x: (x > 0).mean()),
        )
        .sort_values("alpha")
    )
    violations = results[
        results["operational_moved_load"] > results["baseline_moved_load"]
    ]
    if not violations.empty:
        raise RuntimeError("operational assignment exceeded a matched movement budget")
    if (results["predicted_gain_pp"] < -1e-9).any():
        raise RuntimeError("operational assignment violated expected-value dominance")
    summary["planning_batches"] = results.groupby("alpha").size().to_numpy()
    summary["movement_violations"] = 0
    summary.attrs["acceptance_rate"] = float(release["accepted"].mean())
    summary.attrs["mean_release_gain_pp"] = float(
        release.loc[release["accepted"], "accuracy_gain_pp"].mean()
    )
    return summary


def write_table(summary: pd.DataFrame, path: Path) -> None:
    """Write the main matched-budget comparison as a compact LaTeX table."""
    selected = summary[summary["alpha"].isin([0.0, 0.2, 0.5, 0.8, 1.0])]
    rows = [
        (
            f"{row.alpha:.1f} & {100 * row.mean_move_share:.2f} & "
            f"{100 * row.mean_nfr:.2f} & {100 * row.baseline_accuracy:.2f} & "
            f"{100 * row.operational_accuracy:.2f} & {row.realized_gain_pp:.2f} \\\\"
        )
        for row in selected.itertuples(index=False)
    ]
    content = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{NFR interpolation versus exact assignment at the same queue shift}",
            r"\label{tab:stable-update}",
            r"\small",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            "Interpolation & Queue shift & NFR & NFR path & Exact assignment & Gain \\\\",
            "$\\alpha$ & (\\%) & (\\%) & accuracy (\\%) & accuracy (\\%) & (pp) \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.94\linewidth}",
            r"\footnotesize\emph{Note:} The data-generating process is fixed across release and planning samples. The candidate observes an additional independent signal and is deployed only after beating the incumbent on an independent out-of-sample release set. Each row compares probability interpolation with the exact assignment using the candidate posterior at the interpolation point's realized queue shift. Results average 1,000 planning batches from 200 accepted updates. Percentage-point gains use labels revealed only after both assignments are fixed.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_figure(summary: pd.DataFrame, path: Path) -> None:
    """Plot accuracy against queue movement for the two matched policies."""
    x = 100 * summary["mean_move_share"].to_numpy()
    baseline = 100 * summary["baseline_accuracy"].to_numpy()
    operational = 100 * summary["operational_accuracy"].to_numpy()
    baseline_half_width = 1.96 * 100 * summary["baseline_accuracy_se"].to_numpy()
    operational_half_width = 1.96 * 100 * summary["operational_accuracy_se"].to_numpy()

    figure, axis = plt.subplots(figsize=(5.4, 3.3))
    axis.fill_between(
        x,
        operational - operational_half_width,
        operational + operational_half_width,
        color=BLUE,
        alpha=0.14,
        linewidth=0,
    )
    axis.fill_between(
        x,
        baseline - baseline_half_width,
        baseline + baseline_half_width,
        color=GRAY,
        alpha=0.12,
        linewidth=0,
    )
    axis.plot(
        x,
        operational,
        color=BLUE,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Exact assignment",
    )
    axis.plot(
        x,
        baseline,
        color=GRAY,
        marker="s",
        markersize=3.2,
        linewidth=1.4,
        linestyle="--",
        label="NFR interpolation",
    )
    axis.set_xlabel("Forecasted workload moved across queues (%)")
    axis.set_ylabel("Realized accuracy (%)")
    axis.set_xlim(left=0)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_macros(release: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Write manuscript numbers from the same computed summaries."""
    zero = summary.loc[summary["alpha"] == 0].iloc[0]
    midpoint = summary.loc[summary["alpha"] == 0.5].iloc[0]
    content = "\n".join(
        [
            f"\\newcommand{{\\ReleaseAcceptance}}{{{100 * release['accepted'].mean():.0f}\\%}}",
            f"\\newcommand{{\\ReleaseGain}}{{{release.loc[release['accepted'], 'accuracy_gain_pp'].mean():.2f}}}",
            f"\\newcommand{{\\ZeroMoveGain}}{{{zero.realized_gain_pp:.2f}}}",
            f"\\newcommand{{\\MidMoveShare}}{{{100 * midpoint.mean_move_share:.2f}\\%}}",
            f"\\newcommand{{\\MidOperationalGain}}{{{midpoint.realized_gain_pp:.2f}}}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--macros", type=Path)
    return parser.parse_args()


def main() -> None:
    """Generate oracle summaries and exhibits."""
    args = parse_args()
    release = pd.read_csv(args.results_dir / "release.csv")
    results = pd.read_csv(args.results_dir / "matched_budget.csv")
    summary = summarize(release, results)
    print(
        f"release acceptance: {summary.attrs['acceptance_rate']:.1%}; "
        f"mean accepted OOS gain: {summary.attrs['mean_release_gain_pp']:.2f} pp"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False, float_format="%.12g")
    if args.tex:
        write_table(summary, args.tex)
    if args.figure:
        write_figure(summary, args.figure)
    if args.macros:
        write_macros(release, summary, args.macros)


if __name__ == "__main__":
    main()
