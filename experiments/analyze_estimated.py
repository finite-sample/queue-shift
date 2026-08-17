"""Audit and summarize the estimated-model matched-movement validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from experiments.plot_style import BLUE, LIGHT_GRAY, ORANGE, apply_plot_style

mpl.use("Agg")
import matplotlib.pyplot as plt

apply_plot_style(mpl)


PRIMARY_ALPHAS = (0.0, 0.5)
SCENARIO_ORDER = (
    "well_estimated",
    "small_history",
    "weak_innovation",
    "overconfident",
    "underconfident",
)

SCENARIO_LABELS = {
    "well_estimated": "Well estimated",
    "small_history": "Small history",
    "weak_innovation": "Weak innovation",
    "overconfident": "Overconfident",
    "underconfident": "Underconfident",
}


def summarize(release: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Return repetition-level summaries and enforce the design invariants."""
    if results.empty:
        raise ValueError("no estimated update passed the release gate")
    if (results["operational_moved_load"] > results["baseline_moved_load"]).any():
        raise RuntimeError("an exact assignment exceeded its matched movement budget")
    if (results["predicted_gain_pp"] < -1e-9).any():
        raise RuntimeError("supplied-score dominance failed")
    if (results["conditional_gain_pp"] < results["robust_lower_bound_pp"] - 1e-8).any():
        raise RuntimeError("the probability-error lower bound failed")
    endpoint = results[results["alpha"] == 1.0]
    if endpoint.empty or (endpoint["predicted_gain_pp"].abs() > 1e-7).any():
        raise RuntimeError("the raw-candidate endpoint is not the required score null")

    repetition = results.groupby(
        ["scenario", "repetition", "alpha"], as_index=False
    ).agg(
        move_share=("baseline_move_share", "mean"),
        nfr=("baseline_nfr", "mean"),
        predicted_gain_pp=("predicted_gain_pp", "mean"),
        conditional_gain_pp=("conditional_gain_pp", "mean"),
        realized_gain_pp=("realized_gain_pp", "mean"),
        robust_lower_bound_pp=("robust_lower_bound_pp", "mean"),
    )
    summary = repetition.groupby(["scenario", "alpha"], as_index=False).agg(
        repetitions=("realized_gain_pp", "size"),
        move_share=("move_share", "mean"),
        nfr=("nfr", "mean"),
        predicted_gain_pp=("predicted_gain_pp", "mean"),
        conditional_gain_pp=("conditional_gain_pp", "mean"),
        conditional_gain_se=("conditional_gain_pp", "sem"),
        realized_gain_pp=("realized_gain_pp", "mean"),
        realized_gain_se=("realized_gain_pp", "sem"),
        positive_realized_share=("realized_gain_pp", lambda x: (x > 0).mean()),
        robust_lower_bound_pp=("robust_lower_bound_pp", "mean"),
    )
    summary["scenario"] = pd.Categorical(
        summary["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    summary = summary.sort_values(["scenario", "alpha"]).reset_index(drop=True)
    summary["scenario"] = summary["scenario"].astype(str)
    acceptance = release.groupby("scenario")["accepted"].mean()
    summary["acceptance_rate"] = summary["scenario"].map(acceptance)
    primary = summary[summary["alpha"].isin(PRIMARY_ALPHAS)]
    scenario_pass = primary.groupby("scenario")["conditional_gain_pp"].min() >= -1e-9
    summary["stress_pass"] = summary["scenario"].map(scenario_pass)
    return summary


def write_table(summary: pd.DataFrame, path: Path) -> None:
    """Write the scenario comparison table."""
    selected = summary[summary["alpha"].isin([0.0, 0.5, 1.0])]
    rows = []
    for row in selected.itertuples(index=False):
        conditional_low = row.conditional_gain_pp - 1.96 * row.conditional_gain_se
        conditional_high = row.conditional_gain_pp + 1.96 * row.conditional_gain_se
        realized_low = row.realized_gain_pp - 1.96 * row.realized_gain_se
        realized_high = row.realized_gain_pp + 1.96 * row.realized_gain_se
        rows.append(
            f"{SCENARIO_LABELS[row.scenario]} & {row.alpha:.1f} & "
            f"{100 * row.move_share:.2f} & "
            f"{row.predicted_gain_pp:.2f} & "
            f"{row.conditional_gain_pp:.2f} "
            f"[{conditional_low:.2f}, {conditional_high:.2f}] & "
            f"{row.realized_gain_pp:.2f} [{realized_low:.2f}, {realized_high:.2f}] \\\\"
        )
    content = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Estimated-model validation against NFR interpolation}",
            r"\label{tab:estimated-validation}",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            (
                r"Scenario & $\alpha$ & Shift (\%) & Score gain & "
                r"True gain [95\% CI] & Realized gain [95\% CI] \\"
            ),
            r" & & & (pp) & (pp) & (pp) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            (
                r"\footnotesize\emph{Note:} Multinomial logistic models are "
                r"estimated on historical samples and released only after the candidate "
                r"wins on an independent labeled sample. Each row compares exact "
                r"assignment with probability interpolation at the same realized queue "
                r"shift. True gain uses the simulation's conditional probabilities and "
                r"is unavailable in applications. Results average four planning batches "
                r"within each independently trained accepted update before averaging "
                r"across updates."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_figure(summary: pd.DataFrame, path: Path) -> None:
    """Plot matched-movement gains under finite estimation and score distortion."""
    selected = summary[summary["alpha"].isin(PRIMARY_ALPHAS)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharex=True, sharey=True)
    scenario_positions = np.arange(len(SCENARIO_ORDER))

    for axis, alpha in zip(axes, PRIMARY_ALPHAS, strict=True):
        panel = selected[selected["alpha"] == alpha].set_index("scenario")
        panel = panel.loc[list(SCENARIO_ORDER)]
        true_half_width = 1.96 * panel["conditional_gain_se"].to_numpy()
        realized_half_width = 1.96 * panel["realized_gain_se"].to_numpy()
        true_gain = panel["conditional_gain_pp"].to_numpy()
        realized_gain = panel["realized_gain_pp"].to_numpy()

        axis.axvline(0, color="#222222", linewidth=0.8)
        axis.errorbar(
            true_gain,
            scenario_positions - 0.11,
            xerr=true_half_width,
            color=BLUE,
            marker="o",
            markersize=4,
            capsize=2,
            linewidth=1.2,
            label="Conditional expected",
        )
        axis.errorbar(
            realized_gain,
            scenario_positions + 0.11,
            xerr=realized_half_width,
            color=ORANGE,
            marker="s",
            markersize=3.7,
            capsize=2,
            linewidth=1.1,
            label="Realized",
        )
        axis.set_title(f"NFR interpolation weight $\\alpha={alpha:.1f}$")
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(
        scenario_positions,
        [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER],
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.56, -0.01),
        ncol=2,
        frameon=False,
    )
    figure.supxlabel("Queue Shift gain (percentage points)", y=0.09)
    figure.subplots_adjust(bottom=0.25, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_macros(release: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Write validation headline quantities for the manuscript."""
    primary = summary[summary["alpha"].isin(PRIMARY_ALPHAS)]
    minimum_true_gain = primary["conditional_gain_pp"].min()
    minimum_realized_gain = primary["realized_gain_pp"].min()
    minimum_acceptance = release.groupby("scenario")["accepted"].mean().min()
    content = "\n".join(
        [
            f"\\newcommand{{\\ValidationScenarios}}{{{summary['scenario'].nunique()}}}",
            f"\\newcommand{{\\ValidationMinTrueGain}}{{{minimum_true_gain:.2f}}}",
            (
                f"\\newcommand{{\\ValidationMinRealizedGain}}"
                f"{{{minimum_realized_gain:.2f}}}"
            ),
            (
                f"\\newcommand{{\\ValidationMinAcceptance}}"
                f"{{{100 * minimum_acceptance:.0f}\\%}}"
            ),
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
    """Generate estimated-model summaries and exhibits."""
    args = parse_args()
    release = pd.read_csv(args.results_dir / "release.csv")
    results = pd.read_csv(args.results_dir / "matched_budget.csv")
    summary = summarize(release, results)
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
