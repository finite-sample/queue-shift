"""Summarize where exact constrained assignment beats simpler update rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import pandas as pd

from experiments.plot_style import (
    BLUE,
    GRAY,
    GREEN,
    LIGHT_GRAY,
    ORANGE,
    apply_plot_style,
)

mpl.use("Agg")
import matplotlib.pyplot as plt

apply_plot_style(mpl)

SCENARIO_ORDER = ("stable", "confused_pair", "stale_priors", "specialist_queues")
SCENARIO_LABELS = {
    "stable": "Stable",
    "confused_pair": "Confused pair",
    "stale_priors": "Stale priors",
    "specialist_queues": "15 specialist queues",
}
REGIME_ORDER = ("large_stable", "medium_stable", "medium_volatile", "small_volatile")
REGIME_LABELS = {
    "large_stable": "2,000, fixed mix",
    "medium_stable": "300, fixed mix",
    "medium_volatile": "300, volatile mix",
    "small_volatile": "100, volatile mix",
}
FRONTIER_REGIME = "medium_stable"
PRIMARY_ALPHA = 0.5


def add_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    """Add batch-level contrasts in percentage points of cases."""
    out = results.copy()
    for arm in ("operational", "joint", "offsets"):
        out[f"{arm}_true_gain_pp"] = 100 * (
            out[f"{arm}_true_value"] - out["baseline_true_value"]
        )
        out[f"{arm}_realized_gain_pp"] = 100 * (
            out[f"{arm}_accuracy"] - out["baseline_accuracy"]
        )
    out["offsets_overshoot_pp"] = 100 * (
        out["offsets_move_share"] - out["baseline_move_share"]
    )
    out["offsets_over_budget"] = (
        out["offsets_moved_load"] > out["baseline_moved_load"]
    ).astype(float)
    out["offsets_suboptimality_pp"] = 100 * (
        out["offsets_shift_optimum_true_value"] - out["offsets_true_value"]
    )
    return out


def check_invariants(results: pd.DataFrame) -> None:
    """Enforce the guarantees every exact arm must satisfy."""
    if (results["operational_moved_load"] > results["baseline_moved_load"]).any():
        raise RuntimeError("shift-only assignment exceeded its movement budget")
    if (results["predicted_gain_pp"] < -1e-9).any():
        raise RuntimeError("shift-only score dominance failed")
    joint = results.dropna(subset=["joint_moved_load"])
    if (joint["joint_moved_load"] > joint["baseline_moved_load"]).any():
        raise RuntimeError("joint assignment exceeded its movement budget")
    # flip_tolerance at its largest weight sum: one unit per case.
    slack = 2e-6 * (1 + joint["n_cases"])
    if (joint["joint_expected_flips"] > joint["flip_budget"] + slack).any():
        raise RuntimeError("joint assignment exceeded its flip budget")
    if (joint["joint_predicted_gain_pp"] < -1e-9).any():
        raise RuntimeError("joint score dominance failed")
    endpoint = results[results["alpha"] == 1.0]
    if (endpoint["predicted_gain_pp"].abs() > 1e-7).any():
        raise RuntimeError("raw-candidate endpoint is not the required score null")


MEASURES = (
    "raw_move_share",
    "baseline_move_share",
    "offsets_move_share",
    "baseline_nfr",
    "operational_nfr",
    "joint_nfr",
    "offsets_nfr",
    "baseline_true_value",
    "operational_true_value",
    "joint_true_value",
    "offsets_true_value",
    "operational_true_gain_pp",
    "joint_true_gain_pp",
    "offsets_true_gain_pp",
    "operational_realized_gain_pp",
    "joint_realized_gain_pp",
    "offsets_overshoot_pp",
    "offsets_over_budget",
    "offsets_suboptimality_pp",
)


def summarize(release: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Average batches within each model pair, then summarize across pairs."""
    if results.empty:
        raise ValueError("no update passed the release gate")
    check_invariants(results)
    data = add_contrasts(results)
    keys = ["scenario", "regime", "alpha"]
    per_pair = data.groupby([*keys, "repetition"], as_index=False)[
        list(MEASURES)
    ].mean()
    grouped = per_pair.groupby(keys)
    summary = grouped[list(MEASURES)].mean()
    errors = grouped[list(MEASURES)].sem().add_suffix("_se")
    summary = summary.join(errors).reset_index()
    summary["pairs"] = grouped.size().to_numpy()
    summary["acceptance_rate"] = summary["scenario"].map(
        release.groupby("scenario")["accepted"].mean()
    )
    summary["scenario"] = pd.Categorical(summary["scenario"], SCENARIO_ORDER, True)
    summary["regime"] = pd.Categorical(summary["regime"], REGIME_ORDER, True)
    summary = summary.sort_values(keys).reset_index(drop=True)
    summary["scenario"] = summary["scenario"].astype(str)
    summary["regime"] = summary["regime"].astype(str)
    return summary


def load_lock_cost(results: pd.DataFrame) -> pd.DataFrame:
    """Return the true-value cost of holding incumbent totals, by scenario."""
    keys = ["scenario", "regime", "repetition", "batch"]
    locked = results[results["alpha"] == 0.0].set_index(keys)
    raw = results[results["alpha"] == 1.0].set_index(keys)
    batch = pd.DataFrame(
        {
            "cost_pp": 100
            * (raw["baseline_true_value"] - locked["operational_true_value"]),
            "update_pp": 100
            * (raw["baseline_true_value"] - locked["baseline_true_value"]),
        }
    ).reset_index()
    per_pair = batch.groupby(["scenario", "regime", "repetition"], as_index=False)[
        ["cost_pp", "update_pp"]
    ].mean()
    out = per_pair.groupby(["scenario", "regime"], as_index=False).agg(
        cost_pp=("cost_pp", "mean"),
        cost_se=("cost_pp", "sem"),
        update_pp=("update_pp", "mean"),
    )
    out["kept_share"] = 1 - out["cost_pp"] / out["update_pp"]
    return out


def write_frontier(summary: pd.DataFrame, path: Path) -> None:
    """Plot value and NFR against queue movement for every arm."""
    data = summary[summary["regime"] == FRONTIER_REGIME]
    figure, axes = plt.subplots(
        2, len(SCENARIO_ORDER), figsize=(7.2, 4.4), sharex="col", sharey="row"
    )
    arms = (
        ("baseline", "baseline_move_share", "NFR interpolation", GRAY, "o", "-"),
        ("operational", "baseline_move_share", "Queue Shift", BLUE, "s", "-"),
        ("joint", "baseline_move_share", "Queue Shift + flips", GREEN, "D", "--"),
        ("offsets", "offsets_move_share", "Historical prices", ORANGE, "^", ":"),
    )
    for column, scenario in enumerate(SCENARIO_ORDER):
        panel = data[data["scenario"] == scenario].sort_values("alpha")
        incumbent = panel.loc[panel["alpha"] == 0.0, "baseline_true_value"].iloc[0]
        for arm, shift, label, color, marker, style in arms:
            x = 100 * panel[shift]
            value = 100 * (panel[f"{arm}_true_value"] - incumbent)
            nfr = 100 * panel[f"{arm}_nfr"]
            for row, y in enumerate((value, nfr)):
                axes[row, column].plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linestyle=style,
                    linewidth=1.1,
                    label=label,
                )
        axes[0, column].set_title(SCENARIO_LABELS[scenario])
        for axis in axes[:, column]:
            axis.grid(color=LIGHT_GRAY, linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
    figure.supxlabel("Queue shift (% of cases)", y=0.08)
    axes[0, 0].set_ylabel("True accuracy gain\nover incumbent (pp)")
    axes[1, 0].set_ylabel("Negative-flip rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.subplots_adjust(bottom=0.2, hspace=0.12, wspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_mix_gap(summary: pd.DataFrame, path: Path) -> None:
    """Plot matched-budget gains against the candidate's raw queue movement."""
    data = summary[summary["alpha"] == PRIMARY_ALPHA]
    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    markers = dict(zip(SCENARIO_ORDER, ("o", "s", "D", "^"), strict=True))
    for arm, color, label in (
        ("operational", BLUE, "Queue Shift (matched shift)"),
        ("joint", GREEN, "Queue Shift + flips (matched shift and flips)"),
    ):
        for scenario in SCENARIO_ORDER:
            panel = data[data["scenario"] == scenario].dropna(
                subset=[f"{arm}_true_gain_pp"]
            )
            axis.errorbar(
                100 * panel["raw_move_share"],
                panel[f"{arm}_true_gain_pp"],
                yerr=1.96 * panel[f"{arm}_true_gain_pp_se"],
                color=color,
                marker=markers[scenario],
                markersize=4,
                linestyle="none",
                capsize=1.5,
                linewidth=0.9,
                label=label if scenario == SCENARIO_ORDER[0] else None,
            )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_xlabel("Raw candidate's queue shift (% of cases)")
    axis.set_ylabel(f"Gain over interpolation at $\\alpha={PRIMARY_ALPHA}$ (pp)")
    axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_offsets_table(summary: pd.DataFrame, path: Path) -> None:
    """Tabulate how historical prices compare with exact per-batch solutions."""
    data = summary[summary["alpha"] == PRIMARY_ALPHA]
    rows = [
        f"{SCENARIO_LABELS[row.scenario]} & {REGIME_LABELS[row.regime]} & "
        f"{100 * row.baseline_move_share:.1f} & "
        f"{row.offsets_overshoot_pp:.2f} & "
        f"{100 * row.offsets_over_budget:.0f} & "
        f"{row.offsets_suboptimality_pp:.2f} \\\\"
        for row in data.itertuples(index=False)
    ]
    content = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Historical queue prices versus exact per-batch assignment}",
            r"\label{tab:offsets}",
            r"\scriptsize",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Scenario & Batch & Budget & Overshoot & Over budget & Value gap \\",
            r" & & (\% cases) & (pp of cases) & (\% batches) & (pp) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            (
                r"\footnotesize\emph{Note:} Budget is NFR interpolation's queue shift "
                rf"at $\alpha={PRIMARY_ALPHA}$. Prices are learned once on a "
                r"2,000-case history at the same interpolation weight and applied case "
                r"by case. Overshoot is the prices' extra queue shift; value gap is the "
                r"true-accuracy shortfall against the exact assignment at the "
                r"movement the prices actually used."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_lock_table(lock: pd.DataFrame, path: Path) -> None:
    """Tabulate what holding incumbent queue totals costs, by scenario and batch."""
    table = lock.copy()
    table["scenario"] = pd.Categorical(table["scenario"], SCENARIO_ORDER, True)
    table["regime"] = pd.Categorical(table["regime"], REGIME_ORDER, True)
    table = table.sort_values(["scenario", "regime"])
    rows = [
        f"{SCENARIO_LABELS[row.scenario]} & {REGIME_LABELS[row.regime]} & "
        f"{row.update_pp:.2f} & {row.cost_pp:.2f} [{row.cost_pp - 1.96 * row.cost_se:.2f}, "
        f"{row.cost_pp + 1.96 * row.cost_se:.2f}] & {100 * row.kept_share:.0f} \\\\"
        for row in table.itertuples(index=False)
    ]
    content = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Cost of holding every incumbent queue total}",
            r"\label{tab:lock-cost}",
            r"\scriptsize",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Scenario & Batch & Raw update & Cost of zero movement & Update kept \\",
            r" & & (pp) & (pp) [95\% CI] & (\%) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            (
                r"\footnotesize\emph{Note:} Raw update is the raw candidate's true "
                r"accuracy gain over the incumbent. Cost of zero movement is the raw "
                r"candidate's true accuracy minus that of Queue Shift at $B=0$. True "
                r"accuracy uses the simulation's conditional probabilities."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_macros(summary: pd.DataFrame, lock: pd.DataFrame, path: Path) -> None:
    """Write headline quantities for the manuscript."""
    reorder = ("stable", "specialist_queues")
    volume = ("confused_pair", "stale_priors")
    fixed = ("large_stable", "medium_stable")
    reorder_lock = lock[lock["scenario"].isin(reorder) & lock["regime"].isin(fixed)]
    volume_lock = lock[lock["scenario"].isin(volume)]
    small_lock = lock[
        lock["scenario"].isin(reorder) & (lock["regime"] == "small_volatile")
    ]
    joint = summary.dropna(subset=["joint_true_gain_pp"])
    joint_quarter = joint[joint["alpha"] == 0.25]
    joint_half = joint[joint["alpha"] == PRIMARY_ALPHA]
    interior = joint[(joint["alpha"] > 0) & (joint["alpha"] < 1)]
    zero = summary[summary["alpha"] == 0.0]
    raw = summary[summary["alpha"] == 1.0].set_index(["scenario", "regime"])
    zero_ratio = (
        zero.set_index(["scenario", "regime"])["operational_nfr"] / raw["baseline_nfr"]
    )
    half = summary[summary["alpha"] == PRIMARY_ALPHA]
    values = {
        "RegimeRawShiftMin": f"{100 * half['raw_move_share'].min():.0f}",
        "RegimeRawShiftMax": f"{100 * half['raw_move_share'].max():.0f}",
        "LockReorderMin": f"{reorder_lock['cost_pp'].min():.1f}",
        "LockReorderMax": f"{reorder_lock['cost_pp'].max():.1f}",
        "LockReorderKeptMin": f"{100 * reorder_lock['kept_share'].min():.0f}",
        "LockVolumeMin": f"{volume_lock['cost_pp'].min():.1f}",
        "LockVolumeMax": f"{volume_lock['cost_pp'].max():.1f}",
        "LockVolumeKeptMin": f"{100 * volume_lock['kept_share'].min():.0f}",
        "LockVolumeKeptMax": f"{100 * volume_lock['kept_share'].max():.0f}",
        "LockSmallMin": f"{small_lock['cost_pp'].min():.1f}",
        "LockSmallMax": f"{small_lock['cost_pp'].max():.1f}",
        "QSZeroNFRMin": f"{100 * zero['operational_nfr'].min():.0f}",
        "QSZeroNFRMax": f"{100 * zero['operational_nfr'].max():.0f}",
        "QSZeroNFRRawShareMin": f"{100 * zero_ratio.min():.0f}",
        "JointQuarterGainMin": f"{joint_quarter['joint_true_gain_pp'].min():.1f}",
        "JointQuarterGainMax": f"{joint_quarter['joint_true_gain_pp'].max():.1f}",
        "JointHalfGainMin": f"{joint_half['joint_true_gain_pp'].min():.1f}",
        "JointHalfGainMax": f"{joint_half['joint_true_gain_pp'].max():.1f}",
        "JointNFRGapMax": (
            f"{100 * (interior['joint_nfr'] - interior['baseline_nfr']).max():.1f}"
        ),
        "OffsetsOverZeroMin": (f"{zero['offsets_overshoot_pp'].min():.1f}"),
        "OffsetsOverZeroMax": f"{zero['offsets_overshoot_pp'].max():.1f}",
        "OffsetsOverBudgetHalfMin": f"{100 * half['offsets_over_budget'].min():.0f}",
        "OffsetsOverBudgetHalfMax": f"{100 * half['offsets_over_budget'].max():.0f}",
        "OffsetsSuboptMax": (f"{summary['offsets_suboptimality_pp'].max():.1f}"),
    }
    content = "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--lock-out", type=Path)
    parser.add_argument("--lock-tex", type=Path)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--mix-gap", type=Path)
    parser.add_argument("--macros", type=Path)
    return parser.parse_args()


def main() -> None:
    """Generate regime summaries and exhibits."""
    args = parse_args()
    release = pd.read_csv(args.results_dir / "release.csv")
    results = pd.read_csv(args.results_dir / "matched_budget.csv.gz")
    summary = summarize(release, results)
    lock = load_lock_cost(results)
    print(lock.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False, float_format="%.12g")
    if args.lock_out:
        lock.to_csv(args.lock_out, index=False, float_format="%.12g")
    if args.tex:
        write_offsets_table(summary, args.tex)
    if args.lock_tex:
        write_lock_table(lock, args.lock_tex)
    if args.frontier:
        write_frontier(summary, args.frontier)
    if args.mix_gap:
        write_mix_gap(summary, args.mix_gap)
    if args.macros:
        write_macros(summary, lock, args.macros)


if __name__ == "__main__":
    main()
