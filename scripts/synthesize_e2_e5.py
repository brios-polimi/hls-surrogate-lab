#!/usr/bin/env python3
"""Create compact E2-replication and E5 thesis synthesis tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (7, 42, 137)
SEED_COLORS = {7: "#0072B2", 42: "#E69F00", 137: "#009E73"}
METHOD_STYLE = {
    "affine": ("#D55E00", "o", "-"),
    "head": ("#0072B2", "s", "-"),
    "full": ("#CC79A7", "D", "--"),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("artifacts/results")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/thesis_synthesis_2026-09-11"),
    )
    return parser.parse_args()


def _e2_frame(root: Path) -> pd.DataFrame:
    paths = {
        7: root / "e2_replication_v1/analysis/seed7/architecture_cluster_bootstrap.csv",
        42: root / "e2_structural_v1/e2_structural_v1/analysis_final/architecture_cluster_bootstrap.csv",
        137: root / "e2_replication_v1/analysis/seed137/architecture_cluster_bootstrap.csv",
    }
    rows = []
    for seed, path in paths.items():
        frame = pd.read_csv(path)
        frame = frame[
            frame["kernel_family"].eq("all")
            & frame["candidate"].isin(("fusion", "topology_destroyed", "extra_trees"))
        ].copy()
        frame["seed"] = seed
        rows.append(frame)
    no_local = pd.read_csv(
        root / "e2_replication_v1/analysis/seed42_no_local/architecture_cluster_bootstrap.csv"
    )
    no_local = no_local[
        no_local["kernel_family"].eq("all")
        & no_local["candidate"].eq("no_local_message")
    ].copy()
    no_local["seed"] = 42
    rows.append(no_local)
    return pd.concat(rows, ignore_index=True)


def _write_e2_summary(frame: pd.DataFrame, output: Path) -> None:
    summary = (
        frame.groupby("candidate", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            h0_smape_mean=("reference_macro_smape", "mean"),
            candidate_smape_mean=("candidate_macro_smape", "mean"),
            delta_mean=("delta_macro_smape_candidate_minus_reference", "mean"),
            delta_sd_across_seeds=("delta_macro_smape_candidate_minus_reference", "std"),
            delta_min=("delta_macro_smape_candidate_minus_reference", "min"),
            delta_max=("delta_macro_smape_candidate_minus_reference", "max"),
        )
    )
    summary.to_csv(output / "e2_replication_summary.csv", index=False)
    frame.to_csv(output / "e2_seed_effects.csv", index=False)


def _plot_e2(frame: pd.DataFrame, output: Path) -> None:
    order = ["fusion", "topology_destroyed", "no_local_message", "extra_trees"]
    labels = {
        "fusion": "Fusion",
        "topology_destroyed": "Topology destroyed",
        "no_local_message": "No local messages",
        "extra_trees": "Extra Trees",
    }
    offsets = {7: -0.16, 42: 0.0, 137: 0.16}
    figure, axis = plt.subplots(figsize=(9.2, 5.1))
    for index, candidate in enumerate(order):
        rows = frame[frame["candidate"].eq(candidate)].sort_values("seed")
        for row in rows.itertuples():
            y = len(order) - 1 - index + offsets[int(row.seed)]
            value = row.delta_macro_smape_candidate_minus_reference
            low = row.cluster_bootstrap_ci95_low
            high = row.cluster_bootstrap_ci95_high
            axis.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=SEED_COLORS[int(row.seed)],
                capsize=3,
                markersize=6,
                linewidth=1.5,
                label=f"seed {int(row.seed)}" if index == 0 else None,
            )
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(range(len(order)), [labels[name] for name in reversed(order)])
    axis.set_xlabel("Test SMAPE difference from H0 (points; lower is better)")
    axis.set_title("E2: topology and fusion conclusions replicate across training seeds", loc="left")
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.text(
        0.01,
        -0.20,
        "Bars: 95% architecture-cluster bootstrap intervals, conditional on each trained seed. "
        "No-local-message was run only at seed 42.",
        transform=axis.transAxes,
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(output / "e2_replication_effects.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_e5_curves(summary: pd.DataFrame, output: Path) -> None:
    overall = summary[summary["scope"].eq("overall")].copy()
    zero = overall[overall["method"].eq("zero_shot")].iloc[0]
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    axis.errorbar(
        [0], [zero.equal_architecture_smape],
        yerr=[[zero.equal_architecture_smape - zero.ci95_low],
              [zero.ci95_high - zero.equal_architecture_smape]],
        fmt="o", color="#333333", capsize=4, markersize=7,
    )
    axis.annotate("zero-shot", (0, zero.equal_architecture_smape), xytext=(5, 5),
                  textcoords="offset points")
    for method in ("affine", "head"):
        rows = overall[overall["method"].eq(method)].sort_values("budget")
        x = np.r_[0, rows["budget"].to_numpy(float)]
        y = np.r_[zero.equal_architecture_smape,
                  rows["equal_architecture_smape"].to_numpy(float)]
        color, marker, linestyle = METHOD_STYLE[method]
        axis.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                  linewidth=2, markersize=6)
        axis.errorbar(
            rows["budget"], rows["equal_architecture_smape"],
            yerr=np.vstack((
                rows["equal_architecture_smape"] - rows["ci95_low"],
                rows["ci95_high"] - rows["equal_architecture_smape"],
            )), fmt="none", ecolor=color, capsize=3, alpha=0.75,
        )
        last = rows.iloc[-1]
        axis.annotate(method, (last.budget, last.equal_architecture_smape),
                      xytext=(7, 0), textcoords="offset points", va="center",
                      color=color, fontweight="bold")
    full = overall[overall["method"].eq("full")].iloc[0]
    color, marker, _ = METHOD_STYLE["full"]
    axis.errorbar(
        [full.budget], [full.equal_architecture_smape],
        yerr=[[full.equal_architecture_smape - full.ci95_low],
              [full.ci95_high - full.equal_architecture_smape]],
        fmt=marker, color=color, capsize=4, markersize=7,
    )
    axis.annotate("full fine-tune", (full.budget, full.equal_architecture_smape),
                  xytext=(7, -3), textcoords="offset points", va="center",
                  color=color, fontweight="bold")
    axis.set_xticks([0, 4, 16, 32])
    axis.set_xlabel("Labeled support designs per architecture")
    axis.set_ylabel("Equal-architecture query SMAPE")
    axis.set_title("E5: frozen-encoder head adaptation is strongly label-responsive", loc="left")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.01, -0.19,
        "Points average seven held-out architectures; bars are 95% hierarchical bootstrap intervals "
        "over architectures and repeated support draws.",
        transform=axis.transAxes, fontsize=9, color="#444444",
    )
    figure.tight_layout()
    figure.savefig(output / "e5_label_efficiency.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_e5_tradeoff(
    summary: pd.DataFrame, degradation: pd.DataFrame, output: Path
) -> None:
    query = summary[summary["scope"].eq("overall")].copy()
    source = degradation[degradation["scope"].eq("overall")].copy()
    points = query.merge(source, on=["method", "budget"], suffixes=("_query", "_source"))
    zero = query[query["method"].eq("zero_shot")].iloc[0]
    figure, axis = plt.subplots(figsize=(8.6, 6.0))
    axis.scatter(zero.equal_architecture_smape, 0, color="#333333", marker="o", s=55)
    axis.annotate("zero-shot", (zero.equal_architecture_smape, 0), xytext=(6, 5),
                  textcoords="offset points")
    offsets = {
        ("affine", 4): (-48, 7), ("affine", 16): (-18, 10),
        ("affine", 32): (8, -13), ("head", 4): (7, -12),
        ("head", 16): (7, 5), ("head", 32): (7, 5),
        ("full", 32): (7, 5),
    }
    for row in points.itertuples():
        color, marker, _ = METHOD_STYLE[row.method]
        x = row.equal_architecture_smape
        y = row.source_degradation_smape
        axis.errorbar(
            x, y,
            xerr=[[x - row.ci95_low_query], [row.ci95_high_query - x]],
            yerr=[[y - row.ci95_low_source], [row.ci95_high_source - y]],
            fmt=marker, color=color, capsize=3, markersize=7, alpha=0.9,
        )
        axis.annotate(
            f"{row.method} k={int(row.budget)}", (x, y),
            xytext=offsets[(row.method, int(row.budget))],
            textcoords="offset points", color=color, fontsize=9,
        )
    axis.set_xlabel("Target-architecture query SMAPE (lower is better)")
    axis.set_ylabel("Source-test degradation from adaptation (points; lower is better)")
    axis.set_title("E5: adaptation accuracy trades off against a shared global model", loc="left")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.01, -0.18,
        "Bars are 95% hierarchical bootstrap intervals. Separate architecture-specific heads avoid "
        "overwriting the deployable source checkpoint.",
        transform=axis.transAxes, fontsize=9, color="#444444",
    )
    figure.tight_layout()
    figure.savefig(output / "e5_accuracy_retention_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_e5_targets(summary: pd.DataFrame, output: Path) -> None:
    targets = ["lut", "ff", "dsp", "bram", "cycles_max", "interval_max"]
    labels = ["LUT", "FF", "DSP", "BRAM", "Cycles", "Initiation interval"]
    methods = [("zero_shot", 0), ("full", 32), ("head", 32)]
    y = np.arange(len(targets))[::-1]
    offsets = {("zero_shot", 0): 0.18, ("full", 32): 0, ("head", 32): -0.18}
    colors = {"zero_shot": "#333333", "full": "#CC79A7", "head": "#0072B2"}
    markers = {"zero_shot": "o", "full": "D", "head": "s"}
    figure, axis = plt.subplots(figsize=(9.0, 5.7))
    for method, budget in methods:
        rows = summary[(summary["method"].eq(method)) & (summary["budget"].eq(budget))]
        rows = rows.set_index("scope").loc[targets]
        x = rows["equal_architecture_smape"].to_numpy(float)
        low = rows["ci95_low"].to_numpy(float)
        high = rows["ci95_high"].to_numpy(float)
        label = "zero-shot" if method == "zero_shot" else f"{method} k={budget}"
        axis.errorbar(
            x, y + offsets[(method, budget)], xerr=np.vstack((x-low, high-x)),
            fmt=markers[method], color=colors[method], capsize=2.5, markersize=6,
            linewidth=1.4, label=label,
        )
    axis.set_yticks(y, labels)
    axis.set_xlabel("Equal-architecture query SMAPE")
    axis.set_title("E5: head adaptation improves every target; FF and interval remain hardest", loc="left")
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.text(
        0.01, -0.16,
        "Bars: 95% hierarchical bootstrap intervals across seven architectures and three support draws.",
        transform=axis.transAxes, fontsize=9, color="#444444",
    )
    figure.tight_layout()
    figure.savefig(output / "e5_target_profile.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _e5_pairwise_contrasts(per_run: pd.DataFrame, output: Path) -> None:
    query = per_run[per_run["split"].eq("query")].copy()
    comparisons = [
        (("head", 16), ("head", 4)),
        (("head", 32), ("head", 16)),
        (("affine", 16), ("affine", 4)),
        (("affine", 32), ("affine", 16)),
        (("head", 32), ("full", 32)),
        (("head", 16), ("full", 32)),
    ]
    rows = []
    for comparison_index, (left, right) in enumerate(comparisons):
        left_frame = query[
            query["method"].eq(left[0]) & query["budget"].eq(left[1])
        ][["architecture_id", "draw_seed", "smape_overall"]]
        right_frame = query[
            query["method"].eq(right[0]) & query["budget"].eq(right[1])
        ][["architecture_id", "draw_seed", "smape_overall"]]
        paired = left_frame.merge(
            right_frame, on=["architecture_id", "draw_seed"],
            suffixes=("_left", "_right"), validate="one_to_one",
        )
        paired["delta"] = paired["smape_overall_left"] - paired["smape_overall_right"]
        by_architecture = {
            architecture: group["delta"].to_numpy(float)
            for architecture, group in paired.groupby("architecture_id")
        }
        architectures = sorted(by_architecture)
        rng = np.random.default_rng(20260911 + comparison_index)
        estimates = np.empty(10_000)
        for replicate in range(len(estimates)):
            selected = rng.choice(architectures, len(architectures), replace=True)
            estimates[replicate] = np.mean([
                rng.choice(by_architecture[architecture])
                for architecture in selected
            ])
        architecture_means = paired.groupby("architecture_id")["delta"].mean()
        rows.append({
            "left_method": left[0], "left_budget": left[1],
            "right_method": right[0], "right_budget": right[1],
            "delta_smape_left_minus_right": architecture_means.mean(),
            "ci95_low": np.percentile(estimates, 2.5),
            "ci95_high": np.percentile(estimates, 97.5),
            "architecture_win_fraction_left": (architecture_means < 0).mean(),
        })
    pd.DataFrame(rows).to_csv(output / "e5_pairwise_contrasts.csv", index=False)


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e2 = _e2_frame(args.results_root)
    _write_e2_summary(e2, args.output_dir)
    _plot_e2(e2, args.output_dir)
    analysis = args.results_root / "e5_adaptation_v1/analysis"
    summary = pd.read_csv(analysis / "equal_architecture_summary.csv")
    degradation = pd.read_csv(analysis / "source_degradation.csv")
    per_run = pd.read_csv(analysis / "per_run_metrics.csv")
    _plot_e5_curves(summary, args.output_dir)
    _plot_e5_tradeoff(summary, degradation, args.output_dir)
    _plot_e5_targets(summary, args.output_dir)
    _e5_pairwise_contrasts(per_run, args.output_dir)
    print(f"Wrote synthesis to {args.output_dir}")


if __name__ == "__main__":
    main()
