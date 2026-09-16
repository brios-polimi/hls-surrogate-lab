#!/usr/bin/env python3
"""Post-hoc operational analysis and thesis figures for a completed E5c run.

The registered primary tests remain those emitted by ``run_e5c.py``.  This
script adds explicitly descriptive paired comparisons and target-level views;
it does not promote them to preregistered confirmatory claims.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUDGETS = (4, 8, 16, 32, 64)
TARGETS = (
    ("lut", "LUT"),
    ("ff", "FF"),
    ("dsp", "DSP"),
    ("bram", "BRAM"),
    ("cycles_max", "Cycles"),
    ("interval_max", "II"),
)
METHOD_LABELS = {
    "zero_shot": "Zero-shot source model",
    "identity_affine": "Affine calibration",
    "pretrained_residual_ridge": "Pretrained residual ridge",
    "random_encoder_residual_ridge": "Random-feature residual ridge",
    "final_layer_tune": "Final-layer tuning",
    "source_head_tune": "Source-head tuning",
    "fresh_head_standardized_pretrained_encoder": "Fresh head, pretrained encoder",
    "fresh_head_standardized_random_encoder": "Fresh head, random encoder",
    "scratch_full": "Full model from scratch",
    "pretrained_full_tune": "Full pretrained fine-tuning",
}


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _hierarchical_interval(
    frame: pd.DataFrame,
    column: str,
    coverage: float = 95.0,
    replicates: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    groups = {
        architecture: group[column].dropna().to_numpy()
        for architecture, group in frame.groupby("architecture_id")
    }
    architectures = sorted(groups)
    if not architectures or any(len(values) == 0 for values in groups.values()):
        raise ValueError(f"Cannot bootstrap empty groups for {column}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        selected = rng.choice(architectures, len(architectures), replace=True)
        estimates[index] = np.mean([rng.choice(groups[architecture]) for architecture in selected])
    tail = (100.0 - coverage) / 2.0
    low, high = np.percentile(estimates, [tail, 100.0 - tail])
    return float(low), float(high)


def _draw_level(per_run: pd.DataFrame) -> pd.DataFrame:
    return (
        per_run[per_run.method != "zero_shot"]
        .groupby(["architecture_id", "draw_seed", "method", "budget"], as_index=False)
        .mean(numeric_only=True)
    )


def _pair_methods(
    draw_level: pd.DataFrame,
    preferred: str,
    comparator: str,
    budget: int,
    column: str = "smape_overall",
) -> pd.DataFrame:
    preferred_frame = draw_level[
        (draw_level.method == preferred) & (draw_level.budget == budget)
    ][["architecture_id", "draw_seed", column]]
    comparator_frame = draw_level[
        (draw_level.method == comparator) & (draw_level.budget == budget)
    ][["architecture_id", "draw_seed", column]]
    paired = comparator_frame.merge(
        preferred_frame,
        on=["architecture_id", "draw_seed"],
        suffixes=("_comparator", "_preferred"),
        validate="one_to_one",
    )
    paired["advantage_smape"] = (
        paired[f"{column}_comparator"] - paired[f"{column}_preferred"]
    )
    return paired


def _operational_contrasts(draw_level: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows: list[dict] = []
    comparators = (
        "identity_affine",
        "pretrained_residual_ridge",
        "random_encoder_residual_ridge",
        "final_layer_tune",
        "fresh_head_standardized_pretrained_encoder",
        "fresh_head_standardized_random_encoder",
        "scratch_full",
        "pretrained_full_tune",
    )
    for budget in BUDGETS:
        available = set(draw_level[draw_level.budget == budget].method)
        for comparator in comparators:
            if comparator not in available:
                continue
            paired = _pair_methods(draw_level, "source_head_tune", comparator, budget)
            architecture_effect = paired.groupby("architecture_id").advantage_smape.mean()
            low, high = _hierarchical_interval(
                paired,
                "advantage_smape",
                replicates=replicates,
                seed=_seed("operational", comparator, budget),
            )
            rows.append(
                {
                    "preferred_method": "source_head_tune",
                    "comparator_method": comparator,
                    "budget": budget,
                    "advantage_smape": architecture_effect.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "architecture_wins": int((architecture_effect > 0).sum()),
                    "architectures": len(architecture_effect),
                }
            )

    for budget in (32, 64):
        paired = _pair_methods(
            draw_level, "scratch_full", "pretrained_full_tune", budget
        )
        architecture_effect = paired.groupby("architecture_id").advantage_smape.mean()
        low, high = _hierarchical_interval(
            paired,
            "advantage_smape",
            replicates=replicates,
            seed=_seed("scratch-vs-full-tune", budget),
        )
        rows.append(
            {
                "preferred_method": "scratch_full",
                "comparator_method": "pretrained_full_tune",
                "budget": budget,
                "advantage_smape": architecture_effect.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "architecture_wins": int((architecture_effect > 0).sum()),
                "architectures": len(architecture_effect),
            }
        )
    return pd.DataFrame(rows)


def _target_results(
    per_run: pd.DataFrame, draw_level: pd.DataFrame, replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zero = per_run[per_run.method == "zero_shot"].set_index("architecture_id")
    improvement_rows: list[dict] = []
    level_rows: list[dict] = []
    for budget in BUDGETS:
        source = draw_level[
            (draw_level.method == "source_head_tune") & (draw_level.budget == budget)
        ]
        for suffix, label in TARGETS:
            column = f"smape_{suffix}"
            paired = source[["architecture_id", "draw_seed", column]].copy()
            paired["zero_smape"] = paired.architecture_id.map(zero[column])
            paired["advantage_smape"] = paired.zero_smape - paired[column]
            architecture_effect = paired.groupby("architecture_id").advantage_smape.mean()
            low, high = _hierarchical_interval(
                paired,
                "advantage_smape",
                replicates=replicates,
                seed=_seed("target-improvement", budget, suffix),
            )
            improvement_rows.append(
                {
                    "budget": budget,
                    "target": label,
                    "zero_minus_source_head_smape": architecture_effect.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "architecture_wins": int((architecture_effect > 0).sum()),
                    "architectures": len(architecture_effect),
                }
            )

            for method, frame in (("zero_shot", zero.reset_index()), ("source_head_tune", source)):
                mean = frame.groupby("architecture_id")[column].mean().mean()
                level_low, level_high = _hierarchical_interval(
                    frame,
                    column,
                    replicates=replicates,
                    seed=_seed("target-level", method, budget, suffix),
                )
                level_rows.append(
                    {
                        "budget": budget,
                        "target": label,
                        "method": method,
                        "mean_smape": mean,
                        "ci95_low": level_low,
                        "ci95_high": level_high,
                    }
                )
    return pd.DataFrame(improvement_rows), pd.DataFrame(level_rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.pdf")
    figure.savefig(directory / f"{stem}.png", dpi=220)
    plt.close(figure)


def _plot_curves(summary: pd.DataFrame, figure_dir: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    colors = {
        "source_head_tune": "#0072B2",
        "identity_affine": "#E69F00",
        "pretrained_residual_ridge": "#009E73",
        "scratch_full": "#CC79A7",
        "pretrained_full_tune": "#D55E00",
    }
    for method in (
        "source_head_tune",
        "pretrained_residual_ridge",
        "identity_affine",
        "scratch_full",
        "pretrained_full_tune",
    ):
        frame = summary[(summary.method == method) & (summary.scope == "overall")].sort_values("budget")
        axis.errorbar(
            frame.budget,
            frame.mean_smape,
            yerr=[frame.mean_smape - frame.ci95_low, frame.ci95_high - frame.mean_smape],
            color=colors[method],
            marker="o",
            capsize=2.5,
            linewidth=1.7,
            linestyle="--" if method in {"scratch_full", "pretrained_full_tune"} else "-",
            label=METHOD_LABELS[method],
        )
    zero = summary[(summary.method == "zero_shot") & (summary.scope == "overall")].iloc[0]
    axis.axhspan(zero.ci95_low, zero.ci95_high, color="0.75", alpha=0.25)
    axis.axhline(zero.mean_smape, color="0.35", linewidth=1.3, label=METHOD_LABELS["zero_shot"])
    axis.set_xscale("log", base=2)
    axis.set_xticks(BUDGETS, labels=[str(value) for value in BUDGETS])
    axis.set_xlabel("Total target-label budget $k$")
    axis.set_ylabel("Equal-architecture query SMAPE (lower is better)")
    axis.set_title("Honest-budget adaptation on seven exemplar architectures")
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    axis.legend(ncol=2, frameon=False, loc="upper right")
    _save(figure, figure_dir, "e5c_honest_budget_curve")


def _plot_attribution(
    primary: pd.DataFrame, secondary: pd.DataFrame, figure_dir: Path
) -> None:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    neural = primary[primary.claim == "P2_pretrained_vs_random_encoder"].sort_values("budget")
    ridge = secondary[
        secondary.contrast == "pretrained_vs_random_encoder_residual_ridge"
    ].sort_values("budget")
    axis = axes[0]
    axis.errorbar(
        neural.budget,
        neural.advantage_smape,
        yerr=[
            neural.advantage_smape - neural.ci95_low,
            neural.ci95_high - neural.advantage_smape,
        ],
        color="#0072B2",
        marker="o",
        capsize=2.5,
        label="Fresh neural head (95% CI)",
    )
    axis.errorbar(
        neural.budget,
        neural.advantage_smape,
        yerr=[
            neural.advantage_smape - neural.simultaneous_ci99_low,
            neural.simultaneous_ci99_high - neural.advantage_smape,
        ],
        color="#0072B2",
        marker="none",
        linewidth=0.8,
        alpha=0.45,
        capsize=1.5,
        label="99% family-wise interval",
    )
    axis.errorbar(
        ridge.budget,
        ridge.advantage_smape,
        yerr=[ridge.advantage_smape - ridge.ci95_low, ridge.ci95_high - ridge.advantage_smape],
        color="#009E73",
        marker="s",
        capsize=2.5,
        label="Residual ridge (95% CI)",
    )
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_title("Encoder-state attribution")
    axis.set_ylabel("Random $-$ pretrained SMAPE")
    axis.legend(frameon=False, loc="lower right")

    initialization = secondary[
        secondary.contrast == "source_head_initialization_vs_fresh_head"
    ].sort_values("budget")
    axis = axes[1]
    axis.errorbar(
        initialization.budget,
        initialization.advantage_smape,
        yerr=[
            initialization.advantage_smape - initialization.ci95_low,
            initialization.ci95_high - initialization.advantage_smape,
        ],
        color="#D55E00",
        marker="o",
        capsize=2.5,
    )
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_title("Source-head initialization")
    axis.set_ylabel("Fresh head $-$ source head SMAPE")

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, labels=[str(value) for value in BUDGETS])
        axis.set_xlabel("Total target-label budget $k$")
        axis.grid(axis="y", color="0.9", linewidth=0.7)
    _save(figure, figure_dir, "e5c_attribution_decomposition")


def _plot_targets(levels: pd.DataFrame, figure_dir: Path) -> None:
    _style()
    frame = levels[levels.budget == 64]
    target_order = [label for _, label in TARGETS]
    positions = np.arange(len(target_order))
    width = 0.36
    figure, axis = plt.subplots(figsize=(6.6, 3.5))
    for offset, method, color in (
        (-width / 2, "zero_shot", "0.55"),
        (width / 2, "source_head_tune", "#0072B2"),
    ):
        selected = frame[frame.method == method].set_index("target").loc[target_order]
        axis.bar(
            positions + offset,
            selected.mean_smape,
            width,
            color=color,
            label=METHOD_LABELS[method],
            yerr=[
                selected.mean_smape - selected.ci95_low,
                selected.ci95_high - selected.mean_smape,
            ],
            capsize=2,
        )
    axis.set_xticks(positions, target_order)
    axis.set_ylabel("Equal-architecture query SMAPE")
    axis.set_title("Target profile at $k=64$")
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    axis.legend(frameon=False)
    _save(figure, figure_dir, "e5c_k64_target_profile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    analysis_dir = result_dir / "analysis"
    output_dir = (args.output_dir or analysis_dir / "postmortem").resolve()
    figure_dir = (args.figure_dir or output_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run = pd.read_csv(analysis_dir / "per_run_metrics.csv")
    draw_level = _draw_level(per_run)
    operational = _operational_contrasts(draw_level, args.bootstrap_replicates)
    improvements, levels = _target_results(
        per_run, draw_level, args.bootstrap_replicates
    )
    operational.to_csv(output_dir / "operational_contrasts.csv", index=False)
    improvements.to_csv(output_dir / "target_improvements.csv", index=False)
    levels.to_csv(output_dir / "target_levels.csv", index=False)

    summary = pd.read_csv(analysis_dir / "equal_architecture_summary.csv")
    primary = pd.read_csv(analysis_dir / "primary_contrasts.csv")
    secondary = pd.read_csv(analysis_dir / "secondary_contrasts.csv")
    _plot_curves(summary, figure_dir)
    _plot_attribution(primary, secondary, figure_dir)
    _plot_targets(levels, figure_dir)
    print(f"Wrote post-mortem analysis to {output_dir}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
