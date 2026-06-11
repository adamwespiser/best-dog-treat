#!/usr/bin/env python3
"""Generate PDF plots for the dog treat preference experiments."""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.stats import binomtest

from analyze_treats import DEFAULT_DATA_DIR, Trial, fit_bradley_terry, parse_data_dir, predict_prob


DEFAULT_PLOTS_DIR = ROOT / "plots"


def load_treat_metadata(data_dir: Path) -> dict[str, dict[str, str]]:
    path = data_dir / "treats.csv"
    if not path.exists():
        return {}

    metadata: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            clean_row = {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in row.items()
            }
            label = clean_row.get("label")
            if label:
                metadata[label] = clean_row
    return metadata


def options_from_trials(trials: list[Trial]) -> list[str]:
    return sorted({option for trial in trials for option in (trial.first, trial.second)})


def display_label(option: str, metadata: dict[str, dict[str, str]], include_desc: bool = False) -> str:
    meta = metadata.get(option, {})
    brand = meta.get("brand")
    desc = meta.get("desc")

    if brand and include_desc and desc:
        return f"{option}: {brand}\n{desc}"
    if brand:
        return f"{option}: {brand}"
    return option


def treat_key(options: list[str], metadata: dict[str, dict[str, str]]) -> str:
    lines = []
    for option in options:
        meta = metadata.get(option, {})
        brand = meta.get("brand")
        desc = meta.get("desc")
        if brand and desc:
            lines.append(f"{option}: {brand} ({desc})")
        elif brand:
            lines.append(f"{option}: {brand}")
        else:
            lines.append(option)
    return "\n".join(lines)


def winning_hand(trial: Trial) -> str:
    if trial.winner == trial.first:
        return trial.first_hand
    return trial.second_hand


def head_to_head_matrix(trials: list[Trial], options: list[str]) -> np.ndarray:
    index = {option: i for i, option in enumerate(options)}
    matrix = np.zeros((len(options), len(options)), dtype=int)

    for trial in trials:
        winner_index = index[trial.winner]
        loser_index = index[trial.loser]
        matrix[winner_index, loser_index] += 1

    return matrix


def trials_per_pair_matrix(trials: list[Trial], options: list[str]) -> np.ndarray:
    index = {option: i for i, option in enumerate(options)}
    matrix = np.zeros((len(options), len(options)), dtype=int)

    for trial in trials:
        first_index = index[trial.first]
        second_index = index[trial.second]
        matrix[first_index, second_index] += 1
        matrix[second_index, first_index] += 1

    return matrix


def bootstrap_scores(
    trials: list[Trial],
    options: list[str],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, list[float]], Counter[str]]:
    rng = random.Random(seed)
    score_samples: defaultdict[str, list[float]] = defaultdict(list)
    best_counts: Counter[str] = Counter()

    for _ in range(n_boot):
        sample = [rng.choice(trials) for _ in trials]
        scores = fit_bradley_terry(sample, options)

        for option, score in scores.items():
            score_samples[option].append(score)

        best = max(scores.items(), key=lambda item: item[1])[0]
        best_counts[best] += 1

    return dict(score_samples), best_counts


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_score_intervals(
    ax: plt.Axes,
    scores: dict[str, float],
    score_samples: dict[str, list[float]],
    metadata: dict[str, dict[str, str]],
    include_desc: bool,
) -> None:
    ranked = sorted(scores, key=scores.get, reverse=True)
    y = np.arange(len(ranked))
    intervals = {
        option: np.percentile(np.array(score_samples[option]), [2.5, 50, 97.5])
        for option in ranked
    }
    lows = np.array([intervals[option][0] for option in ranked])
    mids = np.array([intervals[option][1] for option in ranked])
    highs = np.array([intervals[option][2] for option in ranked])

    ax.errorbar(
        mids,
        y,
        xerr=np.vstack((mids - lows, highs - mids)),
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        elinewidth=3,
        capsize=4,
    )
    ax.axvline(0, color="#7f7f7f", linewidth=1, linestyle="--")
    ax.set_yticks(y, [display_label(option, metadata, include_desc) for option in ranked])
    ax.invert_yaxis()
    ax.set_xlabel("Bradley-Terry preference score")
    ax.set_title("Modeled Treat Preference")
    ax.grid(axis="x", color="#d9d9d9")


def plot_chance_best(
    ax: plt.Axes,
    options: list[str],
    best_counts: Counter[str],
    n_boot: int,
    metadata: dict[str, dict[str, str]],
    include_desc: bool,
) -> None:
    ranked = sorted(options, key=lambda option: best_counts[option], reverse=True)
    y = np.arange(len(ranked))
    values = np.array([best_counts[option] / n_boot for option in ranked])
    colors = ["#2ca25f" if value == values.max() else "#bdbdbd" for value in values]

    ax.barh(y, values, color=colors)
    ax.set_yticks(y, [display_label(option, metadata, include_desc) for option in ranked])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Bootstrap probability")
    ax.set_title("Chance Each Treat Is Best")
    ax.grid(axis="x", color="#d9d9d9")

    for y_pos, value in zip(y, values):
        ax.text(min(value + 0.025, 0.98), y_pos, f"{value:.1%}", va="center", fontsize=9)


def plot_head_to_head(ax: plt.Axes, matrix: np.ndarray, options: list[str]) -> None:
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="wins")
    ax.set_xticks(np.arange(len(options)), options)
    ax.set_yticks(np.arange(len(options)), options)
    ax.set_xlabel("Column treat")
    ax.set_ylabel("Row treat")
    ax.set_title("Head-to-Head Results: Row Beats Column")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            label = "-" if row == col else str(matrix[row, col])
            color = "white" if matrix[row, col] > matrix.max() * 0.55 else "#252525"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=10)


def plot_hand_bias(ax: plt.Axes, trials: list[Trial]) -> None:
    counts = Counter(winning_hand(trial) for trial in trials)
    labels = ["Left hand", "Right hand"]
    values = np.array([counts["left"], counts["right"]])
    rates = values / values.sum()
    p_value = binomtest(values[0], int(values.sum()), 0.5).pvalue

    bars = ax.bar(labels, rates, color=["#756bb1", "#fdae6b"])
    ax.axhline(0.5, color="#7f7f7f", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of included wins")
    ax.set_title("Winning Hand Diagnostic")
    ax.text(
        0.5,
        0.95,
        f"Two-sided binomial p={p_value:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4d4d4d",
    )

    for bar, count, rate in zip(bars, values, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.035, 0.96),
            f"{count} wins\n{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_pairwise_probabilities(ax: plt.Axes, scores: dict[str, float], options: list[str]) -> None:
    matrix = np.empty((len(options), len(options)))
    for row, row_option in enumerate(options):
        for col, col_option in enumerate(options):
            if row == col:
                matrix[row, col] = np.nan
            else:
                matrix[row, col] = predict_prob(scores[row_option], scores[col_option])

    display_matrix = np.nan_to_num(matrix, nan=0.5)
    image = ax.imshow(display_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="model probability")
    ax.set_xticks(np.arange(len(options)), options)
    ax.set_yticks(np.arange(len(options)), options)
    ax.set_xlabel("Column treat")
    ax.set_ylabel("Row treat")
    ax.set_title("Model-Implied Pairwise Win Probability")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row == col:
                label = "-"
            else:
                label = f"{matrix[row, col]:.0%}"
            ax.text(col, row, label, ha="center", va="center", color="#252525", fontsize=9)


def plot_trials_per_pair(ax: plt.Axes, matrix: np.ndarray, options: list[str]) -> None:
    image = ax.imshow(matrix, cmap="PuBu", vmin=0)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="included trials")
    ax.set_xticks(np.arange(len(options)), options)
    ax.set_yticks(np.arange(len(options)), options)
    ax.set_xlabel("Treat")
    ax.set_ylabel("Treat")
    ax.set_title("Trial Coverage by Pair")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            label = "-" if row == col else str(matrix[row, col])
            color = "white" if matrix[row, col] > matrix.max() * 0.55 else "#252525"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=10)


def plot_raw_win_loss(
    ax: plt.Axes,
    trials: list[Trial],
    scores: dict[str, float],
    metadata: dict[str, dict[str, str]],
) -> None:
    wins = Counter(trial.winner for trial in trials)
    losses = Counter(trial.loser for trial in trials)
    ranked = sorted(scores, key=scores.get, reverse=True)
    y = np.arange(len(ranked))
    win_values = np.array([wins[option] for option in ranked])
    loss_values = np.array([losses[option] for option in ranked])

    ax.barh(y, win_values, color="#2ca25f", label="Wins")
    ax.barh(y, loss_values, left=win_values, color="#d9d9d9", label="Losses")
    ax.set_yticks(y, [display_label(option, metadata, include_desc=True) for option in ranked])
    ax.invert_yaxis()
    ax.set_xlabel("Included trials")
    ax.set_title("Raw Win/Loss Record")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#e5e5e5")

    for y_pos, wins_count, losses_count in zip(y, win_values, loss_values):
        total = wins_count + losses_count
        rate = wins_count / total if total else 0
        ax.text(total + 0.2, y_pos, f"{wins_count}-{losses_count} ({rate:.0%})", va="center", fontsize=9)


def plot_cumulative_win_rate(
    ax: plt.Axes,
    trials: list[Trial],
    options: list[str],
    metadata: dict[str, dict[str, str]],
) -> None:
    ordered_trials = sorted(trials, key=lambda trial: (str(trial.source), trial.day, trial.trial))
    wins = Counter()
    appearances = Counter()
    x_values = np.arange(1, len(ordered_trials) + 1)
    series = {option: [] for option in options}

    for trial in ordered_trials:
        appearances[trial.first] += 1
        appearances[trial.second] += 1
        wins[trial.winner] += 1

        for option in options:
            if appearances[option]:
                series[option].append(wins[option] / appearances[option])
            else:
                series[option].append(np.nan)

    for option in options:
        ax.plot(x_values, series[option], marker="o", linewidth=1.8, label=display_label(option, metadata))

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Included trial number")
    ax.set_ylabel("Cumulative win rate")
    ax.set_title("Cumulative Raw Win Rate")
    ax.grid(color="#e5e5e5")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)


def add_key_footer(fig: plt.Figure, options: list[str], metadata: dict[str, dict[str, str]]) -> None:
    if metadata:
        fig.text(0.01, 0.01, treat_key(options, metadata), ha="left", va="bottom", fontsize=7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PDF plots for dog treat experiments.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing experiment files and optional treats.csv metadata.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory for generated PDF files. Defaults to ./plots.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap iterations for interval and chance-best plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for bootstrap sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be at least 1 for plotting uncertainty.")

    trials, excluded, warnings = parse_data_dir(args.data_dir)
    for warning in warnings:
        print(f"Warning: {warning}")

    metadata = load_treat_metadata(args.data_dir)
    options = options_from_trials(trials)
    scores = fit_bradley_terry(trials, options)
    score_samples, best_counts = bootstrap_scores(trials, options, args.bootstrap, args.seed)
    h2h = head_to_head_matrix(trials, options)
    pair_trials = trials_per_pair_matrix(trials, options)

    plt.style.use("seaborn-v0_8-whitegrid")
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_score_intervals(ax, scores, score_samples, metadata, include_desc=True)
    path = args.plots_dir / "bradley_terry_scores.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_chance_best(ax, options, best_counts, args.bootstrap, metadata, include_desc=True)
    path = args.plots_dir / "bootstrap_chance_best.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    plot_head_to_head(ax, h2h, options)
    add_key_footer(fig, options, metadata)
    path = args.plots_dir / "head_to_head_heatmap.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    plot_hand_bias(ax, trials)
    path = args.plots_dir / "hand_bias.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    plot_pairwise_probabilities(ax, scores, options)
    add_key_footer(fig, options, metadata)
    path = args.plots_dir / "pairwise_probabilities.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    plot_trials_per_pair(ax, pair_trials, options)
    add_key_footer(fig, options, metadata)
    path = args.plots_dir / "trials_per_pair.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    plot_raw_win_loss(ax, trials, scores, metadata)
    path = args.plots_dir / "raw_win_loss.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    plot_cumulative_win_rate(ax, trials, options, metadata)
    path = args.plots_dir / "cumulative_win_rate.pdf"
    save_fig(fig, path)
    generated.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), constrained_layout=True)
    plot_score_intervals(axes[0, 0], scores, score_samples, metadata, include_desc=False)
    plot_chance_best(axes[0, 1], options, best_counts, args.bootstrap, metadata, include_desc=False)
    plot_head_to_head(axes[1, 0], h2h, options)
    plot_hand_bias(axes[1, 1], trials)
    fig.suptitle(
        f"Dog Treat Preference Experiment ({len(trials)} included trials, {len(excluded)} excluded)",
        fontsize=16,
        fontweight="bold",
    )
    add_key_footer(fig, options, metadata)
    path = args.plots_dir / "presentation_dashboard.pdf"
    save_fig(fig, path)
    generated.append(path)

    print("Generated PDF plots:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
