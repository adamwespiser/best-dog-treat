#!/usr/bin/env python3
"""Analyze dog treat preference experiments from files in ./data."""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

try:
    import numpy as np
    from scipy.optimize import minimize
except ImportError as exc:
    raise SystemExit("Install dependencies with: python3 -m pip install numpy scipy") from exc


TRIAL_RE = re.compile(r"^\s*([A-Z])\s*/\s*([A-Z])\s*::\s*([A-ZX])\s*(?:#.*)?$")
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_EXTENSIONS = {".txt", ".md", ".dat"}


@dataclass(frozen=True)
class Trial:
    source: Path
    day: int
    trial: int
    first: str
    second: str
    first_hand: str
    second_hand: str
    winner: str
    loser: str


@dataclass(frozen=True)
class ExcludedTrial:
    source: Path
    day: int
    trial: int
    first: str
    second: str
    first_hand: str
    second_hand: str
    reason: str


def hand_assignment(trial_in_day: int) -> tuple[str, str]:
    """Counterbalance hands by trial order within a day."""
    if trial_in_day % 2 == 1:
        return "right", "left"
    return "left", "right"


def iter_data_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"Data path is not a directory: {data_dir}")

    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    )


def parse_file(path: Path) -> tuple[list[Trial], list[ExcludedTrial], list[str]]:
    trials: list[Trial] = []
    excluded: list[ExcludedTrial] = []
    warnings: list[str] = []

    day = 1
    trial_in_day = 0
    saw_trial_in_block = False

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            if saw_trial_in_block:
                day += 1
                trial_in_day = 0
                saw_trial_in_block = False
            continue

        if line.startswith("#"):
            continue

        match = TRIAL_RE.match(line)
        if not match:
            warnings.append(f"{path}:{line_number}: skipping unparsable line: {raw_line}")
            continue

        first, second, winner = match.groups()
        trial_in_day += 1
        saw_trial_in_block = True
        first_hand, second_hand = hand_assignment(trial_in_day)

        if winner == "X":
            excluded.append(ExcludedTrial(path, day, trial_in_day, first, second, first_hand, second_hand, "X"))
            continue

        if winner not in {first, second}:
            warnings.append(f"{path}:{line_number}: skipping invalid winner: {raw_line}")
            continue

        loser = second if winner == first else first
        trials.append(Trial(path, day, trial_in_day, first, second, first_hand, second_hand, winner, loser))

    return trials, excluded, warnings


def parse_data_dir(data_dir: Path) -> tuple[list[Trial], list[ExcludedTrial], list[str]]:
    trials: list[Trial] = []
    excluded: list[ExcludedTrial] = []
    warnings: list[str] = []

    files = iter_data_files(data_dir)
    if not files:
        raise SystemExit(f"No data files found in {data_dir}")

    for path in files:
        file_trials, file_excluded, file_warnings = parse_file(path)
        trials.extend(file_trials)
        excluded.extend(file_excluded)
        warnings.extend(file_warnings)

    if not trials:
        raise SystemExit(f"No included trials found in {data_dir}")

    return trials, excluded, warnings


def summarize(trials: list[Trial]) -> list[str]:
    options = sorted({option for trial in trials for option in (trial.first, trial.second)})

    wins = Counter(trial.winner for trial in trials)
    losses = Counter(trial.loser for trial in trials)
    h2h: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for trial in trials:
        h2h[trial.winner][trial.loser] += 1

    print("\n=== Overall record ===")
    rows = []
    for option in options:
        win_count = wins[option]
        loss_count = losses[option]
        total = win_count + loss_count
        win_rate = win_count / total if total else 0
        rows.append((option, win_count, loss_count, win_rate))

    for option, win_count, loss_count, win_rate in sorted(rows, key=lambda row: (-row[1], row[2], row[0])):
        print(f"{option}: {win_count:2d}-{loss_count:2d}  win_rate={win_rate:.3f}")

    print("\n=== Head-to-head matrix: row beats column ===")
    print("    " + " ".join(f"{option:>3}" for option in options))
    for row_option in options:
        row = []
        for col_option in options:
            if row_option == col_option:
                row.append("  -")
            else:
                row.append(f"{h2h[row_option][col_option]:3d}")
        print(f"{row_option:>3} " + " ".join(row))

    print("\n=== Pair summaries ===")
    for option_a, option_b in combinations(options, 2):
        a_wins = h2h[option_a][option_b]
        b_wins = h2h[option_b][option_a]
        if a_wins + b_wins > 0:
            leader = option_a if a_wins > b_wins else option_b if b_wins > a_wins else "tie"
            print(f"{option_a}/{option_b}: {option_a} {a_wins} - {b_wins} {option_b}   leader={leader}")

    return options


def summarize_position_bias(trials: list[Trial]) -> None:
    left_wins = sum(
        1
        for trial in trials
        if (trial.winner == trial.first and trial.first_hand == "left")
        or (trial.winner == trial.second and trial.second_hand == "left")
    )
    right_wins = len(trials) - left_wins
    left_rate = left_wins / len(trials)

    print("\n=== Position summary ===")
    print(f"left wins:  {left_wins:2d}  rate={left_rate:.3f}")
    print(f"right wins: {right_wins:2d}  rate={1 - left_rate:.3f}")


def fit_bradley_terry(trials: list[Trial], options: list[str]) -> dict[str, float]:
    idx = {option: index for index, option in enumerate(options)}
    n = len(options)

    def neg_log_likelihood(scores: np.ndarray) -> float:
        log_likelihood = 0.0
        for trial in trials:
            winner_score = scores[idx[trial.winner]]
            loser_score = scores[idx[trial.loser]]
            diff = winner_score - loser_score

            # Stable log(sigmoid(diff)).
            log_likelihood += -math.log1p(math.exp(-diff))

        penalty = 0.01 * float(np.sum(scores**2))
        return -log_likelihood + penalty

    result = minimize(
        neg_log_likelihood,
        x0=np.zeros(n),
        method="BFGS",
    )

    if not result.success:
        raise RuntimeError(f"Bradley-Terry optimization failed: {result.message}")

    scores = result.x
    scores = scores - np.mean(scores)
    return {option: float(scores[idx[option]]) for option in options}


def predict_prob(score_a: float, score_b: float) -> float:
    return 1 / (1 + math.exp(-(score_a - score_b)))


def bootstrap_chance_best(
    trials: list[Trial],
    options: list[str],
    n_boot: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    best_counts: Counter[str] = Counter()
    score_samples: defaultdict[str, list[float]] = defaultdict(list)

    for _ in range(n_boot):
        sample = [rng.choice(trials) for _ in trials]
        scores = fit_bradley_terry(sample, options)

        for option, score in scores.items():
            score_samples[option].append(score)

        best = max(scores.items(), key=lambda item: item[1])[0]
        best_counts[best] += 1

    print("\n=== Bootstrap chance best ===")
    for option, count in best_counts.most_common():
        print(f"{option}: {count / n_boot:.3f}")

    print("\n=== Bootstrap score intervals ===")
    for option in sorted(options):
        vals = np.array(score_samples[option])
        lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
        print(f"{option}: median={mid: .3f}, 95% interval=({lo: .3f}, {hi: .3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dog treat preference trials stored in ./data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing experiment files. Defaults to ./data.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap iterations for chance-best estimates. Use 0 to skip.",
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
    trials, excluded, warnings = parse_data_dir(args.data_dir)

    for warning in warnings:
        print(f"Warning: {warning}")

    print(f"Data directory: {args.data_dir}")
    print(f"Included trials: {len(trials)}")
    print(f"Excluded X trials: {len(excluded)}")

    options = summarize(trials)
    summarize_position_bias(trials)

    scores = fit_bradley_terry(trials, options)

    print("\n=== Bradley-Terry scores ===")
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    for option, score in ranked:
        print(f"{option}: {score: .3f}")

    print("\n=== Model-implied ranking ===")
    print(" > ".join(option for option, _ in ranked))

    print("\n=== Model-implied pairwise probabilities ===")
    ranked_options = [option for option, _ in ranked]
    for option_a, option_b in combinations(ranked_options, 2):
        probability = predict_prob(scores[option_a], scores[option_b])
        print(f"P({option_a} beats {option_b}) = {probability:.3f}")

    if args.bootstrap > 0:
        bootstrap_chance_best(trials, options, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
