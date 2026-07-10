"""
Appendix C proof-of-principle: a four-rule Potts-style extension.

This script keeps the extension's scope deliberately small. It does not replace 
the main balance-scale pymdp model. Instead, it asks whether the same
self-consistency logic used in the two-rule derivation can be implemented with
four competing rules whose evidence counts are updated through Dirichlet learning.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


### DT --> Keep Appendix C outputs separate from the main figures (in manuscrit)
OUT_DIR = Path(__file__).resolve().parent / "Figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RULE_LABELS = ["Rule I", "Rule II", "Rule III", "Rule IV"]
ITEM_LABELS = ["Weight", "Distance", "Conflict", "Torque"]


@dataclass(frozen=True)
class PottsExtensionConfig:
    n_trials: int = 640
    n_agents: int = 180
    gamma: float = 3.3
    alpha_0: float = 1.0
    pref_correct: float = 5.4
    novelty_weight: float = 0.24
    smooth_window: int = 25
    complexity_cost: tuple[float, float, float, float] = (0.00, 0.18, 0.34, 0.50)
    seed: int = 90210


TRUE_SUCCESS = np.array(
    [
        [0.96, 0.50, 0.35, 0.45],
        [0.93, 0.88, 0.48, 0.55],
        [0.90, 0.86, 0.64, 0.63],
        [0.82, 0.82, 0.96, 0.95],
    ],
    dtype=float,
)


CURRICULUM_TIMES = np.array([0.00, 0.30, 0.62, 1.00], dtype=float)
CURRICULUM_MIXES = np.array(
    [
        [0.92, 0.06, 0.015, 0.005],
        [0.54, 0.33, 0.10, 0.03],
        [0.28, 0.25, 0.35, 0.12],
        [0.14, 0.16, 0.43, 0.27],
    ],
    dtype=float,
)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


def curriculum_mix(progress: float) -> np.ndarray:
    vals = np.array(
        [
            np.interp(progress, CURRICULUM_TIMES, CURRICULUM_MIXES[:, idx])
            for idx in range(CURRICULUM_MIXES.shape[1])
        ],
        dtype=float,
    )
    return vals / vals.sum()


def trailing_average_matrix(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=float)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        out[idx] = values[start : idx + 1].mean(axis=0)
    return out


def policy_posterior(p_correct: np.ndarray, counts: np.ndarray, mix: np.ndarray, config: PottsExtensionConfig) -> np.ndarray:
    expected_success = p_correct @ mix
    expected_ambiguity = binary_entropy(p_correct) @ mix
    concentration = counts.sum(axis=2)
    novelty = (1.0 / np.maximum(concentration, 1e-9)) @ mix
    complexity = np.array(config.complexity_cost, dtype=float)
    efe = (
        -config.pref_correct * expected_success
        + expected_ambiguity
        + complexity
        - config.novelty_weight * novelty
    )
    return softmax(-config.gamma * efe)


def run_agent(config: PottsExtensionConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.ones((4, 4, 2), dtype=float) * config.alpha_0
    choices = np.zeros(config.n_trials, dtype=int)
    items = np.zeros(config.n_trials, dtype=int)
    q_rules = np.zeros((config.n_trials, 4), dtype=float)
    learned = np.zeros((config.n_trials, 4, 4), dtype=float)

    for trial in range(config.n_trials):
        progress = trial / max(config.n_trials - 1, 1)
        mix = curriculum_mix(progress)
        p_correct = counts[:, :, 0] / counts.sum(axis=2)
        q = policy_posterior(p_correct, counts, mix, config)
        rule = int(rng.choice(4, p=q))
        item = int(rng.choice(4, p=mix))
        correct = int(rng.random() < TRUE_SUCCESS[rule, item])
        counts[rule, item, 0 if correct else 1] += 1.0

        choices[trial] = rule
        items[trial] = item
        q_rules[trial] = q
        learned[trial] = counts[:, :, 0] / counts.sum(axis=2)

    return {"choices": choices, "items": items, "q_rules": q_rules, "learned": learned}


def run_population(config: PottsExtensionConfig) -> list[dict[str, np.ndarray]]:
    return [run_agent(config, config.seed + idx) for idx in range(config.n_agents)]


def summarize_population(histories: list[dict[str, np.ndarray]], config: PottsExtensionConfig) -> dict[str, np.ndarray]:
    trial_rule = np.zeros((config.n_trials, 4), dtype=float)
    q_rules = np.zeros((config.n_trials, 4), dtype=float)
    final_learned = np.zeros((config.n_agents, 4, 4), dtype=float)

    for agent_idx, history in enumerate(histories):
        choices = history["choices"]
        one_hot = np.zeros((config.n_trials, 4), dtype=float)
        one_hot[np.arange(config.n_trials), choices] = 1.0
        trial_rule += one_hot
        q_rules += history["q_rules"]
        final_learned[agent_idx] = history["learned"][-1]

    trial_rule /= config.n_agents
    q_rules /= config.n_agents
    return {
        "rule_use": trailing_average_matrix(trial_rule, config.smooth_window),
        "q_rules": q_rules,
        "final_learned": final_learned.mean(axis=0),
    }


def plot_potts_extension(config: PottsExtensionConfig | None = None) -> Path:
    config = config or PottsExtensionConfig()
    histories = run_population(config)
    summary = summarize_population(histories, config)
    trial_axis = np.arange(config.n_trials)
    curriculum = np.vstack([curriculum_mix(t / max(config.n_trials - 1, 1)) for t in trial_axis])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.stackplot(trial_axis, curriculum.T, labels=ITEM_LABELS, alpha=0.85)
    ax.set_title("(a) Developmental item ecology")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Item probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    ax = axes[0, 1]
    ax.stackplot(trial_axis, summary["rule_use"].T, labels=RULE_LABELS, alpha=0.85)
    ax.set_title("(b) Population rule use")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Smoothed choice fraction")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    smoothed_q = trailing_average_matrix(summary["q_rules"], config.smooth_window)
    for idx, label in enumerate(RULE_LABELS):
        ax.plot(trial_axis, smoothed_q[:, idx], linewidth=1.6, label=label)
    ax.set_title("(c) Population policy posterior")
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(rule)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    ax = axes[1, 1]
    heat = ax.imshow(summary["final_learned"], vmin=0.3, vmax=1.0, cmap="viridis")
    ax.set_title("(d) Final learned P(correct | rule, item)")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(ITEM_LABELS, rotation=25, ha="right")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(RULE_LABELS)
    for row in range(4):
        for col in range(4):
            ax.text(col, row, f"{summary['final_learned'][row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Appendix Figure B1. Potts-style four-rule extension", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "fig_potts_extension.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    out_path = plot_potts_extension()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
