"""
sim_robustness_appendixB7.py
============================
Appendix B7 ("the transition without external scaffolding")

This script runs the B7 protocol in the *actual* balance-scale engine
(`sim_balance_scale_pymdp.py`), so that the numbers reported in Appendix B7 are 
produced by the same code that produces Figures 4 to 6. 

here is the PROTOCOL:
Base configuration: `_siegler_plot_config()` -- the configuration behind
manuscript Figure 6. Scaffolding is removed by three changes and no others:

    (i)   use_curriculum = False, with a stationary item mixture
    (ii)  policy_prior = late_policy_prior = (0.5, 0.5)   [flat E from trial 1]
    (iii) initial_strategy_prior = (0.5, 0.5)             [flat D over strategy]

Everything else is unchanged (gamma = 14, alpha_action = 7, use_param_info_gain = True,
use_states_info_gain = False, rule2_success = (0.66, 0.87, 0.92),
alpha_outcome = 1.0 (i.e. alpha_0 = 2 per column), lr_pA = 1.0, and the
asymmetric cue emissions).

Analysis uses the SOM conventions: `_trailing_average_matrix` (causal, window 25), 
`_classify_rule_usage` (<0.3 Rule I, >0.7 Rule II), and
`_estimate_transition_trials` (threshold 0.7, window 20).

Usage:  python3 sim_robustness_appendixB7.py [--quick]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import replace

import numpy as np

### DT ---> The engine and the vendored pymdp live alongside this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "third_party")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_balance_scale_pymdp as bs  # noqa: E402

TRACE_DIR = os.path.join(_HERE, "b7_traces")
SMOOTH_WINDOW = 25      # matches _trailing_average_matrix use in Notebook 3
TRANSITION_WINDOW = 20  # matches _estimate_transition_trials in Notebook 3
COMMIT_THRESHOLD = 0.9  # sustained smoothed P(Rule II) defining commitment


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------
def scaffolded_config() -> bs.BalanceScalePymdpConfig:
    """The Figure 6 configuration, unmodified. Serves as the control."""
    return bs._siegler_plot_config()


def robustness_config(problem_mix=(0.50, 0.25, 0.25),
                      conflict_rule_ii: float = 0.92,
                      alpha_outcome: float = 1.0) -> bs.BalanceScalePymdpConfig:
    """Figure 6 configuration with both scaffolds removed."""
    ### DT ---> Only the curriculum, the policy-prior habit, and the initial
    ### DT ---> strategy prior are altered. Everything else is inherited.
    return replace(
        scaffolded_config(),
        rule2_success=(0.66, 0.87, conflict_rule_ii),
        alpha_outcome=alpha_outcome,
        use_curriculum=False,
        problem_mix=problem_mix,
        policy_prior=(0.5, 0.5),
        late_policy_prior=(0.5, 0.5),
        initial_strategy_prior=(0.5, 0.5),
    )


RUNS = {
    ### DT ---> name: (config, n_agents, n_trials, seed_offset)
    "control":  (scaffolded_config(),                              120, 500,  120000),
    "primary":  (robustness_config(),                              120, 2000, 220000),
    "thirds":   (robustness_config((1/3, 1/3, 1/3)),                40, 2000, 330000),
    "conf073":  (robustness_config(conflict_rule_ii=0.73),          40, 2000, 440000),
    "alpha0_4": (robustness_config(alpha_outcome=2.0),              40, 2000, 660000),
    "alpha0_8": (robustness_config(alpha_outcome=4.0),              40, 2000, 770000),
    "alpha0_16": (robustness_config(alpha_outcome=8.0),             40, 2000, 880000),
}


# --------------------------------------------------------------------------
# Simulation (resumable: one compressed .npz per agent)
# --------------------------------------------------------------------------
def simulate(name: str, n_agents: int, n_trials: int) -> None:
    cfg, _, _, seed_offset = (RUNS[name][0], None, None, RUNS[name][3])
    out = os.path.join(TRACE_DIR, name)
    os.makedirs(out, exist_ok=True)
    for i in range(n_agents):
        path = os.path.join(out, f"a{i:03d}.npz")
        if os.path.exists(path):
            continue
        history = bs.run_balance_scale_pymdp(n_trials, config=cfg,
                                             seed=seed_offset + i,
                                             record_steps=False)
        trials = history["trials"]
        np.savez_compressed(
            path,
            p_rule2=np.array([t["p_rule2"] for t in trials], dtype=np.float32),
            problem_type=np.array([t["problem_type"] for t in trials], dtype=np.int8),
            chosen_rule=np.array([t["chosen_rule"] for t in trials], dtype=np.int8),
            correct=np.array([t["correct"] for t in trials], dtype=np.int8),
        )


def load(name: str):
    files = sorted(glob.glob(os.path.join(TRACE_DIR, name, "a*.npz")))
    if not files:
        raise FileNotFoundError(f"No traces for run '{name}'. Run simulate() first.")
    arrays = [np.load(f) for f in files]
    return (np.stack([a["p_rule2"] for a in arrays]).astype(float),
            np.stack([a["problem_type"] for a in arrays]),
            np.stack([a["chosen_rule"] for a in arrays]),
            np.stack([a["correct"] for a in arrays]))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def commitment_trial(smoothed_agent: np.ndarray, to_rule_ii: bool = True):
    """First trial after which smoothed P(Rule II) stays beyond threshold."""
    ### DT ---> A single excursion above 0.9 is not commitment; we require the
    ### DT ---> trajectory never to fall back below 0.85 thereafter.
    series = smoothed_agent if to_rule_ii else 1.0 - smoothed_agent
    for j in np.where(series > COMMIT_THRESHOLD)[0]:
        if (series[j:] > COMMIT_THRESHOLD - 0.05).all():
            return int(j) + 1
    return None


def discrimination_index(problem_type: np.ndarray, chosen_rule: np.ndarray,
                         tail_fraction: float = 0.25) -> np.ndarray:
    """Delta = P(Rule II | hard item) - P(Rule II | Weight item), per agent."""
    n, n_trials = problem_type.shape
    tail = max(int(tail_fraction * n_trials), 1)
    delta = np.full(n, np.nan)
    for i in range(n):
        pt, cr = problem_type[i, -tail:], chosen_rule[i, -tail:]
        if (pt == 0).any() and (pt > 0).any():
            delta[i] = cr[pt > 0].mean() - cr[pt == 0].mean()
    return delta


def _iqr(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)))


def analyse(name: str) -> dict:
    p_rule2, problem_type, chosen_rule, correct = load(name)
    n, n_trials = p_rule2.shape
    smoothed = bs._trailing_average_matrix(p_rule2, window=SMOOTH_WINDOW)

    tail = max(int(0.05 * n_trials), 1)
    final = smoothed[:, -tail:].mean(axis=1)
    rule_ii = np.where(final > 0.7)[0]
    rule_i = np.where(final < 0.3)[0]
    mixed = np.setdiff1d(np.arange(n), np.concatenate([rule_ii, rule_i]))

    delta = discrimination_index(problem_type, chosen_rule)
    locked = delta < 0.2

    acc_tail = max(int(0.25 * n_trials), 1)
    accuracy = correct[:, -acc_tail:].mean(axis=1)

    # rise time of the smoothed policy posterior, 0.2 -> 0.8
    rise = []
    for i in range(n):
        lo = np.where(smoothed[i] > 0.2)[0]
        hi = np.where(smoothed[i] > 0.8)[0]
        if lo.size and hi.size and hi[0] >= lo[0]:
            rise.append(hi[0] - lo[0])

    variance = smoothed.var(axis=0)
    peak = int(np.argmax(variance))

    med2, lo2, hi2 = _iqr([commitment_trial(smoothed[i], True) for i in rule_ii])
    med1, lo1, hi1 = _iqr([commitment_trial(smoothed[i], False) for i in rule_i])
    medr, lor, hir = _iqr(rise)

    def group_accuracy(idx):
        return float(accuracy[idx].mean()) if len(idx) else float("nan")

    res = dict(
        run=name, n_agents=n, n_trials=n_trials,
        prop_rule_ii_locked=len(rule_ii) / n,
        prop_rule_i_locked=len(rule_i) / n,
        prop_cue_contingent=len(mixed) / n,
        commit_trial_rule_ii=[med2, lo2, hi2],
        commit_trial_rule_i=[med1, lo1, hi1],
        rise_time_02_to_08=[medr, lor, hir],
        variance_peak_trial=peak + 1,
        variance_peak_value=float(variance[peak]),
        variance_final=float(variance[-1]),
        mass_in_inaccessible_band_at_peak=float(
            ((smoothed[:, peak] > 0.3) & (smoothed[:, peak] < 0.7)).mean()),
        delta_median=float(np.nanmedian(delta)),
        delta_frac_below_0p2=float(np.nanmean(delta < 0.2)),
        delta_frac_above_0p4=float(np.nanmean(delta > 0.4)),
        delta_mass_between=float(np.nanmean((delta > 0.2) & (delta < 0.4))),
        accuracy_all=float(accuracy.mean()),
        accuracy_rule_locked=group_accuracy(np.where(locked)[0]),
        accuracy_cue_contingent=group_accuracy(np.where(~locked)[0]),
    )

    # accuracy by item type, per regime
    labels = {"rule_ii_locked": rule_ii, "rule_i_locked": rule_i, "cue_contingent": mixed}
    res["accuracy_by_item"] = {}
    res["p_rule_ii_by_item"] = {}
    for label, idx in labels.items():
        if not len(idx):
            continue
        res["accuracy_by_item"][label] = [
            round(float(np.mean([correct[i, -acc_tail:][problem_type[i, -acc_tail:] == pt].mean()
                                 for i in idx])), 4) for pt in range(3)]
        res["p_rule_ii_by_item"][label] = [
            round(float(np.mean([chosen_rule[i, -acc_tail:][problem_type[i, -acc_tail:] == pt].mean()
                                 for i in idx])), 4) for pt in range(3)]
    return res


def report(res: dict) -> None:
    print(f"\n=== {res['run']}  (n = {res['n_agents']}, {res['n_trials']} trials) ===")
    print(f"  Rule II locked      : {res['prop_rule_ii_locked']:.3f}")
    print(f"  Rule I locked       : {res['prop_rule_i_locked']:.3f}")
    print(f"  Cue-contingent      : {res['prop_cue_contingent']:.3f}")
    m, lo, hi = res["commit_trial_rule_ii"]
    print(f"  Commitment trial (Rule II) : {m:.0f}  IQR [{lo:.0f}, {hi:.0f}]")
    m, lo, hi = res["rise_time_02_to_08"]
    print(f"  Rise 0.2 -> 0.8            : {m:.0f}  IQR [{lo:.0f}, {hi:.0f}]")
    print(f"  Variance peak       : t = {res['variance_peak_trial']} "
          f"(var = {res['variance_peak_value']:.4f}); final var = {res['variance_final']:.4f}")
    print(f"  Mass in [0.3, 0.7] at peak : {res['mass_in_inaccessible_band_at_peak']:.3f}")
    print(f"  Discrimination index Delta : median {res['delta_median']:.3f}; "
          f"frac < 0.2 = {res['delta_frac_below_0p2']:.3f}; "
          f"frac > 0.4 = {res['delta_frac_above_0p4']:.3f}; "
          f"mass in (0.2, 0.4) = {res['delta_mass_between']:.3f}")
    print(f"  Accuracy: all {res['accuracy_all']:.4f} | rule-locked "
          f"{res['accuracy_rule_locked']:.4f} | cue-contingent {res['accuracy_cue_contingent']:.4f}")
    for label, vals in res["accuracy_by_item"].items():
        print(f"    accuracy [W, D, C] {label:16s}: {vals}")
    for label, vals in res["p_rule_ii_by_item"].items():
        print(f"    P(Rule II) [W, D, C] {label:16s}: {vals}")


def expected_accuracies(problem_mix, conflict_rule_ii):
    """Delta-c under a stationary mixture; determines the branch ratio (Appendix A7)."""
    mix = np.array(problem_mix, dtype=float)
    mix /= mix.sum()
    rule_i = float((mix * np.array([0.97, 0.58, 0.40])).sum())
    rule_ii = float((mix * np.array([0.66, 0.87, conflict_rule_ii])).sum())
    return rule_i, rule_ii, rule_ii - rule_i


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="control + primary only, 20 agents, 500 trials")
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    args = parser.parse_args()

    print("Stationary-mixture expected accuracies (Appendix B7, Result 3):")
    for mix, label in [((0.50, 0.25, 0.25), "50/25/25"), ((1/3, 1/3, 1/3), "thirds")]:
        for c in (0.92, 0.73):
            r1, r2, dc = expected_accuracies(mix, c)
            print(f"  {label:9s} C = {c:.2f}: Rule I {r1:.4f}  Rule II {r2:.4f}  "
                  f"Delta-c = {dc:+.4f}")

    results = []
    for name in args.runs:
        cfg, n_agents, n_trials, _ = RUNS[name]
        if args.quick:
            if name not in ("control", "primary"):
                continue
            n_agents, n_trials = 20, 500
        print(f"\nsimulating {name} ({n_agents} agents x {n_trials} trials)...", flush=True)
        simulate(name, n_agents, n_trials)
        res = analyse(name)
        report(res)
        results.append(res)

    with open(os.path.join(_HERE, "b7_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {os.path.join(_HERE, 'b7_results.json')}")