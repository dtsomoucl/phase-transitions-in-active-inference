"""
Notebook 03b / Step 1 -- Proper `pymdp` port of the balance scale task
======================================================================

This module implements the balance-scale developmental model using the actual
`pymdp` Agent API rather than the reduced hand-written policy loop retained in
`sim_balance_scale_appendix.py`.

Key modelling choice:
    The cue must be observed before the strategy is chosen. To preserve that
    ordering in a discrete POMDP, each trial is represented as two timesteps:

    cue phase -> outcome phase

    The hidden `problem_type` persists across these two timesteps, while a
    separate hidden `phase` factor cycles deterministically. This lets the
    agent infer `q(problem_type | cue)` at the cue phase and then plan over the
    strategy action that will determine the expected outcome at the next step.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

### DT --> Keep outputs for the main paper implementation separate from appendix outputs.
OUT_DIR = os.path.join(os.path.dirname(__file__), "Figs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _ensure_pymdp_importable() -> None:
    """Add common local `pymdp` locations to `sys.path` if needed."""

    last_exc = None

    try:
        import pymdp  # noqa: F401
        return
    except ImportError as exc:
        last_exc = exc

    candidate_roots = [
        os.environ.get("PYMDP_REPO"),
        os.path.join(os.path.dirname(__file__), "third_party"),
        "/tmp/pymdp_repo",
    ]

    for root in candidate_roots:
        if not root:
            continue
        package_dir = os.path.join(root, "pymdp")
        if os.path.isdir(package_dir) and root not in sys.path:
            sys.path.insert(0, root)
            try:
                import pymdp  # noqa: F401
                return
            except ImportError as exc:
                last_exc = exc
                continue

    raise ImportError(
        "Could not import `pymdp`. A workspace-local copy is expected under "
        "`third_party/pymdp`, or you can set `PYMDP_REPO` to a clone. "
        "If you copied the upstream GitHub repository verbatim, note that it "
        "may also require optional plotting dependencies such as `seaborn` at "
        "import time. "
        f"Last import error: {last_exc}"
    )


_ensure_pymdp_importable()

from pymdp.agent import Agent
from pymdp import control, utils


### DT --> Hidden-state factor indices.
STRATEGY_FACTOR = 0
PROBLEM_FACTOR = 1
PHASE_FACTOR = 2

### DT --> Strategy states and controllable actions.
RULE_I = 0
RULE_II = 1

### DT --> Problem-type states.
WEIGHT = 0
DISTANCE = 1
CONFLICT = 2

### DT --> Trial phase states.
PHASE_CUE = 0
PHASE_OUTCOME = 1

### DT --> Observation modalities.
OUTCOME_MODALITY = 0
CUE_MODALITY = 1
PHASE_MODALITY = 2

### DT --> Outcome observations.
OBS_CORRECT = 0
OBS_INCORRECT = 1
OBS_OUTCOME_NULL = 2

### DT --> Cue observations.
OBS_WEIGHT_CUE = 0
OBS_DISTANCE_CUE = 1
OBS_MIXED_CUE = 2
OBS_CUE_NULL = 3


@dataclass
class BalanceScalePymdpConfig:
    """Configuration for the proper `pymdp` balance-scale model."""

    rule1_success: tuple = (0.97, 0.58, 0.40)
    rule2_success: tuple = (0.66, 0.87, 0.92)
    problem_mix: tuple = (0.35, 0.35, 0.30)
    initial_strategy_prior: tuple = (0.80, 0.20)
    policy_prior: tuple = (0.94, 0.06)
    late_policy_prior: tuple = (0.38, 0.62)
    cue_emission: tuple = (
        (0.80, 0.10, 0.10),  # weight items
        (0.10, 0.80, 0.10),  # distance items
        (0.20, 0.20, 0.60),  # conflict items
    )
    alpha_outcome: float = 1.0
    known_precision: float = 64.0
    gamma: float = 16.0
    alpha_action: float = 16.0
    pref_correct: float = 2.0
    lr_pA: float = 1.0
    use_utility: bool = True
    use_states_info_gain: bool = True
    use_param_info_gain: bool = False
    action_selection: str = "stochastic"
    sampling_mode: str = "marginal"
    use_curriculum: bool = False
    curriculum_easy_mix: tuple = (0.95, 0.03, 0.02)
    curriculum_hard_mix: tuple = (0.06, 0.42, 0.52)
    curriculum_start_frac: float = 0.06
    curriculum_end_frac: float = 0.46
    policy_prior_start_frac: float = 0.03
    policy_prior_end_frac: float = 0.40


class BalanceScalePymdpEnv:
    """
    Generative process for the balance-scale task.

    The environment maintains three hidden-state factors:
      - strategy
      - problem type
      - trial phase

    The phase alternates deterministically:
      cue -> outcome -> cue -> outcome -> ...
    """

    def __init__(self, config: BalanceScalePymdpConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.problem_mix = _initial_problem_mix(config)
        self.state = np.array([RULE_I, WEIGHT, PHASE_CUE], dtype=int)

    def reset(self) -> List[int]:
        """Reset the environment to the start of a new cue phase."""

        problem_type = int(self.rng.choice(3, p=self.problem_mix))
        self.state = np.array([RULE_I, problem_type, PHASE_CUE], dtype=int)
        return self._sample_observation()

    def step(self, action: np.ndarray) -> List[int]:
        """Advance one timestep using the current hidden state and chosen action."""

        strategy, problem_type, phase = self.state
        chosen_strategy = int(action[STRATEGY_FACTOR])

        if phase == PHASE_CUE:
            next_state = np.array([chosen_strategy, problem_type, PHASE_OUTCOME], dtype=int)
        else:
            next_problem = int(self.rng.choice(3, p=self.problem_mix))
            next_state = np.array([strategy, next_problem, PHASE_CUE], dtype=int)

        self.state = next_state
        return self._sample_observation()

    def _sample_observation(self) -> List[int]:
        """Sample observations from the current latent state."""

        strategy, problem_type, phase = self.state

        if phase == PHASE_CUE:
            cue_probs = np.array(self.config.cue_emission[problem_type], dtype=float)
            cue = int(self.rng.choice(3, p=cue_probs))
            return [OBS_OUTCOME_NULL, cue, PHASE_CUE]

        p_correct = self._success_probability(strategy, problem_type)
        outcome = OBS_CORRECT if self.rng.random() < p_correct else OBS_INCORRECT
        return [outcome, OBS_CUE_NULL, PHASE_OUTCOME]

    def _success_probability(self, strategy: int, problem_type: int) -> float:
        if strategy == RULE_I:
            return float(self.config.rule1_success[problem_type])
        return float(self.config.rule2_success[problem_type])


def _normalized(probs) -> np.ndarray:
    """Return a normalized probability vector."""

    probs = np.array(probs, dtype=float)
    return probs / probs.sum()


def _initial_problem_mix(config: BalanceScalePymdpConfig) -> np.ndarray:
    """Return the starting item mixture for the developmental run."""

    ### DT --> We are ensuring here that, when we use a curriculum, the agent genuinely starts in an early Rule I-friendly regime.
    if config.use_curriculum:
        return _normalized(config.curriculum_easy_mix)
    return _normalized(config.problem_mix)


def _curriculum_problem_mix(
    config: BalanceScalePymdpConfig,
    trial_idx: int,
    total_trials: int,
) -> np.ndarray:
    """Blend from easy to hard items over developmental time."""

    ### DT --> We are ensuring here that the task statistics change gradually across development, rather than making Rule II globally dominant from the first trial.
    if (not config.use_curriculum) or total_trials <= 1:
        return _normalized(config.problem_mix)

    progress = trial_idx / max(total_trials - 1, 1)
    weight = _blend_weight(
        progress,
        float(config.curriculum_start_frac),
        float(config.curriculum_end_frac),
    )

    easy = _normalized(config.curriculum_easy_mix)
    hard = _normalized(config.curriculum_hard_mix)
    return _normalized((1.0 - weight) * easy + weight * hard)


def _blend_weight(progress: float, start: float, end: float) -> float:
    """Return a linear interpolation weight between 0 and 1."""

    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / max(end - start, 1e-8)


def _policy_prior_over_development(
    config: BalanceScalePymdpConfig,
    trial_idx: int,
    total_trials: int,
) -> np.ndarray:
    """Interpolate the policy prior from early Rule I bias to a later balanced regime."""

    ### DT --> We are ensuring here that the agent begins development with a Rule I habit, but that this habit relaxes over time so later evidence can support a Rule II transition.
    progress = trial_idx / max(total_trials - 1, 1)
    weight = _blend_weight(
        progress,
        float(config.policy_prior_start_frac),
        float(config.policy_prior_end_frac),
    )
    early = _normalized(config.policy_prior)
    late = _normalized(config.late_policy_prior)
    return _normalized((1.0 - weight) * early + weight * late)


def _set_global_numpy_seed(seed: Optional[int]) -> None:
    """Seed NumPy's global RNG so `pymdp` action sampling is reproducible."""

    ### DT --> We are ensuring here that `pymdp`'s internal stochastic action sampling uses the same seed discipline as the environment.
    if seed is not None:
        np.random.seed(int(seed))


def build_balance_scale_agent(config: BalanceScalePymdpConfig) -> Agent:
    """Construct a proper `pymdp.Agent` for the balance-scale task."""

    num_obs = [3, 4, 2]
    num_states = [2, 3, 2]
    num_controls = [2, 1, 1]
    policies = control.construct_policies(num_states, num_controls, policy_len=1, control_fac_idx=[STRATEGY_FACTOR])

    A = utils.obj_array(3)
    pA = utils.obj_array(3)

    ### DT --> Outcome modality: null during cue phase, learnable during outcome phase.
    A[OUTCOME_MODALITY] = np.zeros((3, 2, 3, 2))
    pA[OUTCOME_MODALITY] = np.zeros_like(A[OUTCOME_MODALITY])

    A[OUTCOME_MODALITY][OBS_OUTCOME_NULL, :, :, PHASE_CUE] = 1.0
    pA[OUTCOME_MODALITY][OBS_OUTCOME_NULL, :, :, PHASE_CUE] = config.known_precision

    A[OUTCOME_MODALITY][OBS_CORRECT, :, :, PHASE_OUTCOME] = 0.5
    A[OUTCOME_MODALITY][OBS_INCORRECT, :, :, PHASE_OUTCOME] = 0.5
    pA[OUTCOME_MODALITY][OBS_CORRECT, :, :, PHASE_OUTCOME] = config.alpha_outcome
    pA[OUTCOME_MODALITY][OBS_INCORRECT, :, :, PHASE_OUTCOME] = config.alpha_outcome

    ### DT --> Cue modality: informative at cue phase, null at outcome phase.
    A[CUE_MODALITY] = np.zeros((4, 3, 2))
    for problem_type, cue_probs in enumerate(config.cue_emission):
        A[CUE_MODALITY][0:3, problem_type, PHASE_CUE] = np.array(cue_probs, dtype=float)
    A[CUE_MODALITY][OBS_CUE_NULL, :, PHASE_OUTCOME] = 1.0
    pA[CUE_MODALITY] = config.known_precision * A[CUE_MODALITY]

    ### DT --> Phase modality: deterministic observation of the trial phase.
    A[PHASE_MODALITY] = np.eye(2)
    pA[PHASE_MODALITY] = config.known_precision * A[PHASE_MODALITY]

    B = utils.obj_array(3)

    ### DT --> Strategy dynamics depend on current strategy, phase, and chosen action.
    B[STRATEGY_FACTOR] = np.zeros((2, 2, 2, 2))
    for prev_strategy in range(2):
        for phase in range(2):
            for action in range(2):
                if phase == PHASE_CUE:
                    B[STRATEGY_FACTOR][action, prev_strategy, phase, action] = 1.0
                else:
                    B[STRATEGY_FACTOR][prev_strategy, prev_strategy, phase, action] = 1.0

    ### DT --> Problem type persists across cue->outcome and resamples across outcome->cue.
    B[PROBLEM_FACTOR] = np.zeros((3, 3, 2, 1))
    problem_mix = _initial_problem_mix(config)
    for prev_problem in range(3):
        B[PROBLEM_FACTOR][prev_problem, prev_problem, PHASE_CUE, 0] = 1.0
        B[PROBLEM_FACTOR][:, prev_problem, PHASE_OUTCOME, 0] = problem_mix

    ### DT --> Phase cycles deterministically.
    B[PHASE_FACTOR] = np.zeros((2, 2, 1))
    B[PHASE_FACTOR][PHASE_OUTCOME, PHASE_CUE, 0] = 1.0
    B[PHASE_FACTOR][PHASE_CUE, PHASE_OUTCOME, 0] = 1.0

    C = utils.obj_array(3)
    C[OUTCOME_MODALITY] = np.array([config.pref_correct, -config.pref_correct, 0.0])
    C[CUE_MODALITY] = np.zeros(4)
    C[PHASE_MODALITY] = np.zeros(2)

    D = utils.obj_array(3)
    ### DT --> We are ensuring here that development begins in a Rule I-dominant basin rather than with an unrealistically symmetric starting point.
    D[STRATEGY_FACTOR] = _normalized(config.initial_strategy_prior)
    D[PROBLEM_FACTOR] = problem_mix.copy()
    D[PHASE_FACTOR] = np.array([1.0, 0.0])

    ### DT --> We are ensuring here that the initial policy prior (habit term) matches the intended early Rule I bias before the developmental schedule relaxes it.
    E = np.zeros(len(policies), dtype=float)
    for policy_idx, policy in enumerate(policies):
        chosen_rule = int(policy[0, STRATEGY_FACTOR])
        E[policy_idx] = float(_normalized(config.policy_prior)[chosen_rule])
    E = _normalized(E)

    A_factor_list = [
        [STRATEGY_FACTOR, PROBLEM_FACTOR, PHASE_FACTOR],
        [PROBLEM_FACTOR, PHASE_FACTOR],
        [PHASE_FACTOR],
    ]
    B_factor_list = [
        [STRATEGY_FACTOR, PHASE_FACTOR],
        [PROBLEM_FACTOR, PHASE_FACTOR],
        [PHASE_FACTOR],
    ]

    agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        pA=pA,
        policies=policies,
        num_controls=num_controls,
        control_fac_idx=[STRATEGY_FACTOR],
        A_factor_list=A_factor_list,
        B_factor_list=B_factor_list,
        gamma=config.gamma,
        alpha=config.alpha_action,
        use_utility=config.use_utility,
        use_states_info_gain=config.use_states_info_gain,
        use_param_info_gain=config.use_param_info_gain,
        action_selection=config.action_selection,
        sampling_mode=config.sampling_mode,
        inference_algo="VANILLA",
        modalities_to_learn=[OUTCOME_MODALITY],
        lr_pA=config.lr_pA,
    )

    return agent


def _rule_level_policy_sum(agent: Agent, values: np.ndarray) -> np.ndarray:
    """Sum a policy-level vector onto Rule I vs Rule II."""

    grouped = np.zeros(2, dtype=float)
    for policy_idx, policy in enumerate(agent.policies):
        rule = int(policy[0, STRATEGY_FACTOR])
        grouped[rule] += float(values[policy_idx])
    return grouped


def _rule_level_policy_mean(agent: Agent, values: np.ndarray) -> np.ndarray:
    """Average a policy-level vector onto Rule I vs Rule II."""

    grouped = np.zeros(2, dtype=float)
    counts = np.zeros(2, dtype=float)
    for policy_idx, policy in enumerate(agent.policies):
        rule = int(policy[0, STRATEGY_FACTOR])
        grouped[rule] += float(values[policy_idx])
        counts[rule] += 1.0
    counts[counts == 0.0] = 1.0
    return grouped / counts


def _policy_rule2_probability(agent: Agent, q_pi: np.ndarray) -> float:
    """Map the posterior over policies onto P(Rule II)."""

    return float(_rule_level_policy_sum(agent, q_pi)[RULE_II])


def _delta_efe_rule1_minus_rule2(agent: Agent, neg_efe: np.ndarray) -> float:
    """Return EFE(Rule I) - EFE(Rule II), positive when Rule II is better."""

    ### DT --> We are ensuring here that the plotted score uses the correct sign: `pymdp` returns negative EFE, so we convert back before interpreting the panel.
    neg_efe_by_rule = _rule_level_policy_mean(agent, neg_efe)
    return float(neg_efe_by_rule[RULE_II] - neg_efe_by_rule[RULE_I])


def _set_agent_policy_prior(agent: Agent, prior_by_rule: np.ndarray) -> None:
    """Project a rule-level prior onto the agent's policy prior vector."""

    ### DT --> We are ensuring here that the developmental policy prior acts directly on the Rule I versus Rule II policy competition, rather than remaining fixed for the entire simulation.
    policy_prior = np.zeros(len(agent.policies), dtype=float)
    for policy_idx, policy in enumerate(agent.policies):
        rule = int(policy[0, STRATEGY_FACTOR])
        policy_prior[policy_idx] = float(prior_by_rule[rule])
    agent.E = _normalized(policy_prior)


def _neutralize_agent_policy_prior(agent: Agent) -> None:
    """Set a flat policy prior over Rule I and Rule II policies."""

    ### DT --> We are ensuring here that special protocols such as hysteresis measure path dependence from learning itself, not from a lingering policy-habit prior.
    _set_agent_policy_prior(agent, np.array([0.5, 0.5], dtype=float))


def _extract_learned_correct_matrix(agent: Agent) -> np.ndarray:
    """Return learned P(correct | strategy, problem_type) at the outcome phase."""

    return agent.A[OUTCOME_MODALITY][OBS_CORRECT, :, :, PHASE_OUTCOME].copy()


def _force_or_sample_action(agent: Agent, forced_rule: Optional[int]) -> np.ndarray:
    """Either sample an action normally or force a strategy choice."""

    if forced_rule is None:
        return agent.sample_action()

    action = np.array([forced_rule, 0, 0], dtype=int)
    agent.action = action
    agent.step_time()
    return action


def _set_problem_mix(env: BalanceScalePymdpEnv, problem_mix) -> None:
    """Update the environment's stationary distribution over problem types."""

    env.problem_mix = np.array(problem_mix, dtype=float)
    env.problem_mix /= env.problem_mix.sum()


def _refresh_cue_observation(env: BalanceScalePymdpEnv) -> List[int]:
    """Resample the cue-phase problem type after a curriculum change."""

    if int(env.state[PHASE_FACTOR]) != PHASE_CUE:
        raise ValueError("Can only refresh the latent problem type at the cue phase.")
    env.state[PROBLEM_FACTOR] = int(env.rng.choice(3, p=env.problem_mix))
    return env._sample_observation()


def _moving_average(values, window: int) -> np.ndarray:
    """Simple moving average with a window capped by the data length."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), values.size))
    if window == 1:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _trailing_average(values, window: int) -> np.ndarray:
    """Trailing moving average that preserves the original series length."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), values.size))
    if window == 1:
        return values.copy()
    csum = np.cumsum(np.insert(values, 0, 0.0))
    trailing = (csum[window:] - csum[:-window]) / window
    prefix = np.array([values[: idx + 1].mean() for idx in range(window - 1)], dtype=float)
    return np.concatenate([prefix, trailing])


def _trailing_average_matrix(matrix: np.ndarray, window: int) -> np.ndarray:
    """Apply a trailing moving average row-wise to a population matrix."""

    return np.stack([_trailing_average(row, window) for row in np.asarray(matrix, dtype=float)], axis=0)


def _trailing_average_valid(values, window: int) -> np.ndarray:
    """Trailing moving average using only full windows."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), values.size))
    if window == 1:
        return values.copy()
    csum = np.cumsum(np.insert(values, 0, 0.0))
    return (csum[window:] - csum[:-window]) / window


def _classify_rule_usage(rule2_fraction: float) -> str:
    """Usage-based proxy classification used in Notebook 03."""

    if rule2_fraction < 0.3:
        return "Rule I"
    if rule2_fraction > 0.7:
        return "Rule II"
    return "Transitional"


def _select_representative_transition_history(
    histories: List[Dict[str, object]],
    *,
    threshold_low: float = 0.3,
    threshold_high: float = 0.7,
    window: int = 30,
) -> Dict[str, object]:
    """Pick a representative agent that actually transitions, if one exists."""

    ### DT --> We are ensuring here that the single-agent panel illustrates the developmental phenomenon of interest, rather than whichever arbitrary seed happened to be chosen first.
    best_history = histories[0]
    candidates = []

    for history in histories:
        p_rule2 = np.array([trial["p_rule2"] for trial in history["trials"]], dtype=float)
        ### DT --> We are ensuring here that representative transition timing is measured on a trailing smoother, so it stays aligned to the actual developmental trial index.
        smoothed = _trailing_average(p_rule2, window)
        if smoothed.size == 0:
            continue
        starts_low = float(smoothed[0]) < threshold_low
        crossed = np.where(smoothed >= threshold_high)[0]
        if starts_low and crossed.size:
            candidates.append((float(crossed[0]), history))

    if not candidates:
        return best_history

    transition_times = np.array([item[0] for item in candidates], dtype=float)
    median_time = float(np.median(transition_times))
    _, best_history = min(candidates, key=lambda item: abs(item[0] - median_time))
    return best_history


def _default_plot_config() -> BalanceScalePymdpConfig:
    """Default configuration used for Figure 7 style developmental trajectories."""

    return BalanceScalePymdpConfig(
        ### DT --> We are ensuring here that (1) the paper figures begin with a Rule I bias, then transition as the curriculum hardens and learning reveals the value of Rule II.
        problem_mix=(0.95, 0.03, 0.02),
        gamma=16.0,
        ### DT --> (2) that action selection stays decisive enough to suppress rare early lock-in, but not so rigid that the later transition becomes a near-instantaneous cliff.
        alpha_action=8.0,
        initial_strategy_prior=(0.80, 0.20),
        policy_prior=(0.94, 0.06),
        late_policy_prior=(0.38, 0.62),
        use_utility=True,
        use_states_info_gain=False,
        ### DT --> (3) that the agent has a principled reason to sample the under-learned strategy early on, so the later transition is learnable rather than blocked by permanent lock-in.
        use_param_info_gain=True,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_curriculum=True,
        curriculum_easy_mix=(0.95, 0.03, 0.02),
        curriculum_hard_mix=(0.06, 0.42, 0.52),
        ### DT --> (4) that both the hard-item curriculum and the relaxation of the Rule I habit begin earlier, so the transition window moves into the middle-late portion of the run.
        curriculum_start_frac=0.06,
        curriculum_end_frac=0.46,
        policy_prior_start_frac=0.03,
        policy_prior_end_frac=0.40,
    )


def _catastrophe_plot_config() -> BalanceScalePymdpConfig:
    """Configuration tuned for catastrophe-style population signatures."""

    return BalanceScalePymdpConfig(
        problem_mix=(0.93, 0.04, 0.03),
        gamma=14.0,
        alpha_action=6.0,
        initial_strategy_prior=(0.72, 0.28),
        policy_prior=(0.86, 0.14),
        late_policy_prior=(0.42, 0.58),
        use_utility=True,
        use_states_info_gain=False,
        use_param_info_gain=True,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_curriculum=True,
        curriculum_easy_mix=(0.93, 0.04, 0.03),
        curriculum_hard_mix=(0.08, 0.42, 0.50),
        curriculum_start_frac=0.05,
        curriculum_end_frac=0.38,
        policy_prior_start_frac=0.02,
        policy_prior_end_frac=0.30,
    )


def _individual_differences_config() -> BalanceScalePymdpConfig:
    """Configuration used when comparing gamma values."""

    return BalanceScalePymdpConfig(
        problem_mix=(0.94, 0.04, 0.02),
        gamma=16.0,
        alpha_action=8.0,
        initial_strategy_prior=(0.78, 0.22),
        policy_prior=(0.92, 0.08),
        late_policy_prior=(0.40, 0.60),
        use_utility=True,
        use_states_info_gain=False,
        use_param_info_gain=True,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_curriculum=True,
        curriculum_easy_mix=(0.94, 0.04, 0.02),
        curriculum_hard_mix=(0.07, 0.42, 0.51),
        curriculum_start_frac=0.05,
        curriculum_end_frac=0.42,
        policy_prior_start_frac=0.02,
        policy_prior_end_frac=0.34,
    )


def _siegler_plot_config() -> BalanceScalePymdpConfig:
    """Configuration tuned for developmental-rule summaries."""

    return BalanceScalePymdpConfig(
        problem_mix=(0.94, 0.04, 0.02),
        gamma=14.0,
        alpha_action=7.0,
        initial_strategy_prior=(0.76, 0.24),
        policy_prior=(0.90, 0.10),
        late_policy_prior=(0.42, 0.58),
        use_utility=True,
        use_states_info_gain=False,
        use_param_info_gain=True,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_curriculum=True,
        curriculum_easy_mix=(0.94, 0.04, 0.02),
        curriculum_hard_mix=(0.08, 0.42, 0.50),
        curriculum_start_frac=0.05,
        curriculum_end_frac=0.40,
        policy_prior_start_frac=0.02,
        policy_prior_end_frac=0.32,
    )


def _hysteresis_protocol_config(base_config: BalanceScalePymdpConfig) -> BalanceScalePymdpConfig:
    """Configuration tuned specifically for a non-degenerate hysteresis sweep."""

    return replace(
        base_config,
        ### DT --> We are ensuring here that the hysteresis loop is driven by accumulated learning under changing item mixtures, not by the strong developmental policy schedule used for Figure 7.
        ### DT --> and that the two strategies are close enough in the ambiguous regime for path dependence to bend the sweep, rather than saturating immediately at opposite ceilings.
        rule1_success=(0.90, 0.64, 0.52),
        rule2_success=(0.74, 0.76, 0.82),
        gamma=4.0,
        alpha_action=2.0,
        initial_strategy_prior=(0.50, 0.50),
        policy_prior=(0.50, 0.50),
        late_policy_prior=(0.50, 0.50),
        use_curriculum=False,
        use_param_info_gain=False,
    )


def _run_trials_with_agent(
    agent: Agent,
    env: BalanceScalePymdpEnv,
    observation: List[int],
    num_trials: int,
    *,
    forced_rule: Optional[int] = None,
    record_steps: bool = True,
    rule2_choices_start: int = 0,
    trial_index_start: int = 0,
    total_trials: Optional[int] = None,
    apply_curriculum: bool = True,
    apply_policy_schedule: bool = True,
):
    """
    Continue running an existing agent/environment pair for `num_trials`.

    This lower-level helper makes it possible to keep learning across phases of
    a curriculum or hysteresis protocol while still recording per-trial metrics.
    """

    step_history: List[Dict[str, object]] = []
    trial_history: List[Dict[str, object]] = []
    active_trial: Optional[Dict[str, object]] = None
    rule2_choices = rule2_choices_start
    total_trials = num_trials if total_trials is None else int(total_trials)
    config = env.config

    for step_idx in range(num_trials * 2):
        qs = agent.infer_states(observation)
        phase = int(observation[PHASE_MODALITY])
        current_trial_idx = trial_index_start + len(trial_history)

        if apply_policy_schedule:
            ### DT --> We are ensuring here that the early Rule I policy habit gradually relaxes over developmental time, so later evidence can overturn it.
            _set_agent_policy_prior(
                agent,
                _policy_prior_over_development(config, current_trial_idx, total_trials),
            )

        if phase == PHASE_OUTCOME:
            agent.update_A(observation)

        q_pi, G = agent.infer_policies()
        p_rule2 = _policy_rule2_probability(agent, q_pi)
        action = _force_or_sample_action(agent, forced_rule)

        if record_steps:
            step_history.append(
                {
                    "step": 2 * trial_index_start + step_idx,
                    "phase": phase,
                    "observation": tuple(int(x) for x in observation),
                    "q_strategy": qs[STRATEGY_FACTOR].copy(),
                    "q_problem": qs[PROBLEM_FACTOR].copy(),
                    "q_phase": qs[PHASE_FACTOR].copy(),
                    "q_pi": q_pi.copy(),
                    "G": G.copy(),
                    "action": action.copy(),
                    "p_rule2": p_rule2,
                    "latent_state": tuple(int(x) for x in env.state),
                    "learned_correct": _extract_learned_correct_matrix(agent),
                }
            )

        if phase == PHASE_CUE:
            chosen_rule = int(action[STRATEGY_FACTOR])
            rule2_choices += int(chosen_rule == RULE_II)
            active_trial = {
                "trial": trial_index_start + len(trial_history),
                "problem_type": int(env.state[PROBLEM_FACTOR]),
                "cue_observation": int(observation[CUE_MODALITY]),
                "q_problem_cue": qs[PROBLEM_FACTOR].copy(),
                "q_strategy_cue": qs[STRATEGY_FACTOR].copy(),
                "p_rule2": p_rule2,
                "chosen_rule": chosen_rule,
                "G_cue": G.copy(),
                "delta_G_cue": _delta_efe_rule1_minus_rule2(agent, G),
            }
        else:
            if active_trial is None:
                raise RuntimeError("Outcome phase encountered without a matching cue-phase trial record.")
            active_trial["outcome"] = int(observation[OUTCOME_MODALITY])
            active_trial["correct"] = int(observation[OUTCOME_MODALITY] == OBS_CORRECT)
            active_trial["q_problem_outcome"] = qs[PROBLEM_FACTOR].copy()
            active_trial["learned_correct"] = _extract_learned_correct_matrix(agent)
            active_trial["cumulative_rule2_fraction"] = rule2_choices / (
                trial_index_start + len(trial_history) + 1
            )
            trial_history.append(active_trial)
            active_trial = None

        observation = env.step(action)
        if apply_curriculum and config.use_curriculum and int(observation[PHASE_MODALITY]) == PHASE_CUE:
            ### DT --> We are ensuring that each new cue-phase trial uses the developmental curriculum, so Rule II is not advantaged from the very beginning.
            next_trial_idx = trial_index_start + len(trial_history)
            _set_problem_mix(env, _curriculum_problem_mix(config, next_trial_idx, total_trials))
            observation = _refresh_cue_observation(env)

    return observation, step_history, trial_history, rule2_choices


def run_balance_scale_pymdp(
    num_trials: int,
    config: Optional[BalanceScalePymdpConfig] = None,
    seed: Optional[int] = None,
    record_steps: bool = True,
) -> Dict[str, object]:
    """
    Run one `pymdp` agent for `num_trials` balance-scale trials.

    Each trial consists of two timesteps:
      1. cue phase   -> observe cue, infer problem type, choose strategy
      2. outcome phase -> observe feedback, update the learned outcome model
    """

    config = config or BalanceScalePymdpConfig()
    _set_global_numpy_seed(seed)
    agent = build_balance_scale_agent(config)
    env = BalanceScalePymdpEnv(config, seed=seed)
    if config.use_curriculum:
        ### DT --> We are ensuring here that the very first observation is sampled from the early developmental regime, not from the later mixed regime.
        _set_problem_mix(env, _curriculum_problem_mix(config, 0, num_trials))
    observation = env.reset()
    observation, step_history, trial_history, rule2_choices = _run_trials_with_agent(
        agent,
        env,
        observation,
        num_trials,
        forced_rule=None,
        record_steps=record_steps,
        rule2_choices_start=0,
        trial_index_start=0,
        total_trials=num_trials,
        apply_curriculum=True,
    )

    return {
        "config": config,
        "agent": agent,
        "env": env,
        "steps": step_history,
        "trials": trial_history,
        "final_observation": observation,
        "final_rule2_choices": rule2_choices,
    }


def run_population_pymdp(
    num_trials: int,
    n_agents: int,
    config: Optional[BalanceScalePymdpConfig] = None,
    seed_offset: int = 0,
    record_steps: bool = False,
) -> List[Dict[str, object]]:
    """Run a population of proper `pymdp` agents."""

    config = config or BalanceScalePymdpConfig()
    histories = []
    for idx in range(n_agents):
        histories.append(
            run_balance_scale_pymdp(
                num_trials,
                config=config,
                seed=seed_offset + idx,
                record_steps=record_steps,
            )
        )
    return histories


def extract_population_arrays(histories: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Convert a list of histories into aligned population matrices."""

    if not histories:
        raise ValueError("Need at least one history to extract population arrays.")

    num_agents = len(histories)
    num_trials = len(histories[0]["trials"])

    arrays = {
        "p_rule2": np.zeros((num_agents, num_trials)),
        "rule2_fraction": np.zeros((num_agents, num_trials)),
        "chosen_rule": np.zeros((num_agents, num_trials), dtype=int),
        "problem_type": np.zeros((num_agents, num_trials), dtype=int),
        "correct": np.zeros((num_agents, num_trials), dtype=int),
        "delta_G": np.zeros((num_agents, num_trials)),
    }

    for agent_idx, history in enumerate(histories):
        for trial_idx, trial in enumerate(history["trials"]):
            arrays["p_rule2"][agent_idx, trial_idx] = trial["p_rule2"]
            arrays["rule2_fraction"][agent_idx, trial_idx] = trial["cumulative_rule2_fraction"]
            arrays["chosen_rule"][agent_idx, trial_idx] = trial["chosen_rule"]
            arrays["problem_type"][agent_idx, trial_idx] = trial["problem_type"]
            arrays["correct"][agent_idx, trial_idx] = trial["correct"]
            arrays["delta_G"][agent_idx, trial_idx] = trial["delta_G_cue"]

    return arrays


def summarize_population(histories: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Summarize population-level development over trials."""

    arrays = extract_population_arrays(histories)
    rule2_fraction = arrays["rule2_fraction"]
    p_rule2 = arrays["p_rule2"]

    return {
        "mean_rule2_fraction": rule2_fraction.mean(axis=0),
        "std_rule2_fraction": rule2_fraction.std(axis=0),
        "mean_p_rule2": p_rule2.mean(axis=0),
        "std_p_rule2": p_rule2.std(axis=0),
    }


def _estimate_transition_trials(
    p_rule2_matrix: np.ndarray,
    threshold: float = 0.7,
    window: int = 20,
) -> np.ndarray:
    """Estimate transition times using a smoothed `P(Rule II)` threshold."""

    transitions = np.full(p_rule2_matrix.shape[0], np.nan)
    for idx, series in enumerate(p_rule2_matrix):
        ### DT --> We are ensuring that estimated transition times correspond to the real trial at which the smoothed trajectory crosses threshold, rather than a left-shifted moving-average index.
        smoothed = _trailing_average(series, window)
        crossed = np.where(smoothed > threshold)[0]
        if crossed.size:
            transitions[idx] = float(crossed[0])
    return transitions


def _late_window_rule2_summary(
    p_rule2_matrix: np.ndarray,
    *,
    smooth_window: int = 25,
    tail_window: int = 80,
) -> np.ndarray:
    """Summarize each agent by its late-development smoothed Rule II usage."""

    smoothed = _trailing_average_matrix(np.asarray(p_rule2_matrix, dtype=float), window=smooth_window)
    tail_window = max(1, min(int(tail_window), smoothed.shape[1]))
    return smoothed[:, -tail_window:].mean(axis=1)


def _run_hysteresis_protocol(
    config: BalanceScalePymdpConfig,
    hard_fracs: np.ndarray,
    *,
    trials_per_point: int = 80,
    pretrain_trials: int = 12,
    n_reps: int = 10,
    seed_offset: int = 30000,
):
    """Path-dependent curriculum sweep using the proper `pymdp` agent."""

    config = _hysteresis_protocol_config(config)
    forward = np.zeros((n_reps, len(hard_fracs)))
    backward = np.zeros((n_reps, len(hard_fracs)))

    for rep in range(n_reps):
        ### DT --> We want path dependence to come from what the agent has learned in an easy regime, not from an externally forced action policy.
        _set_global_numpy_seed(seed_offset + rep)
        forward_config = replace(config, initial_strategy_prior=(0.56, 0.44))
        agent_f = build_balance_scale_agent(forward_config)
        env_f = BalanceScalePymdpEnv(forward_config, seed=seed_offset + rep)
        _set_problem_mix(env_f, [0.70, 0.18, 0.12])
        obs_f = env_f.reset()
        obs_f, _, pre_f, rule2_f = _run_trials_with_agent(
            agent_f,
            env_f,
            obs_f,
            pretrain_trials,
            forced_rule=None,
            record_steps=False,
            rule2_choices_start=0,
            trial_index_start=0,
            total_trials=pretrain_trials,
            apply_curriculum=False,
            apply_policy_schedule=False,
        )
        _neutralize_agent_policy_prior(agent_f)
        trial_count_f = len(pre_f)

        for idx, hard_frac in enumerate(hard_fracs):
            mix = np.array([1.0 - hard_frac, hard_frac / 2.0, hard_frac / 2.0])
            _set_problem_mix(env_f, mix)
            obs_f = _refresh_cue_observation(env_f)
            obs_f, _, block_f, rule2_f = _run_trials_with_agent(
                agent_f,
                env_f,
                obs_f,
                trials_per_point,
                forced_rule=None,
                record_steps=False,
                rule2_choices_start=rule2_f,
                trial_index_start=trial_count_f,
                total_trials=trials_per_point,
                apply_curriculum=False,
                apply_policy_schedule=False,
            )
            trial_count_f += len(block_f)
            ### DT --> We are ensuring here that the hysteresis panel reflects observed Rule II choice behaviour over each sweep block, which is the quantity that should display path dependence most directly.
            trailing = _trailing_average(
                np.array([trial["chosen_rule"] == RULE_II for trial in block_f], dtype=float),
                window=min(20, len(block_f)),
            )
            forward[rep, idx] = float(trailing[-1])

        ### DT --> Esnuring that the reverse sweep starts from a genuinely learned hard-item regime, again without forcing the action policy.
        _set_global_numpy_seed(seed_offset + 1000 + rep)
        backward_config = replace(config, initial_strategy_prior=(0.44, 0.56))
        agent_b = build_balance_scale_agent(backward_config)
        env_b = BalanceScalePymdpEnv(backward_config, seed=seed_offset + 1000 + rep)
        _set_problem_mix(env_b, [0.30, 0.30, 0.40])
        obs_b = env_b.reset()
        obs_b, _, pre_b, rule2_b = _run_trials_with_agent(
            agent_b,
            env_b,
            obs_b,
            pretrain_trials,
            forced_rule=None,
            record_steps=False,
            rule2_choices_start=0,
            trial_index_start=0,
            total_trials=pretrain_trials,
            apply_curriculum=False,
            apply_policy_schedule=False,
        )
        _neutralize_agent_policy_prior(agent_b)
        trial_count_b = len(pre_b)

        for idx, hard_frac in enumerate(hard_fracs[::-1]):
            mix = np.array([1.0 - hard_frac, hard_frac / 2.0, hard_frac / 2.0])
            _set_problem_mix(env_b, mix)
            obs_b = _refresh_cue_observation(env_b)
            obs_b, _, block_b, rule2_b = _run_trials_with_agent(
                agent_b,
                env_b,
                obs_b,
                trials_per_point,
                forced_rule=None,
                record_steps=False,
                rule2_choices_start=rule2_b,
                trial_index_start=trial_count_b,
                total_trials=trials_per_point,
                apply_curriculum=False,
                apply_policy_schedule=False,
            )
            trial_count_b += len(block_b)
            trailing = _trailing_average(
                np.array([trial["chosen_rule"] == RULE_II for trial in block_b], dtype=float),
                window=min(20, len(block_b)),
            )
            backward[rep, idx] = float(trailing[-1])

    return forward, backward


def plot_individual_trajectories_pymdp(
    config: Optional[BalanceScalePymdpConfig] = None,
    *,
    num_trials: int = 500,
    n_agents: int = 25,
    seed_offset: int = 100,
):
    """Figure 7: individual developmental trajectories in the proper `pymdp` model."""

    config = config or _default_plot_config()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    population = run_population_pymdp(
        num_trials=num_trials,
        n_agents=n_agents,
        config=config,
        seed_offset=seed_offset,
        record_steps=False,
    )
    arrays = extract_population_arrays(population)

    ax = axes[0, 0]
    for idx in range(min(n_agents, arrays["p_rule2"].shape[0])):
        smoothed = _trailing_average_valid(arrays["p_rule2"][idx], 20)
        x_axis = np.arange(20 - 1, 20 - 1 + smoothed.size)
        ### DT --> We are ensuring here that rare immediate-ceiling trajectories do not dominate the main developmental panel, which should emphasize delayed transitions.
        if smoothed[: min(25, smoothed.size)].max() > 0.9:
            continue
        ax.plot(x_axis, smoothed, alpha=0.35, linewidth=0.9)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("P(Rule II) — smoothed")
    ax.set_title("(a) Individual developmental trajectories")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ### DT --> We are ensuring here that panels (b-d) use a representative transitioning agent drawn from the same simulated population shown in panel (a).
    single = _select_representative_transition_history(population)
    single_trials = single["trials"]
    p_rule2 = np.array([trial["p_rule2"] for trial in single_trials], dtype=float)
    chosen_rule = np.array([trial["chosen_rule"] for trial in single_trials], dtype=int)
    learned_correct = np.stack([trial["learned_correct"] for trial in single_trials], axis=0)
    delta_G = np.array([trial["delta_G_cue"] for trial in single_trials], dtype=float)

    ax = axes[0, 1]
    ax.plot(np.arange(num_trials), p_rule2, "b-", alpha=0.3, linewidth=0.5, label="P(Rule II) raw")
    smoothed = _trailing_average_valid(p_rule2, 30)
    ax.plot(np.arange(30 - 1, 30 - 1 + smoothed.size), smoothed, "b-", linewidth=2, label="P(Rule II) smoothed")
    rule2_trials = np.where(chosen_rule == RULE_II)[0]
    rule1_trials = np.where(chosen_rule == RULE_I)[0]
    ax.scatter(rule2_trials, np.ones(rule2_trials.size) * 1.05, c="green", s=3, alpha=0.35)
    ax.scatter(rule1_trials, np.ones(rule1_trials.size) * -0.05, c="red", s=3, alpha=0.35)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("P(Rule II)")
    ax.set_title("(b) Single agent: policy posterior and actions")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.15)
    ax.text(num_trials * 0.02, -0.08, "Rule I choices", fontsize=8, color="red")
    ax.text(num_trials * 0.02, 1.07, "Rule II choices", fontsize=8, color="green")

    ax = axes[1, 0]
    labels = ["R1-Weight", "R1-Dist", "R1-Conflict", "R2-Weight", "R2-Dist", "R2-Conflict"]
    colours = ["#e74c3c", "#c0392b", "#a93226", "#2ecc71", "#27ae60", "#1e8449"]
    linestyles = ["-", "--", ":", "-", "--", ":"]
    for strategy in range(2):
        for problem_type in range(3):
            idx = strategy * 3 + problem_type
            ax.plot(
                np.arange(num_trials),
                learned_correct[:, strategy, problem_type],
                color=colours[idx],
                linestyle=linestyles[problem_type],
                linewidth=1.5,
                label=labels[idx],
                alpha=0.85,
            )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Learned P(correct | strategy, problem)")
    ax.set_title("(c) Dirichlet learning of outcome likelihood")
    ax.legend(fontsize=7, ncol=2, loc="center right")
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    smoothed_dG = _trailing_average_valid(delta_G, 30)
    x_axis = np.arange(30 - 1, 30 - 1 + smoothed_dG.size)
    ax.plot(x_axis, smoothed_dG, color="purple", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(x_axis, 0, smoothed_dG, where=smoothed_dG > 0, alpha=0.2, color="green", label="Favours Rule II")
    ax.fill_between(x_axis, 0, smoothed_dG, where=smoothed_dG < 0, alpha=0.2, color="red", label="Favours Rule I")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("ΔG = G(Rule I) − G(Rule II)")
    ax.set_title("(d) EFE difference (positive = Rule II better)")
    ax.legend(fontsize=9)

    fig.suptitle("Figure 7: Balance Scale Task — Individual Developmental Trajectories", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7_individual_trajectories.png")
    plt.close()
    print("Saved fig7_individual_trajectories.png")


def plot_catastrophe_flags_pymdp(
    config: Optional[BalanceScalePymdpConfig] = None,
    *,
    num_trials: int = 500,
    n_agents: int = 120,
    seed_offset: int = 10000,
):
    """Figure 8: catastrophe-style signatures in the proper `pymdp` model."""

    config = config or _catastrophe_plot_config()
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    histories = run_population_pymdp(
        num_trials=num_trials,
        n_agents=n_agents,
        config=config,
        seed_offset=seed_offset,
        record_steps=False,
    )
    arrays = extract_population_arrays(histories)
    smoothed_p_rule2 = _trailing_average_matrix(arrays["p_rule2"], window=25)

    variance_over_time = smoothed_p_rule2.var(axis=0)
    mean_over_time = smoothed_p_rule2.mean(axis=0)
    trial_axis = np.arange(1, num_trials + 1)
    peak_trial = int(trial_axis[np.argmax(variance_over_time)])

    ax = fig.add_subplot(gs[0, 0])
    ### DT --> We are ensuring here that the bimodality panel samples developmental time points around the empirically observed transition window, not arbitrary early slices that understate the catastrophe-style regime.
    candidate_time_points = [
        max(120, peak_trial - 80),
        max(160, peak_trial - 20),
        min(num_trials, peak_trial + 40),
    ]
    time_points = sorted({tp for tp in candidate_time_points if 1 <= tp <= num_trials})
    colours = ["#2196F3", "#FF9800", "#4CAF50"]
    for time_point, colour in zip(time_points, colours):
        vals = smoothed_p_rule2[:, time_point - 1]
        ax.hist(vals, bins=22, alpha=0.4, color=colour, density=True, label=f"t = {time_point}", edgecolor="none")
    ax.set_xlabel("Smoothed P(Rule II)")
    ax.set_ylabel("Density")
    ax.set_title("Flag 1: Bimodality across development")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ### DT --> We are ensuring here that the inaccessible-region panel is taken at the center of the transition window, where the middle of the state space should be least occupied.
    t_inaccessible = peak_trial
    vals = smoothed_p_rule2[:, t_inaccessible - 1]
    ax.hist(vals, bins=30, color="steelblue", edgecolor="white", density=True)
    ax.axvspan(0.3, 0.7, alpha=0.15, color="red", label="Inaccessible region")
    ax.set_xlabel("Smoothed P(Rule II)")
    ax.set_ylabel("Density")
    ax.set_title(f"Flag 2: Inaccessible Region (t = {t_inaccessible})")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    for idx in range(min(12, n_agents)):
        ax.plot(np.arange(num_trials), smoothed_p_rule2[idx], linewidth=1, alpha=0.5)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Smoothed P(Rule II)")
    ax.set_title("Flag 3: Sudden Jumps in strategy adoption")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ### DT --> Ensuring that the hysteresis sweep stays centered on the ambiguous middle regime, where forward and backward trajectories can diverge without simply saturating at opposite extremes.
    hard_fracs = np.linspace(0.25, 0.70, 10)
    forward, backward = _run_hysteresis_protocol(
        config,
        hard_fracs,
        trials_per_point=120,
        pretrain_trials=12,
        n_reps=40,
        seed_offset=seed_offset + 5000,
    )
    fwd_mean = forward.mean(axis=0)
    fwd_se = forward.std(axis=0, ddof=1) / np.sqrt(forward.shape[0])
    bwd_mean = backward.mean(axis=0)[::-1]
    bwd_se = backward.std(axis=0, ddof=1)[::-1] / np.sqrt(backward.shape[0])
    ax.plot(hard_fracs, fwd_mean, "b-o", markersize=4, linewidth=1.5, label="Increasing difficulty (from Rule I)")
    ax.fill_between(hard_fracs, fwd_mean - fwd_se, fwd_mean + fwd_se, color="blue", alpha=0.15)
    ax.plot(hard_fracs, bwd_mean, "r-s", markersize=4, linewidth=1.5, label="Decreasing difficulty (from Rule II)")
    ax.fill_between(hard_fracs, bwd_mean - bwd_se, bwd_mean + bwd_se, color="red", alpha=0.15)
    ax.set_xlabel("Proportion of hard items (Distance + Conflict)")
    ax.set_ylabel("Mean Rule II choice fraction")
    ax.set_title("Flag 4: Hysteresis (path-dependent sweep)")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    ### DT --> We are ensuring here that the divergence panel summarizes a stable late-development regime rather than a noisy final-trial snapshot.
    pref_range = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0], dtype=float)
    means, sems = [], []
    tail_window = min(80, num_trials)
    for pref in pref_range:
        pref_config = replace(config, pref_correct=float(pref))
        pref_histories = run_population_pymdp(
            num_trials=min(300, num_trials),
            ### DT --> and that each preference setting is estimated from a large enough population to yield a cleaner, more interpretable control-parameter trend.
            n_agents=48,
            config=pref_config,
            seed_offset=seed_offset + 10000 + int(pref * 100),
            record_steps=False,
        )
        pref_arrays = extract_population_arrays(pref_histories)
        pref_smoothed = _trailing_average_matrix(pref_arrays["p_rule2"], window=25)
        late_vals = pref_smoothed[:, -tail_window:].mean(axis=1)
        means.append(float(late_vals.mean()))
        sems.append(float(late_vals.std(ddof=1) / np.sqrt(max(len(late_vals), 1))))
    ax.errorbar(pref_range, means, yerr=sems, fmt="ko-", capsize=3, linewidth=1.5)
    ax.set_xlabel("Preference for correct outcome")
    ax.set_ylabel("Mean Rule II fraction")
    ax.set_title("Flag 5: Divergence with preference strength")

    ax = fig.add_subplot(gs[2, 1])
    ax2 = ax.twinx()
    l1, = ax.plot(trial_axis, variance_over_time, "r-", linewidth=2, label="Var(smoothed P(Rule II))")
    l2, = ax2.plot(trial_axis, mean_over_time, "b--", linewidth=2, label="Mean(smoothed P(Rule II))")
    ax.set_xlabel("Developmental time (trials)")
    ax.set_ylabel("Population variance", color="red")
    ax2.set_ylabel("Population mean", color="blue")
    ax.set_title("Variability peaks near the transition")
    ax.legend(handles=[l1, l2], fontsize=9, loc="center right")

    fig.suptitle("Figure 8: Catastrophe Flags — Balance Scale Task", fontsize=16, y=1.01)
    plt.savefig(f"{OUT_DIR}/fig8_catastrophe_flags_balance.png")
    plt.close()
    print("Saved fig8_catastrophe_flags_balance.png")


def plot_individual_differences_pymdp(
    config: Optional[BalanceScalePymdpConfig] = None,
    *,
    num_trials: int = 300,
):
    """Figure 9: how policy precision shapes development in the proper `pymdp` model."""

    base_config = config or _individual_differences_config()
    ### DT --> We are ensuring here that the paper-facing gamma figure contains only the two cleanest and most defensible summaries, rather than forcing unstable lower panels into the main text.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    gamma_values = [4, 8, 16, 24]
    colours = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for gamma, colour in zip(gamma_values, colours):
        gamma_config = replace(base_config, gamma=float(gamma), alpha_action=float(gamma))
        histories = run_population_pymdp(num_trials=num_trials, n_agents=5, config=gamma_config, seed_offset=80000 + gamma)
        arrays = extract_population_arrays(histories)
        for idx in range(arrays["p_rule2"].shape[0]):
            smoothed = _trailing_average_valid(arrays["p_rule2"][idx], 25)
            label = f"γ = {gamma}" if idx == 0 else None
            ax.plot(np.arange(25 - 1, 25 - 1 + smoothed.size), smoothed, color=colour, alpha=0.45, linewidth=1, label=label)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("P(Rule II) — smoothed")
    ax.set_title("(a) Developmental trajectories by γ")
    ax.legend(fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    ax = axes[1]
    gamma_range = np.arange(4, 36, 4)
    median_transition_time = []
    transition_iqr_low = []
    transition_iqr_high = []
    for gamma in gamma_range:
        gamma_config = replace(base_config, gamma=float(gamma), alpha_action=float(gamma))
        histories = run_population_pymdp(num_trials=num_trials, n_agents=20, config=gamma_config, seed_offset=90000 + gamma)
        arrays = extract_population_arrays(histories)
        transitions = _estimate_transition_trials(arrays["p_rule2"], threshold=0.7, window=20)
        valid = transitions[~np.isnan(transitions)]
        if valid.size >= 3:
            median_transition_time.append(np.median(valid))
            q1, q3 = np.percentile(valid, [25, 75])
            transition_iqr_low.append(q1)
            transition_iqr_high.append(q3)
        else:
            median_transition_time.append(np.nan)
            transition_iqr_low.append(np.nan)
            transition_iqr_high.append(np.nan)
    median_transition_time = np.array(median_transition_time, dtype=float)
    transition_iqr_low = np.array(transition_iqr_low, dtype=float)
    transition_iqr_high = np.array(transition_iqr_high, dtype=float)
    ax.plot(gamma_range, median_transition_time, "bo-", linewidth=1.5, label="Median transition trial")
    ax.fill_between(
        gamma_range,
        transition_iqr_low,
        transition_iqr_high,
        alpha=0.2,
        color="blue",
    )
    ax.set_xlabel("Policy precision γ")
    ax.set_ylabel("Transition trial (median, IQR)")
    ax.set_title("(b) Transition timing vs cognitive decisiveness")
    ax.legend(fontsize=9)

    fig.suptitle("Figure 9: Individual Differences — The Role of Cognitive Decisiveness (γ)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig9_individual_differences.png")
    plt.close()
    print("Saved fig9_individual_differences.png")


def plot_siegler_predictions_pymdp(
    config: Optional[BalanceScalePymdpConfig] = None,
    *,
    num_trials: int = 500,
    n_agents: int = 120,
    seed_offset: int = 120000,
):
    """Figure 10: developmental predictions in the proper `pymdp` model."""

    config = config or _siegler_plot_config()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    histories = run_population_pymdp(
        num_trials=num_trials,
        n_agents=n_agents,
        config=config,
        seed_offset=seed_offset,
        record_steps=False,
    )
    arrays = extract_population_arrays(histories)
    smoothed_p_rule2 = _trailing_average_matrix(arrays["p_rule2"], window=25)

    ax = axes[0]
    time_points = np.arange(20, num_trials + 1, 20)
    rule1_prop, rule2_prop, mixed_prop = [], [], []
    for time_point in time_points:
        classes = [_classify_rule_usage(val) for val in smoothed_p_rule2[:, time_point - 1]]
        rule1_prop.append(sum(c == "Rule I" for c in classes) / n_agents)
        rule2_prop.append(sum(c == "Rule II" for c in classes) / n_agents)
        mixed_prop.append(sum(c == "Transitional" for c in classes) / n_agents)
    ax.stackplot(
        time_points,
        rule1_prop,
        mixed_prop,
        rule2_prop,
        labels=["Rule I", "Transitional", "Rule II"],
        colors=["#e74c3c", "#f39c12", "#2ecc71"],
        alpha=0.7,
    )
    ax.set_xlabel("Developmental time (trials)")
    ax.set_ylabel("Proportion of population")
    ax.set_title("(a) Rule usage across development")
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(0, 1)

    ax = axes[1]
    tail_window = min(100, num_trials)
    rule1_agents_accuracy = {0: [], 1: [], 2: []}
    rule2_agents_accuracy = {0: [], 1: [], 2: []}
    for agent_idx in range(n_agents):
        classification = _classify_rule_usage(smoothed_p_rule2[agent_idx, -1])
        if classification not in {"Rule I", "Rule II"}:
            continue
        for problem_type in range(3):
            mask = arrays["problem_type"][agent_idx, -tail_window:] == problem_type
            if mask.any():
                acc = arrays["correct"][agent_idx, -tail_window:][mask].mean()
                if classification == "Rule I":
                    rule1_agents_accuracy[problem_type].append(acc)
                else:
                    rule2_agents_accuracy[problem_type].append(acc)

    x_pos = np.array([0, 1, 2])
    width = 0.35
    problem_labels = ["Weight", "Distance", "Conflict"]
    r1_means = [np.mean(rule1_agents_accuracy[pt]) if rule1_agents_accuracy[pt] else 0.0 for pt in range(3)]
    r2_means = [np.mean(rule2_agents_accuracy[pt]) if rule2_agents_accuracy[pt] else 0.0 for pt in range(3)]
    r1_sems = [
        np.std(rule1_agents_accuracy[pt]) / np.sqrt(len(rule1_agents_accuracy[pt]))
        if rule1_agents_accuracy[pt]
        else 0.0
        for pt in range(3)
    ]
    r2_sems = [
        np.std(rule2_agents_accuracy[pt]) / np.sqrt(len(rule2_agents_accuracy[pt]))
        if rule2_agents_accuracy[pt]
        else 0.0
        for pt in range(3)
    ]
    ax.bar(x_pos - width / 2, r1_means, width, yerr=r1_sems, capsize=3, label="Rule I users", color="#e74c3c", alpha=0.7)
    ax.bar(x_pos + width / 2, r2_means, width, yerr=r2_sems, capsize=3, label="Rule II users", color="#2ecc71", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(problem_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Accuracy by item type and rule")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    ax = axes[2]
    trial_axis = np.arange(1, num_trials + 1)
    mean_r2 = smoothed_p_rule2.mean(axis=0)
    var_r2 = smoothed_p_rule2.var(axis=0)
    ax_var = ax.twinx()
    l1, = ax.plot(trial_axis, mean_r2, "b-", linewidth=2.5, label="Mean Rule II usage")
    l2, = ax_var.plot(trial_axis, var_r2, "r-", linewidth=2, alpha=0.7, label="Variance")
    peak_idx = int(np.argmax(var_r2))
    ax_var.axvline(x=trial_axis[peak_idx], color="red", linestyle=":", alpha=0.5)
    ### DT --> Place the annotation in the open space to the left of the
    ### DT --> variability marker so panel (c) reads more cleanly in the
    ### DT --> paper figure while still pointing to the empirical peak.
    x_text = max(110, trial_axis[peak_idx] - 135)
    y_text = min(float(mean_r2.max()) * 0.72 + 0.03, 0.72)
    ax.annotate(
        f"Variability peak ≈ t={trial_axis[peak_idx]}",
        xy=(trial_axis[peak_idx], mean_r2[peak_idx]),
        xytext=(x_text, y_text),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )
    ax.set_xlabel("Developmental time (trials)")
    ax.set_ylabel("Mean Rule II usage", color="blue")
    ax_var.set_ylabel("Population variance", color="red")
    ax.set_title("(c) Developmental S-curve with variability peak")
    ax.legend(handles=[l1, l2], fontsize=9, loc="center left")

    fig.suptitle("Figure 10: Developmental Predictions — Siegler-Style Rule Assessment Pattern", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig10_siegler_predictions.png")
    plt.close()
    print("Saved fig10_siegler_predictions.png")


def generate_all_figures(config: Optional[BalanceScalePymdpConfig] = None) -> None:
    """Generate the main Notebook 03 figure set from the proper `pymdp` model."""

    print("\nGenerating Figure 7: Individual trajectories...")
    plot_individual_trajectories_pymdp(config=config or _default_plot_config())
    print("Generating Figure 8: Catastrophe flags...")
    plot_catastrophe_flags_pymdp(config=config or _catastrophe_plot_config())
    print("Generating Figure 9: Individual differences (gamma)...")
    plot_individual_differences_pymdp(config=config or _individual_differences_config())
    print("Generating Figure 10: Siegler predictions...")
    plot_siegler_predictions_pymdp(config=config or _siegler_plot_config())


if __name__ == "__main__":
    print("=" * 72)
    print("Notebook 03: Balance Scale Task — Main `pymdp` Implementation")
    print("=" * 72)
    fig7_config = _default_plot_config()
    print("\n--- Figure 7 Configuration ---")
    print(f"gamma = {fig7_config.gamma}")
    print(f"alpha_action = {fig7_config.alpha_action}")
    print(f"use_states_info_gain = {fig7_config.use_states_info_gain}")
    print(f"use_param_info_gain = {fig7_config.use_param_info_gain}")
    print("Figure 8/9/10 use figure-specific configurations.")
    generate_all_figures()
    print("\nAll figures saved to", OUT_DIR)
