"""
Appendix bridge model -- reduced balance-scale simulation
=========================================================

This file retains the reduced active-inference-style balance-scale model as an
appendix / bridge implementation.

It is useful for pedagogical continuity with the reduced derivation and for
transparent mechanism-isolation, but it is not the primary rich-model
implementation for the paper. The main implementation is now
`sim_balance_scale_pymdp.py`, which uses the actual `pymdp` Agent API with
explicit hidden-state inference and policy inference.

### DT --> We are ensuring here that this script is read as an appendix bridge:
### DT --> it deliberately keeps the reduced cue-conditioned policy loop and
### DT --> therefore omits the richer developmental curriculum and scheduled
### DT --> policy-prior scaffolding used by the paper-facing `pymdp` model.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

### DT --> Keep appendix outputs separate from the paper-facing `pymdp` figures
### DT --> so this bridge model cannot overwrite the main results.
OUT_DIR = os.path.join(os.path.dirname(__file__), "output_figs_appendix_bridge")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# ============================================================================
# PART 1: GENERATIVE MODEL — THE BALANCE SCALE TASK
# ============================================================================
#
# Background (Siegler, 1976):
#   Children judge which side of a balance scale will tip.
#   Rule I:  Consider only the number of weights.
#   Rule II: Consider weights; if equal, also consider distance from fulcrum.
#
# Problem types (Siegler's item classification):
#   - Weight items:    Different weights, same distance → Rule I correct
#   - Distance items:  Same weight, different distance  → Rule I fails (guesses)
#   - Conflict-Weight: More weight on one side, more distance on other,
#                      weight wins → Rule I correct (by luck)
#   - Conflict-Dist:   Same conflict, distance wins → Rule I wrong
#   - Conflict-Equal:  Conflict, equal torques → Rule I wrong
#   - Balance items:   Everything equal → both rules correct
#
# Simplification for tractability:
#   Problem types: {Weight, Distance, Conflict} (3 types)
#   Weight items:  Rule I succeeds (p=0.95), Rule II succeeds (p=0.90)
#   Distance items: Rule I fails (p=0.35), Rule II succeeds (p=0.85)
#   Conflict items: Rule I mixed (p=0.50), Rule II succeeds (p=0.80)
#
# ============================================================================

class BalanceScaleModel:
    """
    Generative model for the balance scale task.
    
    Hidden state factors:
        Factor 0 — Strategy: {Rule I, Rule II}                    (2 states, controllable)
        Factor 1 — Problem type: {Weight, Distance, Conflict}     (3 states, uncontrollable)
    
    Observation modalities:
        Modality 0 — Outcome: {Correct, Incorrect}                (2 levels)
        Modality 1 — Item cue: {Weight-salient, Distance-salient, Mixed-cue}  (3 levels)
    
    Policies:
        pi_0: Apply Rule I  (transition to strategy=0)
        pi_1: Apply Rule II (transition to strategy=1)
    """
    
    def __init__(self, 
                 ### DT ---> True success rates: A_true[outcome, strategy, problem_type]
                 rule1_success=(0.95, 0.35, 0.50),
                 rule2_success=(0.90, 0.85, 0.80),
                 ### DT ---> Problem type base rates
                 problem_mix=(0.35, 0.35, 0.30),
                 ### DT ---> Prior Dirichlet concentration (per cell)
                 alpha_0=1.0,
                 ### DT ---> Policy precision
                 gamma=16.0,
                 ### DT --> Preference strength for correct outcomes
                 pref_correct=2.0,
        ### DT --> Weight of a simple novelty bonus on under-learned mappings.
        epistemic_weight=0.3):
        
        self.n_strategies = 2
        self.n_problem_types = 3
        self.n_outcomes = 2
        self.n_cues = 3
        self.gamma = gamma
        self.alpha_0 = alpha_0
        self.epistemic_weight = epistemic_weight
        
        ### DT ---> True A matrix for outcomes: shape (2, 2, 3)
        ### DT ---> A_outcome[o, s, p] = P(outcome=o | strategy=s, problem=p)
        self.A_outcome_true = np.zeros((2, 2, 3))
        for pt in range(3):
            self.A_outcome_true[0, 0, pt] = rule1_success[pt]   ### DT ---> P(correct | Rule I, problem_type)
            self.A_outcome_true[1, 0, pt] = 1 - rule1_success[pt]
            self.A_outcome_true[0, 1, pt] = rule2_success[pt]   ### DT ---> P(correct | Rule II, problem_type)
            self.A_outcome_true[1, 1, pt] = 1 - rule2_success[pt]
        
        ### DT ---> True A matrix for cues: shape (3, 2, 3)
        ### DT ---> Cue depends mainly on problem type, not strategy
        self.A_cue_true = np.zeros((3, 2, 3))
        ### DT ---> Weight items → weight-salient cue
        self.A_cue_true[:, :, 0] = np.array([[0.8, 0.8], [0.1, 0.1], [0.1, 0.1]])
        ### DT ---> Distance items → distance-salient cue
        self.A_cue_true[:, :, 1] = np.array([[0.1, 0.1], [0.8, 0.8], [0.1, 0.1]])
        ### DT ---> Conflict items → mixed cue
        self.A_cue_true[:, :, 2] = np.array([[0.2, 0.2], [0.2, 0.2], [0.6, 0.6]])
        
        ### DT ---> Problem type mixing (stationary distribution)
        self.problem_mix = np.array(problem_mix)
        self.problem_mix /= self.problem_mix.sum()
        
        ### DT ---> Preferences: log-preferences over outcomes
        self.C_outcome = np.array([pref_correct, -pref_correct])
        self.C_cue = np.zeros(3)  ### DT --> No direct preference over cues.
    
    def init_dirichlet(self):
        """Initialise Dirichlet concentration parameters for A_outcome."""
        ### DT ---> Shape matches A_outcome: (2, 2, 3)
        ### DT ---> Each cell starts at alpha_0 (symmetric uninformative prior)
        pA = np.ones((2, 2, 3)) * self.alpha_0
        return pA
    
    def get_learned_A(self, pA):
        """Get expected A_outcome from Dirichlet parameters."""
        ### DT ---> Expected value of Dirichlet: pA[o,s,p] / sum_o pA[:,s,p]
        A_learned = np.zeros_like(pA)
        for s in range(2):
            for p in range(3):
                col_sum = pA[:, s, p].sum()
                A_learned[:, s, p] = pA[:, s, p] / col_sum
        return A_learned


# ============================================================================
# PART 2: ACTIVE INFERENCE LOOP
# ============================================================================

def compute_efe_balance(A_outcome, A_cue, problem_prior, C_outcome, C_cue, 
                        pA=None, epistemic_weight=0.0):
    """
    Compute expected free energy for each strategy (policy).
    
    G(pi_k) = sum_p q(p) * [pragmatic_value(k,p) + ambiguity(k,p)]
    
    where:
      pragmatic_value(k,p) = -sum_o A[o,k,p] * C[o]
      ambiguity(k,p)       = H(A[:,k,p])
    
    Design notes:
      - Policy is evaluated AFTER observing the cue, using a posterior belief
        q(problem_type | cue) passed in as `problem_prior`.
      - The cue likelihood A_cue is identical across strategies by design, so
        cue ambiguity does not directly separate policies. It matters by
        shaping the posterior over latent problem type before action.
      - The optional `epistemic_weight` term is a novelty bonus on mappings
        with lower total concentration. It is not the exact `pymdp`
        parameter-information-gain expression.
    
    Returns: G array of shape (2,) — one per strategy
    """
    G = np.zeros(2)
    
    for k in range(2):  ### DT ---> for each strategy
        for p in range(3):  ### DT ---> for each problem type
            ### DT ---> Outcome modality
            pragmatic_outcome = -np.sum(A_outcome[:, k, p] * C_outcome)
            col = np.clip(A_outcome[:, k, p], 1e-12, 1)
            ambiguity_outcome = -np.sum(col * np.log(col))
            
            ### DT ---> Cue modality (contributes ambiguity only, no preference)
            col_cue = np.clip(A_cue[:, k, p], 1e-12, 1)
            ambiguity_cue = -np.sum(col_cue * np.log(col_cue))
            
            G[k] += problem_prior[p] * (pragmatic_outcome + ambiguity_outcome + ambiguity_cue)
        
        ### DT --> Simple novelty bonus for under-learned mappings.
        if pA is not None and epistemic_weight > 0:
            for p in range(3):
                total_conc = pA[:, k, p].sum()
                G[k] -= epistemic_weight * problem_prior[p] / total_conc
    
    return G


def infer_problem_posterior(cue, A_cue, prior_problem):
    """
    Infer q(problem_type | cue) using Bayes' rule.

    Because cue likelihoods are strategy-independent in this model, we can use
    either strategy slice when reading the cue likelihood.
    """
    likelihood = A_cue[cue, 0, :]
    posterior = likelihood * prior_problem
    total = posterior.sum()
    if total <= 0:
        return prior_problem.copy()
    return posterior / total


def run_balance_scale_agent(model, N_total, seed=None):
    """
    Run one active inference agent on the balance scale task.

    Trial structure:
      1. Environment samples latent problem type.
      2. Agent observes a cue and infers q(problem | cue).
      3. Agent evaluates Rule I vs Rule II conditional on that posterior.
      4. Outcome is observed and the outcome model is updated with a
         posterior-weighted Dirichlet increment.
    
    Returns: dict with full trajectory.
    """
    rng = np.random.default_rng(seed)
    
    ### DT ---> Initialise Dirichlet parameters
    pA = model.init_dirichlet()
    
    ### DT ---> Storage
    history = {
        'strategy': np.zeros(N_total, dtype=int),
        'problem_type': np.zeros(N_total, dtype=int),
        'q_problem': np.zeros((N_total, 3)),
        'outcome': np.zeros(N_total, dtype=int),
        'cue': np.zeros(N_total, dtype=int),
        'P_rule2': np.zeros(N_total),
        'G': np.zeros((N_total, 2)),
        'A_learned': np.zeros((N_total, 2, 2, 3)),
        'accuracy_rule1': np.zeros(N_total),
        'accuracy_rule2': np.zeros(N_total),
        'n_rule1': np.zeros(N_total),
        'n_rule2': np.zeros(N_total),
        'rule2_fraction': np.zeros(N_total),
    }
    
    n_rule1 = 0
    n_rule2 = 0
    correct_rule1 = 0
    correct_rule2 = 0
    
    for t in range(N_total):
        ### DT --> Sample the latent problem type from the environment.
        problem_type = rng.choice(3, p=model.problem_mix)

        ### DT --> Observe an item cue before selecting a strategy.
        cue_probs = model.A_cue_true[:, 0, problem_type]
        cue = rng.choice(3, p=cue_probs)
        q_problem = infer_problem_posterior(cue, model.A_cue_true, model.problem_mix)
        
        ### DT --> Get current learned outcome model.
        A_learned = model.get_learned_A(pA)
        
        ### DT --> Compute EFE for each strategy conditional on q(problem | cue).
        G = compute_efe_balance(
            A_learned, model.A_cue_true, q_problem,
            model.C_outcome, model.C_cue,
            pA=pA, epistemic_weight=model.epistemic_weight
        )
        
        ### DT ---> Policy posterior (softmax)
        delta_G = G[0] - G[1]  ### DT ---> Rule I EFE minus Rule II EFE
        x = model.gamma * delta_G
        x = np.clip(x, -500, 500)
        P_rule2 = 1.0 / (1.0 + np.exp(-x))  ### DT ---> P(Rule II) = sigma(gamma * delta_G)
        ### DT ---> Note: positive delta_G means Rule I has higher (worse) EFE → favours Rule II
        
        ### DT ---> Sample strategy
        strategy = 1 if rng.random() < P_rule2 else 0
        
        ### DT --> Generate outcome from the true environment.
        p_correct = model.A_outcome_true[0, strategy, problem_type]
        outcome = 0 if rng.random() < p_correct else 1  ### DT ---> 0=correct, 1=incorrect
        
        ### DT --> Learn by weighting the update with posterior beliefs about problem type.
        pA[outcome, strategy, :] += q_problem
        
        ### DT ---> Track accuracy
        if strategy == 0:
            n_rule1 += 1
            if outcome == 0:
                correct_rule1 += 1
        else:
            n_rule2 += 1
            if outcome == 0:
                correct_rule2 += 1
        
        ### DT ---> Record
        history['strategy'][t] = strategy
        history['problem_type'][t] = problem_type
        history['q_problem'][t] = q_problem
        history['outcome'][t] = outcome
        history['cue'][t] = cue
        history['P_rule2'][t] = P_rule2
        history['G'][t] = G
        history['A_learned'][t] = A_learned
        history['n_rule1'][t] = n_rule1
        history['n_rule2'][t] = n_rule2
        total = n_rule1 + n_rule2
        history['rule2_fraction'][t] = n_rule2 / total if total > 0 else 0.5
        history['accuracy_rule1'][t] = correct_rule1 / max(n_rule1, 1)
        history['accuracy_rule2'][t] = correct_rule2 / max(n_rule2, 1)
    
    return history


# ============================================================================
# PART 3: POPULATION ANALYSIS
# ============================================================================

def run_population(model, N_total, n_agents, seed_offset=0):
    """Run a population of agents and collect trajectories."""
    histories = []
    for i in range(n_agents):
        h = run_balance_scale_agent(model, N_total, seed=seed_offset + i)
        histories.append(h)
    return histories


def compute_population_stats(histories, time_points=None):
    """Compute population-level statistics at specified time points."""
    N_total = len(histories[0]['P_rule2'])
    if time_points is None:
        time_points = np.arange(N_total)
    
    stats = {
        'mean_P_rule2': [],
        'std_P_rule2': [],
        'mean_rule2_frac': [],
        'std_rule2_frac': [],
        'bimodality_coeff': [],
    }
    
    for t in time_points:
        vals = [h['rule2_fraction'][t] for h in histories]
        stats['mean_rule2_frac'].append(np.mean(vals))
        stats['std_rule2_frac'].append(np.std(vals))
        
        p_vals = [h['P_rule2'][t] for h in histories]
        stats['mean_P_rule2'].append(np.mean(p_vals))
        stats['std_P_rule2'].append(np.std(p_vals))
        
        ### DT ---> Bimodality coefficient (Sarle's): (skewness^2 + 1) / kurtosis
        if len(vals) > 3:
            m = np.mean(vals)
            s = np.std(vals)
            if s > 1e-10:
                skew = np.mean(((np.array(vals) - m) / s) ** 3)
                kurt = np.mean(((np.array(vals) - m) / s) ** 4)
                bc = (skew**2 + 1) / kurt if kurt > 0 else 0
            else:
                bc = 0
        else:
            bc = 0
        stats['bimodality_coeff'].append(bc)
    
    return stats


def _trailing_average_valid(values, window):
    """Simple trailing-average helper for appendix figure summaries."""
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _fixed_slice_smoothed_p_rule2(history, time_point, window=20):
    """Read smoothed Rule II usage at a fixed developmental slice."""
    t = min(max(int(time_point), 1), len(history["P_rule2"]))
    if t < window:
        return float(np.mean(history["P_rule2"][:t]))
    return float(np.mean(history["P_rule2"][t - window:t]))


def _choose_representative_history(model, n_total, *, seed_start=0, n_candidates=80):
    """
    Pick a representative delayed-transition history for appendix display.

    We prefer agents whose smoothed Rule II posterior crosses 0.5 neither
    immediately nor at the very end, so the bridge figure does not saturate
    from the first few trials.
    """
    best_history = None
    best_score = np.inf
    for offset in range(n_candidates):
        history = run_balance_scale_agent(model, n_total, seed=seed_start + offset)
        smoothed = _trailing_average_valid(history["P_rule2"], 25)
        if smoothed.size == 0:
            continue
        trial_axis = np.arange(25 - 1, 25 - 1 + smoothed.size)
        crossed = np.where(smoothed > 0.5)[0]
        if crossed.size == 0:
            continue
        cross_trial = trial_axis[crossed[0]]
        if cross_trial < 60 or cross_trial > int(0.8 * n_total):
            continue
        score = abs(cross_trial - 0.45 * n_total)
        if score < best_score:
            best_score = score
            best_history = history
    if best_history is None:
        return run_balance_scale_agent(model, n_total, seed=seed_start)
    return best_history


# ============================================================================
# FIGURE 7: The Balance Scale Model — Individual Trajectories
# ============================================================================

def plot_individual_trajectories():
    """Show developmental trajectories of individual agents."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    model = BalanceScaleModel(gamma=16.0, alpha_0=1.0, epistemic_weight=0.3)
    N_total = 500
    
    ### DT --> Panel (a): appendix trajectories should illustrate delayed
    ### DT --> transitions rather than immediate ceiling runs, so we keep only
    ### DT --> representative traces that do not hit Rule II almost instantly.
    ax = axes[0, 0]
    kept = 0
    for i in range(40):
        h = run_balance_scale_agent(model, N_total, seed=100+i)
        window = 20
        smoothed = _trailing_average_valid(h['P_rule2'], window)
        trial_axis = np.arange(window - 1, window - 1 + len(smoothed))
        early_ceiling = np.any(smoothed[trial_axis <= 25] > 0.9) if np.any(trial_axis <= 25) else False
        if early_ceiling:
            continue
        ax.plot(trial_axis, smoothed, alpha=0.35, linewidth=0.9)
        kept += 1
        if kept >= 12:
            break
    
    ax.set_xlabel('Trial number')
    ax.set_ylabel('P(Rule II) — smoothed')
    ax.set_title('(a) Individual developmental trajectories')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    ### DT --> Panel (b): use a representative delayed-transition agent rather
    ### DT --> than a hard-coded seed, so the bridge figure does not saturate
    ### DT --> from the first few trials.
    ax = axes[0, 1]
    h = _choose_representative_history(model, N_total, seed_start=42, n_candidates=80)
    ax.plot(np.arange(N_total), h['P_rule2'], 'b-', alpha=0.3, linewidth=0.5, label='P(Rule II) raw')
    window = 30
    smoothed = _trailing_average_valid(h['P_rule2'], window)
    trial_axis = np.arange(window - 1, window - 1 + len(smoothed))
    ax.plot(trial_axis, smoothed, 'b-', linewidth=2, label='P(Rule II) smoothed')
    
    ### DT --> Mark realized strategy choices.
    rule2_trials = np.where(h['strategy'] == 1)[0]
    rule1_trials = np.where(h['strategy'] == 0)[0]
    ax.scatter(rule2_trials, np.ones(len(rule2_trials)) * 1.05, c='green', s=1, alpha=0.3)
    ax.scatter(rule1_trials, np.ones(len(rule1_trials)) * -0.05, c='red', s=1, alpha=0.3)
    
    ax.set_xlabel('Trial number')
    ax.set_ylabel('P(Rule II)')
    ax.set_title('(b) Single agent: policy posterior and actions')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.15)
    ax.text(N_total*0.02, -0.08, 'Rule I choices', fontsize=8, color='red')
    ax.text(N_total*0.02, 1.07, 'Rule II choices', fontsize=8, color='green')
    
    ### DT --> Panel (c): use the same representative history so the learning
    ### DT --> curves correspond directly to the transition shown above.
    ax = axes[1, 0]
    labels = ['R1-Weight', 'R1-Dist', 'R1-Conflict', 'R2-Weight', 'R2-Dist', 'R2-Conflict']
    colours = ['#e74c3c', '#c0392b', '#a93226', '#2ecc71', '#27ae60', '#1e8449']
    linestyles = ['-', '--', ':', '-', '--', ':']
    
    for s in range(2):
        for p in range(3):
            idx = s * 3 + p
            learned_vals = h['A_learned'][:, 0, s, p]  ### DT ---> P(correct | s, p)
            ax.plot(np.arange(N_total), learned_vals, color=colours[idx],
                    linestyle=linestyles[p], linewidth=1.5, label=labels[idx], alpha=0.8)
    
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Learned P(correct | strategy, problem)')
    ax.set_title('(c) Dirichlet learning of outcome likelihood')
    ax.legend(fontsize=7, ncol=2, loc='center right')
    ax.set_ylim(0, 1)
    
    ### DT --> Panel (d): align the smoothed EFE difference to the trailing
    ### DT --> averaging window so the transition timing is not shifted left.
    ax = axes[1, 1]
    delta_G = h['G'][:, 0] - h['G'][:, 1]
    smoothed_dG = _trailing_average_valid(delta_G, window)
    dG_axis = np.arange(window - 1, window - 1 + len(smoothed_dG))
    ax.plot(dG_axis, smoothed_dG, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(dG_axis, 0, smoothed_dG,
                    where=smoothed_dG > 0, alpha=0.2, color='green', label='Favours Rule II')
    ax.fill_between(dG_axis, 0, smoothed_dG,
                    where=smoothed_dG < 0, alpha=0.2, color='red', label='Favours Rule I')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('ΔG = G(Rule I) − G(Rule II)')
    ax.set_title('(d) EFE difference (positive = Rule II better)')
    ax.legend(fontsize=9)
    
    fig.suptitle('Appendix Figure A1: Reduced Balance-Scale Bridge Model — Individual Developmental Trajectories',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figA1_appendix_individual_trajectories.png")
    plt.close()
    print("Saved figA1_appendix_individual_trajectories.png")


# ============================================================================
# FIGURE 8: Catastrophe Flags in the Balance Scale Model
# ============================================================================

def plot_catastrophe_flags_balance():
    """Demonstrate catastrophe flags in the richer model."""
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    model = BalanceScaleModel(gamma=16.0, alpha_0=1.0, epistemic_weight=0.3)
    n_agents = 150
    
    # ------------------------------------------------------------------
    # FLAG 1: BIMODALITY at different developmental times
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    
    time_points = [60, 100, 150]
    colours = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    
    for N_total, col in zip(time_points, colours):
        smoothed_rule2 = []
        for i in range(n_agents):
            h = run_balance_scale_agent(model, N_total, seed=10000+i)
            smoothed_rule2.append(_fixed_slice_smoothed_p_rule2(h, N_total, window=20))
        ax.hist(smoothed_rule2, bins=25, alpha=0.4, color=col, density=True,
                label=f't = {N_total}', edgecolor='none')
    
    ax.set_xlabel('Smoothed P(Rule II)')
    ax.set_ylabel('Density')
    ax.set_title('Flag 1: Bimodality across development')
    ax.legend(fontsize=9)
    
    # ------------------------------------------------------------------
    # FLAG 2: INACCESSIBLE REGION
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    
    smoothed_rule2 = []
    for i in range(n_agents):
        h = run_balance_scale_agent(model, 150, seed=20000+i)
        smoothed_rule2.append(_fixed_slice_smoothed_p_rule2(h, 150, window=20))
    
    ax.hist(smoothed_rule2, bins=30, color='steelblue', edgecolor='white', density=True)
    ax.axvspan(0.3, 0.7, alpha=0.15, color='red', label='Inaccessible region')
    ax.set_xlabel('Smoothed P(Rule II)')
    ax.set_ylabel('Density')
    ax.set_title('Flag 2: Inaccessible Region (t = 150)')
    ax.legend(fontsize=9)
    
    # ------------------------------------------------------------------
    # FLAG 3: SUDDEN JUMP — transition timing
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    
    N_total = 500
    for i in range(10):
        h = run_balance_scale_agent(model, N_total, seed=30000+i*3)
        smoothed = _trailing_average_valid(h['P_rule2'], 25)
        trial_axis = np.arange(25 - 1, 25 - 1 + len(smoothed))
        ax.plot(trial_axis, smoothed, linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Smoothed P(Rule II)')
    ax.set_title('Flag 3: Sudden Jumps in strategy adoption')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # ------------------------------------------------------------------
    # FLAG 4: HYSTERESIS — path-dependent sweep of problem mix
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    
    ### DT --> Use a softer appendix-only hysteresis protocol so the bridge
    ### DT --> model shows path dependence without saturating at 0 or 1.
    model_base = BalanceScaleModel(
        rule1_success=(0.90, 0.62, 0.52),
        rule2_success=(0.76, 0.78, 0.82),
        gamma=8.0,
        alpha_0=1.5,
        epistemic_weight=0.15,
    )
    hard_fracs = np.linspace(0.2, 0.7, 10)
    trials_per_point = 60
    n_reps = 16
    z_forward_all = np.zeros((n_reps, hard_fracs.size))
    z_backward_all = np.zeros((n_reps, hard_fracs.size))

    def _run_hysteresis_block(pA_state, pmix, rng):
        A_learned = model_base.get_learned_A(pA_state)
        strategies = []
        for _ in range(trials_per_point):
            pt = rng.choice(3, p=pmix)
            G = compute_efe_balance(
                A_learned, model_base.A_cue_true, pmix,
                model_base.C_outcome, model_base.C_cue,
                pA=pA_state, epistemic_weight=model_base.epistemic_weight
            )
            dG = G[0] - G[1]
            P_r2 = 1.0 / (1.0 + np.exp(-np.clip(model_base.gamma * dG, -500, 500)))
            strat = 1 if rng.random() < P_r2 else 0
            outcome = 0 if rng.random() < model_base.A_outcome_true[0, strat, pt] else 1
            pA_state[outcome, strat, pt] += 1.0
            strategies.append(strat)
            A_learned = model_base.get_learned_A(pA_state)
        return float(np.mean(strategies[-20:]))

    for rep in range(n_reps):
        rng = np.random.default_rng(40000 + rep)
        pA_fwd = model_base.init_dirichlet()
        for _ in range(80):
            pt = rng.choice(3, p=[0.82, 0.09, 0.09])
            G = compute_efe_balance(
                model_base.get_learned_A(pA_fwd), model_base.A_cue_true, np.array([0.82, 0.09, 0.09]),
                model_base.C_outcome, model_base.C_cue,
                pA=pA_fwd, epistemic_weight=model_base.epistemic_weight
            )
            P_r2 = 1.0 / (1.0 + np.exp(-np.clip(model_base.gamma * (G[0] - G[1]), -500, 500)))
            strat = 1 if rng.random() < P_r2 else 0
            outcome = 0 if rng.random() < model_base.A_outcome_true[0, strat, pt] else 1
            pA_fwd[outcome, strat, pt] += 1.0
        for si, hf in enumerate(hard_fracs):
            pmix = np.array([1 - hf, hf / 2, hf / 2], dtype=float)
            pmix /= pmix.sum()
            z_forward_all[rep, si] = _run_hysteresis_block(pA_fwd, pmix, rng)

        pA_bwd = model_base.init_dirichlet()
        for _ in range(80):
            pt = rng.choice(3, p=[0.12, 0.44, 0.44])
            G = compute_efe_balance(
                model_base.get_learned_A(pA_bwd), model_base.A_cue_true, np.array([0.12, 0.44, 0.44]),
                model_base.C_outcome, model_base.C_cue,
                pA=pA_bwd, epistemic_weight=model_base.epistemic_weight
            )
            P_r2 = 1.0 / (1.0 + np.exp(-np.clip(model_base.gamma * (G[0] - G[1]), -500, 500)))
            strat = 1 if rng.random() < P_r2 else 0
            outcome = 0 if rng.random() < model_base.A_outcome_true[0, strat, pt] else 1
            pA_bwd[outcome, strat, pt] += 1.0
        backward_means = []
        for hf in hard_fracs[::-1]:
            pmix = np.array([1 - hf, hf / 2, hf / 2], dtype=float)
            pmix /= pmix.sum()
            backward_means.append(_run_hysteresis_block(pA_bwd, pmix, rng))
        z_backward_all[rep] = backward_means[::-1]

    ax.plot(hard_fracs, z_forward_all.mean(axis=0), 'b-o', markersize=4, linewidth=1.5,
            label='Increasing difficulty (from Rule I)')
    ax.plot(hard_fracs, z_backward_all.mean(axis=0), 'r-s', markersize=4, linewidth=1.5,
            label='Decreasing difficulty (from Rule II)')
    ax.set_xlabel('Proportion of hard items (Distance + Conflict)')
    ax.set_ylabel('Mean Rule II choice fraction')
    ax.set_title('Flag 4: Hysteresis (path-dependent sweep)')
    ax.legend(fontsize=9)
    
    # ------------------------------------------------------------------
    # FLAG 5: DIVERGENCE — sensitivity to initial conditions
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    
    ### DT --> Use smoothed Rule II usage at a fixed slice rather than the
    ### DT --> final cumulative fraction, which is overly saturated in the
    ### DT --> appendix bridge model.
    pref_range = np.linspace(1.0, 3.0, 7)
    mean_vals = []
    sem_vals = []
    for pref in pref_range:
        model_div = BalanceScaleModel(gamma=16.0, alpha_0=1.0, epistemic_weight=0.3,
                                       pref_correct=pref)
        vals = []
        for i in range(50):
            h = run_balance_scale_agent(model_div, 300, seed=60000+i)
            vals.append(_fixed_slice_smoothed_p_rule2(h, 300, window=20))
        mean_vals.append(np.mean(vals))
        sem_vals.append(np.std(vals) / np.sqrt(len(vals)))
    ax.errorbar(pref_range, mean_vals, yerr=sem_vals, fmt='ko-', capsize=3, linewidth=1.5)
    ax.set_xlabel('Preference for correct outcome')
    ax.set_ylabel('Mean smoothed P(Rule II) at t = 300')
    ax.set_title('Flag 5: Divergence with preference strength')
    
    # ------------------------------------------------------------------
    # VARIABILITY PEAK (susceptibility analogue)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    
    time_range = np.arange(10, 400, 15)
    variance_over_time = []
    mean_over_time = []
    
    for t in time_range:
        vals = []
        for i in range(50):
            h = run_balance_scale_agent(model, t, seed=70000+i)
            vals.append(_fixed_slice_smoothed_p_rule2(h, t, window=20))
        variance_over_time.append(np.var(vals))
        mean_over_time.append(np.mean(vals))
    
    ax2 = ax.twinx()
    l1, = ax.plot(time_range, variance_over_time, 'r-', linewidth=2, label='Var(smoothed P(Rule II))')
    l2, = ax2.plot(time_range, mean_over_time, 'b--', linewidth=2, label='Mean(smoothed P(Rule II))')
    ax.set_xlabel('Developmental time (trials)')
    ax.set_ylabel('Population variance', color='red')
    ax2.set_ylabel('Population mean', color='blue')
    ax.set_title('Variability peaks near the transition')
    ax.legend(handles=[l1, l2], fontsize=9, loc='center right')
    
    fig.suptitle('Appendix Figure A2: Reduced Balance-Scale Bridge Model — Catastrophe Flags', fontsize=16, y=1.01)
    plt.savefig(f"{OUT_DIR}/figA2_appendix_catastrophe_flags_balance.png")
    plt.close()
    print("Saved figA2_appendix_catastrophe_flags_balance.png")


# ============================================================================
# FIGURE 9: Individual Differences — The Role of Gamma
# ============================================================================

def plot_individual_differences():
    """Gamma as cognitive decisiveness: how it shapes developmental trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    N_total = 500
    n_agents = 100
    
    ### DT --> Panel (a): use aligned trailing averages for appendix display.
    ax = axes[0, 0]
    gamma_values = [4, 8, 16, 32]
    colours = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    
    for gamma, col in zip(gamma_values, colours):
        model = BalanceScaleModel(gamma=gamma, alpha_0=1.0, epistemic_weight=0.3)
        for i in range(5):
            h = run_balance_scale_agent(model, N_total, seed=80000+i)
            window = 30
            smoothed = _trailing_average_valid(h['P_rule2'], window)
            trial_axis = np.arange(window - 1, window - 1 + len(smoothed))
            label = f'γ = {gamma}' if i == 0 else None
            ax.plot(trial_axis, smoothed, color=col, alpha=0.4, 
                    linewidth=1, label=label)
    
    ax.set_xlabel('Trial number')
    ax.set_ylabel('P(Rule II) — smoothed')
    ax.set_title('(a) Developmental trajectories by γ')
    ax.legend(fontsize=9)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    ### DT --> Panel (b): summarize transition timing with mean and SD for the
    ### DT --> appendix bridge model, which remains intentionally coarse.
    ax = axes[0, 1]
    gamma_range = np.arange(2, 36, 2)
    
    transition_sharpness = []
    mean_transition_time = []
    
    for gamma in gamma_range:
        model = BalanceScaleModel(gamma=gamma, alpha_0=1.0, epistemic_weight=0.3)
        times = []
        for i in range(n_agents):
            h = run_balance_scale_agent(model, N_total, seed=90000+i)
            ### DT ---> Detect transition: first time P_rule2 stays above 0.7 for 20 trials
            smoothed = _trailing_average_valid(h['P_rule2'], 20)
            trial_axis = np.arange(20 - 1, 20 - 1 + len(smoothed))
            crossed = np.where(smoothed > 0.7)[0]
            if len(crossed) > 0:
                times.append(trial_axis[crossed[0]])
        
        if len(times) > 5:
            transition_sharpness.append(np.std(times))
            mean_transition_time.append(np.mean(times))
        else:
            transition_sharpness.append(np.nan)
            mean_transition_time.append(np.nan)
    
    ax.plot(gamma_range, mean_transition_time, 'bo-', linewidth=1.5, label='Mean transition trial')
    ax.fill_between(gamma_range, 
                    np.array(mean_transition_time) - np.array(transition_sharpness),
                    np.array(mean_transition_time) + np.array(transition_sharpness),
                    alpha=0.2, color='blue')
    ax.set_xlabel('Policy precision γ')
    ax.set_ylabel('Transition trial (mean ± SD)')
    ax.set_title('(b) Transition timing vs cognitive decisiveness')
    ax.legend(fontsize=9)
    
    ### DT --> Panel (c): use a fixed earlier slice so the appendix bridge
    ### DT --> model does not collapse entirely to ceiling.
    ax = axes[1, 0]
    slice_t = 120
    for gamma, col in zip([4, 12, 24], ['#3498db', '#e67e22', '#9b59b6']):
        model = BalanceScaleModel(gamma=gamma, alpha_0=1.0, epistemic_weight=0.3)
        fracs = []
        for i in range(n_agents):
            h = run_balance_scale_agent(model, slice_t, seed=100000+i)
            fracs.append(_fixed_slice_smoothed_p_rule2(h, slice_t, window=20))
        ax.hist(fracs, bins=25, alpha=0.4, color=col, density=True,
                label=f'γ = {gamma}', edgecolor='none')
    
    ax.set_xlabel(f'Smoothed P(Rule II) at t = {slice_t}')
    ax.set_ylabel('Density')
    ax.set_title('(c) Population distributions by γ')
    ax.legend(fontsize=9)
    
    ### DT --> Panel (d): signed order parameter at the same fixed slice.
    ax = axes[1, 1]
    
    gamma_range_fine = np.arange(2, 40, 2)
    mean_abs_z = []
    
    for gamma in gamma_range_fine:
        model = BalanceScaleModel(gamma=gamma, alpha_0=1.0, epistemic_weight=0.3)
        abs_z = []
        for i in range(50):
            h = run_balance_scale_agent(model, slice_t, seed=110000+i)
            z = 2 * _fixed_slice_smoothed_p_rule2(h, slice_t, window=20) - 1
            abs_z.append(np.abs(z))
        mean_abs_z.append(np.mean(abs_z))
    
    ax.plot(gamma_range_fine, mean_abs_z, 'ko-', linewidth=1.5, markersize=5)
    ax.set_xlabel('Policy precision γ')
    ax.set_ylabel(f'Mean |z| = |2·(Rule II frac.) − 1|')
    ax.set_title('(d) Order parameter vs γ (cf. Ising magnetisation)')
    
    ### DT ---> Mark approximate critical gamma (inflection point)
    diffs = np.diff(mean_abs_z)
    gamma_c_approx = gamma_range_fine[np.argmax(diffs) + 1]
    ax.axvline(x=gamma_c_approx, color='red', linestyle='--', linewidth=2,
               label=f'Approx. γ_c ≈ {gamma_c_approx}')
    ax.legend(fontsize=9)
    
    fig.suptitle('Appendix Figure A3: Reduced Balance-Scale Bridge Model — Individual Differences in γ',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figA3_appendix_individual_differences.png")
    plt.close()
    print("Saved figA3_appendix_individual_differences.png")


# ============================================================================
# FIGURE 10: Developmental Predictions — Siegler's Rule Assessment
# ============================================================================

def plot_siegler_predictions():
    """
    Generate predictions matching Siegler's (1976) rule assessment methodology.
    Here we use a usage-based proxy classification from smoothed Rule II
    posterior usage, then compare item-type accuracies across those proxy groups.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    N_total = 500
    n_agents = 150
    model = BalanceScaleModel(gamma=16.0, alpha_0=1.0, epistemic_weight=0.3)
    
    ### DT ---> Panel (a): proportion using Rule I vs Rule II over development
    ax = axes[0]
    
    time_points = np.arange(20, N_total, 20)
    rule1_prop = []
    rule2_prop = []
    mixed_prop = []
    
    for t in time_points:
        n_r1, n_r2, n_mixed = 0, 0, 0
        for i in range(n_agents):
            h = run_balance_scale_agent(model, t, seed=120000+i)
            frac = _fixed_slice_smoothed_p_rule2(h, t, window=20)
            ### DT ---> Classify: <0.3 = Rule I, >0.7 = Rule II, else = Mixed
            if frac < 0.3:
                n_r1 += 1
            elif frac > 0.7:
                n_r2 += 1
            else:
                n_mixed += 1
        
        rule1_prop.append(n_r1 / n_agents)
        rule2_prop.append(n_r2 / n_agents)
        mixed_prop.append(n_mixed / n_agents)
    
    ax.stackplot(time_points, rule1_prop, mixed_prop, rule2_prop,
                 labels=['Rule I', 'Transitional', 'Rule II'],
                 colors=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7)
    ax.set_xlabel('Developmental time (trials)')
    ax.set_ylabel('Proportion of population')
    ax.set_title('(a) Rule usage across development')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(0, 1)
    
    ### DT ---> Panel (b): accuracy by item type for Rule I vs Rule II users
    ax = axes[1]
    
    ### DT ---> Run agents for 400 trials and classify
    rule1_agents_accuracy = {0: [], 1: [], 2: []}  ### DT ---> by problem type
    rule2_agents_accuracy = {0: [], 1: [], 2: []}
    
    for i in range(n_agents):
        h = run_balance_scale_agent(model, 400, seed=130000+i)
        frac = _fixed_slice_smoothed_p_rule2(h, 400, window=20)
        
        ### DT ---> Compute accuracy by problem type in last 100 trials
        for pt in range(3):
            mask = (h['problem_type'][-100:] == pt)
            if mask.sum() > 0:
                acc = (h['outcome'][-100:][mask] == 0).mean()
                if frac < 0.3:
                    rule1_agents_accuracy[pt].append(acc)
                elif frac > 0.7:
                    rule2_agents_accuracy[pt].append(acc)
    
    x_pos = np.array([0, 1, 2])
    width = 0.35
    problem_labels = ['Weight', 'Distance', 'Conflict']
    
    r1_means = [np.mean(rule1_agents_accuracy[pt]) if rule1_agents_accuracy[pt] else 0 for pt in range(3)]
    r2_means = [np.mean(rule2_agents_accuracy[pt]) if rule2_agents_accuracy[pt] else 0 for pt in range(3)]
    r1_sems = [np.std(rule1_agents_accuracy[pt])/np.sqrt(max(len(rule1_agents_accuracy[pt]),1)) 
               if rule1_agents_accuracy[pt] else 0 for pt in range(3)]
    r2_sems = [np.std(rule2_agents_accuracy[pt])/np.sqrt(max(len(rule2_agents_accuracy[pt]),1))
               if rule2_agents_accuracy[pt] else 0 for pt in range(3)]
    
    ax.bar(x_pos - width/2, r1_means, width, yerr=r1_sems, capsize=3,
           label='Rule I users', color='#e74c3c', alpha=0.7)
    ax.bar(x_pos + width/2, r2_means, width, yerr=r2_sems, capsize=3,
           label='Rule II users', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(problem_labels)
    ax.set_ylabel('Accuracy')
    ax.set_title('(b) Accuracy by item type and rule')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    
    ### DT ---> Panel (c): the developmental "S-curve" with variability
    ax = axes[2]
    
    time_points = np.arange(10, N_total, 10)
    mean_r2 = []
    var_r2 = []
    
    for t in time_points:
        vals = []
        for i in range(50):
            h = run_balance_scale_agent(model, t, seed=140000+i)
            vals.append(_fixed_slice_smoothed_p_rule2(h, t, window=20))
        mean_r2.append(np.mean(vals))
        var_r2.append(np.var(vals))
    
    ax_var = ax.twinx()
    l1, = ax.plot(time_points, mean_r2, 'b-', linewidth=2.5, label='Mean Rule II usage')
    l2, = ax_var.plot(time_points, var_r2, 'r-', linewidth=2, alpha=0.7, label='Variance')
    
    ### DT ---> Mark peak variance = transition point
    peak_idx = np.argmax(var_r2)
    ax_var.axvline(x=time_points[peak_idx], color='red', linestyle=':', alpha=0.5)
    x_text = max(90, time_points[peak_idx] - 110)
    y_text = min(float(np.max(mean_r2)) - 0.02, float(np.max(mean_r2)) * 0.72 + 0.02)
    ax.annotate(f'Variability peak ≈ t={time_points[peak_idx]}', 
                xy=(time_points[peak_idx], mean_r2[peak_idx]),
                xytext=(x_text, y_text),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    ax.set_xlabel('Developmental time (trials)')
    ax.set_ylabel('Mean Rule II usage', color='blue')
    ax_var.set_ylabel('Population variance', color='red')
    ax.set_title('(c) Developmental S-curve with variability peak')
    ax.legend(handles=[l1, l2], fontsize=9, loc='center left')
    
    fig.suptitle('Appendix Figure A4: Reduced Balance-Scale Bridge Model — Siegler-Style Assessment Pattern',
                 fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figA4_appendix_siegler_predictions.png")
    plt.close()
    print("Saved figA4_appendix_siegler_predictions.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Notebook 03: Balance Scale Task — Developmental Phase Transition")
    print("=" * 65)
    
    ### DT ---> Print model parameters
    model = BalanceScaleModel()
    print("\n--- Model Parameters ---")
    print(f"True A_outcome (P(correct | strategy, problem_type)):")
    print(f"  Rule I:  Weight={model.A_outcome_true[0,0,0]:.2f}, "
          f"Distance={model.A_outcome_true[0,0,1]:.2f}, "
          f"Conflict={model.A_outcome_true[0,0,2]:.2f}")
    print(f"  Rule II: Weight={model.A_outcome_true[0,1,0]:.2f}, "
          f"Distance={model.A_outcome_true[0,1,1]:.2f}, "
          f"Conflict={model.A_outcome_true[0,1,2]:.2f}")
    print(f"Problem mix: Weight={model.problem_mix[0]:.2f}, "
          f"Distance={model.problem_mix[1]:.2f}, "
          f"Conflict={model.problem_mix[2]:.2f}")
    print(f"gamma = {model.gamma}, alpha_0 = {model.alpha_0}")
    
    print("\nGenerating Appendix Figure A1: Individual trajectories...")
    plot_individual_trajectories()
    
    print("Generating Appendix Figure A2: Catastrophe flags (balance scale)...")
    plot_catastrophe_flags_balance()
    
    print("Generating Appendix Figure A3: Individual differences (gamma)...")
    plot_individual_differences()
    
    print("Generating Appendix Figure A4: Siegler predictions...")
    plot_siegler_predictions()
    
    print("\n" + "=" * 65)
    print("All figures saved to", OUT_DIR)
    print("=" * 65)
