"""
Notebook 02 — Simulation: Bifurcation in a Reduced Active-Inference Model
==========================================================================
Implements the minimal 2-state, 2-observation learning/policy-feedback model
from Notebook 01. This is a reduced, active-inference-inspired mean-field
system rather than a full `pymdp` agent with hidden-state inference.

The code is used to probe the local bifurcation structure, its primary
developmental window, and selected catastrophe-style phenomena under explicit
simulation protocols.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq
import os

### DT --> Keep outputs inside the sandbox workspace.
OUT_DIR = os.path.join(os.path.dirname(__file__), "Figs")
os.makedirs(OUT_DIR, exist_ok=True)

### DT ---> Global plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ============================================================================
# PART 1: Core Mathematical Functions
# ============================================================================

def binary_entropy(x):
    """Binary entropy function H(x) = -x ln(x) - (1-x) ln(1-x)."""
    ### DT ---> Clip to avoid log(0)
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log(x) - (1 - x) * np.log(1 - x)

def binary_entropy_deriv(x):
    """Derivative of binary entropy: H'(x) = ln((1-x)/x)."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return np.log((1 - x) / x)

def learned_discriminability(phi_or_z_based_n, alpha_0, p):
    """
    Compute learned discriminability a(n) given:
      - n: evidence count for this column
      - alpha_0: prior concentration (total for one column)
      - p: true discriminability
    Returns the expected point estimate of the A-matrix diagonal.
    """
    ### DT ---> a = (alpha_0/2 + n*p) / (alpha_0 + n)
    ### DT ---> which simplifies to 0.5 + (p - 0.5) * n / (alpha_0 + n)
    return 0.5 + (p - 0.5) * phi_or_z_based_n / (alpha_0 + phi_or_z_based_n)

def coupling_function(tau, p):
    """
    Compute the coupling function G(tau; p) from Notebook 01.
    tau = N / (2 * alpha_0), the rescaled developmental time.
    p = true discriminability.
    """
    if tau < 1e-12:
        return 0.0
    a_bar = 0.5 + (p - 0.5) * tau / (1 + tau)
    lam = (p - 0.5) * tau / (1 + tau)**2
    log_ratio = np.abs(np.log((1 - a_bar) / a_bar))
    return lam * log_ratio

def find_gamma_c(p, tau_range=np.linspace(0.01, 50, 5000)):
    """Find critical precision gamma_c = 1 / max_tau G(tau; p)."""
    G_vals = np.array([coupling_function(t, p) for t in tau_range])
    G_max = np.max(G_vals)
    tau_max = tau_range[np.argmax(G_vals)]
    if G_max < 1e-12:
        return np.inf, 0, 0
    return 1.0 / G_max, tau_max, G_max

def compute_efe_difference(a, b, delta_c):
    """
    Compute Delta_G = G(pi_1) - G(pi_2).
    = (1 - a - b) * delta_c + H(a) - H(b)
    """
    risk_diff = (1 - a - b) * delta_c
    ambiguity_diff = binary_entropy(a) - binary_entropy(b)
    return risk_diff + ambiguity_diff

def policy_posterior_pi1(delta_G, gamma):
    """P(pi_1) = sigma(-gamma * Delta_G) = 1 / (1 + exp(gamma * Delta_G))."""
    ### DT ---> Use numerically stable sigmoid
    x = gamma * delta_G
    if x > 500:
        return 0.0
    elif x < -500:
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(x))

def self_consistency_rhs(z, tau, p, gamma, delta_c):
    """
    RHS of the self-consistency equation: tanh(-gamma/2 * Delta_G(z)).
    """
    ### DT ---> Compute a(z) and b(z) from Notebook 01 eqs
    n1 = (1 + z) * tau  ### DT ---> rescaled evidence for state 1
    n2 = (1 - z) * tau  ### DT ---> rescaled evidence for state 2
    a = 0.5 + (p - 0.5) * n1 / (1 + n1) if n1 > 0 else 0.5
    b = 0.5 + (p - 0.5) * n2 / (1 + n2) if n2 > 0 else 0.5
    dG = compute_efe_difference(a, b, delta_c)
    return np.tanh(-gamma / 2 * dG)


# ============================================================================
# PART 2: Single-Agent Simulation Loop
# ============================================================================

def run_single_agent(p, alpha_0, gamma, delta_c, N_total, seed=None,
                     use_epistemic=False, epistemic_weight=0.1):
    """
    Run a single active inference agent with Dirichlet learning.
    
    Parameters:
        p: true discriminability (a* = b* = p)
        alpha_0: prior Dirichlet concentration (per column)
        gamma: policy precision
        delta_c: preference asymmetry (c1 - c2)
        N_total: total number of time steps
        seed: random seed
        use_epistemic: whether to include parameter information gain
        epistemic_weight: scaling of epistemic term
    
    Returns: dict with full trajectory
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    ### DT ---> Initialise Dirichlet parameters (symmetric prior)
    alpha1, beta1 = alpha_0 / 2, alpha_0 / 2  ### DT ---> column 1: P(o|s1)
    alpha2, beta2 = alpha_0 / 2, alpha_0 / 2  ### DT ---> column 2: P(o|s2)
    
    ### DT ---> True observation model
    A_true = np.array([[p, 1 - p],
                       [1 - p, p]])
    
    ### DT ---> Storage
    history = {
        'a': np.zeros(N_total),      ### DT ---> learned P(o1|s1)
        'b': np.zeros(N_total),      ### DT ---> learned P(o2|s2)
        'phi': np.zeros(N_total),    ### DT ---> fraction of experience with s1
        'z': np.zeros(N_total),      ### DT ---> order parameter 2*phi - 1
        'delta_G': np.zeros(N_total),### DT ---> EFE difference
        'P_pi1': np.zeros(N_total),  ### DT ---> P(pi_1)
        'action': np.zeros(N_total, dtype=int),  ### DT ---> chosen policy
        'n1': np.zeros(N_total),     ### DT ---> evidence for s1
        'n2': np.zeros(N_total),     ### DT ---> evidence for s2
    }
    
    n1_count = 0
    n2_count = 0
    
    for t in range(N_total):
        ### DT ---> Current learned A matrix (expected value of Dirichlet)
        a = alpha1 / (alpha1 + beta1)
        b = beta2 / (alpha2 + beta2)
        
        ### DT ---> Compute EFE difference
        dG = compute_efe_difference(a, b, delta_c)
        
        ### DT ---> Add epistemic term if requested
        if use_epistemic and t > 0:
            n1_total = alpha1 + beta1
            n2_total = alpha2 + beta2
            ### DT ---> Info gain REDUCES G(pi_k) by w/n_k, so DG_info = (-w/n1) - (-w/n2) = w(1/n2 - 1/n1)
            ### DT ---> When n1 >> n2: DG_info > 0, making P(pi1) lower, favouring under-explored pi2
            epistemic_diff = epistemic_weight * (1.0/n2_total - 1.0/n1_total)
            dG += epistemic_diff
        
        ### DT ---> Policy posterior
        P_pi1 = policy_posterior_pi1(dG, gamma)
        
        ### DT ---> Sample action
        action = 0 if rng.random() < P_pi1 else 1
        
        ### DT ---> Determine state visited and generate observation
        if action == 0:
            ### DT ---> Policy pi_1: visit s1
            obs = 0 if rng.random() < A_true[0, 0] else 1
            ### DT ---> Update Dirichlet for column 1
            if obs == 0:
                alpha1 += 1
            else:
                beta1 += 1
            n1_count += 1
        else:
            ### DT ---> Policy pi_2: visit s2
            obs = 0 if rng.random() < A_true[0, 1] else 1
            ### DT ---> Update Dirichlet for column 2
            if obs == 0:
                alpha2 += 1
            else:
                beta2 += 1
            n2_count += 1
        
        ### DT ---> Record
        N_so_far = n1_count + n2_count
        history['a'][t] = a
        history['b'][t] = b
        history['phi'][t] = n1_count / max(N_so_far, 1)
        history['z'][t] = 2 * history['phi'][t] - 1
        history['delta_G'][t] = dG
        history['P_pi1'][t] = P_pi1
        history['action'][t] = action
        history['n1'][t] = n1_count
        history['n2'][t] = n2_count
    
    return history


def late_window_mean(values, window=30):
    """Return the mean of the final `window` entries of a trajectory."""

    values = np.asarray(values, dtype=float)
    window = max(1, min(int(window), values.size))
    return float(values[-window:].mean())


# ============================================================================
# PART 3: Analytical Predictions
# ============================================================================

def find_fixed_points(tau, p, gamma, delta_c, z_grid=np.linspace(-0.999, 0.999, 2000)):
    """
    Find fixed points of z = tanh(-gamma/2 * Delta_G(z)) by scanning.
    Returns list of (z_fp, stable) tuples.
    """
    ### DT ---> Evaluate f(z) = tanh(...) - z
    f_vals = np.array([self_consistency_rhs(z, tau, p, gamma, delta_c) - z 
                       for z in z_grid])
    
    ### DT ---> Find sign changes
    fps = []
    for i in range(len(f_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            ### DT --> Refine with bisection.
            try:
                z_fp = brentq(lambda z: self_consistency_rhs(z, tau, p, gamma, delta_c) - z,
                              z_grid[i], z_grid[i+1])
                ### DT --> Stable if |d(rhs)/dz| < 1 at the fixed point.
                eps = 1e-5
                rhs_plus = self_consistency_rhs(z_fp + eps, tau, p, gamma, delta_c)
                rhs_minus = self_consistency_rhs(z_fp - eps, tau, p, gamma, delta_c)
                deriv = (rhs_plus - rhs_minus) / (2 * eps)
                stable = abs(deriv) < 1
                fps.append((z_fp, stable))
            except:
                pass

    ### DT --> Deduplicate numerically adjacent roots from dense scans.
    fps.sort(key=lambda item: item[0])
    unique_fps = []
    for z_fp, stable in fps:
        if not unique_fps or abs(z_fp - unique_fps[-1][0]) > 1e-4:
            unique_fps.append((z_fp, stable))
    return unique_fps


# ============================================================================
# FIGURE 1: The Coupling Function and Critical Condition
# ============================================================================

def plot_coupling_function():
    """Plot G(tau; p) for several values of p, with gamma_c marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    tau_range = np.linspace(0.01, 20, 500)
    p_values = [0.75, 0.80, 0.85, 0.90, 0.95]
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(p_values)))
    
    ### DT ---> Left panel: coupling function
    ax = axes[0]
    for p, col in zip(p_values, colours):
        G_vals = [coupling_function(t, p) for t in tau_range]
        ax.plot(tau_range, G_vals, color=col, linewidth=2, label=f'p = {p:.2f}')
    
    ### DT ---> Mark 1/gamma = 1/16 line (pymdp default)
    ax.axhline(y=1/16, color='red', linestyle='--', alpha=0.7, label='1/γ = 1/16 (pymdp default)')
    ax.set_xlabel('Rescaled developmental time τ = N/(2α₀)')
    ax.set_ylabel('Coupling function G(τ; p)')
    ax.set_title('(a) Coupling function')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.2)
    
    ### DT ---> Right panel: gamma_c vs p
    ax = axes[1]
    p_range = np.linspace(0.55, 0.99, 200)
    gamma_c_vals = []
    for p in p_range:
        gc, _, _ = find_gamma_c(p)
        gamma_c_vals.append(gc)
    
    ax.plot(p_range, gamma_c_vals, 'k-', linewidth=2)
    ax.axhline(y=16, color='red', linestyle='--', alpha=0.7, label='γ = 16 (pymdp default)')
    ax.fill_between(p_range, 0, gamma_c_vals, alpha=0.15, color='blue', label='Bifurcation region (γ > γ_c)')
    ax.set_xlabel('True discriminability p')
    ax.set_ylabel('Critical precision γ_c')
    ax.set_title('(b) Critical precision vs discriminability')
    ### DT --> We are ensuring here that the low-p end of the critical curve is not clipped against the top axis limit.
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9)
    
    fig.suptitle('Figure 1: Coupling Function and Bifurcation Condition', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_coupling_function.png")
    plt.close()
    print("Saved fig1_coupling_function.png")


# ============================================================================
# FIGURE 2: Bifurcation Diagram (analytical fixed points vs tau)
# ============================================================================

def plot_bifurcation_diagram():
    """Plot the primary local bifurcation window of the fixed-point equation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gamma = 16.0
    p = 0.85
    
    ### DT ---> Left panel: symmetric case (delta_c = 0)
    ax = axes[0]
    tau_range = np.linspace(0.1, 4.0, 220)
    
    for tau in tau_range:
        fps = find_fixed_points(tau, p, gamma, delta_c=0.0)
        for z_fp, stable in fps:
            marker = 'o' if stable else 'x'
            color = 'blue' if stable else 'red'
            ax.plot(tau, z_fp, marker, markersize=1.5, color=color, alpha=0.6)
    
    ### DT ---> Mark critical tau
    gc, tau_max, G_max = find_gamma_c(p)
    ### DT ---> Find tau where gamma * G(tau) = 1
    tau_scan = np.linspace(0.01, 20, 5000)
    gG_vals = [gamma * coupling_function(t, p) for t in tau_scan]
    ### DT ---> Find crossings of gamma*G = 1
    crossings = []
    for i in range(len(gG_vals)-1):
        if (gG_vals[i] - 1) * (gG_vals[i+1] - 1) < 0:
            crossings.append(tau_scan[i])
    if crossings:
        ### DT --> We are ensuring here that the analytical panel marks only the primary bifurcation crossing discussed in the text, not later re-entrant structure.
        ax.axvline(x=crossings[0], color='green', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Rescaled developmental time τ')
    ax.set_ylabel('Order parameter z = 2φ − 1')
    ax.set_title(f'(a) Symmetric (Δc = 0), γ = {gamma}, p = {p}')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ### DT --> Right panel: small asymmetry, showing the local cusp/fold slice.
    ax = axes[1]
    for tau in tau_range:
        fps = find_fixed_points(tau, p, gamma, delta_c=0.01)
        for z_fp, stable in fps:
            marker = 'o' if stable else 'x'
            color = 'blue' if stable else 'red'
            ax.plot(tau, z_fp, marker, markersize=1.5, color=color, alpha=0.6)
    
    ax.set_xlabel('Rescaled developmental time τ')
    ax.set_ylabel('Order parameter z = 2φ − 1')
    ax.set_title(f'(b) Local asymmetric (Δc = 0.01), γ = {gamma}, p = {p}')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    fig.suptitle('Figure 2: Primary Bifurcation Window — Analytical Fixed Points', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_bifurcation_diagram.png")
    plt.close()
    print("Saved fig2_bifurcation_diagram.png")


# ============================================================================
# FIGURE 3: Single-Agent Trajectories (simulation)
# ============================================================================

def plot_single_agent_trajectories():
    """Run multiple agents and overlay trajectories to show the transition."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    ### DT ---> Parameters
    p = 0.85
    alpha_0 = 2.0
    gamma = 16.0
    N_total = 200
    n_agents = 36
    
    ### DT ---> Panel (a): z trajectories, symmetric case
    ax = axes[0, 0]
    z_histories = []
    for i in range(n_agents):
        h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                             seed=1000+i, use_epistemic=False)
        z_histories.append(h['z'])
    z_histories = np.array(z_histories)
    show_idx = np.linspace(0, n_agents - 1, 10, dtype=int)
    for idx in show_idx:
        ax.plot(np.arange(N_total), z_histories[idx], alpha=0.25, linewidth=0.9)
    ### DT --> We are ensuring here that the panel presents a readable ensemble summary rather than an overly dense spaghetti plot.
    ax.plot(np.arange(N_total), z_histories.mean(axis=0), color='black', linewidth=2.0, label='Population mean')
    ax.set_xlabel('Time step t')
    ax.set_ylabel('z = 2φ − 1')
    ax.set_title(f'(a) Strategy trajectories (Δc = 0, γ = {gamma})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc='lower right')
    
    ### DT ---> Panel (b): z trajectories, weak asymmetry
    ax = axes[0, 1]
    z_histories = []
    for i in range(n_agents):
        h = run_single_agent(p, alpha_0, gamma, delta_c=0.3, N_total=N_total,
                             seed=2000+i, use_epistemic=False)
        z_histories.append(h['z'])
    z_histories = np.array(z_histories)
    for idx in show_idx:
        ax.plot(np.arange(N_total), z_histories[idx], alpha=0.25, linewidth=0.9)
    ax.plot(np.arange(N_total), z_histories.mean(axis=0), color='black', linewidth=2.0, label='Population mean')
    ax.set_xlabel('Time step t')
    ax.set_ylabel('z = 2φ − 1')
    ax.set_title(f'(b) Strategy trajectories (Δc = 0.3, γ = {gamma})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc='lower right')
    
    ### DT ---> Panel (c): learned discriminabilities a, b for one agent
    ax = axes[1, 0]
    h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                         seed=42, use_epistemic=False)
    ax.plot(np.arange(N_total), h['a'], label='a = P̂(o₁|s₁)', linewidth=2)
    ax.plot(np.arange(N_total), h['b'], label='b = P̂(o₂|s₂)', linewidth=2)
    ax.axhline(y=p, color='gray', linestyle=':', alpha=0.5, label=f'True p = {p}')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('Learned discriminability')
    ax.set_title('(c) Learning dynamics (single agent)')
    ax.legend(fontsize=9)
    
    ### DT ---> Panel (d): P(pi_1) and action history
    ax = axes[1, 1]
    ax.plot(np.arange(N_total), h['P_pi1'], 'b-', linewidth=1.5, alpha=0.7, label='P(π₁)')
    ### DT ---> Running average of actions
    window = 20
    action_avg = np.convolve(1 - h['action'], np.ones(window)/window, mode='valid')
    ax.plot(np.arange(window-1, N_total), action_avg, 'r-', linewidth=1.5, alpha=0.7,
            label=f'Rule I action rate (window={window})')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('Probability / frequency')
    ax.set_title('(d) Policy posterior and action frequency')
    ax.legend(fontsize=9)
    
    fig.suptitle('Figure 3: Single-Agent Simulation Trajectories', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_single_agent.png")
    plt.close()
    print("Saved fig3_single_agent.png")


# ============================================================================
# FIGURE 4: Catastrophe Flags
# ============================================================================

def plot_catastrophe_flags():
    """Demonstrate all five catastrophe flags in population simulations."""
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    p = 0.85
    alpha_0 = 2.0
    gamma = 16.0
    n_agents = 500
    
    # ------------------------------------------------------------------
    # FLAG 1: BIMODALITY
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    
    ### DT ---> Run population at several developmental times
    ### DT --> We are ensuring here that the bimodality panel uses a smaller set of diagnostically distinct slices near the transition window, rather than an over-cluttered overlay.
    time_points = [60, 100, 150]
    colours = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    
    for N_total, col in zip(time_points, colours[:len(time_points)]):
        z_finals = []
        for i in range(n_agents):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=3000+i, use_epistemic=False)
            z_finals.append(h['z'][-1])
        ax.hist(z_finals, bins=30, alpha=0.4, color=col, density=True,
                label=f't = {N_total}', edgecolor='none')
    
    ax.set_xlabel('z = 2φ − 1')
    ax.set_ylabel('Density')
    ax.set_title('Flag 1: Bimodality')
    ax.legend(fontsize=9)
    
    # ------------------------------------------------------------------
    # FLAG 2: INACCESSIBLE REGION
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    
    ### DT ---> Collect z values at t=150 across many agents
    z_all = []
    for i in range(n_agents):
        h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=150,
                             seed=4000+i, use_epistemic=False)
        z_all.append(h['z'][-1])
    
    ax.hist(z_all, bins=40, color='steelblue', edgecolor='white', density=True)
    ### DT ---> Mark inaccessible region
    ax.axvspan(-0.3, 0.3, alpha=0.15, color='red', label='Inaccessible region')
    ax.set_xlabel('z = 2φ − 1')
    ax.set_ylabel('Density')
    ax.set_title('Flag 2: Inaccessible Region (t = 150)')
    ax.legend(fontsize=9)
    
    # ------------------------------------------------------------------
    # FLAG 3: SUDDEN JUMP
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    
    ### DT ---> Show individual trajectories with clear jumps
    N_total = 250
    for i in range(8):
        h = run_single_agent(p, alpha_0, gamma, delta_c=0.15, N_total=N_total,
                             seed=5000+i*7, use_epistemic=False)
        ax.plot(np.arange(N_total), h['z'], linewidth=1.2, alpha=0.6)
    
    ax.set_xlabel('Time step t')
    ax.set_ylabel('z = 2φ − 1')
    ax.set_title('Flag 3: Sudden Jumps')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    # ------------------------------------------------------------------
    # FLAG 4: HYSTERESIS
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    
    ### DT ---> Sweep delta_c forward then backward with PERSISTENT Dirichlet state
    ### DT ---> At each sweep point, run agent for some trials, carrying forward learned parameters
    n_sweep = 30
    dc_forward = np.linspace(-1.5, 1.5, n_sweep)
    dc_backward = np.linspace(1.5, -1.5, n_sweep)
    trials_per_point = 15
    rng_hyst = np.random.default_rng(999)
    
    ### DT ---> Forward sweep: start from pre-trained Phase A state (heavy s1 experience)
    alpha1_f, beta1_f = alpha_0/2 + 25*p, alpha_0/2 + 25*(1-p)
    alpha2_f, beta2_f = alpha_0/2 + 3*(1-p), alpha_0/2 + 3*p
    A_true = np.array([[p, 1-p], [1-p, p]])
    
    z_forward = []
    for dc in dc_forward:
        ### DT ---> Run trials at this dc, updating Dirichlet state
        for _ in range(trials_per_point):
            a_f = alpha1_f / (alpha1_f + beta1_f)
            b_f = beta2_f / (alpha2_f + beta2_f)
            dG = compute_efe_difference(a_f, b_f, dc)
            P_pi1 = policy_posterior_pi1(dG, gamma)
            action = 0 if rng_hyst.random() < P_pi1 else 1
            if action == 0:
                obs = 0 if rng_hyst.random() < A_true[0, 0] else 1
                if obs == 0: alpha1_f += 1
                else: beta1_f += 1
            else:
                obs = 0 if rng_hyst.random() < A_true[0, 1] else 1
                if obs == 0: alpha2_f += 1
                else: beta2_f += 1
        ### DT ---> Record instantaneous policy posterior as z
        a_f = alpha1_f / (alpha1_f + beta1_f)
        b_f = beta2_f / (alpha2_f + beta2_f)
        dG = compute_efe_difference(a_f, b_f, dc)
        z_forward.append(2*policy_posterior_pi1(dG, gamma) - 1)
    
    ### DT ---> Backward sweep: start from pre-trained Phase B state (heavy s2 experience)
    alpha1_b, beta1_b = alpha_0/2 + 3*p, alpha_0/2 + 3*(1-p)
    alpha2_b, beta2_b = alpha_0/2 + 25*(1-p), alpha_0/2 + 25*p
    
    z_backward = []
    for dc in dc_backward:
        for _ in range(trials_per_point):
            a_b = alpha1_b / (alpha1_b + beta1_b)
            b_b = beta2_b / (alpha2_b + beta2_b)
            dG = compute_efe_difference(a_b, b_b, dc)
            P_pi1 = policy_posterior_pi1(dG, gamma)
            action = 0 if rng_hyst.random() < P_pi1 else 1
            if action == 0:
                obs = 0 if rng_hyst.random() < A_true[0, 0] else 1
                if obs == 0: alpha1_b += 1
                else: beta1_b += 1
            else:
                obs = 0 if rng_hyst.random() < A_true[0, 1] else 1
                if obs == 0: alpha2_b += 1
                else: beta2_b += 1
        a_b = alpha1_b / (alpha1_b + beta1_b)
        b_b = beta2_b / (alpha2_b + beta2_b)
        dG = compute_efe_difference(a_b, b_b, dc)
        z_backward.append(2*policy_posterior_pi1(dG, gamma) - 1)
    
    ax.plot(dc_forward, z_forward, 'b-o', markersize=3, linewidth=1.5, label='Forward sweep (A→B)')
    ax.plot(dc_backward, z_backward, 'r-s', markersize=3, linewidth=1.5, label='Backward sweep (B→A)')
    ax.set_xlabel('Preference asymmetry Δc')
    ax.set_ylabel('Strategy allocation z')
    ax.set_title('Flag 4: Hysteresis')
    ax.legend(fontsize=9)
    
    # ------------------------------------------------------------------
    # FLAG 5: DIVERGENCE
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    
    ### DT ---> Two agents with nearly identical initial conditions diverge
    N_total = 200
    delta_c_values = np.linspace(-0.05, 0.05, 11)
    z_medians = []
    z_iqr_low = []
    z_iqr_high = []
    for dc in delta_c_values:
        late_vals = []
        for seed_idx in range(40):
            h = run_single_agent(p, alpha_0, gamma, delta_c=dc, N_total=N_total,
                                 seed=4200 + seed_idx, use_epistemic=False)
            late_vals.append(late_window_mean(h['z'], window=30))
        late_vals = np.array(late_vals)
        z_medians.append(np.median(late_vals))
        q1, q3 = np.percentile(late_vals, [25, 75])
        z_iqr_low.append(q1)
        z_iqr_high.append(q3)
    
    ax.plot(delta_c_values, z_medians, 'ko-', markersize=5, linewidth=1.5)
    ax.fill_between(delta_c_values, z_iqr_low, z_iqr_high, color='gray', alpha=0.2)
    ax.set_xlabel('Initial preference asymmetry Δc')
    ax.set_ylabel('Late-window mean z at t = 200')
    ax.set_title('Flag 5: Divergence')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # ------------------------------------------------------------------
    # SUMMARY: Variability peaks at transition
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    
    ### DT ---> Compute population variance of z at each time point
    ### DT ---> Using symmetric case (Δc=0) for the sharpest susceptibility peak
    time_range = np.arange(5, 300, 5)
    z_var = []
    z_mean_abs = []
    for N_total in time_range:
        z_at_t = []
        for i in range(300):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=6000+i, use_epistemic=False)
            z_at_t.append(h['z'][-1])
        z_var.append(np.var(z_at_t))
        z_mean_abs.append(np.mean(z_at_t))
    
    ax2 = ax.twinx()
    ax.plot(time_range, z_var, 'r-', linewidth=2, label='Var(z)')
    ax2.plot(time_range, z_mean_abs, 'b--', linewidth=2, label='Mean (z)')
    ax.set_xlabel('Developmental time t')
    ax.set_ylabel('Variance of z (red)', color='red')
    ax2.set_ylabel('Mean (z) (blue)', color='blue')
    ax2.set_ylim(-0.5, 0.5)
    ax.set_title('Variability peaks near transition (Δc = 0; cf. susceptibility)')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('Figure 4: The Five Catastrophe Flags', fontsize=16, y=1.01)
    plt.savefig(f"{OUT_DIR}/fig4_catastrophe_flags.png")
    plt.close()
    print("Saved fig4_catastrophe_flags.png")


# ============================================================================
# FIGURE 5: Analytical vs Simulation Verification
# ============================================================================

def plot_verification():
    """Compare analytical critical condition with simulation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    alpha_0 = 2.0
    N_total = 200
    n_agents = 200
    
    ### DT ---> Left panel: sweep gamma, measure final |z|
    ax = axes[0]
    gamma_range = np.arange(4, 30, 2)
    p = 0.85
    
    mean_abs_z = []
    sem_abs_z = []
    
    for gamma in gamma_range:
        abs_z = []
        for i in range(n_agents):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=7000+i, use_epistemic=False)
            abs_z.append(late_window_mean(np.abs(h['z']), window=30))
        mean_abs_z.append(np.mean(abs_z))
        sem_abs_z.append(np.std(abs_z, ddof=1) / np.sqrt(len(abs_z)))
    
    ax.errorbar(gamma_range, mean_abs_z, yerr=sem_abs_z, fmt='ko-', 
                capsize=3, linewidth=1.5, label='Simulation mean |z|')
    
    ### DT ---> Mark analytical gamma_c
    gc, _, _ = find_gamma_c(p)
    ax.axvline(x=gc, color='red', linestyle='--', linewidth=2, 
               label=f'Analytical γ_c = {gc:.1f}')
    ax.set_xlabel('Policy precision γ')
    ax.set_ylabel('Mean |z| at t = 200')
    ax.set_title(f'(a) Bifurcation in γ (p = {p})')
    ax.legend(fontsize=9)
    
    ### DT ---> Right panel: sweep p, measure final |z|
    ax = axes[1]
    p_range = np.arange(0.60, 0.98, 0.03)
    gamma = 16.0
    
    mean_abs_z = []
    sem_abs_z = []
    analytical_gc = []
    
    for p in p_range:
        abs_z = []
        for i in range(n_agents):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=8000+i, use_epistemic=False)
            abs_z.append(late_window_mean(np.abs(h['z']), window=30))
        mean_abs_z.append(np.mean(abs_z))
        sem_abs_z.append(np.std(abs_z, ddof=1) / np.sqrt(len(abs_z)))
        gc_val, _, _ = find_gamma_c(p)
        analytical_gc.append(gc_val)
    
    ax.errorbar(p_range, mean_abs_z, yerr=sem_abs_z, fmt='ko-',
                capsize=3, linewidth=1.5, label='Simulation mean |z|')
    
    ### DT ---> Mark analytical p_c where gamma_c = gamma
    p_critical = None
    for i in range(len(analytical_gc)-1):
        if analytical_gc[i] > gamma and analytical_gc[i+1] <= gamma:
            ### DT ---> Linear interpolation
            frac = (gamma - analytical_gc[i]) / (analytical_gc[i+1] - analytical_gc[i])
            p_critical = p_range[i] + frac * (p_range[i+1] - p_range[i])
            break
    
    if p_critical:
        ax.axvline(x=p_critical, color='red', linestyle='--', linewidth=2,
                   label=f'Analytical p_c ≈ {p_critical:.2f}')
    
    ax.set_xlabel('True discriminability p')
    ax.set_ylabel('Mean |z| at t = 200')
    ax.set_title(f'(b) Bifurcation in p (γ = {gamma})')
    ax.legend(fontsize=9)
    
    fig.suptitle('Figure 5: Analytical Prediction vs Simulation', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_verification.png")
    plt.close()
    print("Saved fig5_verification.png")


# ============================================================================
# FIGURE 6: The Ising Mapping — Visual Summary
# ============================================================================

def plot_ising_mapping_visual():
    """Create a visual summary of the local Ising-style mapping."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ### DT ---> Panel (a): Mean-field self-consistency equation
    ax = axes[0]
    z_range = np.linspace(-0.99, 0.99, 500)
    
    tau_values = [0.5, 1.5, 2.5, 5.0]
    p = 0.85
    gamma = 16.0
    colours = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(tau_values)))
    
    ax.plot(z_range, z_range, 'k-', linewidth=1, alpha=0.5, label='y = z')
    
    for tau, col in zip(tau_values, colours):
        rhs = [self_consistency_rhs(z, tau, p, gamma, 0.0) for z in z_range]
        ax.plot(z_range, rhs, color=col, linewidth=2, label=f'τ = {tau:.1f}')
    
    ax.set_xlabel('z')
    ax.set_ylabel('tanh(−γΔG(z)/2)')
    ax.set_title('(a) Self-consistency equation')
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    ### DT ---> Panel (b): Free energy landscape (Landau potential analogue)
    ax = axes[1]
    
    for tau, col in zip(tau_values, colours):
        ### DT ---> Integrate to get potential V(z) = integral of (z - tanh(...)) dz
        dz = z_range[1] - z_range[0]
        integrand = [z - self_consistency_rhs(z, tau, p, gamma, 0.0) for z in z_range]
        V = np.cumsum(integrand) * dz
        V -= np.min(V)
        ax.plot(z_range, V, color=col, linewidth=2, label=f'τ = {tau:.1f}')
    
    ax.set_xlabel('Order parameter z')
    ax.set_ylabel('Effective potential V(z)')
    ax.set_title('(b) Free energy landscape')
    ax.legend(fontsize=8)
    
    ### DT --> Panel (c): local phase diagram in the primary bifurcation window
    ax = axes[2]
    
    ### DT --> We are ensuring here that the cusp boundary is resolved on a finer grid, so the multistable region does not look jagged in the publication figure.
    tau_range = np.linspace(0.4, 3.0, 220)
    dc_range = np.linspace(-0.05, 0.05, 220)
    n_solutions = np.zeros((len(dc_range), len(tau_range)))
    
    for i, dc in enumerate(dc_range):
        for j, tau in enumerate(tau_range):
            fps = find_fixed_points(tau, p, gamma, dc,
                                   z_grid=np.linspace(-0.99, 0.99, 500))
            n_solutions[i, j] = len(fps)
    
    ax.contourf(tau_range, dc_range, n_solutions, levels=[2.5, 3.5], 
                colors=['salmon'], alpha=0.5)
    ax.contour(tau_range, dc_range, n_solutions, levels=[2.5], 
               colors=['red'], linewidths=2)
    ax.set_xlabel('Rescaled developmental time τ')
    ax.set_ylabel('Preference asymmetry Δc')
    ax.set_title('(c) Local cusp / fold region')
    ### DT --> Label regions
    ax.text(0.8, 0.0, 'Monostable', ha='center', va='center', fontsize=10, color='gray')
    ax.text(2.2, 0.0, 'Multistable', ha='center', va='center', fontsize=10, color='red',
            fontweight='bold')
    
    fig.suptitle('Figure 6: The Ising Mapping — Visual Summary', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig6_ising_mapping.png")
    plt.close()
    print("Saved fig6_ising_mapping.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Notebook 02: Simulation of Bifurcation in Active Inference")
    print("=" * 60)
    
    ### DT ---> Print analytical predictions first
    print("\n--- Analytical Predictions ---")
    for p in [0.75, 0.80, 0.85, 0.90, 0.95]:
        gc, tau_max, G_max = find_gamma_c(p)
        print(f"p = {p:.2f}: gamma_c = {gc:.1f}, tau_max = {tau_max:.2f}, G_max = {G_max:.4f}")
    
    print(f"\npymdp default gamma = 16")
    print(f"Bifurcation occurs for p > ~0.81 (at gamma = 16)\n")
    
    ### DT ---> Generate all figures
    print("Generating Figure 1: Coupling function...")
    plot_coupling_function()
    
    print("Generating Figure 2: Bifurcation diagram...")
    plot_bifurcation_diagram()
    
    print("Generating Figure 3: Single-agent trajectories...")
    plot_single_agent_trajectories()
    
    print("Generating Figure 4: Catastrophe flags...")
    plot_catastrophe_flags()
    
    print("Generating Figure 5: Analytical verification...")
    plot_verification()
    
    print("Generating Figure 6: Ising mapping visual...")
    plot_ising_mapping_visual()
    
    print("\n" + "=" * 60)
    print("All figures saved to", OUT_DIR)
    print("=" * 60)
