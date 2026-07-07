"""
Appendix — Trial-Level Stochastic Verification with the Novelty Term Included
=============================================================================
Companion to `sim_bifurcation.py` (Notebook 02) and Notebook 05.

This script reruns ONLY the two population figures of the minimal model —
the five catastrophe flags (manuscript Figure 2) and the analytical-vs-
simulation verification sweeps (manuscript Figure 3a/b) — with the EXACT
Dirichlet parameter information-gain ("novelty") term of Notebook 05
switched on in the trial-level stochastic agent.

Implementation note (exactness). `run_single_agent` in `sim_bifurcation.py`
already contains an epistemic option:
    epistemic_diff = w * (1/n2_total - 1/n1_total),
with n_k_total = alpha_0 + n_k (prior included). Notebook 05 (Eq. 3) shows
the exact novelty term for a two-outcome Dirichlet column is
    W(pi_k) = 1 / (2 (alpha_0 + n_k)),
so the exact extension of the EFE difference is
    Delta G -> Delta G - Delta W = Delta G + (1/2) (1/n2_total - 1/n1_total).
Hence the ORIGINAL engine implements the exact novelty term when called with
`use_epistemic=True, epistemic_weight=0.5`. No modification of the engine is
needed; exactness is obtained by parameter choice, and the closed form is
composition-independent (it depends only on the column totals).

Parameter choice (alpha_0 = 8). The corrected condition of Notebook 05,
    gamma * [G(tau; p) - N(tau; alpha_0)] = 1,   N = tau / (2 alpha_0 (1+tau)^2),
implies that at p = 0.85 and gamma = 16 commitment is possible only for
alpha_0 above about 6.39 (gamma_c(0.85, 8) = 15.2 < 16 < 32.4 = gamma_c(0.85, 2)).
The flags figure therefore uses alpha_0 = 8; the verification figure sweeps
gamma and p at alpha_0 = 8, and additionally sweeps alpha_0 itself — the
distinctive new prediction of the corrected condition.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

### DT ---> Reuse the original engine and analytics unchanged
from sim_bifurcation import (run_single_agent, compute_efe_difference,
                             policy_posterior_pi1, coupling_function,
                             late_window_mean)

### DT ---> Keep outputs inside the sandbox workspace
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs")
os.makedirs(OUT_DIR, exist_ok=True)

### DT ---> Exact novelty = original heuristic with w = 1/2 (see docstring)
NOVELTY_KWARGS = dict(use_epistemic=True, epistemic_weight=0.5)

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
# PART 1: Corrected analytical predictions (Notebook 05, Eqs. 9-13)
# ============================================================================

def novelty_coupling(tau, alpha_0):
    """N(tau; alpha_0) = tau / (2 alpha_0 (1 + tau)^2)  (Notebook 05, Eq. 9)."""
    return tau / (2.0 * alpha_0 * (1.0 + tau) ** 2)

def corrected_coupling(tau, p, alpha_0):
    """G(tau; p) - N(tau; alpha_0), the curiosity-corrected coupling."""
    return coupling_function(tau, p) - novelty_coupling(tau, alpha_0)

_TAU_GRID = np.linspace(0.01, 60, 24000)

def find_gamma_c_corrected(p, alpha_0):
    """gamma_c(p, alpha_0) = 1 / max_tau [G - N]  (Notebook 05, Eq. 13)."""
    vals = np.array([corrected_coupling(t, p, alpha_0) for t in _TAU_GRID])
    m = vals.max()
    if m <= 0:
        return np.inf, np.nan, m
    return 1.0 / m, _TAU_GRID[vals.argmax()], m

def find_p_c_corrected(gamma, alpha_0, p_lo=0.55, p_hi=0.99):
    """Smallest p at which gamma_c(p, alpha_0) = gamma (bisection)."""
    def f(p):
        gc, _, _ = find_gamma_c_corrected(p, alpha_0)
        return gc - gamma
    ### DT ---> gamma_c decreases in p; bisect on the sign change
    lo, hi = p_lo, p_hi
    if f(lo) < 0 or f(hi) > 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def find_alpha0_c(gamma, p, a_lo=0.5, a_hi=64.0):
    """Smallest alpha_0 at which gamma_c(p, alpha_0) = gamma (bisection)."""
    def f(a0):
        gc, _, _ = find_gamma_c_corrected(p, a0)
        return gc - gamma
    lo, hi = a_lo, a_hi
    if f(lo) < 0 or f(hi) > 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================================
# PART 2: Figure A1 — the five catastrophe flags with the novelty term
# ============================================================================

def plot_catastrophe_flags_novelty(alpha_0=8.0, p=0.85, gamma=16.0):
    """Replicates the Figure-2 protocols with the exact novelty term ON."""
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    n_agents = 500

    # ------------------------------------------------------------------
    # FLAG 1: BIMODALITY
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ### DT ---> Same diagnostic slices as the consolidation-only figure
    time_points = [60, 100, 150]
    colours = ['#2196F3', '#FF9800', '#4CAF50']
    for N_total, col in zip(time_points, colours):
        z_finals = []
        for i in range(n_agents):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=3000 + i, **NOVELTY_KWARGS)
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
    z_all = []
    for i in range(n_agents):
        h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=150,
                             seed=4000 + i, **NOVELTY_KWARGS)
        z_all.append(h['z'][-1])
    ax.hist(z_all, bins=40, color='steelblue', edgecolor='white', density=True)
    ax.axvspan(-0.3, 0.3, alpha=0.15, color='red', label='Inaccessible region')
    ax.set_xlabel('z = 2φ − 1')
    ax.set_ylabel('Density')
    ax.set_title('Flag 2: Inaccessible Region (t = 150)')
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # FLAG 3: SUDDEN JUMP
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    N_total = 250
    for i in range(8):
        h = run_single_agent(p, alpha_0, gamma, delta_c=0.15, N_total=N_total,
                             seed=5000 + i * 7, **NOVELTY_KWARGS)
        ax.plot(np.arange(N_total), h['z'], linewidth=1.2, alpha=0.6)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('z = 2φ − 1')
    ax.set_title('Flag 3: Sudden Jumps')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # ------------------------------------------------------------------
    # FLAG 4: HYSTERESIS (sweep with persistent Dirichlet state)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    n_sweep = 30
    dc_forward = np.linspace(-1.5, 1.5, n_sweep)
    dc_backward = np.linspace(1.5, -1.5, n_sweep)
    trials_per_point = 15
    rng_hyst = np.random.default_rng(999)
    A_true = np.array([[p, 1 - p], [1 - p, p]])

    def novelty_diff(a1, b1, a2, b2):
        ### DT ---> -Delta W = (1/2)(1/n2_total - 1/n1_total), exact (Notebook 05, Eq. 3)
        n1_total = a1 + b1
        n2_total = a2 + b2
        return 0.5 * (1.0 / n2_total - 1.0 / n1_total)

    def sweep(dc_values, heavy_first):
        ### DT ---> Pre-train: heavy experience with one strategy, light with the other
        if heavy_first:
            a1, b1 = alpha_0 / 2 + 25 * p, alpha_0 / 2 + 25 * (1 - p)
            a2, b2 = alpha_0 / 2 + 3 * (1 - p), alpha_0 / 2 + 3 * p
        else:
            a1, b1 = alpha_0 / 2 + 3 * p, alpha_0 / 2 + 3 * (1 - p)
            a2, b2 = alpha_0 / 2 + 25 * (1 - p), alpha_0 / 2 + 25 * p
        z_trace = []
        for dc in dc_values:
            for _ in range(trials_per_point):
                af = a1 / (a1 + b1)
                bf = b2 / (a2 + b2)
                dG = compute_efe_difference(af, bf, dc) + novelty_diff(a1, b1, a2, b2)
                P1 = policy_posterior_pi1(dG, gamma)
                action = 0 if rng_hyst.random() < P1 else 1
                if action == 0:
                    obs = 0 if rng_hyst.random() < A_true[0, 0] else 1
                    if obs == 0: a1 += 1
                    else: b1 += 1
                else:
                    obs = 0 if rng_hyst.random() < A_true[0, 1] else 1
                    if obs == 0: a2 += 1
                    else: b2 += 1
            af = a1 / (a1 + b1)
            bf = b2 / (a2 + b2)
            dG = compute_efe_difference(af, bf, dc) + novelty_diff(a1, b1, a2, b2)
            z_trace.append(2 * policy_posterior_pi1(dG, gamma) - 1)
        return z_trace

    z_forward = sweep(dc_forward, heavy_first=True)
    z_backward = sweep(dc_backward, heavy_first=False)
    ax.plot(dc_forward, z_forward, 'b-o', markersize=3, linewidth=1.5,
            label='Forward sweep (A→B)')
    ax.plot(dc_backward, z_backward, 'r-s', markersize=3, linewidth=1.5,
            label='Backward sweep (B→A)')
    ax.set_xlabel('Preference asymmetry Δc')
    ax.set_ylabel('Strategy allocation z')
    ax.set_title('Flag 4: Hysteresis')
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # FLAG 5: DIVERGENCE
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    N_total = 200
    delta_c_values = np.linspace(-0.05, 0.05, 11)
    z_medians, z_iqr_low, z_iqr_high = [], [], []
    for dc in delta_c_values:
        late_vals = []
        for seed_idx in range(40):
            h = run_single_agent(p, alpha_0, gamma, delta_c=dc, N_total=N_total,
                                 seed=4200 + seed_idx, **NOVELTY_KWARGS)
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
    # SUMMARY: Variability peaks at the (delayed) transition
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    time_range = np.arange(5, 300, 5)
    z_var, z_mean = [], []
    for N_total in time_range:
        z_at_t = []
        for i in range(300):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=6000 + i, **NOVELTY_KWARGS)
            z_at_t.append(h['z'][-1])
        z_var.append(np.var(z_at_t))
        z_mean.append(np.mean(z_at_t))
    ax2 = ax.twinx()
    ax.plot(time_range, z_var, 'r-', linewidth=2, label='Var(z)')
    ax2.plot(time_range, z_mean, 'b--', linewidth=2, label='Mean (z)')
    ### DT ---> Mark the predicted (delayed) linear-instability window in absolute trials
    vals = np.array([corrected_coupling(t, p, alpha_0) for t in _TAU_GRID])
    inside = _TAU_GRID[vals > 1.0 / gamma]
    if inside.size:
        t_lo, t_hi = 2 * alpha_0 * inside[0], 2 * alpha_0 * inside[-1]
        ax.axvspan(t_lo, t_hi, color='green', alpha=0.08,
                   label='Predicted window (Eq. 13)')
    ax.set_xlabel('Developmental time t')
    ax.set_ylabel('Variance of z (red)', color='red')
    ax2.set_ylabel('Mean (z) (blue)', color='blue')
    ax2.set_ylim(-0.5, 0.5)
    ax.set_title('Variability peaks near transition (Δc = 0; cf. susceptibility)')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'Figure A1: The Five Catastrophe Flags with the Novelty Term '
                 f'(exact ΔW; α₀ = {alpha_0:.0f}, p = {p}, γ = {gamma:.0f})',
                 fontsize=15, y=1.01)
    plt.savefig(f"{OUT_DIR}/figA1_flags_novelty.png")
    plt.close()
    print("Saved figA1_flags_novelty.png")


# ============================================================================
# PART 3: Figure A2 — verification sweeps against the corrected condition
# ============================================================================

def plot_verification_novelty(alpha_0=8.0, N_total=200, n_agents=200):
    """Sweeps of gamma and p with the novelty term ON, compared with the
    corrected analytical thresholds of Notebook 05 (Eq. 13). Mirrors the
    consolidation-only verification figure (manuscript Figure 3a/b)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------------- Panel (a): sweep gamma at p = 0.85 ----------------
    ax = axes[0]
    p = 0.85
    gamma_range = np.arange(4, 30, 2)
    mean_abs_z, sem_abs_z = [], []
    for gamma in gamma_range:
        abs_z = []
        for i in range(n_agents):
            h = run_single_agent(p, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=7000 + i, **NOVELTY_KWARGS)
            abs_z.append(late_window_mean(np.abs(h['z']), window=30))
        mean_abs_z.append(np.mean(abs_z))
        sem_abs_z.append(np.std(abs_z, ddof=1) / np.sqrt(len(abs_z)))
    ax.errorbar(gamma_range, mean_abs_z, yerr=sem_abs_z, fmt='ko-',
                capsize=3, linewidth=1.5, label='Simulation mean |z|')
    gc_corr, _, _ = find_gamma_c_corrected(p, alpha_0)
    ax.axvline(x=gc_corr, color='red', linestyle='--', linewidth=2,
               label=f'Corrected γ_c = {gc_corr:.1f} (Eq. 13)')
    ax.set_xlabel('Policy precision γ')
    ax.set_ylabel(f'Mean |z| at t = {N_total}')
    ax.set_title(f'(a) Bifurcation in γ (p = {p}, α₀ = {alpha_0:.0f})')
    ax.legend(fontsize=9)

    # ---------------- Panel (b): sweep p at gamma = 16 ----------------
    ax = axes[1]
    gamma = 16.0
    p_range = np.arange(0.60, 0.98, 0.03)
    mean_abs_z, sem_abs_z = [], []
    for p_val in p_range:
        abs_z = []
        for i in range(n_agents):
            h = run_single_agent(p_val, alpha_0, gamma, delta_c=0.0, N_total=N_total,
                                 seed=8000 + i, **NOVELTY_KWARGS)
            abs_z.append(late_window_mean(np.abs(h['z']), window=30))
        mean_abs_z.append(np.mean(abs_z))
        sem_abs_z.append(np.std(abs_z, ddof=1) / np.sqrt(len(abs_z)))
    ax.errorbar(p_range, mean_abs_z, yerr=sem_abs_z, fmt='ko-',
                capsize=3, linewidth=1.5, label='Simulation mean |z|')
    pc_corr = find_p_c_corrected(gamma, alpha_0)
    if np.isfinite(pc_corr):
        ax.axvline(x=pc_corr, color='red', linestyle='--', linewidth=2,
                   label=f'Corrected p_c ≈ {pc_corr:.2f} (Eq. 13)')
    ax.set_xlabel('True discriminability p')
    ax.set_ylabel(f'Mean |z| at t = {N_total}')
    ax.set_title(f'(b) Bifurcation in p (γ = {gamma:.0f}, α₀ = {alpha_0:.0f})')
    ax.legend(fontsize=9)

    fig.suptitle('Figure A2: Stochastic Verification of the Novelty-Corrected '
                 'Condition (exact ΔW switched on)', fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figA2_verification_novelty.png")
    plt.close()
    print("Saved figA2_verification_novelty.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    ### DT ---> Console report of the corrected analytical values used above
    p = 0.85
    print("Corrected analytical predictions (Notebook 05, Eq. 13):")
    for a0 in [1.0, 2.0, 8.0, 16.0]:
        gc, tau_m, m = find_gamma_c_corrected(p, a0)
        print(f"  alpha_0 = {a0:5.1f}:  max(G - N) = {m:8.5f}  "
              f"gamma_c = {gc if np.isfinite(gc) else float('inf'):7.1f}  "
              f"(tau_max = {tau_m if np.isfinite(gc) else float('nan'):.2f})")
    print(f"  p_c(gamma = 16, alpha_0 = 8)  = {find_p_c_corrected(16.0, 8.0):.3f}")
    print(f"  alpha_0*(gamma = 16, p = .85) = {find_alpha0_c(16.0, 0.85):.2f}")
    vals = np.array([corrected_coupling(t, p, 8.0) for t in _TAU_GRID])
    inside = _TAU_GRID[vals > 1.0 / 16.0]
    print(f"  window (alpha_0 = 8, gamma = 16): tau in [{inside[0]:.2f}, {inside[-1]:.2f}]"
          f"  ->  t in [{16 * inside[0]:.0f}, {16 * inside[-1]:.0f}]")

    print("\nGenerating Figure A1 (flags with novelty)...")
    plot_catastrophe_flags_novelty()
    print("Generating Figure A2 (verification sweeps with novelty)...")
    plot_verification_novelty()
    print("Done.")
