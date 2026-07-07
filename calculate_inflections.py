"""
calculate_inflections.py
===================
Reproduces the two sweeps of Figure 3 (manuscript) / plot_verification()
in sim_bifurcation.py — identical parameters and identical random seeds —
and prints the location of steepest increase (the empirical inflection)
for the caption. (No figures are produced or modified.)

Also prints the same quantities for the novelty-corrected sweeps of
Figure A2 (Appendix A, Section A8), for the matching caption sentence.

"""

import numpy as np
from sim_bifurcation import run_single_agent, late_window_mean

### DT ---> Common settings, matching plot_verification() exactly
N_TOTAL = 200
N_AGENTS = 200

def sweep_gamma(alpha_0, use_novelty):
    gamma_range = np.arange(4, 30, 2)
    kw = dict(use_epistemic=True, epistemic_weight=0.5) if use_novelty else dict(use_epistemic=False)
    means = []
    for gamma in gamma_range:
        vals = [late_window_mean(np.abs(run_single_agent(0.85, alpha_0, gamma, delta_c=0.0,
                                                         N_total=N_TOTAL, seed=7000 + i, **kw)['z']),
                                 window=30)
                for i in range(N_AGENTS)]
        means.append(np.mean(vals))
    return gamma_range, np.array(means)

def sweep_p(alpha_0, use_novelty):
    p_range = np.arange(0.60, 0.98, 0.03)
    kw = dict(use_epistemic=True, epistemic_weight=0.5) if use_novelty else dict(use_epistemic=False)
    means = []
    for p in p_range:
        vals = [late_window_mean(np.abs(run_single_agent(p, alpha_0, 16.0, delta_c=0.0,
                                                         N_total=N_TOTAL, seed=8000 + i, **kw)['z']),
                                 window=30)
                for i in range(N_AGENTS)]
        means.append(np.mean(vals))
    return p_range, np.array(means)

def inflection(x, y):
    ### DT ---> Numerical derivative over the sweep grid, lightly smoothed against grid noise
    d = np.gradient(y, x)
    d = np.convolve(d, np.ones(3) / 3, mode='same')
    ### DT ---> Ignore the two edge points, where the smoothed derivative is unreliable
    k = 1 + np.argmax(d[1:-1])
    return x[k], d



def half_rise(x, y):
    ### DT ---> Midpoint between sub-critical baseline and saturation plateau (pseudo-critical point)
    m = 0.5 * (y[0] + y[-1])
    k = np.argmax(y >= m)              ### firstt grid point at or above the midpoint
    if k == 0:
        return x[0]
    ### DT ---> Linear interpolation between the bracketing grid points
    return x[k-1] + (m - y[k-1]) / (y[k] - y[k-1]) * (x[k] - x[k-1])

if __name__ == "__main__":
    print("Figure 3 (consolidation-only, alpha_0 = 2, DW = 0):")
    g, mg = sweep_gamma(2.0, use_novelty=False)
    gi, _ = inflection(g, mg)
    print(f"  (a) gamma sweep: mean |z| = {np.round(mg, 3).tolist()}")
    print(f"      steepest increase at gamma = {gi:g}   (analytical gamma_c = 12.7)")
    p, mp = sweep_p(2.0, use_novelty=False)
    pi, _ = inflection(p, mp)
    print(f"  (b) p sweep: mean |z| = {np.round(mp, 3).tolist()}")
    print(f"      steepest increase at p = {pi:.2f}   (analytical p_c = 0.81)")

    print("\nFigure A2 (novelty on, alpha_0 = 8, exact DW):")
    g, mg = sweep_gamma(8.0, use_novelty=True)
    gi, _ = inflection(g, mg)
    print(f"  (a) gamma sweep: steepest increase at gamma = {gi:g}   (corrected gamma_c = 15.2)")
    p, mp = sweep_p(8.0, use_novelty=True)
    pi, _ = inflection(p, mp)
    print(f"  (b) p sweep: steepest increase at p = {pi:.2f}   (corrected p_c = 0.84)")
