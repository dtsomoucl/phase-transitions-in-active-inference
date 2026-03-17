# ReducedBifurcationDemo

This is a Lean 4 + mathlib starter project for the reduced bifurcation model derived from the two notebooks with the key derivation (main result):

- `Notebook_01_Bifurcation_Conditions.md`
- `Notebook_04_Verified_Derivation_Steps.md`

The project has two goals:

1. provide a Lean theorem skeleton for the **core reduced deterministic mathematics**, and
2. provide a small executable so we can **run something immediately** and see numerical output.

---

## Overview 

Here we are looking to formalise specific claims to prove with lean code. As a top-level overview, this is the sort of claim we formalise:

"Under the reduced two-state model, with expected-count closure and the stated definitions of a,b,ΔG, the fixed-point equation has the stated form; in the symmetric case the origin loses stability when γG(τ,p)=1; and near that point the local equation is equivalent to a pitchfork, with a cusp-type unfolding under small asymmetry."

---

## Folder layout

- `ReducedBifurcationDemo/Defs.lean`
- `ReducedBifurcationDemo/Algebra.lean`
- `ReducedBifurcationDemo/Domain.lean`
- `ReducedBifurcationDemo/Calculus.lean`
- `ReducedBifurcationDemo/Bifurcation.lean`
- `ReducedBifurcationDemo/Symmetry.lean`
- `ReducedBifurcationDemo/NormalForm.lean`
- `ReducedBifurcationDemo.lean` (library umbrella import)
- `Main.lean` (numerical demo executable)

## What is fully proved vs. what is scaffolded (IMPORTANT!)

This project is designed to **build and open cleanly in Lean with placeholders**.
Some easy lemmas are proved. The harder analytic steps are intentionally left as `by sorry` in various stages, and as we make progress we gradually replace them with lemmas that are needed to beuild to the full proof.

That makes the project practical as a starting point:
- it opens,
- the theorem statements elaborate,
- we can click on the `sorry`s and continue the proofs.

---

## How to run

Inside the project folder:

```bash
lake exe cache get
lake build
lake exe reducedBifurcationMain
```

---

## What output to expect

The executable prints a short numerical demo for
- `τ = 2.0`
- `p = 0.9`
- `γ = 16.0`

Then one should see lines of the form:

```text
ReducedBifurcationDemo

abar(tau,p) ≈ ...
lambda(tau,p) ≈ ...
couplingG(tau,p) ≈ ...
gamma * couplingG(tau,p) ≈ ...
Is gamma * couplingG > 1? true/false
```

The theorem files themselves do not print runtime output. Their “output” is:
- elaborated theorem states,
- goal windows for incomplete proofs,
- and Lean diagnostics.

(I used Visual Studio Code and LeanInfo view to debug).

---

## Suggested first file to inspect

Open:

- `ReducedBifurcationDemo/Bifurcation.lean`

The main theorem target is:

```lean
theorem deriv_F_zero_symm_eq_gamma_couplingG_sub_one
    {τ p γ : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ) / 2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => F z τ p γ 0) 0 = γ * couplingG τ p - 1 := by
  sorry
```

This is the cleanest formal statement of the local criticality condition from the notebooks.

---

## Files replaced in this last iteration

- `ReducedBifurcationDemo/Algebra.lean`
- `ReducedBifurcationDemo/Domain.lean`

## What changed in the final fixes

### Algebra
This replacement removes the placeholder `sorry`s from the placeholder/stepwise-checks version by filling in:
- `abar_eq`
- `a_zero`
- `b_zero`
- `a_neg_eq_b`
- `b_neg_eq_a`
- `deltaG_neg_symm`
- `sigmoid_eq_half_one_add_tanh_half`
- `phi_eq_sigmoid_iff_z_eq_tanh`

The most delicate lemma remains:
- `sigmoid_eq_half_one_add_tanh_half`

That is the robust version for the current local build.

## Recommended commands

From the project root:

```bash
lake clean
lake build
lake env lean --run Main.lean
```

---

**For Notebooks 01-04**: Phase Transitions in Early Learning — An Active Inference Approach

D. I. Tsomokos

**Date**: 12 March 2026

**Author**: Dr Dimitris I. Tsomokos

Psychology & Human Development, Institute of Education, University College London
