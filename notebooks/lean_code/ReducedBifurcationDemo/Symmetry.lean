import Mathlib
import ReducedBifurcationDemo.Bifurcation

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

lemma fixedMap_neg_symm (z τ p γ : ℝ) :
    fixedMap (-z) τ p γ 0 = - fixedMap z τ p γ 0 := by
  simp [fixedMap, deltaG_neg_symm, Real.tanh_neg]

lemma F_neg_symm (z τ p γ : ℝ) :
    F (-z) τ p γ 0 = - F z τ p γ 0 := by
  simp [F, fixedMap_neg_symm]
  ring

theorem second_deriv_F_zero_symm
    {τ p γ : ℝ}
    (hreg : ContDiff ℝ 3 (fun z => F z τ p γ 0)) :
    deriv (fun z => deriv (fun z => F z τ p γ 0) z) 0 = 0 := by
  let f : ℝ → ℝ := fun z => F z τ p γ 0

  have hfodd : ∀ z : ℝ, f (-z) = - f z := by
    intro z
    simpa [f] using F_neg_symm z τ p γ

  have hf_diff : Differentiable ℝ f := by
    simpa [f] using
      (hreg.differentiable_iteratedDeriv (m := 0)
        (by norm_num : ((0 : ℕ) : WithTop ℕ∞) < (3 : WithTop ℕ∞)))

  have hg_diff : Differentiable ℝ (fun z => deriv f z) := by
    simpa [f, iteratedDeriv_one] using
      (hreg.differentiable_iteratedDeriv (m := 1)
        (by norm_num : ((1 : ℕ) : WithTop ℕ∞) < (3 : WithTop ℕ∞)))

  have hgeven : ∀ z : ℝ, deriv f (-z) = deriv f z := by
    intro z
    have hf_negz : HasDerivAt f (deriv f (-z)) (-z) := by
      simpa using (hf_diff (-z)).hasDerivAt
    have hleft : HasDerivAt (fun x : ℝ => f (-x)) (-(deriv f (-z))) z := by
      simpa [mul_comm, mul_left_comm, mul_assoc]
        using hf_negz.comp z ((hasDerivAt_id z).neg)
    have hf_z : HasDerivAt f (deriv f z) z := by
      simpa using (hf_diff z).hasDerivAt
    have hright : HasDerivAt (fun x : ℝ => - f x) (-(deriv f z)) z := by
      simpa using hf_z.neg
    have hleft' : HasDerivAt (fun x : ℝ => - f x) (-(deriv f (-z))) z := by
      simpa [hfodd] using hleft
    have hEq : -(deriv f z) = -(deriv f (-z)) := by
      exact hright.unique hleft'
    linarith

  have hg0 : HasDerivAt (fun z => deriv f z) (deriv (fun z => deriv f z) 0) 0 := by
    simpa using (hg_diff 0).hasDerivAt

  have hneg0 : HasDerivAt (fun x : ℝ => -x) (-1) 0 := by
    simpa using (hasDerivAt_id 0).neg

  have hg0' : HasDerivAt (fun z => deriv f z) (deriv (fun z => deriv f z) 0) (-0) := by
    simpa using hg0

  have hleft0 : HasDerivAt (fun x : ℝ => deriv f (-x))
      (-(deriv (fun z => deriv f z) 0)) 0 := by
    simpa [mul_comm, mul_left_comm, mul_assoc]
      using hg0'.comp 0 hneg0

  have hleft0' : HasDerivAt (fun x : ℝ => deriv f x)
      (-(deriv (fun z => deriv f z) 0)) 0 := by
    simpa [hgeven] using hleft0

  have hEq0 : deriv (fun z => deriv f z) 0 = -(deriv (fun z => deriv f z) 0) := by
    exact hg0.unique hleft0'

  linarith

end ReducedBifurcationDemo
