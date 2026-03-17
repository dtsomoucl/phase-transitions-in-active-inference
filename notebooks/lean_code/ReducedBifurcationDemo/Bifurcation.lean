import Mathlib
import ReducedBifurcationDemo.Calculus

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

lemma deltaG_zero_symm (τ p : ℝ) :
    deltaG 0 τ p 0 = 0 := by
  simp [deltaG, a_zero, b_zero]

lemma F_zero_symm (τ p γ : ℝ) :
    F 0 τ p γ 0 = 0 := by
  simp [F, fixedMap, deltaG_zero_symm]

lemma hasDerivAt_deltaG_zero_symm
    {τ p : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ) / 2 < p)
    (hp₂ : p < 1) :
    HasDerivAt (fun z => deltaG z τ p 0)
      (deriv (fun z => deltaG z τ p 0) 0) 0 := by
  have hab : abar τ p ∈ Set.Ioo 0 1 := abar_mem_Ioo hτ hp₁ hp₂
  have hbin :
      HasDerivAt binEntropy
        (Real.log (1 - abar τ p) - Real.log (abar τ p))
        (abar τ p) := by
    exact hasDerivAt_binEntropy hab.1 hab.2

  have ha0 : HasDerivAt (fun z => aFun z τ p)
      (((p - (1 : ℝ) / 2) * τ) / (1 + τ)^2) 0 := by
    simpa using hasDerivAt_aFun τ p 0 (denom_a_zero_ne_zero hτ)

  have hb0 : HasDerivAt (fun z => bFun z τ p)
      ((-(p - (1 : ℝ) / 2) * τ) / (1 + τ)^2) 0 := by
    simpa using hasDerivAt_bFun τ p 0 (denom_b_zero_ne_zero hτ)

  have hbinA :
      HasDerivAt binEntropy
        (Real.log (1 - abar τ p) - Real.log (abar τ p))
        (aFun 0 τ p) := by
    simpa [a_zero τ p] using hbin

  have hbinB :
      HasDerivAt binEntropy
        (Real.log (1 - abar τ p) - Real.log (abar τ p))
        (bFun 0 τ p) := by
    simpa [b_zero τ p] using hbin

  have hA :
      HasDerivAt (fun z => binEntropy (aFun z τ p))
        ((Real.log (1 - abar τ p) - Real.log (abar τ p)) * lambda τ p) 0 := by
    have hcomp := hbinA.comp 0 ha0
    simpa [lambda, mul_comm, mul_left_comm, mul_assoc] using hcomp

  have hB :
      HasDerivAt (fun z => binEntropy (bFun z τ p))
        ((Real.log (1 - abar τ p) - Real.log (abar τ p)) * (- lambda τ p)) 0 := by
    have hcomp := hbinB.comp 0 hb0
    have : (((Real.log (1 - abar τ p) - Real.log (abar τ p)) *
        (-(p - (1 : ℝ) / 2) * τ / (1 + τ) ^ 2)))
        = (Real.log (1 - abar τ p) - Real.log (abar τ p)) * (- lambda τ p) := by
      simp [lambda]
      ring
    exact this ▸ hcomp

  have hΔ :
      HasDerivAt (fun z => deltaG z τ p 0)
        (((Real.log (1 - abar τ p) - Real.log (abar τ p)) * lambda τ p) -
          ((Real.log (1 - abar τ p) - Real.log (abar τ p)) * (- lambda τ p))) 0 := by
    simpa [deltaG] using hA.sub hB

  have hΔ' :
      HasDerivAt (fun z => deltaG z τ p 0)
        (2 * lambda τ p * (Real.log (1 - abar τ p) - Real.log (abar τ p))) 0 := by
    convert hΔ using 1
    ring_nf

  exact (deriv_deltaG_zero_symm hτ hp₁ hp₂).symm ▸ hΔ'

lemma hasDerivAt_tanh_zero :
    HasDerivAt (fun x : ℝ => Real.tanh x) 1 0 := by
  have hsinh : HasDerivAt (fun x : ℝ => Real.sinh x) (Real.cosh 0) 0 := by
    simpa using (Real.hasDerivAt_sinh (x := (0 : ℝ)))
  have hcosh : HasDerivAt (fun x : ℝ => Real.cosh x) (Real.sinh 0) 0 := by
    simpa using (Real.hasDerivAt_cosh (x := (0 : ℝ)))
  have hq :
      HasDerivAt (fun x : ℝ => Real.sinh x / Real.cosh x)
        ((Real.cosh 0 * Real.cosh 0 - Real.sinh 0 * Real.sinh 0) / Real.cosh 0 ^ 2) 0 := by
    exact hsinh.div hcosh (by simpa using Real.cosh_ne_zero (0 : ℝ))
  convert hq using 1
  · funext x
    simp [Real.tanh_eq_sinh_div_cosh]
  · norm_num [Real.sinh_zero, Real.cosh_zero]

lemma deriv_F_zero_symm
    {τ p γ : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ)/2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => F z τ p γ 0) 0
      = - (γ / 2) * deriv (fun z => deltaG z τ p 0) 0 - 1 := by
  have hΔ : HasDerivAt (fun z => deltaG z τ p 0)
      (deriv (fun z => deltaG z τ p 0) 0) 0 := by
    exact hasDerivAt_deltaG_zero_symm hτ hp₁ hp₂
  have hinner :
      HasDerivAt (fun z => (γ / 2) * deltaG z τ p 0)
        ((γ / 2) * deriv (fun z => deltaG z τ p 0) 0) 0 := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hΔ.const_mul (γ / 2)
  have houter :
      HasDerivAt (fun x : ℝ => - Real.tanh x) (-1) ((γ / 2) * deltaG 0 τ p 0) := by
    simpa [deltaG_zero_symm] using (hasDerivAt_tanh_zero.neg :
      HasDerivAt (fun x : ℝ => - Real.tanh x) (-1) 0)
  have hfix :
      HasDerivAt (fun z => fixedMap z τ p γ 0)
        (-(γ / 2) * deriv (fun z => deltaG z τ p 0) 0) 0 := by
    have hcomp := houter.comp 0 hinner
    simpa [fixedMap, deltaG_zero_symm, mul_comm, mul_left_comm, mul_assoc] using hcomp
  have hid : HasDerivAt (fun z : ℝ => z) 1 0 := hasDerivAt_id 0
  have hFraw :
      HasDerivAt (fun z => -z + fixedMap z τ p γ 0)
        (-1 + -(γ / 2) * deriv (fun z => deltaG z τ p 0) 0) 0 := by
    exact hid.neg.add hfix
  have hF :
      HasDerivAt (fun z => F z τ p γ 0)
        (-(γ / 2) * deriv (fun z => deltaG z τ p 0) 0 - 1) 0 := by
    convert hFraw using 1
    · funext z
      simp [F, sub_eq_add_neg, add_comm, add_left_comm, add_assoc]
    · ring
  simpa using hF.deriv

theorem deriv_F_zero_symm_eq_gamma_couplingG_sub_one
    {τ p γ : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ)/2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => F z τ p γ 0) 0 = γ * couplingG τ p - 1 := by
  rw [deriv_F_zero_symm hτ hp₁ hp₂]
  rw [deriv_deltaG_zero_symm_eq_neg_two_couplingG hτ hp₁ hp₂]
  ring

theorem local_stability_if_lt
    {τ p γ : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ)/2 < p)
    (hp₂ : p < 1)
    (hcrit : γ * couplingG τ p < 1) :
    deriv (fun z => F z τ p γ 0) 0 < 0 := by
  rw [deriv_F_zero_symm_eq_gamma_couplingG_sub_one hτ hp₁ hp₂]
  linarith

theorem local_instability_if_gt
    {τ p γ : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ)/2 < p)
    (hp₂ : p < 1)
    (hcrit : 1 < γ * couplingG τ p) :
    0 < deriv (fun z => F z τ p γ 0) 0 := by
  rw [deriv_F_zero_symm_eq_gamma_couplingG_sub_one hτ hp₁ hp₂]
  linarith

end ReducedBifurcationDemo
