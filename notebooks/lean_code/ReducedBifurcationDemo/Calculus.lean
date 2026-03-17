import Mathlib
import ReducedBifurcationDemo.Defs
import ReducedBifurcationDemo.Domain
import ReducedBifurcationDemo.Algebra

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

lemma hasDerivAt_binEntropy
    {x : ℝ} (hx0 : 0 < x) (hx1 : x < 1) :
    HasDerivAt binEntropy (Real.log (1 - x) - Real.log x) x := by
  have hxne : x ≠ 0 := ne_of_gt hx0
  have h1mx_pos : 0 < 1 - x := by
    linarith
  have h1mx_ne : 1 - x ≠ 0 := ne_of_gt h1mx_pos

  have hid : HasDerivAt (fun y : ℝ => y) 1 x := hasDerivAt_id x
  have h1my : HasDerivAt (fun y : ℝ => 1 - y) (-1) x := by
    simpa using ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_id x))

  have hlogx : HasDerivAt (fun y : ℝ => Real.log y) (1 / x) x := by
    simpa using Real.hasDerivAt_log hxne

  have hlog1mx :
      HasDerivAt (fun y : ℝ => Real.log (1 - y)) (-(1 - x)⁻¹) x := by
    simpa [Function.comp] using (Real.hasDerivAt_log h1mx_ne).comp x h1my

  have hterm1_raw :
      HasDerivAt (fun y : ℝ => Real.log y * y) (x⁻¹ * x + Real.log x) x := by
    simpa using hlogx.mul hid

  have hterm1 :
      HasDerivAt (fun y : ℝ => -(Real.log y * y)) (-(x⁻¹ * x + Real.log x)) x := by
    exact hterm1_raw.neg

  have hterm2_raw :
      HasDerivAt (fun y : ℝ => (1 - y) * Real.log (1 - y))
        ((-1) * Real.log (1 - x) + (1 - x) * (-(1 - x)⁻¹)) x := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using h1my.mul hlog1mx

  have hterm2 :
      HasDerivAt (fun y : ℝ => -((1 - y) * Real.log (1 - y)))
        (Real.log (1 - x) + 1) x := by
    have hscalar :
        -(((-1) * Real.log (1 - x) + (1 - x) * (-(1 - x)⁻¹)))
          = Real.log (1 - x) + 1 := by
      field_simp [h1mx_ne]
      ring_nf
    exact hscalar ▸ hterm2_raw.neg

  have hsum :
      HasDerivAt
        (fun y : ℝ => -(Real.log y * y) - ((1 - y) * Real.log (1 - y)))
        (-(x⁻¹ * x + Real.log x) + (Real.log (1 - x) + 1)) x := by
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using hterm1.add hterm2

  convert hsum using 1
  · funext y
    simp [binEntropy, sub_eq_add_neg]
    ring
  · have hxinv : x⁻¹ * x = 1 := by
      field_simp [hxne]
    rw [hxinv]
    ring_nf

lemma aFun_u_hasDerivAt (τ z : ℝ) :
    HasDerivAt (fun x => (1 + x) * τ) τ z := by
  simpa [mul_comm, mul_left_comm, mul_assoc]
    using (((hasDerivAt_id z).const_add 1).mul_const τ)

lemma aFun_v_hasDerivAt (τ z : ℝ) :
    HasDerivAt (fun x => 1 + (1 + x) * τ) τ z := by
  simpa [add_comm, add_left_comm, add_assoc, mul_comm, mul_left_comm, mul_assoc]
    using ((aFun_u_hasDerivAt τ z).const_add 1)

lemma aFun_quot_hasDerivAt
    (τ z : ℝ) (hden : 1 + (1 + z) * τ ≠ 0) :
    HasDerivAt
      (fun x => ((1 + x) * τ) / (1 + (1 + x) * τ))
      ((τ * (1 + (1 + z) * τ) - (1 + z) * τ * τ) / (1 + (1 + z) * τ)^2) z := by
  exact (aFun_u_hasDerivAt τ z).div (aFun_v_hasDerivAt τ z) hden

lemma aFun_quot_deriv_simplifies (τ z : ℝ) :
    (τ * (1 + (1 + z) * τ) - (1 + z) * τ * τ) / (1 + (1 + z) * τ)^2
      = τ / (1 + (1 + z) * τ)^2 := by
  have hnum : τ * (1 + (1 + z) * τ) - (1 + z) * τ * τ = τ := by
    ring_nf
  rw [hnum]

lemma hasDerivAt_aFun
    (τ p z : ℝ)
    (hden : 1 + (1 + z) * τ ≠ 0) :
    HasDerivAt (fun z => aFun z τ p)
      ((p - (1 : ℝ) / 2) * τ / (1 + (1 + z) * τ)^2) z := by
  simp [aFun]
  have hquot := aFun_quot_hasDerivAt τ z hden
  have hmul :
      HasDerivAt
        (fun x => (p - (1 : ℝ) / 2) * (((1 + x) * τ) / (1 + (1 + x) * τ)))
        ((p - (1 : ℝ) / 2) *
          ((τ * (1 + (1 + z) * τ) - (1 + z) * τ * τ) / (1 + (1 + z) * τ)^2)) z := by
    simpa [mul_comm, mul_left_comm, mul_assoc]
      using (hquot.const_mul (p - (1 : ℝ) / 2))
  have hsimp :
      (p - (1 : ℝ) / 2) *
        ((τ * (1 + (1 + z) * τ) - (1 + z) * τ * τ) / (1 + (1 + z) * τ)^2)
      = ((p - (1 : ℝ) / 2) * τ) / (1 + (1 + z) * τ)^2 := by
    rw [aFun_quot_deriv_simplifies]
    ring_nf
  have hmul' :
      HasDerivAt
        (fun x => (p - (1 : ℝ) / 2) * (((1 + x) * τ) / (1 + (1 + x) * τ)))
        (((p - (1 : ℝ) / 2) * τ) / (1 + (1 + z) * τ)^2) z := by
    exact hsimp ▸ hmul
  simpa [one_div] using hmul'

lemma bFun_u_hasDerivAt (τ z : ℝ) :
    HasDerivAt (fun x => (1 - x) * τ) (-τ) z := by
  simpa [sub_eq_add_neg, mul_comm, mul_left_comm, mul_assoc]
    using ((((hasDerivAt_const z (1 : ℝ)).sub (hasDerivAt_id z))).mul_const τ)

lemma bFun_v_hasDerivAt (τ z : ℝ) :
    HasDerivAt (fun x => 1 + (1 - x) * τ) (-τ) z := by
  simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc, mul_comm, mul_left_comm, mul_assoc]
    using ((bFun_u_hasDerivAt τ z).const_add 1)

lemma bFun_quot_hasDerivAt
    (τ z : ℝ) (hden : 1 + (1 - z) * τ ≠ 0) :
    HasDerivAt
      (fun x => ((1 - x) * τ) / (1 + (1 - x) * τ))
      (((-τ) * (1 + (1 - z) * τ) - (1 - z) * τ * (-τ)) / (1 + (1 - z) * τ)^2) z := by
  exact (bFun_u_hasDerivAt τ z).div (bFun_v_hasDerivAt τ z) hden

lemma bFun_quot_deriv_simplifies (τ z : ℝ) :
    ((-τ) * (1 + (1 - z) * τ) - (1 - z) * τ * (-τ)) / (1 + (1 - z) * τ)^2
      = (-τ) / (1 + (1 - z) * τ)^2 := by
  have hnum : (-τ) * (1 + (1 - z) * τ) - (1 - z) * τ * (-τ) = -τ := by
    ring_nf
  rw [hnum]

lemma hasDerivAt_bFun
    (τ p z : ℝ)
    (hden : 1 + (1 - z) * τ ≠ 0) :
    HasDerivAt (fun z => bFun z τ p)
      (-(p - (1 : ℝ) / 2) * τ / (1 + (1 - z) * τ)^2) z := by
  simp [bFun]
  have hquot := bFun_quot_hasDerivAt τ z hden
  have hmul :
      HasDerivAt
        (fun x => (p - (1 : ℝ) / 2) * (((1 - x) * τ) / (1 + (1 - x) * τ)))
        ((p - (1 : ℝ) / 2) *
          (((-τ) * (1 + (1 - z) * τ) - (1 - z) * τ * (-τ)) / (1 + (1 - z) * τ)^2)) z := by
    simpa [mul_comm, mul_left_comm, mul_assoc]
      using (hquot.const_mul (p - (1 : ℝ) / 2))
  have hsimp :
      (p - (1 : ℝ) / 2) *
        (((-τ) * (1 + (1 - z) * τ) - (1 - z) * τ * (-τ)) / (1 + (1 - z) * τ)^2)
      = (-(p - (1 : ℝ) / 2) * τ) / (1 + (1 - z) * τ)^2 := by
    rw [bFun_quot_deriv_simplifies]
    ring_nf
  have hmul' :
      HasDerivAt
        (fun x => (p - (1 : ℝ) / 2) * (((1 - x) * τ) / (1 + (1 - x) * τ)))
        ((-(p - (1 : ℝ) / 2) * τ) / (1 + (1 - z) * τ)^2) z := by
    exact hsimp ▸ hmul
  simpa [one_div] using hmul'

lemma deriv_aFun_zero
    {τ p : ℝ} (hτ : 0 < τ) :
    deriv (fun z => aFun z τ p) 0 = lambda τ p := by
  have h := hasDerivAt_aFun τ p 0 (denom_a_zero_ne_zero hτ)
  calc
    deriv (fun z => aFun z τ p) 0
        = ((p - (1 : ℝ) / 2) * τ) / (1 + τ)^2 := by
            simpa using h.deriv
    _ = lambda τ p := by
          simp [lambda]

lemma deriv_bFun_zero
    {τ p : ℝ} (hτ : 0 < τ) :
    deriv (fun z => bFun z τ p) 0 = - lambda τ p := by
  have h := hasDerivAt_bFun τ p 0 (denom_b_zero_ne_zero hτ)
  calc
    deriv (fun z => bFun z τ p) 0
        = (-(p - (1 : ℝ) / 2) * τ) / (1 + τ)^2 := by
            simpa using h.deriv
    _ = - lambda τ p := by
          unfold lambda
          ring

lemma deriv_deltaG_zero_symm
    {τ p : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ) / 2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => deltaG z τ p 0) 0
      = 2 * lambda τ p * (Real.log (1 - abar τ p) - Real.log (abar τ p)) := by
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

  have hmain := hΔ.deriv
  calc
    deriv (fun z => deltaG z τ p 0) 0
        = ((Real.log (1 - abar τ p) - Real.log (abar τ p)) * lambda τ p) -
          ((Real.log (1 - abar τ p) - Real.log (abar τ p)) * (- lambda τ p)) := by
            exact hmain
    _ = 2 * lambda τ p * (Real.log (1 - abar τ p) - Real.log (abar τ p)) := by
          ring

lemma deriv_deltaG_zero_symm_eq_neg_two_couplingG
    {τ p : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ) / 2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => deltaG z τ p 0) 0 = - 2 * couplingG τ p := by
  have hab : abar τ p ∈ Set.Ioo 0 1 := abar_mem_Ioo hτ hp₁ hp₂
  have habhalf : (1 : ℝ) / 2 < abar τ p := abar_gt_half hτ hp₁
  have habne : abar τ p ≠ 0 := ne_of_gt hab.1
  have hnumne : 1 - abar τ p ≠ 0 := by
    linarith [hab.2]
  have hratio_pos : 0 < ((1 - abar τ p) / abar τ p) := by
    exact div_pos (sub_pos.mpr hab.2) hab.1
  have hratio_lt : ((1 - abar τ p) / abar τ p) < 1 := by
    have hlt : 1 - abar τ p < abar τ p := by
      linarith
    have hpos : 0 < abar τ p := hab.1
    have hdiv : (1 - abar τ p) / abar τ p < abar τ p / abar τ p := by
      exact div_lt_div_of_pos_right hlt hpos
    simpa [habne] using hdiv
  have hlogeq :
      Real.log ((1 - abar τ p) / abar τ p)
        = Real.log (1 - abar τ p) - Real.log (abar τ p) := by
    rw [Real.log_div hnumne habne]
  have hlogneg : Real.log ((1 - abar τ p) / abar τ p) < 0 := by
    exact Real.log_neg hratio_pos hratio_lt

  rw [deriv_deltaG_zero_symm hτ hp₁ hp₂]
  unfold couplingG
  rw [← hlogeq]
  rw [abs_of_neg hlogneg]
  ring

lemma deriv_deltaG_zero_symm_neg
    {τ p : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ) / 2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => deltaG z τ p 0) 0 < 0 := by
  rw [deriv_deltaG_zero_symm_eq_neg_two_couplingG hτ hp₁ hp₂]
  unfold couplingG
  have hlam_pos : 0 < lambda τ p := by
    unfold lambda
    have hp' : 0 < p - (1 : ℝ) / 2 := by linarith
    have hden : 0 < (1 + τ)^2 := by positivity
    exact div_pos (mul_pos hp' hτ) hden
  have hab : abar τ p ∈ Set.Ioo 0 1 := abar_mem_Ioo hτ hp₁ hp₂
  have habhalf : (1 : ℝ) / 2 < abar τ p := abar_gt_half hτ hp₁
  have habne : abar τ p ≠ 0 := ne_of_gt hab.1
  have hratio_pos : 0 < ((1 - abar τ p) / abar τ p) := by
    exact div_pos (sub_pos.mpr hab.2) hab.1
  have hratio_lt : ((1 - abar τ p) / abar τ p) < 1 := by
    have hlt : 1 - abar τ p < abar τ p := by
      linarith
    have hpos : 0 < abar τ p := hab.1
    have hdiv : (1 - abar τ p) / abar τ p < abar τ p / abar τ p := by
      exact div_lt_div_of_pos_right hlt hpos
    simpa [habne] using hdiv
  have hlogneg : Real.log ((1 - abar τ p) / abar τ p) < 0 := by
    exact Real.log_neg hratio_pos hratio_lt
  have habs_pos : 0 < |Real.log ((1 - abar τ p) / abar τ p)| := by
    rw [abs_of_neg hlogneg]
    linarith
  have hcg_pos : 0 < couplingG τ p := by
    unfold couplingG
    exact mul_pos hlam_pos habs_pos
  have hneg2 : (-2 : ℝ) < 0 := by norm_num
  exact mul_neg_of_neg_of_pos hneg2 hcg_pos

end ReducedBifurcationDemo
