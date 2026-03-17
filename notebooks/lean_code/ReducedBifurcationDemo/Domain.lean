import Mathlib
import ReducedBifurcationDemo.Defs
import ReducedBifurcationDemo.Algebra

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

lemma frac_a_nonneg
    {z τ : ℝ} (hz : -1 ≤ z) (hτ : 0 < τ) :
    0 ≤ ((1 + z) * τ) / (1 + (1 + z) * τ) := by
  have hnum : 0 ≤ (1 + z) * τ := by nlinarith
  have hden : 0 < 1 + (1 + z) * τ := by nlinarith
  exact div_nonneg hnum (le_of_lt hden)

lemma frac_b_nonneg
    {z τ : ℝ} (hz : z ≤ 1) (hτ : 0 < τ) :
    0 ≤ ((1 - z) * τ) / (1 + (1 - z) * τ) := by
  have hnum : 0 ≤ (1 - z) * τ := by nlinarith
  have hden : 0 < 1 + (1 - z) * τ := by nlinarith
  exact div_nonneg hnum (le_of_lt hden)

lemma denom_a_zero_ne_zero {τ : ℝ} (hτ : 0 < τ) :
    1 + (1 + (0 : ℝ)) * τ ≠ 0 := by
  have : 0 < 1 + (1 + (0 : ℝ)) * τ := by nlinarith
  linarith

lemma denom_b_zero_ne_zero {τ : ℝ} (hτ : 0 < τ) :
    1 + (1 - (0 : ℝ)) * τ ≠ 0 := by
  have : 0 < 1 + (1 - (0 : ℝ)) * τ := by nlinarith
  linarith

lemma abar_gt_half
    {τ p : ℝ} (hτ : 0 < τ) (hp : (1 : ℝ) / 2 < p) :
    (1 : ℝ) / 2 < abar τ p := by
  rw [abar_eq]
  have hp' : 0 < p - (1 : ℝ) / 2 := by linarith
  have hfrac : 0 < τ / (1 + τ) := by
    have : 0 < 1 + τ := by linarith
    exact div_pos hτ this
  have : 0 < (p - (1 : ℝ) / 2) * (τ / (1 + τ)) := by
    exact mul_pos hp' hfrac
  linarith

lemma abar_lt_one
    {τ p : ℝ} (hτ : 0 < τ) (hp : p < 1) :
    abar τ p < 1 := by
  rw [abar_eq]
  have hden : 0 < 1 + τ := by linarith
  have hfrac_pos : 0 < τ / (1 + τ) := by
    exact div_pos hτ hden
  have hfrac_lt' : τ / (1 + τ) < (1 + τ) / (1 + τ) := by
    have : τ < 1 + τ := by linarith
    exact div_lt_div_of_pos_right this hden
  have hfrac_lt : τ / (1 + τ) < 1 := by
    have hone : (1 + τ) / (1 + τ) = (1 : ℝ) := by
      field_simp
    rw [hone] at hfrac_lt'
    exact hfrac_lt'
  have hp' : p - (1 : ℝ) / 2 < (1 : ℝ) / 2 := by
    linarith
  have hmul_lt : (p - (1 : ℝ) / 2) * (τ / (1 + τ)) < (1 : ℝ) / 2 := by
    by_cases hx : p - (1 : ℝ) / 2 ≤ 0
    · nlinarith [hfrac_pos]
    · have hx_pos : 0 < p - (1 : ℝ) / 2 := by linarith
      have hlt1 : (p - (1 : ℝ) / 2) * (τ / (1 + τ)) < p - (1 : ℝ) / 2 := by
        have := mul_lt_mul_of_pos_left hfrac_lt hx_pos
        simpa [mul_comm, mul_left_comm, mul_assoc] using this
      linarith
  linarith

lemma abar_mem_Ioo
    {τ p : ℝ} (hτ : 0 < τ) (hp₁ : (1 : ℝ) / 2 < p) (hp₂ : p < 1) :
    abar τ p ∈ Set.Ioo 0 1 := by
  constructor
  · have hhalf : (0 : ℝ) < (1 : ℝ) / 2 := by norm_num
    have hab : (1 : ℝ) / 2 < abar τ p := abar_gt_half hτ hp₁
    linarith
  · exact abar_lt_one hτ hp₂

end ReducedBifurcationDemo
