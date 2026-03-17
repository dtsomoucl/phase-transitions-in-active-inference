import Mathlib
import ReducedBifurcationDemo.Defs

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

def G1 (a c₁ c₂ : ℝ) : ℝ :=
  -(a * c₁ + (1 - a) * c₂) + binEntropy a

def G2 (b c₁ c₂ : ℝ) : ℝ :=
  -((1 - b) * c₁ + b * c₂) + binEntropy b

lemma deltaG_from_raw
    (a b c₁ c₂ : ℝ) :
    G1 a c₁ c₂ - G2 b c₁ c₂
      = (1 - a - b) * (c₁ - c₂) + binEntropy a - binEntropy b := by
  unfold G1 G2
  ring_nf

lemma abar_eq (τ p : ℝ) :
    abar τ p = (1 : ℝ) / 2 + (p - (1 : ℝ) / 2) * (τ / (1 + τ)) := by
  simp [abar, aFun]

lemma a_zero (τ p : ℝ) :
    aFun 0 τ p = abar τ p := by
  rfl

lemma b_zero (τ p : ℝ) :
    bFun 0 τ p = abar τ p := by
  simp [abar, aFun, bFun]

lemma a_neg_eq_b (z τ p : ℝ) :
    aFun (-z) τ p = bFun z τ p := by
  have hz : (1 + (-z : ℝ)) = 1 - z := by ring
  simp [aFun, bFun, hz]

lemma b_neg_eq_a (z τ p : ℝ) :
    bFun (-z) τ p = aFun z τ p := by
  have hz : (1 - (-z : ℝ)) = 1 + z := by ring
  simp [aFun, bFun, hz]

lemma deltaG_neg_symm (z τ p : ℝ) :
    deltaG (-z) τ p 0 = - deltaG z τ p 0 := by
  simp [deltaG, a_neg_eq_b, b_neg_eq_a]

def sigmoid (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

lemma sigmoid_eq_half_one_add_tanh_half (x : ℝ) :
    sigmoid x = (1 + Real.tanh (x / 2)) / 2 := by
  unfold sigmoid
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have hexp : Real.exp (-x) = Real.exp (-(x / 2)) * Real.exp (-(x / 2)) := by
    calc
      Real.exp (-x) = Real.exp (-(x / 2) + -(x / 2)) := by
        congr 1
        ring
      _ = Real.exp (-(x / 2)) * Real.exp (-(x / 2)) := by
        rw [Real.exp_add]
  have hneg : Real.exp (-(x / 2)) = (Real.exp (x / 2))⁻¹ := by
    rw [Real.exp_neg]
  have hpos : 0 < Real.exp (x / 2) := by
    positivity
  have hne : Real.exp (x / 2) ≠ 0 := ne_of_gt hpos
  simp [hexp, hneg]
  field_simp [hne]
  ring

lemma phi_eq_sigmoid_iff_z_eq_tanh
    (φ z x : ℝ) (hz : z = 2 * φ - 1) :
    φ = sigmoid x ↔ z = Real.tanh (x / 2) := by
  constructor
  · intro hφ
    rw [hz, hφ, sigmoid_eq_half_one_add_tanh_half]
    ring
  · intro hz'
    have hφ' : φ = (1 + Real.tanh (x / 2)) / 2 := by
      linarith [hz, hz']
    rw [sigmoid_eq_half_one_add_tanh_half]
    exact hφ'

end ReducedBifurcationDemo
