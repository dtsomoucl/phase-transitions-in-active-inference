import Mathlib
import ReducedBifurcationDemo.Calculus

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

-- DT ---> This optional file is NOT imported by the main project.
-- DT ---> It isolates the first substantive derivative proofs for aFun and bFun.
-- DT ---> The proofs are split into small helper lemmas to avoid simplifier noise.

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

theorem hasDerivAt_aFun_experimental
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

theorem hasDerivAt_bFun_experimental
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

end ReducedBifurcationDemo
