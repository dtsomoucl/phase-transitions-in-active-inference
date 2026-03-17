import Mathlib
import ReducedBifurcationDemo.Calculus

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

-- DT ---> This optional file is NOT imported by the main project.
-- DT ---> Its purpose is to isolate the entropy derivative proof, which is the
-- DT ---> next genuine bottleneck after proving the derivatives of aFun and bFun.
-- DT ---> We keep it separate so the main package remains build-stable.

-- DT ---> If binEntropy is defined as
-- DT --->   - x * log x - (1 - x) * log (1 - x),
-- DT ---> then the target derivative is
-- DT --->   log (1 - x) - log x.
-- DT ---> This file is the safe place to prove that cleanly.

theorem hasDerivAt_binEntropy_experimental
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


-- DT ---> A weaker checkpoint target, useful while interactively proving the
-- DT ---> theorem above, is simply to inspect the derivative after unfolding.
theorem hasDerivAt_binEntropy_checkpoint
    {x : ℝ} (hx0 : 0 < x) (hx1 : x < 1) :
    ∃ y : ℝ, HasDerivAt binEntropy y x := by
  refine ⟨Real.log (1 - x) - Real.log x, ?_⟩
  exact hasDerivAt_binEntropy_experimental hx0 hx1

end ReducedBifurcationDemo
