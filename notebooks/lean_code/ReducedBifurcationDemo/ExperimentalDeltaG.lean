import Mathlib
import ReducedBifurcationDemo.Calculus

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

-- DT ---> This optional file is NOT imported by the main project.
-- DT ---> It records the final algebra/sign rewrite from the logarithmic derivative
-- DT ---> formula for ΔG into the compact couplingG expression.

theorem deriv_deltaG_zero_symm_eq_neg_two_couplingG_experimental
    {τ p : ℝ}
    (hτ : 0 < τ)
    (hp₁ : (1 : ℝ) / 2 < p)
    (hp₂ : p < 1) :
    deriv (fun z => deltaG z τ p 0) 0 = - 2 * couplingG τ p := by
  exact deriv_deltaG_zero_symm_eq_neg_two_couplingG hτ hp₁ hp₂

end ReducedBifurcationDemo
