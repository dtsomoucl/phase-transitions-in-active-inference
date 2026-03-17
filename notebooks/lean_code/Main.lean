import ReducedBifurcationDemo

open ReducedBifurcationDemo

/-- DT ---> Runnable numerical demo matching Notebook 01's reduced scalar quantities. -/
def main : IO Unit := do
  let τ : Float := 2.19
  let p : Float := 0.9
  let γ : Float := 16.0
  let abarF : Float := 0.5 + (p - 0.5) * (τ / (1.0 + τ))
  let lambdaF : Float := (p - 0.5) * τ / ((1.0 + τ) * (1.0 + τ))
  let couplingG : Float :=
    lambdaF * Float.abs (Float.log ((1.0 - abarF) / abarF))
  let prod : Float := γ * couplingG
  let exceeded : String := if prod > 1.0 then "true" else "false"
  IO.println "Reduced bifurcation demo"
  IO.println s!"tau = {τ}"
  IO.println s!"p = {p}"
  IO.println s!"gamma = {γ}"
  IO.println s!"abar(tau,p) = {abarF}"
  IO.println s!"lambda(tau,p) = {lambdaF}"
  IO.println s!"couplingG(tau,p) = {couplingG}"
  IO.println s!"gamma * couplingG(tau,p) = {prod}"
  IO.println s!"bifurcation threshold exceeded? {exceeded}"
