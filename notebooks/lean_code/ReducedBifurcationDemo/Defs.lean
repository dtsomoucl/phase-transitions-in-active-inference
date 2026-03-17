import Mathlib

noncomputable section
open Real
open scoped Topology

namespace ReducedBifurcationDemo

-- DT ---> Definitions extracted from Notebook 01's reduced scalar model.

def binEntropy (x : ℝ) : ℝ :=
  -x * Real.log x - (1 - x) * Real.log (1 - x)

def aFun (z τ p : ℝ) : ℝ :=
  (1 : ℝ)/2 + (p - (1 : ℝ)/2) * (((1 + z) * τ) / (1 + (1 + z) * τ))

def bFun (z τ p : ℝ) : ℝ :=
  (1 : ℝ)/2 + (p - (1 : ℝ)/2) * (((1 - z) * τ) / (1 + (1 - z) * τ))

def deltaG (z τ p Δc : ℝ) : ℝ :=
  (1 - aFun z τ p - bFun z τ p) * Δc
  + binEntropy (aFun z τ p) - binEntropy (bFun z τ p)

def fixedMap (z τ p γ Δc : ℝ) : ℝ :=
  Real.tanh (-(γ / 2) * deltaG z τ p Δc)

def F (z τ p γ Δc : ℝ) : ℝ :=
  fixedMap z τ p γ Δc - z

def abar (τ p : ℝ) : ℝ :=
  aFun 0 τ p

def lambda (τ p : ℝ) : ℝ :=
  (p - (1 : ℝ)/2) * τ / (1 + τ)^2

def couplingG (τ p : ℝ) : ℝ :=
  lambda τ p * |Real.log ((1 - abar τ p) / abar τ p)|

structure ValidParams where
  τ : ℝ
  p : ℝ
  γ : ℝ
  Δc : ℝ
  hτ : 0 < τ
  hp_lower : (1 : ℝ)/2 < p
  hp_upper : p < 1
  hγ : 0 < γ

structure ValidSymmetricParams extends ValidParams where
  hΔc : Δc = 0

end ReducedBifurcationDemo
