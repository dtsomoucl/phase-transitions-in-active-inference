# Notebook 01 — Bifurcation Conditions in a Reduced Active-Inference Model with Dirichlet Learning

**Series:** *Phase Transitions in Early Learning — An Active Inference Approach*

**Date:** 09 March 2026

**Status:** Analytical derivation of the main result

**Author:** Dr Dimitris I. Tsomokos

Psychology & Human Development, Institute of Education, University College London

---

## 1. Objective 

We derive the mathematical conditions under which a **reduced active-inference-style** agent undergoing Dirichlet parameter learning can exhibit a phase transition in policy selection — switching from a "simple" strategy (Phase A) to a "complex" strategy (Phase B) after a period of smooth, incremental learning.

The key result is that, after a mean-field reduction, the coupled dynamics of learning and policy selection produce a **self-consistency equation** formally identical to the Curie-Weiss / Ising mean-field equation. Near the primary symmetry-breaking point, the local normal form is a pitchfork in the symmetric case and a cusp-type unfolding under small asymmetry. The two control parameters map onto (i) accumulated evidence (developmental time) and (ii) preference or environmental asymmetry.

### Approximation status

This notebook mixes three levels of argument that should be kept distinct:

1. **Exact algebra** for the reduced one-step EFE difference under the specified two-state model.
2. **Mean-field / large-sample closure** when random Dirichlet updates are replaced by expected counts and a deterministic allocation variable ($\phi$).
3. **Local normal-form analysis** near the primary bifurcation, which yields pitchfork/cusp structure but does not by itself characterise the full global fixed-point landscape.

Accordingly, the derivation below should be read as a mathematically explicit reduction of a simplified model, not as an exact theorem about the full stochastic active-inference process or about generic `pymdp` agents.

---

## 2. High-Level Summary of the Derivation

1. **Model specification.** We define a minimal two-state, two-observation POMDP with two competing policies and Dirichlet learning on the observation likelihood (A matrix).

2. **Expected free energy and policy selection.** We compute the expected free energy (EFE) for each policy and express the policy posterior as a softmax over the EFE difference ΔG.

3. **Learning–policy feedback loop.** We show that the agent's policy choice determines which column of A is updated, creating a self-reinforcing feedback loop between learning and action.

4. **Mean-field reduction.** We reduce the coupled stochastic dynamics to a single order-parameter equation for φ (the fraction of experience allocated to each strategy).

5. **Self-consistency equation.** The equilibrium condition takes the form z = tanh(−γ ΔG(z)/2), which is the mean-field equation of statistical mechanics.

6. **Bifurcation analysis.** We compute the critical condition on the Jacobian and show that a pitchfork bifurcation (symmetric case) or cusp catastrophe (asymmetric case) occurs when a composite function of policy precision γ, true discriminability p, and developmental time N exceeds a critical threshold.

7. **Local mapping to the cusp catastrophe.** Near the primary bifurcation, we identify the splitting variable (γ × discriminability), the normal variable (preference asymmetry Δc), and the behavioural variable (strategy allocation z), giving the local cusp normal form.

8. **Numerical signatures.** Under suitable simulation protocols, the reduced model can generate catastrophe-style signatures such as bimodality, sudden jumps, path dependence, and amplified variability.

---

## 3. Model Specification

### 3.1 Generative model

We define the simplest POMDP that can exhibit strategy competition with learning.

| Component | Specification |
|---|---|
| Hidden states | s ∈ {s₁, s₂} |
| Observations | o ∈ {o₁, o₂} |
| Policies | π₁ (leads to s₁), π₂ (leads to s₂) |
| Preferences | C = [c₁, c₂], encoding prior log-preferences over observations |

The two policies represent two qualitatively different developmental strategies. For concreteness, one may think of s₁ as a "simple perceptual rule" state and s₂ as a "complex integrative rule" state (cf. Rule I vs Rule II on the balance scale task; Jansen & van der Maas, 2007).

### 3.2 Observation likelihood (A matrix)

The agent's learned observation model is a 2×2 matrix:

```math
\mathbf{A} = \begin{pmatrix} a & 1-b \\ 1-a & b \end{pmatrix}
```

where:

- a = P̂(o₁ | s₁) — learned probability of observation o₁ given hidden state s₁
- b = P̂(o₂ | s₂) — learned probability of observation o₂ given hidden state s₂

Both a and b start near 1/2 (maximum uncertainty) and, through Dirichlet learning, approach their true values a\*, b\* ∈ (1/2, 1).

### 3.3 Dirichlet parameterisation

Each column of A is parameterised by Dirichlet concentration parameters. For column j (state sⱼ), the concentrations are (αⱼ, βⱼ) with:

```math
a = \frac{\alpha_1}{\alpha_1 + \beta_1}, \quad b = \frac{\beta_2}{\alpha_2 + \beta_2}
```

We define **n₁** and **n₂** as the number of observations (data points) accumulated from states s₁ and s₂ respectively, **excluding** the prior. The total Dirichlet concentration for column 1 is then α₀ + n₁, and the expected point estimates are:

```math
a = \frac{\alpha_0/2 + n_1 p}{\alpha_0 + n_1} = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{n_1}{\alpha_0 + n_1}
```

where p = a\* is the true discriminability and α₀ is the prior concentration sum (strength of the prior). Analogously for b with n₂. We also define N = n₁ + n₂ as the total number of observations (excluding the prior).

### 3.4 Transition model (B matrix)

For analytical tractability, we assume each policy deterministically selects its target state:

```math
B(\pi_1) = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}, \quad B(\pi_2) = \begin{pmatrix} 0 & 0 \\ 1 & 1 \end{pmatrix}
```

That is, π₁ transitions the agent to s₁ regardless of the current state, and π₂ transitions to s₂. This captures the idea that each policy commits the agent to operating within a particular cognitive strategy.

### 3.5 Prior preferences (C vector)

The C vector encodes log-preferences over observations:

```math
\mathbf{C} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix}
```

We parameterise this as c₁ = c̄ + Δc/2, c₂ = c̄ − Δc/2, where Δc = c₁ − c₂ captures the asymmetry of preferences and c̄ is an irrelevant constant (it cancels in the EFE difference).

---

## 4. Expected Free Energy and Policy Selection

### 4.1 EFE for each policy

Under the deterministic transition model, policy πₖ places the agent in state sₖ with certainty. In this reduced one-step setting, the policy score decomposes into **negative expected pragmatic value** and **ambiguity**:

```math
G(\pi_k) = \underbrace{-\sum_i A_{ik} \, C_i}_{\text{negative expected pragmatic value}} + \underbrace{H(\mathbf{A}_{:,k})}_{\text{ambiguity (entropy of likelihood column)}}
```

*Note on terminology.* In strict FEP notation (Parr et al., 2022), "risk" refers to the KL divergence D_KL[q(o|π) ‖ P(o)]. Our first term is instead the negative expected utility under the likelihood column. The two coincide only in this deterministic one-step reduction, so we keep the more precise label "negative expected pragmatic value."

where H(·) denotes Shannon entropy:

```math
H(\mathbf{A}_{:,1}) = -a \ln a - (1-a) \ln(1-a) \equiv \mathcal{H}(a)
```

```math
H(\mathbf{A}_{:,2}) = -(1-b) \ln(1-b) - b \ln b \equiv \mathcal{H}(b)
```

and 𝓗(·) is the binary entropy function.

### 4.2 EFE difference

The quantity that determines policy selection is:

```math
\Delta G = G(\pi_1) - G(\pi_2)
```

**Risk difference:**

```math
\Delta G_{\text{risk}} = -[a \, c_1 + (1-a) \, c_2] + [(1-b) \, c_1 + b \, c_2]
```

Expanding:

```math
\Delta G_{\text{risk}} = (1 - a - b)(c_1 - c_2) = (1 - a - b) \, \Delta c
```

Since a, b > 1/2, we have (1 − a − b) < 0. If Δc > 0 (preference for o₁), then ΔG_risk < 0, favouring π₁ — because π₁ leads to s₁, which produces the preferred observation o₁ more reliably.

**Ambiguity difference:**

```math
\Delta G_{\text{amb}} = \mathcal{H}(a) - \mathcal{H}(b)
```

If a and b are equally well-learned (a = b), this term vanishes.

**Full EFE difference:**

```math
\boxed{\Delta G = (1 - a - b) \, \Delta c + \mathcal{H}(a) - \mathcal{H}(b)}
```

### 4.3 Policy posterior

The posterior probability of policy π₁ is:

```math
P(\pi_1) = \sigma(-\gamma \, \Delta G) = \frac{1}{1 + \exp(\gamma \, \Delta G)}
```

where γ > 0 is the policy precision (inverse temperature). In pymdp, the default is γ = 16.

---

## 5. The Learning–Policy Feedback Loop

### 5.1 Policy-dependent learning

This is the critical ingredient for the phase transition. In the reduced model, the agent's policy choice determines which state it visits, and therefore which column of A gets updated:

- If the agent selects π₁ → visits s₁ → observes a sample from A\*_{:,1} → updates Dirichlet parameters (α₁, β₁) → n₁ increases by 1, n₂ unchanged.
- If the agent selects π₂ → visits s₂ → observes a sample from A\*_{:,2} → updates Dirichlet parameters (α₂, β₂) → n₂ increases by 1, n₁ unchanged.

This creates **positive feedback**: an agent that uses strategy π₁ becomes better at discriminating observations from s₁ (a sharpens toward a\*), which reduces the ambiguity of π₁, making π₁ more attractive, leading to even more experience with s₁.

### 5.2 Mean-field learning dynamics

Let N = n₁ + n₂ denote total accumulated evidence (a proxy for developmental time), and let φ = n₁/N denote the fraction of experience allocated to strategy π₁.

Replacing the random update process by its expectation gives the mean-field dynamics:

```math
\frac{dN}{dt} = 1 \quad \text{(evidence accumulates at unit rate)}
```

```math
\frac{d\phi}{dt} = \frac{1}{N}\left[\sigma(-\gamma \, \Delta G) - \phi\right]
```

The second equation states that φ is pulled toward the current policy probability P(π₁) = σ(−γ ΔG), with a rate that slows as N grows. This is an approximation to the underlying stochastic recursion, not an exact identity.

### 5.3 Dependence of a, b on φ and N

From Section 3.3, the learned discriminabilities are:

```math
a(\phi, N) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{\phi N}{\alpha_0 + \phi N}
```

```math
b(\phi, N) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1-\phi) N}{\alpha_0 + (1-\phi) N}
```

where p is the true discriminability (assumed symmetric: a\* = b\* = p) and α₀ is the prior concentration.

These are saturating functions: they approach p as evidence accumulates and remain near 1/2 when evidence is scarce.

---

## 6. Self-Consistency Equation and Bifurcation Analysis

### 6.1 The order parameter

We define the **order parameter**:

```math
z = 2\phi - 1 \in [-1, 1]
```

This measures the degree of asymmetry in the agent's experience: z = 0 means equal experience with both strategies; z → ±1 means exclusive commitment to one strategy.

In terms of z:

```math
\phi = \frac{1+z}{2}, \quad n_1 = \frac{(1+z)N}{2}, \quad n_2 = \frac{(1-z)N}{2}
```

### 6.2 Equilibrium condition

At equilibrium (dφ/dt = 0), the self-consistency condition is:

```math
\phi^* = \sigma(-\gamma \, \Delta G(\phi^*, N))
```

In terms of z:

```math
\frac{1+z}{2} = \sigma\!\left(-\gamma \, \Delta G(z, N, \Delta c)\right)
```

Using the identity σ(x) = (1 + tanh(x/2))/2, this becomes:

```math
\boxed{z = \tanh\!\left(-\frac{\gamma}{2} \, \Delta G(z, N, \Delta c)\right)}
```

This is **identical in form to the mean-field equation of the Ising model** in statistical mechanics, where z is the magnetisation, γ/2 plays the role of inverse temperature × coupling strength, and ΔG plays the role of the effective field.

### 6.3 Symmetric case (Δc = 0): primary pitchfork bifurcation

When preferences are symmetric (Δc = 0), the EFE difference reduces to:

```math
\Delta G\big|_{\Delta c = 0} = \mathcal{H}(a) - \mathcal{H}(b)
```

At z = 0 (equal experience), a = b, so ΔG = 0 and z = 0 is always a fixed point.

To determine stability, we linearise. Define:

```math
\mathcal{F}(z) = \tanh\!\left(-\frac{\gamma}{2} \, \Delta G(z)\right) - z
```

The fixed point z = 0 loses stability when:

```math
\left.\frac{\partial \mathcal{F}}{\partial z}\right|_{z=0} = 0 \quad \Longleftrightarrow \quad \frac{\gamma}{2} \left|\frac{\partial \Delta G}{\partial z}\right|_{z=0} = 1
```

We now compute ∂ΔG/∂z at z = 0.

### 6.4 Computing the critical derivative

**Step 1: Express a and b in terms of z.**

Define τ = N/(2α₀) (rescaled developmental time). Then:

```math
a(z) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1+z)\tau}{1 + (1+z)\tau}
```

```math
b(z) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1-z)\tau}{1 + (1-z)\tau}
```

At z = 0:

```math
\bar{a} = a(0) = b(0) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{\tau}{1+\tau}
```

**Step 2: Compute ∂a/∂z and ∂b/∂z at z = 0.**

Differentiating:

```math
\left.\frac{\partial a}{\partial z}\right|_{z=0} = \left(p - \frac{1}{2}\right) \frac{\tau}{(1+\tau)^2} \equiv \lambda
```

```math
\left.\frac{\partial b}{\partial z}\right|_{z=0} = -\lambda
```

(The negative sign arises because b depends on (1−z).)

**Step 3: Differentiate ΔG.**

For the symmetric case (Δc = 0):

```math
\frac{\partial \Delta G}{\partial z} = \mathcal{H}'(a) \frac{\partial a}{\partial z} - \mathcal{H}'(b) \frac{\partial b}{\partial z}
```

where 𝓗'(x) = ln((1−x)/x) is the derivative of binary entropy.

At z = 0, a = b = ā, so:

```math
\left.\frac{\partial \Delta G}{\partial z}\right|_{z=0} = \mathcal{H}'(\bar{a}) \cdot (\lambda - (-\lambda)) = 2\lambda \, \mathcal{H}'(\bar{a})
```

Since ā > 1/2, we have 𝓗'(ā) = ln((1−ā)/ā) < 0, so the derivative is negative.

**Step 4: The bifurcation condition.**

Substituting into the critical condition:

```math
\boxed{\frac{\gamma}{2} \cdot 2\lambda \cdot \left|\ln\frac{1-\bar{a}}{\bar{a}}\right| = 1}
```

That is:

```math
\gamma \cdot \left(p - \frac{1}{2}\right) \cdot \frac{\tau}{(1+\tau)^2} \cdot \left|\ln\frac{1-\bar{a}}{\bar{a}}\right| = 1
```

where $\bar{a} = \frac{1}{2} + (p - \frac{1}{2})\frac{\tau}{1+\tau}$ and τ = N/(2α₀).

We define the **coupling function**:

```math
\mathcal{G}(\tau; p) = \left(p - \frac{1}{2}\right) \cdot \frac{\tau}{(1+\tau)^2} \cdot \left|\ln\frac{1-\bar{a}(\tau)}{\bar{a}(\tau)}\right|
```

The bifurcation occurs at the critical developmental time τ\* where:

```math
\gamma \cdot \mathcal{G}(\tau^*; p) = 1
```

### 6.5 Properties of the coupling function 𝓖(τ; p)

The coupling function has the following key properties for the **primary bifurcation window**:

1. **𝓖(0; p) = 0.** At τ = 0 (no evidence), ā = 1/2, and |ln 1| = 0. No bifurcation is possible at birth.

2. **𝓖(τ; p) → 0 as τ → ∞.** The factor τ/(1+τ)² → 0 as τ → ∞ (diminishing learning rate), even though |ln((1−ā)/ā)| → |ln((1−p)/p)| > 0. The system eventually settles.

3. **𝓖 has a unique interior maximum at some τ_max.** Since 𝓖 starts at 0, rises, and returns to 0, it attains a maximum 𝓖_max = 𝓖(τ_max; p) at an intermediate developmental time.

4. **Primary bifurcation requires γ > γ_c = 1/𝓖_max.** If the policy precision γ exceeds this critical value, the self-consistency equation first develops three solutions (bistability) in a range of τ around τ_max.

The critical precision γ_c depends only on the true discriminability p (since 𝓖 is a function of τ and p alone, with α₀ absorbed into the rescaling τ = N/(2α₀)). What α₀ affects is the **critical number of observations** N* = 2α₀τ* at which the primary bifurcation window is reached. In the full nonlinear fixed-point equation, additional re-entrant structure can appear at larger τ; the Ising-style argument here concerns the first loss of stability at the origin.

### 6.6 Numerical illustration

For p = 0.85 (moderately discriminable), α₀ = 2 (weak prior):

| τ = N/(2α₀) | ā | λ | \|ln((1−ā)/ā)\| | 𝓖(τ) |
|---|---|---|---|---|
| 0.5 | 0.617 | 0.078 | 0.477 | 0.037 |
| 1.0 | 0.675 | 0.088 | 0.731 | 0.064 |
| 2.0 | 0.733 | 0.078 | 1.011 | 0.079 |
| 3.0 | 0.763 | 0.066 | 1.171 | 0.077 |
| 5.0 | 0.792 | 0.049 | 1.337 | 0.065 |
| 10.0 | 0.818 | 0.029 | 1.503 | 0.043 |

The maximum 𝓖_max ≈ 0.079 occurs near τ ≈ 2 (i.e., N = 2α₀τ = 2 × 2 × 2 = 8 observations beyond the prior).

The critical precision: γ_c ≈ 1/0.079 ≈ 12.7.

Since pymdp uses γ = 16 by default, **the bifurcation condition is satisfied** for this parameter regime. The phase transition will occur.

---

## 7. Local Cusp-Type Unfolding

### 7.1 Breaking symmetry: the normal variable

When Δc ≠ 0 (asymmetric preferences), the fixed point at z = 0 is perturbed. The self-consistency equation becomes:

```math
z = \tanh\!\left(-\frac{\gamma}{2}\left[(1 - a(z) - b(z)) \, \Delta c + \mathcal{H}(a(z)) - \mathcal{H}(b(z))\right]\right)
```

Near z = 0, the EFE difference has a linear term in Δc:

```math
\Delta G(0) = (1 - 2\bar{a}) \, \Delta c
```

This acts as an external field in the mean-field analogy, breaking the z → −z symmetry.

### 7.2 Mapping to the cusp canonical form

The cusp catastrophe has the canonical potential:

```math
V(z) = \frac{z^4}{4} + \frac{x \, z^2}{2} + y \, z
```

with equilibria satisfying z³ + xz + y = 0. We now show that the **local normal form** of the reduced system takes this form near the primary bifurcation.

**Normal-form expansion.** Write the self-consistency equation as z = tanh(f(z)), where f(z) = −(γ/2)ΔG(z, Δc). Expand f in a Taylor series around z = 0:

f(z) = f₀ + f₁z + f₂z²/2 + f₃z³/6 + ...

The coefficients are:

- f₀ = −(γ/2)ΔG(0) = −(γ/2)(1 − 2ā)Δc. This is zero when Δc = 0 and proportional to Δc otherwise.
- f₁ = −(γ/2)(∂ΔG/∂z)|₀ = γ𝓖(τ; p). This is the quantity whose crossing of 1 triggers the bifurcation.
- f₂ = −(γ/2)(∂²ΔG/∂z²)|₀. At Δc = 0, this is exactly zero because ΔG is an odd function of z when preferences are symmetric (swapping z → −z swaps a and b, negating 𝓗(a)−𝓗(b)). At Δc ≠ 0, f₂ is proportional to Δc.

Expanding tanh(x) ≈ x − x³/3 and substituting:

z ≈ f₀ + f₁z + (f₃/6)z³ − (f₁z)³/3 + ...

(The f₂z² term, when present, contributes at higher order and can be absorbed by a coordinate shift; see below.)

Rearranging:

0 = f₀ + (f₁ − 1)z + (f₃/6 − f₁³/3)z³

**Symmetric case (Δc = 0).** Here f₀ = 0, and we have:

0 = (f₁ − 1)z + bz³

where b = f₃/6 − f₁³/3 < 0 (verified numerically). This is the standard **pitchfork** normal form. It has solutions z = 0 (always) and z² = −(f₁ − 1)/b (when f₁ > 1, i.e., γ𝓖 > 1).

**Asymmetric case (Δc ≠ 0, small).** Now f₀ ≠ 0 and f₂ ≠ 0 (both proportional to Δc). The equation becomes:

0 = f₀ + (f₁ − 1)z + (f₂/2)z² + bz³

The quadratic term is removed by the substitution z̃ = z + f₂/(6b), yielding:

**0 = ỹ + x̃ z̃ + z̃³**

where x̃ = (f₁ − 1)/b − f₂²/(12b²) and ỹ = f₀/b + corrections. This is the **canonical cusp equation**. The bifurcation set is 4x̃³ + 27ỹ² = 0.

In our system:

| Cusp variable | Active inference quantity | Interpretation |
|---|---|---|
| z̃ (behaviour) | ≈ 2φ − 1 (strategy allocation) | Degree of commitment to π₁ vs π₂ |
| x̃ (splitting) | Dominated by (1 − γ𝓖(τ; p))/b | Exceeds 0 → unique solution; below 0 → bistability |
| ỹ (normal) | Dominated by f₀/b ∝ Δc | Preference asymmetry breaking the symmetry |

The bifurcation set in the (x̃, ỹ) control plane — the region where local multistability and sudden jumps can occur — is delimited by the standard cusp curve 4x̃³ + 27ỹ² = 0.

### 7.3 Catastrophe-style signatures

The reduced cusp-style picture motivates five empirically testable signatures. In our developmental context:

**1. Bimodality.** In a population of agents (children) at a given developmental time N near the critical point, the distribution of z (strategy allocation) is bimodal: some children are in Phase A (z < 0, using the simple strategy), others in Phase B (z > 0, using the complex strategy), with few in between.

**2. Inaccessible region.** Intermediate strategy mixtures (z ≈ 0) are unstable and therefore rarely observed — the agent is either committed to one strategy or the other.

**3. Sudden jump.** As developmental time N increases (τ crosses through the bifurcation set), the agent's strategy switches abruptly from one attractor to the other. This is the stage transition.

**4. Hysteresis.** If the control parameter (e.g., environmental bias Δc) is swept forward and then backward, the transition from Phase A → B occurs at a different Δc value than the reverse transition B → A. The agent "resists" switching back because it has accumulated more evidence for its current strategy.

**5. Divergence.** Two agents with slightly different initial conditions or preference parameters can end up in qualitatively different phases — one in A, the other in B — if their trajectories straddle the bifurcation set.

---

## 8. Extension: A Heuristic Novelty / Epistemic Term

### 8.1 The exploration problem

The derivation above assumes no epistemic (information-seeking) component. In full active inference, the EFE includes a **parameter information gain** term that drives the agent to visit states where it can learn more about A. Parameter information gain *reduces* G(πₖ) (makes the policy more attractive) by an amount proportional to 1/nₖ, where nₖ is the evidence count for the state visited by πₖ. This adds a term to ΔG:

```math
\Delta G_{\text{info}} = G_{\text{info}}(\pi_1) - G_{\text{info}}(\pi_2) = \left(-\frac{w}{n_1}\right) - \left(-\frac{w}{n_2}\right) = w\left(\frac{1}{n_2} - \frac{1}{n_1}\right)
```

where w > 0 is the epistemic weight. (The sign follows because information gain *lowers* EFE, entering G as a negative term.)

When the agent has committed to π₁ (n₁ ≫ n₂), this term becomes:

```math
\Delta G_{\text{info}} \approx \frac{w}{n_2} > 0
```

which makes ΔG more positive — and since P(π₁) = σ(−γΔG), a more positive ΔG *lowers* P(π₁), thereby favouring π₂ (the under-explored strategy). This is the correct epistemic drive: curiosity about the strategy with less evidence.

### 8.2 Role in the transition

In the reduced simulations we implement this term heuristically as a novelty bonus for under-sampled mappings. It is useful computationally, but it should not be identified with the exact `pymdp` parameter-information-gain expression without a separate derivation.

This creates a characteristic temporal signature: long periods of stable Phase A behaviour, punctuated by increasing variability (more frequent exploratory excursions to Phase B), culminating in a sudden switch. This matches the empirical pattern described by van der Maas and Molenaar (1992) and Siegler (1996) as increased response variability preceding stage transitions.

---

## 9. Summary of Key Results

### 9.1 The core bifurcation condition

A phase transition in policy selection occurs when the policy precision γ exceeds the critical value:

```math
\boxed{\gamma_c = \frac{1}{\max_\tau \, \mathcal{G}(\tau; p)}}
```

where:

```math
\mathcal{G}(\tau; p) = \left(p - \frac{1}{2}\right) \frac{\tau}{(1+\tau)^2} \left|\ln \frac{1 - \bar{a}(\tau)}{\bar{a}(\tau)}\right|, \quad \bar{a}(\tau) = \frac{1}{2} + \left(p - \frac{1}{2}\right)\frac{\tau}{1+\tau}
```

### 9.2 Conditions favouring bifurcation (i.e., stage-like development)

The bifurcation is more likely (lower γ_c) when:

- **True discriminability p is high.** When the environment offers clearly distinguishable observations under each strategy, the positive feedback loop is stronger.
- **Policy precision γ is high.** More "decisive" agents (those who strongly commit to the best policy) are more prone to sudden switches.

The transition occurs **earlier in developmental time** (lower N*) when:

- **Prior α₀ is small.** Weaker priors mean that fewer observations are needed to reach the bifurcation window (N* = 2α₀τ*). Note that α₀ does not affect γ_c itself (which depends only on p), but it determines the absolute number of observations required.

### 9.3 Conditions favouring smooth (continuous) development

Conversely, the transition is smooth when:

- p is close to 1/2 (low discriminability; the two strategies look similar)
- γ is small (the agent hedges between strategies rather than committing)

And the transition is **delayed** (occurs after more observations) when:

- α₀ is large (strong prior beliefs mean more data are needed to shift the agent into the bifurcation window)

This provides a principled explanation for the coexistence of stage-like and continuous developmental trajectories across different cognitive domains and individual children — a long-standing puzzle in developmental psychology (Siegler, 1996; van der Maas & Molenaar, 1992).

---

## 10. Mapping to Developmental Phenomena

| Mathematical quantity | Developmental interpretation |
|---|---|
| N (total evidence) | Developmental time / cumulative experience |
| τ = N/(2α₀) | Rescaled developmental time (relative to prior strength) |
| φ = n₁/N | Proportion of experience with current strategy |
| z = 2φ − 1 | Degree of strategic commitment |
| γ | Cognitive decisiveness / policy precision |
| p | Environmental discriminability |
| Δc | Asymmetry in task demands or motivational bias |
| γ_c | Critical threshold for stage-like transition |
| τ\* | Age at transition |

---

## Appendix A: Detailed Computation of the Coupling Function

### A.1 Definitions

Let τ = N/(2α₀), p ∈ (1/2, 1), and:

```math
\bar{a}(\tau) = \frac{1}{2} + \left(p - \frac{1}{2}\right)\frac{\tau}{1+\tau}
```

```math
\lambda(\tau) = \left(p - \frac{1}{2}\right)\frac{\tau}{(1+\tau)^2}
```

Note that λ is the derivative ∂a/∂z at z = 0 and captures the sensitivity of learned discriminability to experience allocation.

### A.2 The coupling function

```math
\mathcal{G}(\tau; p) = \lambda(\tau) \cdot \left|\ln\frac{1 - \bar{a}(\tau)}{\bar{a}(\tau)}\right|
```

### A.3 Finding the maximum

Setting d𝓖/dτ = 0 and solving numerically for given p:

For p = 0.80: τ_max ≈ 2.13, 𝓖_max ≈ 0.057, γ_c ≈ 17.7
For p = 0.85: τ_max ≈ 2.19, 𝓖_max ≈ 0.079, γ_c ≈ 12.7
For p = 0.90: τ_max ≈ 2.28, 𝓖_max ≈ 0.106, γ_c ≈ 9.4
For p = 0.95: τ_max ≈ 2.42, 𝓖_max ≈ 0.140, γ_c ≈ 7.1

Note that τ_max *increases* with p (the peak shifts to later rescaled time for more discriminable environments).

The pymdp default γ = 16 therefore produces bifurcations for p ≳ 0.81.

---

## Appendix B: Connection to the Ising Model

The self-consistency equation z = tanh(−γΔG(z)/2) is formally identical to the Curie–Weiss mean-field equation for the Ising model:

```math
m = \tanh(\beta J m + \beta h)
```

where:

| Ising model | Active inference |
|---|---|
| m (magnetisation) | z (strategy allocation) |
| β (inverse temperature) | γ (policy precision) |
| J (coupling constant) | 𝒢(τ;p) = (1/2) partial deriv dΔG/dz at z = 0 (1/2 feedback strength) |
| h (external field) | −ΔG(0)/2 = −(1−2ā)Δc/2 (preference asymmetry) |

The Ising model undergoes a ferromagnetic phase transition at βJ = 1. In our system, the corresponding condition is γ𝓖(τ) = 1.

This mapping is not only formal, but actually reflects a deeper, **structural** analogy. In both systems, a positive feedback loop (spin alignment, i.e. _strategy reinforcement_) competes with disorder (thermal fluctuations, i.e. _ambiguity_), and a phase transition occurs when the feedback exceeds a critical strength.

---

## Appendix C: Relation to van der Maas & Molenaar's Catastrophe Model

Van der Maas and Molenaar (1992) proposed the cusp catastrophe as a model for Piagetian stage transitions, with:

- The "normal" control variable identified with cognitive level (or accumulated knowledge)
- The "splitting" control variable identified with perceptual salience (or task difficulty)
- The behavioural variable identified with conservation performance

Our model recovers this structure from first principles:

- The normal variable Δc corresponds to environmental bias (which regime is currently more rewarding), analogous to their "cognitive level"
- The splitting variable γ𝓖(τ) corresponds to the strength of the learning–action feedback loop, which grows with experience — analogous to their "perceptual salience" in the sense that increased experience makes the distinction between strategies more salient
- The behavioural variable z is the strategy allocation, directly analogous to conservation performance

The key advance is that our model derives the cusp structure from the mechanics of active inference rather than postulating it. The five catastrophe flags are not assumed but emerge as consequences of free energy minimisation with Dirichlet learning.

---

## Key References

(from manuscript, v. mid-March)

Bowers, J. S., & Davis, C. J. (2012). Bayesian just-so stories in psychology and neuroscience. Psychological Bulletin, 138(3), 389. 

Case, R. (2013). Neo-Piagetian theories of intellectual development. In Piaget's theory (pp. 61-104). Psychology Press. 

Courage, M. L., & Howe, M. L. (2002). From infant to child: The dynamics of cognitive change in the second year of life. Psychological Bulletin, 128(2), 250. 

Fischer, K. W., & Bidell, T. R. (2006). Dynamic development of action, thought, and emotion. In W. Damon & R. M. Lerner (Eds.), Theoretical models of human development. Handbook of child psychology (Vol. 1, pp. 313-399). Wiley. 

Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138. https://doi.org/10.1038/nrn2787 

Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. Neural computation, 29(1), 1-49.

Hayes, A. M., Laurenceau, J.-P., Feldman, G., Strauss, J. L., & Cardaciotto, L. (2007). Change is not always linear: The study of nonlinear and discontinuous patterns of change in psychotherapy. Clinical Psychology Review, 27(6), 715-723. 

Heins, C., Klein, B., Demekas, D., Aguilera, M., & Buckley, C. L. (2023). Spin Glass Systems as Collective Active Inference. In C. L. Buckley, D. Cialfi, P. Lanillos, M. Ramstead, N. Sajid, H. Shimazaki, & T. Verbelen, Active Inference Cham.

Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I., & Tschantz, A. (2022). pymdp: A Python library for active inference in discrete state spaces. Journal of Open Source Software, 7(73). https://doi.org/10.21105/joss.04098 

Hengen, K. B., & Shew, W. L. (2025). Is criticality a unified setpoint of brain function? Neuron, 113(16), 2582-2598. e2582. 

Jansen, B. R., & Van der Maas, H. L. (2001). Evidence for the phase transition from Rule I to Rule II on the balance scale task. Developmental Review, 21(4), 450-494. 

Jansen, B. R., & Van der Maas, H. L. (2002). The development of children's rule use on the balance scale task. Journal of Experimental Child Psychology, 81(4), 383-416. 

Kim, S., & Carlson, S. M. (2024). Understanding explore-exploit dynamics in child development: current insights and future directions [Mini Review]. Frontiers in Developmental Psychology, 2, 1467880. https://doi.org/10.3389/fdpys.2024.1467880 

Morra, S., Gobbo, C., Marini, Z., & Sheese, R. (2012). Cognitive development: neo-Piagetian perspectives. Psychology Press. 

Nishimori, H. (2001). Statistical physics of spin glasses and information processing: an introduction. Clarendon Press. 

Nussenbaum, K., & Hartley, C. A. (2019). Reinforcement learning across development: What insights can we draw from a decade of research? Developmental Cognitive Neuroscience, 40, 100733. 

O’Byrne, J., & Jerbi, K. (2022). How critical is brain criticality? Trends in Neurosciences, 45(11), 820-837. 

Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active inference: the free energy principle in mind, brain, and behavior. MIT Press. 

Pezzulo, G., Rigoli, F., & Friston, K. J. (2018). Hierarchical active inference: A theory of motivated control. Trends in Cognitive Sciences, 22(4), 294-306. https://doi.org/https://doi.org/10.1016/j.tics.2018.01.009 

Piaget, J. (1962). The stages of the intellectual development of the child. Bulletin of the Menninger Clinic, 26(3), 120. 

Schiepek, G., Eckert, H., Aas, B., Wallot, S., & Wallot, A. (2016). Integrative psychotherapy: A feedback-driven dynamic systems approach. Hogrefe Publishing GmbH. 

Schwartenbeck, P., FitzGerald, T., Dolan, R. J., & Friston, K. (2013). Exploration, novelty, surprise, and free energy minimization. Frontiers in Psychology, 4, 710. 

Seizer, L., Kratzer, L., Löchner, J., Schöller, H., Aichhorn, W., & Schiepek, G. (2026). Psychotherapy Process Dynamics and Their Relation to Treatment Success Do Not Differ Across Diagnoses. Clinical Psychology & Psychotherapy, 33(1), e70222. 

Seth, A. K. (2014). The cybernetic Bayesian brain. In Open mind. Open MIND. Frankfurt am Main: MIND Group. 

Shapiro, Y., & Scott, J. R. (2018). Dynamical systems therapy (DST): Complex adaptive systems in psychiatry and psychotherapy. In Handbook of Research Methods in Complexity Science (pp. 567-590). Edward Elgar Publishing. 

Siegler, R. S. (1976). Three aspects of cognitive development. Cognitive Psychology, 8(4), 481-520. 

Siegler, R. S. (1995). How does change occur: A microgenetic study of number conservation. Cognitive Psychology, 28(3), 225-273. 

Siegler, R. S. (1996). Emerging minds: The process of change in children's thinking. Oxford University Press. 

Tenenbaum, J. B., Kemp, C., Griffiths, T. L., & Goodman, N. D. (2011). How to grow a mind: Statistics, structure, and abstraction. Science, 331(6022), 1279-1285. 

Thelen, E., & Smith, L. B. (1994). A dynamic systems approach to the development of cognition and action. MIT Press. 

Van der Maas, H. L. (2024). Complex-Systems Research in Psychology. SFI Press. **ONLINE VERSION:** https://santafeinstitute.github.io/ComplexPsych

Van der Maas, H. L., & Molenaar, P. C. (1992). Stagewise cognitive development: an application of catastrophe theory. Psychological review, 99(3), 395-417. 

Van Geert, P. (1991). A dynamic systems model of cognitive and language growth. Psychological review, 98(1), 3. 

Van Geert, P. (1998). A dynamic systems model of basic developmental mechanisms: Piaget, Vygotsky, and beyond. Psychological review, 105(4), 634-677. 

Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning, 1(1-2), 1-305. 

