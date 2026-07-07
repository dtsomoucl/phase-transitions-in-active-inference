# Notebook 05 — Parameter Novelty and the Exploration Phase (the Novelty-Corrected Bifurcation Condition)

**Series:** *Phase Transitions in Early Learning — An Active Inference Approach*

**Date:** 07 July 2026

**Status:** Extension of the core formalism (Notebook 01, and the step-by-step derivation in Notebook 04) to include the parameter information-gain (novelty) term of expected free energy.

**Author:** Dr Dimitris I. Tsomokos

Psychology & Human Development, Institute of Education, University College London

---

## 1. Objective

Notebooks 01 and 04 derive the bifurcation condition γ𝒢(τ; p) = 1 for a reduced active-inference agent whose policy score has the **negative expected pragmatic value** and **ambiguity** terms only. That analysis isolated the *consolidating* arm of the learning–action coupling and deliberately omitted the epistemic, _information-seeking component_ of expected free energy (EFE). Notebook 01 (§8) sketched this component heuristically, as a novelty bonus proportional to w/nₖ (and therefore flagged that it "should not be identified with the exact _pymdp_ parameter-information-gain expression without a separate derivation.").

This notebook extends the primary work by supplying that derivation. Here, therefore, we do the following:

1. Derive the **exact closed form** of the Dirichlet parameter information-gain (_novelty_) term for the two-outcome model: W(πₖ) = 1/(2(α₀ + nₖ)). This replaces, and regularises, the heuristic w/nₖ of Notebook 01 §8 (which diverges at nₖ = 0; the exact form does not, because the prior counts are included).
2. Restore this term in the EFE difference ΔG and repeat, step by step, the linear stability analysis of Notebooks 01 (§6) and 04 (Steps 5–6).
3. Obtain the **revised bifurcation condition** (i.e., corrected by the "curly" N term):

   γ · [𝒢(τ; p) − 𝒩(τ; α₀)] = 1,  with 𝒩(τ; α₀) = τ / (2α₀(1 + τ)²),

   where 𝒩 ("curly" N) is a **novelty coupling** that opposes the consolidation coupling 𝒢.
4. Derive three consequences: an obligatory exploration-first phase; a necessary condition on (p, α₀) for any transition to be possible at all; and a corrected critical precision γ_c that now depends on the prior strength α₀ as well as the environmental discriminability p.
5. Verify the corrected condition numerically against the exact fixed points of the full self-consistency equation, and describe the standalone R simulation (`sim_novelty_correction.R`) that produces the associated figure.

**Why we need this extension.** It answers a reasonable objection that the original positive feedback loop (of Notebooks 01 and 04) predicts agents who "imprint" on the first predictable corner of the state space they find (i.e., can just stop there, namely, an inherently "anti-curiosity" reading of the original equation). With the novelty term included, exploration-first behaviour is a *derived* property (no need to assume it): early in learning the novelty coupling dominates and the strategy-mixing state is strongly stable; commitment becomes possible only once the curiosity bonus attached to under-sampled strategies has decayed. It also completes the specification of the reduced policy score, since the parameter information-gain term survives under every decomposition of the EFE.

### Approximation status

The conventions of Notebook 01 §1 apply. In addition:

- **Exact algebra:** the closed form W(πₖ) = 1/(2(α₀ + nₖ)) is exact for a two-outcome Dirichlet column under the standard matrix expression for parameter information gain — it holds for *any* composition of the counts within the column, not only in expectation.
- **Mean-field closure:** as before, the counts nₖ are replaced by their allocation-variable expressions nₖ = (1 ± z)N/2 when passing to the description based on the order parameter.
- **Local normal-form analysis:** the pitchfork/cusp analysis is local to the primary bifurcation; whereas global fixed-point structure is checked numerically (see Section 8).

---

## 2. Where We Start From (Recap of the Consolidation-Only Result)

From Notebook 01 (boxed results of §4.2, §6.2, §6.4) and Notebook 04 (Eqs. 2, 8, 16–18):

- EFE difference (consolidation-only): ΔG = (1 − a − b)Δc + 𝓗(a) − 𝓗(b).
- Self-consistency equation: z = tanh(−γ ΔG(z)/2).
- Learned discriminabilities under the mean-field counts, with τ = N/(2α₀):

```math
a(z) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1+z)\tau}{1 + (1+z)\tau}, \qquad
b(z) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1-z)\tau}{1 + (1-z)\tau}
```

- Critical derivative at the symmetric point: (∂ΔG/∂z)|₀ = 2λ ln((1−ā)/ā), with λ = (p − ½)τ/(1+τ)² and ā = ½ + (p − ½)τ/(1+τ).
- Bifurcation condition: γ𝒢(τ; p) = 1, with the **consolidation coupling**

```math
\mathcal{G}(\tau; p) = \left(p - \frac{1}{2}\right) \frac{\tau}{(1+\tau)^2} \left|\ln\frac{1-\bar{a}(\tau)}{\bar{a}(\tau)}\right|
```

- Critical precision: γ_c = 1/max_τ 𝒢(τ; p), which depends **only on p** in the consolidation-only analysis.

Everything below modifies exactly one ingredient — the policy score — and traces the consequences through the same steps.

---

## 3. Step 1: The Exact Novelty Term for a Dirichlet Column

### 3.1 Definition

In full active inference, the EFE of a policy includes a **parameter information gain** ("novelty") term, which quantifies how much the agent expects to learn about the likelihood parameters by executing that policy. For Dirichlet-parameterised likelihoods, the standard matrix expression is (Friston et al., 2017; Schwartenbeck et al., 2019; Smith, Friston & Whyte, 2022, Eqs. 39–40):

```math
W(\pi_k) = \sum_i A_{ik} \cdot \frac{1}{2}\left(\frac{1}{\alpha_{ik}} - \frac{1}{\alpha_{\cdot k}}\right) \qquad \text{...(Eq.\;1)}
```

where, for the column k visited under policy πₖ:

- αᵢₖ is the Dirichlet concentration for outcome i in column k,
- α·ₖ = Σᵢ αᵢₖ is the total concentration of the column,
- Āᵢₖ = αᵢₖ/α·ₖ is the posterior-mean likelihood.

--> Note that we are simply writing this out a little differently here from the matrix notation used in Smith, Friston & Whyte, 2022 (Eqs. 39–40); the essential meaning of this term is that it "scores how much beliefs within the matrix are expected to change after receiving a new observation" (Smith, Friston & Whyte, 2022) and, due to this, "the agent will seek out state-observation pairings that will maximize the difference in concentration parameters between posterior and prior distributions over **A**. This difference quantifies the drive or epistemic affordance of finding out ‘what would happen if I do that?’." (Smith, Friston & Whyte, 2022, p. 34). Hence, for our purposes, we are perfectly entitled to consider this novelty-seeking term as indexing **curiosity**.

The term enters the policy score with a **negative** sign (information gain lowers the EFE, making the policy more attractive):

```math
G(\pi_k) = -\sum_i A_{ik}\, C_i + H(\bar{\mathbf{A}}_{:,k}) - W(\pi_k) \qquad \text{...(Eq.\;2)}
```

### 3.2 The closed form

For our **two-outcome** columns, Eq. 1 telescopes (many terms cancel out). Substitute Āᵢₖ = αᵢₖ/α·ₖ:

```math
W(\pi_k) = \frac{1}{2}\sum_{i=1}^{2} \frac{\alpha_{ik}}{\alpha_{\cdot k}}\left(\frac{1}{\alpha_{ik}} - \frac{1}{\alpha_{\cdot k}}\right)
= \frac{1}{2}\left[\sum_{i=1}^{2} \frac{1}{\alpha_{\cdot k}} \;-\; \sum_{i=1}^{2} \frac{\alpha_{ik}}{\alpha_{\cdot k}^{2}}\right]
```

The first sum has two identical terms, giving 2/α·ₖ. In the second sum, Σᵢ αᵢₖ = α·ₖ, so it equals 1/α·ₖ. Hence:

```math
W(\pi_k) = \frac{1}{2}\left[\frac{2}{\alpha_{\cdot k}} - \frac{1}{\alpha_{\cdot k}}\right] = \frac{1}{2\,\alpha_{\cdot k}}
```

With the notation of Notebooks 01/04, column k has total concentration α·ₖ = α₀ + nₖ (prior plus evidence). Therefore:

```math
\boxed{\;W(\pi_k) = \frac{1}{2(\alpha_0 + n_k)}\;} \qquad \text{...(Eq.\;3)}
```

### 3.3 Checks

1. **Hyperbolic decay.** The curiosity bonus attached to a strategy decays as 1/nₖ with the evidence accumulated under it — fast at first, with a long tail. ✓ (This is the qualitative behaviour anticipated heuristically in Notebook 01 §8.)
2. **No divergence at nₖ = 0.** W = 1/(2α₀) at the start of learning: the prior regularises the bonus. The heuristic w/nₖ of Notebook 01 §8.1 diverges here; the exact form supersedes it.
3. **Composition-independence.** Remarkably, W depends only on the *total* concentration α·ₖ, not on how the counts are split between outcomes within the column. The mean-field closure is therefore needed only for the counts nₖ themselves, not for W given the counts — the closed form is exact along every sample path.
4. **Sign of the drive.** Suppose the agent has committed to π₁ (n₁ ≫ n₂). Then W(π₁) ≈ 0 and W(π₂) ≈ 1/(2(α₀+n₂)) is comparatively large, so ΔW = W(π₁) − W(π₂) < 0 and the score difference ΔG (Eq. 4 below) acquires a **positive** contribution −ΔW > 0. Since P(π₁) = σ(−γΔG), this lowers P(π₁): the term favours the under-sampled strategy. ✓ This matches the sign analysis of Notebook 01 §8.1.

---

## 4. Step 2: The Extended EFE Difference

Subtracting the scores of Eq. 2 for the two policies, the pragmatic and ambiguity parts are unchanged from Notebook 04 §2.3, and the novelty parts contribute −ΔW:

```math
\boxed{\;\Delta G = (1 - a - b)\,\Delta c + \mathcal{H}(a) - \mathcal{H}(b) - \Delta W\;}, \qquad \Delta W = W(\pi_1) - W(\pi_2) \qquad \text{...(Eq.\;4)}
```

When the two strategies are equally preferred (Δc = 0), policy selection is driven by **two competing epistemic pressures**:

- the **ambiguity difference** 𝓗(a) − 𝓗(b), which favours the strategy the agent currently understands *better* (the consolidating pressure of Notebooks 01/04), and
- the **novelty difference** ΔW, which favours the strategy it understands *worse* (the exploratory pressure added/derived here).

The consolidation-only analysis of Notebooks 01/04 is recovered exactly in the limit ΔW → 0, which — as Eq. 3 makes precise — is the **large-α₀ limit** (i.e., under _massive_ prior learning, and hence consolidated experience, the novelty/curiosity term becomes negligible).

---

## 5. Step 3: The Novelty Difference in Order-Parameter Form

Substitute the mean-field counts n₁ = (1+z)N/2 and n₂ = (1−z)N/2 (Notebook 04, Eq. 6 and §4.1), together with τ = N/(2α₀), so that nₖ = (1 ± z)τα₀ and

```math
\alpha_0 + n_1 = \alpha_0\bigl[1 + (1+z)\tau\bigr], \qquad \alpha_0 + n_2 = \alpha_0\bigl[1 + (1-z)\tau\bigr]
```

Then from Eq. 3:

```math
\Delta W(z) = \frac{1}{2\alpha_0}\left[\frac{1}{1 + (1+z)\tau} - \frac{1}{1 + (1-z)\tau}\right] \qquad \text{...(Eq.\;5)}
```

### 5.1 Two structural properties

1. **ΔW(0) = 0.** At the symmetric point both strategies have been sampled equally, so the novelty bonuses cancel. The symmetric solution z = 0 therefore remains a fixed point of the self-consistency equation when Δc = 0, and the constant term of the normal-form expansion is unchanged: f₀ = −(γ/2)(1 − 2ā)Δc, exactly as in Notebook 01 §7.2.
2. **ΔW is odd in z.** Sending z → −z swaps the two bracketed terms in Eq. 5 and hence negates ΔW. Consequently (see Step 6 below) the even coefficients of the expansion — in particular f₂ — receive **no** novelty contribution at Δc = 0, and the pitchfork/cusp symmetry structure of Notebook 01 §7 and Notebook 04 Step 7 is preserved intact.

---

## 6. Step 4: The Corrected Critical Derivative

We repeat Notebook 04, Step 5, for the extended ΔG of Eq. 4. Steps A–D of Notebook 04 are unchanged; we need one new step.

**Step D′: Differentiate the novelty term −ΔW.**

Write Eq. 5 as ΔW(z) = [g(1+z) − g(1−z)]/(2α₀) with g(u) = 1/(1 + uτ). Then g′(u) = −τ/(1 + uτ)², and by the chain rule:

```math
\frac{d\,\Delta W}{dz} = \frac{1}{2\alpha_0}\Bigl[g'(1+z)\cdot(+1) - g'(1-z)\cdot(-1)\Bigr] = \frac{g'(1+z) + g'(1-z)}{2\alpha_0}
```

At z = 0 both arguments equal 1, so g′(1+z) = g′(1−z) = −τ/(1+τ)², and:

```math
\left.\frac{d\,\Delta W}{dz}\right|_{z=0} = \frac{1}{2\alpha_0}\cdot\left(-\frac{2\tau}{(1+\tau)^2}\right) = -\frac{\tau}{\alpha_0 (1+\tau)^2} \qquad \text{...(Eq.\;6)}
```

**Step E′: Combine with the consolidation part.**

Adding the −ΔW contribution to Notebook 04's Eq. 15:

```math
\left.\frac{\partial \Delta G}{\partial z}\right|_{z=0} = \underbrace{2\lambda \ln\!\frac{1-\bar{a}}{\bar{a}}}_{\text{consolidation}\;(<0)} \;+\; \underbrace{\frac{\tau}{\alpha_0(1+\tau)^2}}_{\text{novelty}\;(>0)} \qquad \text{...(Eq.\;7)}
```

The two contributions have **opposite signs**: the ambiguity feedback destabilises the symmetric state (it amplifies asymmetries in experience), whilst the novelty feedback restores it (it penalises them). This is the mathematical expression of consolidation versus curiosity.

**Step F′: The corrected slope of the self-consistency map.**

Exactly as in Notebook 04 §6.1, the linearised slope of z ↦ tanh(−γΔG(z)/2) at the origin is f₁ = −(γ/2)(∂ΔG/∂z)|₀. Substituting Eq. 7:

```math
f_1 = \frac{\gamma}{2}\left[2\lambda\left|\ln\frac{1-\bar{a}}{\bar{a}}\right| - \frac{\tau}{\alpha_0(1+\tau)^2}\right] = \gamma\left[\mathcal{G}(\tau; p) - \mathcal{N}(\tau; \alpha_0)\right] \qquad \text{...(Eq.\;8)}
```

where 𝒢 is the consolidation coupling of Section 2 and we define the **novelty coupling**:

```math
\boxed{\;\mathcal{N}(\tau; \alpha_0) = \frac{\tau}{2\alpha_0 (1+\tau)^2}\;} \qquad \text{...(Eq.\;9)}
```

The symmetric fixed point loses stability when f₁ crosses 1, giving the **curiosity-corrected bifurcation condition**:

```math
\boxed{\;\gamma \cdot \bigl[\mathcal{G}(\tau; p) - \mathcal{N}(\tau; \alpha_0)\bigr] = 1\;} \qquad \text{...(Eq.\;10)}
```

In the language of the Ising mapping (Notebook 01, Appendix B), 𝒩 is an **anti-consolidation term**: it subtracts from the effective ferromagnetic coupling J and stabilises the mixed-strategy (paramagnetic) state. The consolidation-only condition γ𝒢 = 1 is fully recovered as α₀ → ∞, where 𝒩 → 0.

---

## 7. Step 5: Consequences of the Corrected Condition

Both couplings share the prefactor τ/(1+τ)², so their difference factorises:

```math
\mathcal{G} - \mathcal{N} = \frac{\tau}{(1+\tau)^2}\left[\left(p - \frac{1}{2}\right)\ln\frac{\bar{a}}{1-\bar{a}} - \frac{1}{2\alpha_0}\right] \qquad \text{...(Eq.\;11)}
```

(using |ln((1−ā)/ā)| = ln(ā/(1−ā)) for ā > ½). Three interesting consequences follow from this.

### 7.1 Early development is exploration-dominated

As τ → 0, ā → ½ and ln(ā/(1−ā)) ≈ 4(p − ½)τ, so:

```math
\mathcal{G} \approx 4\left(p - \tfrac{1}{2}\right)^2 \tau^2, \qquad \mathcal{N} \approx \frac{\tau}{2\alpha_0}
```

The consolidation coupling vanishes **quadratically** in τ, whilst the novelty coupling vanishes only **linearly**. Hence 𝒢/𝒩 → 0 as τ → 0: early in learning the novelty term *always* dominates, for every p, γ, and α₀. In this regime f₁ < 0, so the symmetric (strategy-mixing) solution is even more strongly stable than in the consolidation-only analysis — the agent samples both strategies, exactly as an epistemically driven learner should. Commitment is impossible until the (hyperbolically decaying) curiosity bonus has been substantially spent. Therefore, in this extended model, **exploration-first behaviour becomes a theorem**, and it does not have to be assumed/added to the main model.

### 7.2 A necessary condition for any transition to occur

The bracket in Eq. 11 is increasing in τ (since ā increases towards p), with supremum (p − ½)ln(p/(1−p)). A bifurcation is possible at *any* precision only if the bracket becomes positive at some τ, that is, only if:

```math
\boxed{\;\left(p - \frac{1}{2}\right)\ln\frac{p}{1-p} > \frac{1}{2\alpha_0}\;} \qquad \text{...(Eq.\;12)}
```

Highly **plastic** agents (small α₀), whose novelty drive is strongest, therefore cannot be captured by the consolidation loop alone: in the absence of genuine _pragmatic differences_ between the strategies (Δc = 0), they will keep exploring indefinitely. Numerically, at p = 0.85 the left-hand side is 0.607, so Eq. 12 requires α₀ > 0.824: an agent with a maximally weak prior never undergoes the symmetric transition at this discriminability, however precise its policy selection.

### 7.3 The corrected critical precision depends on the prior

Where Eq. 12 holds, the corrected critical precision is:

```math
\boxed{\;\gamma_c(p, \alpha_0) = \frac{1}{\max_\tau \bigl[\mathcal{G}(\tau; p) - \mathcal{N}(\tau; \alpha_0)\bigr]}\;} \qquad \text{...(Eq.\;13)}
```

This **qualifies the consolidation-only statement** (Notebook 01 §6.5, point 4; Notebook 04 §6.2) that γ_c depends only on p: with novelty included, the prior also gates whether commitment is possible, because _weak priors sustain a strong curiosity drive_. The consolidation-only threshold is recovered in the large-α₀ limit. In addition, wherever the corrected condition is satisfied, the bifurcation **window is delayed and narrowed** relative to the consolidation-only window (the curves 𝒢 − 𝒩 lie strictly below 𝒢 and cross the 1/γ threshold later and exit it earlier).

---

## 8. Step 6: The Normal Form and Numerical Verification

### 8.1 Symmetry structure of the cusp is preserved

We can repeat the expansion of Notebook 04 Step 7 with f(z) = −(γ/2)ΔG(z), ΔG now given by Eq. 4:

- **f₀** = −(γ/2)(1 − 2ā)Δc: unchanged, because ΔW(0) = 0 (Section 5.1).
- **f₁** = γ[𝒢 − 𝒩]: the revised (curiosity-corrected) slope of Eq. 8. The pitchfork occurs at f₁ = 1, i.e. Eq. 10.
- **f₂**: at Δc = 0 this remains exactly zero, because ΔW is odd in z (its second derivative at 0 vanishes term by term, just as for the ambiguity part in Notebook 04 §7.1).
- **f₃**: acquires a novelty contribution. From ΔW(z)·2α₀ = g(1+z) − g(1−z) with g(u) = 1/(1+uτ), the third derivative at z = 0 is 2g‴(1) = −12τ³/(1+τ)⁴, so d³ΔW/dz³|₀ = −6τ³/(α₀(1+τ)⁴). This shifts the cubic coefficient b = f₃/6 − f₁³/3 quantitatvely but does not alter the local structure; the numerics below confirm the pitchfork remains supercritical at onset in the regimes examined.

Hence the entire pitchfork/cusp apparatus of Notebook 01 §7 and Notebook 04 Step 7 carries over exactly but with the single substitution:

```math
\gamma\,\mathcal{G}(\tau; p) \;\longrightarrow\; \gamma\,\bigl[\mathcal{G}(\tau; p) - \mathcal{N}(\tau; \alpha_0)\bigr]
```

in the splitting variable. The normal variable (∝ Δc) and behavioural variable (z) are unchanged.

### 8.2 Numerical illustration (p = 0.85, γ = 16)

Given the manuscript's running example, p = 0.85 with the pymdp default γ = 16 (threshold 1/γ = 0.0625). First, the corrected coupling at α₀ = 8:

| τ | ā | 𝒢(τ) | 𝒩(τ; 8) | 𝒢 − 𝒩 |
|---|---|---|---|---|
| 0.5 | 0.617 | 0.0370 | 0.0139 | 0.0231 |
| 1.0 | 0.675 | 0.0640 | 0.0156 | 0.0483 |
| 2.0 | 0.733 | 0.0787 | 0.0139 | 0.0648 |
| 2.45 | 0.749 | 0.0786 | 0.0129 | **0.0657** (max) |
| 3.0 | 0.762 | 0.0765 | 0.0117 | 0.0648 |
| 5.0 | 0.792 | 0.0649 | 0.0087 | 0.0562 |
| 10.0 | 0.818 | 0.0435 | 0.0052 | 0.0383 |

Note the maximum of 𝒢 − 𝒩 shifts later (τ ≈ 2.45) than that of 𝒢 alone (τ ≈ 2.19) and is lower (0.0657 vs 0.0789).

Second, the corrected critical precision as a function of the prior (Eq. 13):

| α₀ | max_τ (𝒢 − 𝒩) | γ_c(0.85, α₀) | Bifurcation at γ = 16? |
|---|---|---|---|
| 0.5 | ≤ 0 (Eq. 12 violated) | ∞ | never (at any γ) |
| 1 | 0.0030 | 337.5 | no |
| 2 | 0.0309 | 32.4 | no |
| 6.39 | 0.0625 | 16.0 | marginal |
| 8 | 0.0657 | 15.2 | yes |
| ∞ | 0.0789 | 12.7 | yes (consolidation-only value ✓) |

The consolidation-only value γ_c = 12.7 of Notebook 01, Appendix A.3 is recovered as α₀ → ∞. ✓ At the notebook default α₀ = 2, the corrected threshold (32.4) exceeds the pymdp default γ = 16: **with the novelty term included, this agent mixes strategies indefinitely** in the symmetric setting. Commitment at γ = 16 requires α₀ ≳ 6.4.

Third, the bifurcation windows at γ = 16 (where γ(𝒢 − 𝒩) > 1):

| Condition | Linear-instability window (τ) |
|---|---|
| Consolidation-only (ΔW = 0) | [0.96, 5.42] |
| Novelty, α₀ = 8 | [1.68, 3.63] — delayed and narrowed |
| Novelty, α₀ = 2 | none |

### 8.3 Verification against the exact self-consistency equation

The predictions above rest on the *linearised* analysis. As a check, we solve the full fixed-point equation z = tanh(−γΔG(z)/2), with ΔG from Eq. 4 and a(z), b(z), ΔW(z) as in Sections 2 and 5, by iterating the map from z = 0.999 at each τ (this converges to the stable non-negative fixed point). At p = 0.85, γ = 16:

| Condition | Nonzero stable branch (exact) | Linear prediction |
|---|---|---|
| Consolidation-only | τ ∈ [0.97, ≈8.6] | entry 0.96 ✓ |
| Novelty, α₀ = 8 | τ ∈ [1.72, 3.65] | [1.68, 3.63] ✓ |
| Novelty, α₀ = 2 | no nonzero branch | none ✓ |

The window *entries* match the linear predictions to grid resolution. (For the consolidation-only case the committed branch persists somewhat beyond the linear window's exit, τ ≈ 8.6 vs 5.42: the branch there disappears by a fold rather than by re-absorption at the origin, the "re-entrant structure at larger τ" already flagged in Notebook 01 §6.5. This concerns the closing of the window and does not affect the onset analysis.)

---

## 9. Summary of Key Results

1. **Exact novelty term (manuscript Eq. 12):** W(πₖ) = 1/(2(α₀ + nₖ)) — the curiosity bonus decays hyperbolically with evidence, is regularised by the prior, and is exact for any within-column count composition.
2. **Extended EFE difference:** ΔG = (1 − a − b)Δc + 𝓗(a) − 𝓗(b) − ΔW, with the ambiguity and novelty differences as competing epistemic pressures.
3. **Curiosity-corrected bifurcation condition:** γ[𝒢(τ; p) − 𝒩(τ; α₀)] = 1, with 𝒩 = τ/(2α₀(1+τ)²) acting as an anti-consolidation (paramagnetic) coupling.
4. **Exploration-first phase:** 𝒢 ~ τ² and 𝒩 ~ τ as developmental time τ → 0, so curiosity always dominates early; and the prevalence of commitment depends on the decay of the novelty bonus.
5. **Existence condition:** (p − ½)ln(p/(1−p)) > 1/(2α₀). Highly plastic (small α₀) agents never commit under symmetric preferences.
6. **Corrected critical precision:** γ_c = 1/max_τ[𝒢 − 𝒩] now depends on both p and α₀, recovering the consolidation-only threshold as α₀ → ∞; where transitions occur, the window is delayed and narrowed.
7. **Normal form preserved:** ΔW is odd in z with ΔW(0) = 0, so the pitchfork/cusp structure of Notebooks 01/04 carries over with the splitting variable γ𝒢 simply replaced by γ(𝒢 − 𝒩).

### Additions to the notation table (_cf._ Notebook 04)

| Symbol | Meaning | Where defined |
|---|---|---|
| W(πₖ) | Parameter information gain (novelty) for policy πₖ | Eq. 3 (main manuscript Eq. 12) |
| ΔW = W(π₁) − W(π₂) | Novelty difference (favours the under-sampled strategy) | Eq. 4 |
| 𝒩(τ; α₀) = τ/(2α₀(1+τ)²) | Novelty coupling (anti-consolidation) | Eq. 9 (main manuscript Eq. 13) |
| γ_c(p, α₀) | Corrected critical precision | Eq. 13 |

---

## 10. Standalone Simulation in R (`sim_novelty_correction.R`)

The accompanying R script is fully standalone (base R only, no package dependencies, deterministic — no random seeds required), and it produces:

1. **Figure 1e (manuscript panel).** The curiosity-corrected coupling 𝒢(τ; p) − 𝒩(τ; α₀) as a function of rescaled developmental time τ, at p = 0.85, for α₀ ∈ {1, 2, 8} together with the consolidation-only curve 𝒢 (the α₀ → ∞ limit), and the horizontal threshold 1/γ at γ = 16. What this visualises: weak priors (small α₀) sustain a curiosity drive that lowers the whole coupling curve, thereby **delaying, narrowing, or even entirely removing** the bifurcation window — the α₀ = 1 and α₀ = 2 curves never reach threshold, whilst the α₀ = 8 curve crosses it over a window that is visibly later and narrower than the consolidation-only window.
2. **Supplementary verification figure.** The exact stable fixed-point branches z\*(τ) of the full self-consistency equation z = tanh(−γΔG(z)/2) (extended ΔG, Eq. 4) for three conditions at p = 0.85, γ = 16: consolidation-only; novelty with α₀ = 8; novelty with α₀ = 2. The predicted linear-instability windows are marked. What this verifies: commitment (a nonzero branch) exists precisely where the corrected condition γ(𝒢 − 𝒩) > 1 predicts — over the full consolidation-only window, over the delayed/narrowed α₀ = 8 window, and nowhere at all for α₀ = 2.
3. **Console output.** The numerical values quoted in Section 8 (γ_c table, windows, existence threshold), so that for convenience and ease of reference, every number in this notebook and in the manuscript text can be reproduced by running one script.

**What exactly is being simulated?** Nothing stochastic: the script evaluates the *derived* mean-field objects — the two coupling functions and the exact fixed points of the self-consistency map — over a grid of developmental times. It is the novelty/curiosity-corrected analogue of Notebook 01's Appendix A numerics, and it deliberately stays within the mean-field description so that the figure tests the derivation itself rather than a particular stochastic protocol. (Note. Trial-level stochastic simulations at small α₀ are dominated by the O(N^{-1/2}) fluctuations discussed in the manuscript's stochastic-approximation subsection, since small α₀ implies small N at fixed τ; the mean-field statements here concern the deterministic flow that those trajectories track.)

---

## References

Friston, K. J., Lin, M., Frith, C. D., Pezzulo, G., Hobson, J. A., & Ondobaka, S. (2017). Active inference, curiosity and insight. *Neural Computation, 29*(10), 2633–2683. https://discovery.ucl.ac.uk/id/eprint/1570070/1/Friston_Active%20Inference%20Curiosity%20and%20Insight.pdf

Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I., & Tschantz, A. (2022). pymdp: A Python library for active inference in discrete state spaces. *Journal of Open Source Software, 7*(73). https://arxiv.org/pdf/2201.03904

Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active inference: The free energy principle in mind, brain, and behavior.* MIT Press. 

Schwartenbeck, P., FitzGerald, T., Dolan, R. J., & Friston, K. (2013). Exploration, novelty, surprise, and free energy minimization. *Frontiers in Psychology, 4*, 710. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00710/full

Schwartenbeck, P., Passecker, J., Hauser, T. U., FitzGerald, T. H. B., Kronbichler, M., & Friston, K. J. (2019). Computational mechanisms of curiosity and goal-directed exploration. *eLife, 8*, e41703. https://elifesciences.org/articles/41703.pdf

Siegler, R. S. (1996). *Emerging minds: The process of change in children's thinking.* Oxford University Press.

Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. *Journal of Mathematical Psychology, 107*, 102632. https://www.sciencedirect.com/science/article/pii/S0022249621000973

Van der Maas, H. L., & Molenaar, P. C. (1992). Stagewise cognitive development: An application of catastrophe theory. *Psychological Review, 99*(3), 395–417. https://doi.org/10.1037//0033-295X.99.3.395 (possibly access via https://www.academia.edu/81088335/Stagewise_cognitive_development_An_application_of_catastrophe_theory)
