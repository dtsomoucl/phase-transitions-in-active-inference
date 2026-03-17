# Notebook 04 — From Dirichlet Learning to a Local Cusp Normal Form (step-by-step derivation of the main result)

**Series:** *Phase Transitions in Early Learning — An Active Inference Approach*

**Date:** 07 March 2026

**Status:** Step-by-step, explicit version of the key derivation

**Author:** Dr Dimitris I. Tsomokos

Psychology & Human Development, Institute of Education, University College London

---

## Part I: Derivation, Step by Step

### What Is Exact and What Is Approximate

This notebook verifies the algebra of the reduced model and makes explicit where approximations enter.

- **Exact:** the reduced one-step policy score, the EFE difference ΔG, the softmax-to-tanh identity, and the derivatives used in the local stability analysis.
- **Mean-field approximation:** replacing stochastic Dirichlet updates by expected counts and expressing development through the deterministic allocation variable φ.
- **Local normal-form approximation:** reducing the fixed-point equation to pitchfork/cusp form near the primary bifurcation. This does not fully characterise the global fixed-point landscape at large developmental time.

### Preamble: What We Are Trying to Show

We have an agent that:

1. Chooses between two strategies (policies π₁ and π₂) at each time step.
2. Gets feedback (observations) depending on which strategy it uses.
3. Learns from this feedback by updating its beliefs (Dirichlet parameters).
4. Uses its updated beliefs to choose the next strategy.

We want to show that steps 1–4, repeated many times, can produce a sudden switch from one strategy to the other — even though each individual learning step (step 3) is tiny and smooth — in the reduced mean-field description of the process.

---

### Step 1: The Agent's Beliefs About the World

#### 1.1 What the agent knows

The agent believes the world has two possible hidden states, s₁ and s₂, and two possible observations, o₁ and o₂. The agent's belief about the connection between states and observations is captured by the **A matrix**:

```
           s₁      s₂
    o₁ [   a     1-b  ]
    o₂ [ 1-a      b   ]
```

Here:

- a = the agent's current estimate of P(o₁ | s₁). "If I'm in state s₁, how likely am I to see observation o₁?"
- b = the agent's current estimate of P(o₂ | s₂). "If I'm in state s₂, how likely am I to see observation o₂?"

Both a and b are numbers between 0 and 1. When a = b = 1/2, the agent has no idea which observations go with which states (maximum ignorance). As a and b approach their true values (which we call p, assumed the same for both), the agent's beliefs become accurate.

#### 1.2 How beliefs are stored: Dirichlet parameters

The agent doesn't store a and b as fixed numbers. Instead, it stores *counts* that encode how confident it is. For column 1 of A (state s₁), the agent maintains two counts:

- α₁ = "how many times I've seen o₁ from s₁" (plus a prior)
- β₁ = "how many times I've seen o₂ from s₁" (plus a prior)

The agent's best estimate of a is then:

```math
a = \frac{\alpha_1}{\alpha_1 + \beta_1}
```

This is just "the fraction of o₁ observations out of all observations from s₁." Similarly for column 2 with counts α₂, β₂ giving b = β₂/(α₂ + β₂).

#### 1.3 The prior

Before seeing any data, the agent starts with a symmetric prior: α₁ = β₁ = α₀/2 and α₂ = β₂ = α₀/2, where α₀ is the "prior strength." This gives a = b = 1/2 initially (complete ignorance).

We define:

- **n₁** = number of observations from s₁ (not counting the prior)
- **n₂** = number of observations from s₂ (not counting the prior)
- **N = n₁ + n₂** = total observations

After n₁ observations from s₁, the total concentration is α₁ + β₁ = α₀ + n₁. On average, n₁ · p of these observations will have been o₁ (where p is the true probability), so:

```math
a \approx \frac{\alpha_0/2 + n_1 p}{\alpha_0 + n_1}
```

A little algebra gives the cleaner form:

```math
a = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{n_1}{\alpha_0 + n_1} \qquad \text{...(Eq.\;1)}
```

**Check:** When n₁ = 0, a = 1/2. ✓ When n₁ → ∞, a → p. ✓ The formula smoothly interpolates between ignorance and truth.

---

### Step 2: How the Agent Chooses a Strategy

#### 2.1 Expected free energy (EFE)

The agent evaluates each strategy by computing its **expected free energy** G(πₖ). This has two parts:

**Pragmatic value** (the "reward" part): How likely is this strategy to produce observations the agent prefers? This is computed as:

```math
\text{Pragmatic value of } \pi_k = \sum_i A_{ik} \, C_i
```

where C is a vector of log-preferences (positive means "I like this observation," negative means "I dislike it").

**Ambiguity** (the "uncertainty" part): How uncertain are the observations from this strategy? This is measured by the Shannon entropy of column k of A:

```math
\text{Ambiguity of } \pi_k = H(\mathbf{A}_{:,k}) = -\sum_i A_{ik} \ln A_{ik}
```

The EFE combines these:

```math
G(\pi_k) = -\sum_i A_{ik} \, C_i + H(\mathbf{A}_{:,k})
```

Lower G means "better strategy" — more rewarding and less uncertain.

#### 2.2 Computing the EFE for our specific model

For policy π₁ (which takes the agent to state s₁):

```math
G(\pi_1) = -\bigl[a \, c_1 + (1-a) \, c_2\bigr] + \mathcal{H}(a)
```

where 𝒽(a) = −a ln a − (1−a) ln(1−a) is the **binary entropy function**.

For policy π₂ (which takes the agent to state s₂):

```math
G(\pi_2) = -\bigl[(1-b) \, c_1 + b \, c_2\bigr] + \mathcal{H}(b)
```

#### 2.3 The EFE difference

What matters for the agent's choice is the *difference*:

```math
\Delta G = G(\pi_1) - G(\pi_2)
```

If ΔG < 0, then π₁ has lower (better) EFE, so the agent prefers π₁.
If ΔG > 0, then π₂ has lower EFE, so the agent prefers π₂.

Let's compute ΔG by expanding:

```math
\Delta G = \bigl\{-[a \, c_1 + (1-a) c_2] + \mathcal{H}(a)\bigr\} - \bigl\{-[(1-b) c_1 + b \, c_2] + \mathcal{H}(b)\bigr\}
```

Rearranging the pragmatic terms:

Pragmatic part of ΔG = −a c₁ − (1−a)c₂ + (1−b)c₁ + b c₂

Group by c₁ and c₂:

= c₁ [−a + (1−b)] + c₂ [−(1−a) + b]
= c₁ (1 − a − b) + c₂ (a + b − 1)
= (1 − a − b)(c₁ − c₂)
= (1 − a − b) Δc

where Δc = c₁ − c₂. So the full EFE difference is:

```math
\Delta G = (1 - a - b)\,\Delta c + \mathcal{H}(a) - \mathcal{H}(b) \qquad \text{...(Eq.\;2)}
```

**Check the signs make sense.** Suppose a > b (the agent has learned more about s₁ than s₂), so 𝒽(a) < 𝒽(b) (lower entropy = less ambiguity). Then 𝒽(a) − 𝒽(b) < 0, making ΔG more negative, favouring π₁. ✓ This is the ambiguity-driven feedback: the better-learned strategy has lower ambiguity, making it more attractive.

#### 2.4 The policy posterior

The agent doesn't deterministically pick the best strategy. It samples from a softmax distribution:

```math
P(\pi_1) = \frac{1}{1 + \exp(\gamma \, \Delta G)} \qquad \text{...(Eq.\;3)}
```

where γ > 0 is the **policy precision** (how "decisive" the agent is).

When ΔG = 0: P(π₁) = 1/2 (indifferent).
When ΔG < 0: P(π₁) > 1/2 (prefers π₁).
When ΔG > 0: P(π₁) < 1/2 (prefers π₂).
Larger γ makes the choice sharper (more committed to the better option).

---

### Step 3: Closing the Loop — Learning Depends on Choice

#### 3.1 The feedback mechanism

Here is the key insight. In the reduced model, the agent's choice of strategy determines what it *experiences*, which determines what it *learns*:

- If the agent picks π₁ → it visits s₁ → it gets an observation from column 1 of A → **n₁ goes up by 1** → a moves closer to its true value p.
- If the agent picks π₂ → it visits s₂ → it gets an observation from column 2 of A → **n₂ goes up by 1** → b moves closer to p.

This creates a **positive feedback loop**:

Using π₁ more → a sharpens (𝒽(a) decreases) → ΔG decreases → P(π₁) increases → using π₁ even more

#### 3.2 The experience allocation variable

Define:

```math
\phi = \frac{n_1}{N} \qquad \text{(fraction of total experience allocated to strategy 1)}
```

At each time step, the agent adds one observation. The probability that this observation comes from s₁ is P(π₁), which depends on φ (through a, b, and ΔG). So φ evolves as:

```math
\phi_{\text{new}} \approx \phi + \frac{1}{N}\bigl[P(\pi_1) - \phi\bigr] \qquad \text{...(Eq.\;4)}
```

This says φ is pulled toward P(π₁) at a rate 1/N (which slows as N grows, because each new observation is a smaller fraction of the total).

#### 3.3 The fixed-point condition

At a **fixed point** (equilibrium), φ stops changing. From Eq. 4:

```math
\phi^* = P(\pi_1) \Big|_{a(\phi^*),\; b(\phi^*)} \qquad \text{...(Eq.\;5)}
```

This is a **self-consistency equation**: the experience allocation φ must be consistent with the policy it generates. This closes the mean-field approximation.

---

### Step 4: The Order Parameter and the Ising Equation

#### 4.1 Change of variable

Define:

```math
z = 2\phi - 1 \in [-1,\; 1] \qquad \text{...(Eq.\;6)}
```

This is the **order parameter**. z = 0 means equal experience with both strategies. z = +1 means exclusive commitment to π₁. z = −1 means exclusive commitment to π₂.

From Eq. 6: φ = (1+z)/2, so:

```math
n_1 = \frac{(1+z)N}{2}, \qquad n_2 = \frac{(1-z)N}{2}
```

#### 4.2 The σ-to-tanh identity

There is a standard mathematical identity:

```math
\sigma(x) = \frac{1}{1+e^{-x}} = \frac{1 + \tanh(x/2)}{2} \qquad \text{...(Eq.\;7)}
```

**Proof of Eq. 7.** Starting from tanh(x/2):

```math
\tanh(x/2) = \frac{e^{x/2} - e^{-x/2}}{e^{x/2} + e^{-x/2}}
```

Multiply top and bottom by e^{x/2}:

```math
= \frac{e^x - 1}{e^x + 1}
```

Now:

```math
\frac{1 + \tanh(x/2)}{2} = \frac{1 + \frac{e^x - 1}{e^x + 1}}{2} = \frac{\frac{e^x + 1 + e^x - 1}{e^x + 1}}{2} = \frac{2e^x}{2(e^x + 1)} = \frac{e^x}{e^x + 1} = \frac{1}{1 + e^{-x}} = \sigma(x) \quad \checkmark
```

#### 4.3 Deriving the self-consistency equation

The fixed-point condition (Eq. 5) is φ* = σ(−γΔG). Using Eq. 7:

```math
\frac{1 + z}{2} = \frac{1 + \tanh(-\gamma \Delta G / 2)}{2}
```

Cancel (1/2) from both sides, then subtract 1:

```math
z = \tanh\!\left(-\frac{\gamma}{2}\,\Delta G(z)\right) \qquad \text{...(Eq.\;8)}
```

This is **formally identical to the Curie–Weiss mean-field equation** of the Ising model:

```math
m = \tanh(\beta J m + \beta h)
```

where m is the magnetisation, β is inverse temperature, J is the coupling constant, and h is the external magnetic field.

---

### Step 5: What Does ΔG Look Like as a Function of z?

#### 5.1 Expressing a and b in terms of z

We introduce a **rescaled time** τ = N/(2α₀) to absorb α₀. Then n₁ = (1+z)N/2 and:

```math
a(z) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1+z)\tau}{1 + (1+z)\tau} \qquad \text{...(Eq.\;9a)}
```

```math
b(z) = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{(1-z)\tau}{1 + (1-z)\tau} \qquad \text{...(Eq.\;9b)}
```

**Derivation of Eq. 9a.** From Eq. 1: a = 1/2 + (p−1/2) · n₁/(α₀+n₁). With n₁ = (1+z)N/2 and α₀ = N/(2τ):

```math
\frac{n_1}{\alpha_0 + n_1} = \frac{(1+z)N/2}{N/(2\tau) + (1+z)N/2}
```

Cancel N:

```math
= \frac{(1+z)/2}{1/(2\tau) + (1+z)/2} = \frac{(1+z)\tau}{1 + (1+z)\tau} \quad \checkmark
```

#### 5.2 ΔG at z = 0

At z = 0, a(0) = b(0). Call this common value ā:

```math
\bar{a} = \frac{1}{2} + \left(p - \frac{1}{2}\right) \frac{\tau}{1+\tau} \qquad \text{...(Eq.\;10)}
```

Since a = b at z = 0, the ambiguity terms cancel: 𝒽(a) − 𝒽(b) = 0. And (1−a−b) = (1−2ā). So:

```math
\Delta G(z{=}0) = (1 - 2\bar{a})\,\Delta c \qquad \text{...(Eq.\;11)}
```

When Δc = 0 (symmetric preferences), ΔG(0) = 0, confirming z = 0 is always a fixed point in the symmetric case.

#### 5.3 How ΔG changes with z near z = 0 (the crucial derivative)

We need ∂ΔG/∂z evaluated at z = 0.

**Step A: Differentiate a(z).**

Write a = 1/2 + (p−1/2) · u/(1+u) where u = (1+z)τ. Then:

```math
\frac{da}{dz} = (p - \tfrac{1}{2}) \cdot \frac{du/dz}{(1+u)^2} = (p - \tfrac{1}{2}) \cdot \frac{\tau}{(1+u)^2}
```

At z = 0: u = τ, so:

```math
\left.\frac{da}{dz}\right|_{z=0} = (p - \tfrac{1}{2}) \cdot \frac{\tau}{(1+\tau)^2} \equiv \lambda \qquad \text{...(Eq.\;12)}
```

**Step B: Differentiate b(z).**

b = 1/2 + (p−1/2) · v/(1+v) where v = (1−z)τ. Then dv/dz = −τ, so:

```math
\frac{db}{dz} = (p - \tfrac{1}{2}) \cdot \frac{-\tau}{(1+v)^2}
```

At z = 0: v = τ, so:

```math
\left.\frac{db}{dz}\right|_{z=0} = -\lambda \qquad \text{...(Eq.\;13)}
```

The negative sign is physically sensible: when z increases (more experience with s₁), a sharpens but b blunts.

**Step C: Differentiate the ambiguity 𝒽(a) − 𝒽(b).**

```math
\frac{d}{dz}\bigl[\mathcal{H}(a) - \mathcal{H}(b)\bigr] = \mathcal{H}'(a)\,\frac{da}{dz} - \mathcal{H}'(b)\,\frac{db}{dz}
```

The derivative of binary entropy is 𝒽'(x) = −ln x − 1 + ln(1−x) + 1 = ln((1−x)/x).

At z = 0: a = b = ā, da/dz = λ, db/dz = −λ. So:

```math
= \mathcal{H}'(\bar{a}) \cdot \lambda - \mathcal{H}'(\bar{a}) \cdot (-\lambda) = 2\lambda\,\mathcal{H}'(\bar{a}) = 2\lambda\,\ln\!\frac{1-\bar{a}}{\bar{a}} \qquad \text{...(Eq.\;14)}
```

Since ā > 1/2, we have (1−ā)/ā < 1, so ln((1−ā)/ā) < 0, so this whole expression is negative.

**Step D: Differentiate the pragmatic term (1−a−b)Δc.**

```math
\frac{d}{dz}\bigl[(1 - a - b)\,\Delta c\bigr] = \left(-\frac{da}{dz} - \frac{db}{dz}\right)\Delta c = (-\lambda - (-\lambda))\,\Delta c = 0
```

At z = 0, the pragmatic term's derivative vanishes! (Because da/dz and db/dz are equal and opposite, their sum is zero.)

**Step E: Combine.**

```math
\left.\frac{\partial \Delta G}{\partial z}\right|_{z=0} = 2\lambda\,\ln\!\frac{1-\bar{a}}{\bar{a}} \qquad \text{...(Eq.\;15)}
```

This is negative (for ā > 1/2). Define its absolute value:

```math
\left|\frac{\partial \Delta G}{\partial z}\right|_{z=0} = 2\lambda\,\left|\ln\!\frac{1-\bar{a}}{\bar{a}}\right|
```

---

### Step 6: The Bifurcation Condition

#### 6.1 When does z = 0 become unstable?

Define F(z) = tanh(−γΔG(z)/2) − z. The fixed point z = 0 is stable when dF/dz|₀ < 0 and unstable when dF/dz|₀ > 0. The transition occurs at:

```math
\left.\frac{dF}{dz}\right|_{z=0} = 0
```

Computing:

```math
\frac{dF}{dz} = \left(-\frac{\gamma}{2}\right) \frac{\partial \Delta G}{\partial z} \cdot \text{sech}^2\!\left(-\frac{\gamma \Delta G}{2}\right) - 1
```

At z = 0 with Δc = 0: ΔG(0) = 0, so sech²(0) = 1, and:

```math
\left.\frac{dF}{dz}\right|_0 = \left(-\frac{\gamma}{2}\right)\left.\frac{\partial \Delta G}{\partial z}\right|_0 - 1
```

Since (∂ΔG/∂z)|₀ is negative (Eq. 15), the first term is positive:

```math
= \frac{\gamma}{2} \cdot 2\lambda\,\left|\ln\!\frac{1-\bar{a}}{\bar{a}}\right| - 1 = \gamma \cdot \lambda \cdot \left|\ln\!\frac{1-\bar{a}}{\bar{a}}\right| - 1
```

Setting this to zero:

```math
\gamma \cdot \left(p - \frac{1}{2}\right) \cdot \frac{\tau}{(1+\tau)^2} \cdot \left|\ln\!\frac{1-\bar{a}}{\bar{a}}\right| = 1 \qquad \text{...(Eq.\;16)}
```

That is: **γ · 𝒢(τ; p) = 1**, where:

```math
\mathcal{G}(\tau;\, p) = \left(p - \frac{1}{2}\right) \frac{\tau}{(1+\tau)^2}\,\left|\ln\!\frac{1-\bar{a}}{\bar{a}}\right| \qquad \text{...(Eq.\;17)}
```

This is the **coupling function**. It depends only on τ and p.

#### 6.2 The critical precision

The bifurcation occurs when γ exceeds:

```math
\gamma_c = \frac{1}{\max_\tau\, \mathcal{G}(\tau;\, p)} \qquad \text{...(Eq.\;18)}
```

Note: γ_c depends **only on p** (the environmental discriminability), not on α₀. The prior strength α₀ determines the *timing* of the primary transition via N* = 2α₀τ* but not whether the first loss of stability occurs.

---

### Step 7: The Local Cusp Normal Form

#### 7.1 Taylor expansion

We expand the self-consistency equation z = tanh(f(z)) around z = 0, where f(z) = −γΔG(z)/2.

**Expand tanh:** tanh(x) = x − x³/3 + O(x⁵)

**Expand f(z):** f(z) = f₀ + f₁z + f₂z²/2 + f₃z³/6 + ...

where:

- f₀ = −γΔG(0)/2 = −γ(1−2ā)Δc/2. This is zero when Δc = 0.
- f₁ = −(γ/2)(∂ΔG/∂z)|₀ = γ𝒢(τ; p). This is positive.
- f₂ = −(γ/2)(∂²ΔG/∂z²)|₀. When Δc = 0, this is exactly zero because ΔG is an odd function of z (a(z) and b(z) are swapped under z → −z when Δc = 0).

**Verification that f₂ = 0 when Δc = 0:** The ambiguity part 𝒽(a) − 𝒽(b) is odd in z (because swapping z → −z swaps a and b). Its second derivative at z = 0 involves terms 𝒽''(ā)λ² + 𝒽'(ā)a'' − 𝒽''(ā)λ² − 𝒽'(ā)a'' = 0. ✓

#### 7.2 The pitchfork (Δc = 0)

With Δc = 0: f₀ = 0, f₂ = 0. So f(z) = f₁z + f₃z³/6 + O(z⁵).

Substituting into z = tanh(f(z)) ≈ f − f³/3:

```math
z \approx f_1 z + \left(\frac{f_3}{6} - \frac{f_1^3}{3}\right) z^3
```

Rearranging:

```math
0 = (f_1 - 1)\,z + \left(\frac{f_3}{6} - \frac{f_1^3}{3}\right) z^3 \qquad \text{...(Eq.\;19)}
```

This is the **normal form for the pitchfork bifurcation**. Solutions:

- z = 0 (always exists)
- z² = −(f₁ − 1) / (f₃/6 − f₁³/3) (exists only when f₁ > 1, provided the cubic coefficient is negative, which can be verified numerically)

The bifurcation occurs at f₁ = 1, i.e., γ𝒢 = 1. ✓

#### 7.3 The cusp (Δc ≠ 0 small)

With Δc ≠ 0 but small, f₀ is small and proportional to Δc, and f₂ is also proportional to Δc. The equation becomes:

```math
0 = f_0 + (f_1 - 1)\,z + \frac{f_2}{2}\,z^2 + b\,z^3
```

where b = f₃/6 − f₁³/3. The quadratic term can be removed by the substitution z → z − f₂/(6b), yielding:

```math
0 = y + x\,\tilde{z} + \tilde{z}^3 \qquad \text{...(Eq.\;20)}
```

where x = (f₁ − 1)/b + correction and y = f₀/b + correction. This is the **canonical cusp equation** z³ + xz + y = 0.

The bifurcation set (the boundary where the number of real solutions changes) is:

```math
4x^3 + 27y^2 = 0
```

which defines the characteristic cusp-shaped curve in the (x, y) control plane. Inside the cusp, there are three real solutions (bistability); outside, one.

In our system:

- x is controlled primarily by 1 − γ𝒢(τ; p): the "splitting variable"
- y is controlled primarily by Δc: the "normal variable" (asymmetry)

This completes the derivation of the **local cusp normal form** of the reduced system. Additional global roots can appear away from this local regime and must be checked numerically rather than inferred from the cubic normal form alone.

---

## Part II: Summary of the Core Argument in Plain Language

1. An agent learns about two strategies by trying them. Each time it uses a strategy, it gets a bit better at evaluating it (Dirichlet learning).

2. The agent chooses strategies in proportion to how good they seem (softmax policy selection). Better-evaluated strategies get chosen more often.

3. This creates a feedback loop: using a strategy → better evaluation of it → choosing it more → even better evaluation. The loop is self-reinforcing.

4. At equilibrium, the fraction of experience devoted to each strategy must be *consistent* with the policy probabilities that follow from that experience. This self-consistency condition is Eq. 8: z = tanh(−γΔG(z)/2).

5. When the feedback is weak (early in development, or low policy precision γ), the only solution is z = 0 (equal use of both strategies). But when the feedback strength γ𝒢(τ; p) exceeds 1, the z = 0 solution becomes unstable and two new solutions appear at z ≈ ±z* (committed to one or the other strategy). The agent spontaneously "chooses" one.

6. This is a phase transition — identical in form to the magnetisation transition in the Ising model, and locally a cusp-type unfolding under asymmetry. It motivates the developmental signatures discussed in the simulation notebooks, but those signatures still need to be demonstrated numerically under explicit protocols.

---

## Appendix: Summary of Notation

| Symbol | Meaning | Convention in this notebook |
|---|---|---|
| n₁, n₂ | Number of observations from s₁, s₂ | **Excludes** prior |
| N = n₁ + n₂ | Total observations | Excludes prior |
| α₀ | Total prior concentration (per column) | The "strength" of initial ignorance |
| τ = N/(2α₀) | Rescaled developmental time | Absorbs α₀ |
| φ = n₁/N | Experience fraction for strategy 1 | ∈ [0, 1] |
| z = 2φ − 1 | Order parameter | ∈ [−1, 1] |
| a, b | Learned discriminabilities | Eq. 1 |
| ā | Common value at z = 0 | Eq. 10 |
| λ | Sensitivity of a to experience allocation | Eq. 12 |
| 𝒢(τ; p) | Coupling function | Eq. 17 |
| γ | Policy precision | Inverse temperature analogue |
| γ_c | Critical precision | Eq. 18; depends only on p |
| Δc = c₁ − c₂ | Preference asymmetry | External field analogue |
| ΔG | EFE difference G(π₁) − G(π₂) | Eq. 2 |
| 𝒽(x) | Binary entropy | −x ln x − (1−x) ln(1−x) |
