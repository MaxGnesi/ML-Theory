# Statistics Foundations for Machine Learning: A Cheatsheet

---

## 1. Core Concepts

### Expectation (Mean)

```python
E[X] = Σ xᵢ P(xᵢ)           # Discrete
E[X] = ∫ x f(x) dx          # Continuous

# Example: Die roll
E[X] = (1+2+3+4+5+6)/6 = 3.5
```

**ML Application:** Loss functions minimize expected loss; feature means for normalization.

---

### Variance and Standard Deviation

```python
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
σ = √Var(X)

# Example: Die roll
E[X²] = (1+4+9+16+25+36)/6 = 15.17
Var(X) = 15.17 - 3.5² = 2.92
σ = 1.71
```

**ML Application:** Batch normalization, uncertainty quantification, regularization strength.

---

### Covariance and Correlation

```python
Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]

Correlation: ρ = Cov(X,Y) / (σₓ σᵧ)    # Normalized to [-1, 1]

# Example:
X = [1, 2, 3], Y = [2, 4, 5]
E[X] = 2, E[Y] = 3.67, E[XY] = 8.33
Cov(X,Y) = 8.33 - 2×3.67 = 1.0
```

**ML Application:** Feature selection, PCA, portfolio optimization, multicollinearity detection.

---

## 2. Key Theorems

### Law of Large Numbers (LLN)

Sample mean converges to true mean as n → ∞.

```python
# Example: Coin flips (p=0.5)
n = 10:     sample mean might be 0.6
n = 100:    sample mean ≈ 0.52
n = 10000:  sample mean ≈ 0.5003

X̄ₙ → E[X] as n → ∞
```

**ML Application:** Why training on more data works; Monte Carlo methods converge.

---

### Central Limit Theorem (CLT)

Sum/mean of many independent random variables → Normal distribution, regardless of original distribution.

```python
# Example: Sum of 30 dice rolls
# Each die: uniform on {1,2,3,4,5,6}, μ=3.5, σ²=2.92

Sum of 30 dice ~ Normal(μ=105, σ²=87.5)
                 Normal(105, 9.35)

# 95% of sums fall in [105 ± 1.96×9.35] = [86.7, 123.3]
```

**ML Application:** Why gradient descent works (sum of many small updates); confidence intervals; batch statistics.

---

### Bayes' Theorem

```python
P(A|B) = P(B|A) × P(A) / P(B)

         posterior ∝ likelihood × prior

# Example: Disease test
# P(disease) = 0.01 (prior)
# P(positive|disease) = 0.95 (sensitivity)
# P(positive|healthy) = 0.05 (false positive)

P(positive) = 0.95×0.01 + 0.05×0.99 = 0.059

P(disease|positive) = (0.95 × 0.01) / 0.059 = 0.16

# Only 16% chance of disease despite positive test!
```

**ML Application:** Naive Bayes classifier, Bayesian neural networks, probabilistic inference, spam filtering.

---

## 3. Discrete Distributions

### Bernoulli Distribution

Single binary trial with probability p.

```python
X ~ Bernoulli(p)

P(X=1) = p
P(X=0) = 1-p
E[X] = p
Var(X) = p(1-p)

# Example: Single coin flip, p=0.3
P(heads) = 0.3, P(tails) = 0.7
E[X] = 0.3, Var(X) = 0.21
```

**ML Application:** Binary classification output, dropout (each neuron is Bernoulli).

---

### Binomial Distribution

Number of successes in n independent Bernoulli trials.

```python
X ~ Binomial(n, p)

P(X=k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ
E[X] = np
Var(X) = np(1-p)

# Example: 10 coin flips, p=0.3
P(exactly 3 heads) = C(10,3) × 0.3³ × 0.7⁷
                   = 120 × 0.027 × 0.082 = 0.267

E[X] = 3, Var(X) = 2.1
```

**ML Application:** A/B testing, classification accuracy distribution, ensemble voting.

---

### Poisson Distribution

Count of events in fixed interval (rare events).

```python
X ~ Poisson(λ)

P(X=k) = (λᵏ e⁻λ) / k!
E[X] = λ
Var(X) = λ

# Example: Website gets 4 errors/hour on average
P(exactly 2 errors) = (4² × e⁻⁴) / 2!
                    = 16 × 0.0183 / 2 = 0.147

P(0 errors) = e⁻⁴ = 0.0183
```

**ML Application:** Count data modeling, rare event detection, queuing models, text word frequencies.

---

### Categorical (Multinoulli) Distribution

Single trial with k possible outcomes.

```python
X ~ Categorical(p₁, p₂, ..., pₖ)   where Σpᵢ = 1

# Example: Die roll
P(X=1) = P(X=2) = ... = P(X=6) = 1/6

# Example: Sentiment classification
P(positive) = 0.6, P(neutral) = 0.3, P(negative) = 0.1
```

**ML Application:** Softmax output layer, multi-class classification, topic models.

---

### Multinomial Distribution

Counts across k categories in n trials (multivariate Binomial).

```python
X ~ Multinomial(n, p₁, ..., pₖ)

# Example: 100 rolls of biased die
# p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]

E[count of 1s] = 100 × 0.1 = 10
E[count of 3s] = 100 × 0.2 = 20
```

**ML Application:** Document word counts (bag of words), multi-class confusion matrices.

---

## 4. Continuous Distributions

### Uniform Distribution

Equal probability across interval [a, b].

```python
X ~ Uniform(a, b)

f(x) = 1/(b-a)  for x ∈ [a,b]
E[X] = (a+b)/2
Var(X) = (b-a)²/12

# Example: Random number in [0, 10]
E[X] = 5, Var(X) = 8.33
P(X < 3) = 3/10 = 0.3
```

**ML Application:** Weight initialization, random search hyperparameters, random sampling.

---

### Normal (Gaussian) Distribution

The "default" distribution—appears everywhere via CLT.

```python
X ~ Normal(μ, σ²)

f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
E[X] = μ
Var(X) = σ²

# Standard normal: Z ~ N(0,1)

# Example: Heights with μ=170cm, σ=10cm
P(height > 190) = P(Z > 2) ≈ 0.023  (2.3%)
P(150 < height < 190) = P(-2 < Z < 2) ≈ 0.954

# 68-95-99.7 rule:
# 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ
```

**ML Application:** Weight initialization, Gaussian noise, likelihood functions, GPs, VAE latent space.

---

### Multivariate Normal

n-dimensional extension of Normal.

```python
X ~ MVN(μ, Σ)

# μ = mean vector (n×1)
# Σ = covariance matrix (n×n, must be PSD)

# Example: 2D
μ = [0, 0]
Σ = [[1.0, 0.8],
     [0.8, 1.0]]   # Correlated variables

# Diagonal Σ → independent dimensions
# Off-diagonal → correlations
```

**ML Application:** Gaussian Processes, GMMs, Bayesian linear regression, PCA assumes MVN.

---

### Exponential Distribution

Time until first event (continuous analog of geometric).

```python
X ~ Exponential(λ)

f(x) = λ e⁻λˣ  for x ≥ 0
E[X] = 1/λ
Var(X) = 1/λ²

# Memoryless property: P(X > s+t | X > s) = P(X > t)

# Example: Server requests at 5/minute → time between requests
λ = 5, E[X] = 0.2 minutes = 12 seconds
P(wait > 30 sec) = P(X > 0.5) = e⁻⁵ˣ⁰·⁵ = e⁻²·⁵ = 0.082
```

**ML Application:** Survival analysis, queuing theory, time-to-event modeling.

---

### Beta Distribution

Distribution over probabilities [0, 1]—conjugate prior for Bernoulli/Binomial.

```python
X ~ Beta(α, β)

f(x) ∝ x^(α-1) (1-x)^(β-1)  for x ∈ [0,1]
E[X] = α / (α+β)
Mode = (α-1) / (α+β-2)  for α,β > 1

# Example: Prior belief about coin fairness
Beta(1, 1)   = Uniform (no prior knowledge)
Beta(10, 10) = Centered at 0.5, fairly confident
Beta(2, 8)   = Believe p ≈ 0.2

# After observing 7 heads, 3 tails with Beta(1,1) prior:
Posterior = Beta(1+7, 1+3) = Beta(8, 4)
E[p|data] = 8/12 = 0.67
```

**ML Application:** Bayesian A/B testing, Thompson sampling, prior for probabilities.

---

### Gamma Distribution

Generalization of Exponential; conjugate prior for Poisson rate.

```python
X ~ Gamma(α, β)    # α = shape, β = rate

E[X] = α/β
Var(X) = α/β²

# Special cases:
Gamma(1, λ) = Exponential(λ)
Gamma(n/2, 1/2) = Chi-squared(n)

# Example: Prior for Poisson rate
# Believe rate ≈ 5 with some uncertainty
Gamma(α=25, β=5) → E[λ] = 5, Var = 1
```

**ML Application:** Prior for precision/variance, Bayesian inference, survival models.

---

### Chi-Squared Distribution

Sum of squared standard normals.

```python
If Z₁, ..., Zₖ ~ N(0,1) independent, then:
X = Σ Zᵢ² ~ χ²(k)

E[X] = k
Var(X) = 2k

# Example: k=5
E[X] = 5, Var(X) = 10
P(X > 11.07) = 0.05  (critical value for hypothesis test)
```

**ML Application:** Hypothesis testing, goodness-of-fit, feature selection (chi-squared test).

---

### Student's t-Distribution

Normal with uncertain variance—heavier tails.

```python
t ~ t(ν)    # ν = degrees of freedom

E[X] = 0 (for ν > 1)
Var(X) = ν/(ν-2) (for ν > 2)

# As ν → ∞, t → Normal(0,1)
# Small ν → heavy tails (robust to outliers)

# Example: ν = 3
# Much heavier tails than Normal
# P(|t| > 2) ≈ 0.14 vs Normal P(|Z| > 2) ≈ 0.05
```

**ML Application:** Small sample inference, robust regression, Bayesian posteriors with unknown variance.

---

### Log-Normal Distribution

If log(X) is Normal, then X is Log-Normal.

```python
X ~ LogNormal(μ, σ²)

If Y ~ Normal(μ, σ²), then X = eʸ ~ LogNormal

E[X] = exp(μ + σ²/2)
Var(X) = (exp(σ²) - 1) × exp(2μ + σ²)

# Example: Stock prices, μ=0, σ=0.2
# Multiplicative growth → log-normal
E[X] = exp(0.02) = 1.02
```

**ML Application:** Financial modeling, any positive quantity with multiplicative noise, income distribution.

---

## 5. Estimation

### Maximum Likelihood Estimation (MLE)

Find parameters that maximize probability of observed data.

```python
θ_MLE = argmax P(data | θ)
      = argmax Σ log P(xᵢ | θ)   # Log-likelihood

# Example: Estimate μ of Normal (known σ²=1)
# Data: [2.1, 1.8, 2.3, 1.9, 2.0]

# Log-likelihood: -n/2 log(2π) - Σ(xᵢ-μ)²/2
# Derivative = 0 → μ_MLE = (1/n)Σxᵢ = sample mean

μ_MLE = (2.1+1.8+2.3+1.9+2.0)/5 = 2.02
```

**ML Application:** Foundation of most ML training—minimizing cross-entropy = maximizing likelihood.

---

### Maximum A Posteriori (MAP)

MLE + prior belief.

```python
θ_MAP = argmax P(θ | data)
      = argmax [log P(data | θ) + log P(θ)]
                    ↑                ↑
                likelihood        prior

# Example: Same data, but prior μ ~ Normal(0, 1)
# MAP adds regularization term: -μ²/2

θ_MAP = (Σxᵢ + 0) / (n + 1) = 10.1/6 = 1.68

# Prior pulls estimate toward 0
```

**ML Application:** L2 regularization = Gaussian prior; L1 = Laplace prior; Bayesian neural nets.

---

### Confidence Intervals

Range that contains true parameter with probability (1-α).

```python
# For mean with known σ:
CI = X̄ ± z_(α/2) × σ/√n

# Example: n=100, X̄=50, σ=10, 95% CI
CI = 50 ± 1.96 × 10/√100
   = 50 ± 1.96
   = [48.04, 51.96]

# For mean with unknown σ (use t-distribution):
CI = X̄ ± t_(α/2, n-1) × s/√n
```

**ML Application:** Model uncertainty, A/B test results, hyperparameter sensitivity.

---

## 6. Hypothesis Testing

### Framework

```python
H₀: Null hypothesis (status quo)
H₁: Alternative hypothesis

# Procedure:
1. Assume H₀ is true
2. Compute test statistic
3. Find p-value = P(statistic this extreme | H₀)
4. If p-value < α (typically 0.05), reject H₀

# Errors:
Type I:  Reject H₀ when true  (false positive) — rate = α
Type II: Accept H₀ when false (false negative) — rate = β
Power = 1 - β = P(reject H₀ | H₁ true)
```

### Example: Two-Sample t-Test

```python
# Group A: [5.1, 4.8, 5.3, 5.0]  → X̄_A = 5.05
# Group B: [4.2, 4.5, 4.3, 4.1]  → X̄_B = 4.28
# Is there a significant difference?

t = (X̄_A - X̄_B) / √(s²_A/n_A + s²_B/n_B)
  = (5.05 - 4.28) / √(0.04/4 + 0.03/4)
  = 0.77 / 0.132 = 5.83

# With df ≈ 6, p-value < 0.001 → Significant difference
```

**ML Application:** A/B testing, feature importance, model comparison.

---

## 7. Information Theory

### Entropy

Measure of uncertainty/information content.

```python
H(X) = -Σ P(xᵢ) log₂ P(xᵢ)

# Example: Fair coin
H = -[0.5 log₂(0.5) + 0.5 log₂(0.5)] = 1 bit

# Biased coin (p=0.9)
H = -[0.9 log₂(0.9) + 0.1 log₂(0.1)] = 0.47 bits

# More certain → lower entropy
```

**ML Application:** Decision tree splits, information gain, model complexity.

---

### Cross-Entropy

Expected bits to encode from Q when true distribution is P.

```python
H(P, Q) = -Σ P(xᵢ) log Q(xᵢ)

# Example: True P = [0.7, 0.2, 0.1], Model Q = [0.6, 0.3, 0.1]
H(P,Q) = -[0.7 log(0.6) + 0.2 log(0.3) + 0.1 log(0.1)]
       = 0.36 + 0.24 + 0.23 = 0.83

# If Q = P: H(P,P) = H(P) = 0.80 (minimum)
```

**ML Application:** Classification loss function—cross-entropy loss.

---

### KL Divergence

How much Q differs from P (asymmetric).

```python
D_KL(P || Q) = Σ P(xᵢ) log[P(xᵢ)/Q(xᵢ)]
             = H(P, Q) - H(P)

# Always ≥ 0, equals 0 iff P = Q

# Example: P = [0.7, 0.3], Q = [0.5, 0.5]
D_KL = 0.7 log(0.7/0.5) + 0.3 log(0.3/0.5)
     = 0.7(0.336) + 0.3(-0.511)
     = 0.082

# Note: D_KL(P||Q) ≠ D_KL(Q||P)
```

**ML Application:** VAE loss, policy gradient (KL constraint), distribution matching.

---

## 8. Conjugate Priors (Quick Reference)

When prior and posterior are same family → closed-form updates.

| Likelihood | Prior | Posterior |
|------------|-------|-----------|
| Bernoulli/Binomial | Beta(α, β) | Beta(α + successes, β + failures) |
| Poisson | Gamma(α, β) | Gamma(α + Σxᵢ, β + n) |
| Normal (known σ²) | Normal(μ₀, σ₀²) | Normal(weighted mean, smaller variance) |
| Normal (known μ) | Inv-Gamma(α, β) | Inv-Gamma(α + n/2, β + Σ(xᵢ-μ)²/2) |
| Multinomial | Dirichlet(α) | Dirichlet(α + counts) |

```python
# Example: Beta-Binomial
Prior: Beta(2, 2)  # Mild belief in fairness
Data: 8 heads, 2 tails
Posterior: Beta(2+8, 2+2) = Beta(10, 4)
E[p] = 10/14 = 0.71
```

**ML Application:** Online learning, Bayesian updates, Thompson sampling.

---

## 9. Quick Reference Table

| Concept | Formula | ML Use |
|---------|---------|--------|
| E[X] | Σ xᵢ P(xᵢ) | Loss functions |
| Var(X) | E[X²] - (E[X])² | Uncertainty |
| Bayes | P(A\|B) = P(B\|A)P(A)/P(B) | All Bayesian ML |
| CLT | Σ Xᵢ → Normal | Batch statistics |
| MLE | argmax log P(data\|θ) | Training |
| Cross-entropy | -Σ P log Q | Classification loss |
| KL divergence | Σ P log(P/Q) | VAE, regularization |

---

## 10. Distribution Selection Guide

```
Is it discrete or continuous?

DISCRETE:
├── Binary outcome? → Bernoulli
├── Count of successes in n trials? → Binomial
├── Count of events in interval? → Poisson
├── Multiple categories, one trial? → Categorical
└── Multiple categories, n trials? → Multinomial

CONTINUOUS:
├── Unbounded, symmetric? → Normal
├── Strictly positive?
│   ├── Multiplicative process? → Log-Normal
│   ├── Time until event? → Exponential / Gamma
│   └── Rate or variance? → Gamma / Inv-Gamma
├── Bounded [0, 1]? → Beta
├── Heavy tails / outliers? → Student's t
└── No prior knowledge? → Uniform
```

---

*End of Cheatsheet*
