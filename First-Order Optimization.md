# 🎯 **First-Order Optimization**
### *A Sequential Walkthrough from SGD to Adam*

---

## 📋 Setup: Data & Goal

| | |
|---|---|
|**Data Points**|Point 1: (2,10), Point 2: (1,5), Point 3: (3,15), Point 4: (0.5,2.5)|
|**Starting Weight**|w = 3|
|**True Weight**|w = 5 (our target)|
|**Loss Function**|L = (prediction - y_true)²|
|**Gradient Formula**|∂L/∂w = 2 × (prediction - y_true) × x|

---

## 🏗️ **The Optimizer Family Tree**

Understanding how these optimizers evolved helps clarify what each one "inherited" and what it added:

```
                        SGD
                         │
                         │ + velocity accumulation
                         ▼
                     Momentum
                         │
                         │
         ┌───────────────┴───────────────┐
         │                               │
         │ + per-parameter LR            │ + decaying memory
         │   (but accumulates forever)   │   (fixes the freezing)
         ▼                               ▼
      AdaGrad ─────────────────────► RMSprop
         │                               │
         │                               │
         └───────────┬───────────────────┘
                     │
                     │ + momentum (from left branch)
                     │ + adaptive LR (from right branch)
                     │ + bias correction
                     ▼
                    Adam
```

**Reading the tree:**

| Optimizer | What It Inherited | What It Added |
|-----------|-------------------|---------------|
| **SGD** | - | Baseline: η × gradient |
| **Momentum** | SGD's update | Velocity accumulation (β) |
| **AdaGrad** | SGD's update | Per-parameter learning rate via accumulated G |
| **RMSprop** | AdaGrad's adaptive LR | Decaying memory (ρ) to prevent freezing |
| **Adam** | Momentum's velocity + RMSprop's adaptive LR | Bias correction for early timesteps |

**The key insight**: Adam isn't magic - it's the logical endpoint of combining two independent improvements (momentum and adaptive learning rates) with a fix for initialization bias.

---

## 🔧 **Parameter Guide: What Are These Numbers?**

Parameters fall into two categories:

**🎛️ Hyperparameters** = Values YOU choose before training (tunable)  
**📊 State Variables** = Values the algorithm computes and updates during training

---

### **Method 1: SGD (Stochastic Gradient Descent)**

| Parameter | Type | Value | What It Represents |
|-----------|------|-------|-------------------|
| **η** (eta) | 🎛️ Hyperparameter | 0.1 | **Learning rate** - How big a step to take. Too high → overshoot. Too low → slow learning. Typical range: 0.001 to 0.1 |

**Update Rule**: `w_new = w_old - η × gradient`

*SGD is the simplest - just one knob to turn!*

---

### **Method 2: Momentum**

| Parameter | Type | Value | What It Represents |
|-----------|------|-------|-------------------|
| **η** | 🎛️ Hyperparameter | 0.1 | Learning rate |
| **β** (beta) | 🎛️ Hyperparameter | 0.9 | **Momentum coefficient** - How much "memory" of past gradients to keep. 0.9 means 90% of previous velocity carries forward. Range: 0.5 to 0.99 |
| **v** | 📊 State Variable | starts at 0 | **Velocity** - Accumulated momentum from past gradients. Think of it as the "speed" the optimizer has built up |

**Update Rules**:
```
v_new = β × v_old + gradient          ← Accumulate momentum
w_new = w_old - η × v_new              ← Apply velocity to weights
```

**Intuition**: Like a ball rolling downhill - it builds speed and can roll through small bumps (noisy gradients), but might overshoot valleys.

---

### **Method 3: AdaGrad (Adaptive Gradient)**

| Parameter | Type | Value | What It Represents |
|-----------|------|-------|-------------------|
| **η** | 🎛️ Hyperparameter | 0.1 | Base learning rate (gets divided by √G) |
| **ε** (epsilon) | 🎛️ Hyperparameter | 1e-8 | **Numerical stability constant** - Prevents division by zero. Always tiny (10⁻⁸) |
| **G** | 📊 State Variable | starts at 0 | **Accumulated squared gradients** - Sum of ALL squared gradients seen so far. Only grows, never shrinks! |

**Update Rules**:
```
G_new = G_old + gradient²              ← Accumulate squared gradients (forever!)
effective_lr = η / √(G + ε)            ← Learning rate shrinks as G grows
w_new = w_old - effective_lr × gradient
```

**Intuition**: Parameters that get large gradients frequently get smaller learning rates. Good for sparse data, but G never stops growing → learning eventually freezes.

---

### **Method 4: RMSprop (Root Mean Square Propagation)**

| Parameter | Type | Value | What It Represents |
|-----------|------|-------|-------------------|
| **η** | 🎛️ Hyperparameter | 0.1 | Base learning rate |
| **ρ** (rho) | 🎛️ Hyperparameter | 0.9 | **Decay rate** - How quickly to "forget" old gradients. 0.9 keeps 90% of old average, adds 10% of new. Range: 0.9 to 0.99 |
| **ε** | 🎛️ Hyperparameter | 1e-8 | Numerical stability constant |
| **E[g²]** | 📊 State Variable | starts at 0 | **Exponential moving average of squared gradients** - Unlike AdaGrad's G, this DECAYS over time |

**Update Rules**:
```
E[g²]_new = ρ × E[g²]_old + (1-ρ) × gradient²   ← Decaying average
effective_lr = η / √(E[g²] + ε)
w_new = w_old - effective_lr × gradient
```

**Intuition**: Fixes AdaGrad's freezing problem by using a "leaky" memory. Old large gradients fade away, so learning rate can recover.

**Why ρ = 0.9?** This means the effective "window" of memory is about 10 gradients. After ~10 updates, old gradients have faded to <35% influence.

---

### **Method 5: Adam (Adaptive Moment Estimation)**

| Parameter | Type | Value | What It Represents |
|-----------|------|-------|-------------------|
| **η** | 🎛️ Hyperparameter | 0.1 | Base learning rate |
| **β₁** | 🎛️ Hyperparameter | 0.9 | **First moment decay** - Controls momentum-like behavior. Same as RMSprop's ρ but for the gradient itself |
| **β₂** | 🎛️ Hyperparameter | 0.999 | **Second moment decay** - Controls gradient magnitude scaling. Higher than β₁ = longer memory for variance |
| **ε** | 🎛️ Hyperparameter | 1e-8 | Numerical stability constant |
| **m** | 📊 State Variable | starts at 0 | **First moment (mean)** - Exponential moving average of gradients (like momentum's velocity) |
| **v** | 📊 State Variable | starts at 0 | **Second moment (uncentered variance)** - Exponential moving average of squared gradients (like RMSprop's E[g²]) |
| **t** | 📊 State Variable | starts at 0 | **Timestep counter** - Counts how many updates we've done. Used for bias correction |

**Update Rules**:
```
t = t + 1                                        ← Increment timestep

m_new = β₁ × m_old + (1-β₁) × gradient          ← Update first moment (direction)
v_new = β₂ × v_old + (1-β₂) × gradient²         ← Update second moment (magnitude)

m̂ = m_new / (1 - β₁ᵗ)                           ← Bias-corrected first moment
v̂ = v_new / (1 - β₂ᵗ)                           ← Bias-corrected second moment

w_new = w_old - η × m̂ / √(v̂ + ε)               ← Final update
```

**Why bias correction?** At t=1, if β₁=0.9:
- `m = 0.9 × 0 + 0.1 × gradient = 0.1 × gradient` (way too small!)
- Dividing by `(1 - 0.9¹) = 0.1` recovers the full gradient magnitude

As t → ∞, `(1 - β₁ᵗ) → 1`, so correction fades away.

**Why β₂ = 0.999 > β₁ = 0.9?** We want the variance estimate to be more stable (longer memory) than the momentum (can respond faster to direction changes).

---

## 📊 **Parameter Summary Table**

| Method | Hyperparameters | State Variables | Key Idea |
|--------|-----------------|-----------------|----------|
| **SGD** | η | none | Simple scaled step |
| **Momentum** | η, β | v (velocity) | Build up speed |
| **AdaGrad** | η, ε | G (sum of g²) | Adapt LR per-parameter, but freezes |
| **RMSprop** | η, ρ, ε | E[g²] (decaying avg of g²) | AdaGrad + forgetting |
| **Adam** | η, β₁, β₂, ε | m, v, t | Momentum + RMSprop + bias correction |

---

## 🎯 **Standard Default Values**

These values work well in ~80% of cases:

| Hyperparameter | Default | When to Change |
|----------------|---------|----------------|
| **η** (learning rate) | 0.001 | Increase if learning too slow, decrease if unstable |
| **β** (momentum) | 0.9 | Lower (0.5-0.8) if oscillating too much |
| **ρ** (RMSprop decay) | 0.9 | Rarely changed |
| **β₁** (Adam momentum) | 0.9 | Rarely changed |
| **β₂** (Adam variance) | 0.999 | Rarely changed |
| **ε** | 1e-8 | Almost never changed |

---

Now that we understand where every parameter comes from, let's see them in action!

---

## 📍 **POINT 1: (x=2, y=10)**

### **Method 1: SGD**

```
Parameters used: η = 0.1

Current w: 3
Prediction: w × x = 3 × 2 = 6
Error: prediction - y = 6 - 10 = -4
Gradient: 2 × error × x = 2 × (-4) × 2 = -16

Update: w = w - η × gradient
        w = 3 - (0.1 × -16) = 3 + 1.6 = 4.6
```

✅ **Result**: w = **4.6**

---

### **Method 2: Momentum**

```
Parameters used: η = 0.1, β = 0.9
State variables: v = 0 (initialized)

Current w: 3
Prediction: 3 × 2 = 6
Error: 6 - 10 = -4
Gradient: 2 × (-4) × 2 = -16

Velocity Update: v = β × v_old + gradient
                 v = 0.9 × 0 + (-16) = -16

Weight Update: w = w - η × v
               w = 3 - (0.1 × -16) = 4.6
```

✅ **Result**: w = **4.6**, v = **-16**

*Same as SGD on first step since v started at 0!*

---

### **Method 3: AdaGrad**

```
Parameters used: η = 0.1, ε = 1e-8
State variables: G = 0 (initialized)

Current w: 3
Prediction: 3 × 2 = 6
Error: 6 - 10 = -4
Gradient: 2 × (-4) × 2 = -16

G Update: G = G_old + gradient²
          G = 0 + (-16)² = 256

Effective LR: η / √(G + ε) = 0.1 / √256 = 0.1 / 16 = 0.00625

Weight Update: w = w - effective_lr × gradient
               w = 3 - (0.00625 × -16) = 3 + 0.1 = 3.1
```

✅ **Result**: w = **3.1**, G = **256**

*Much smaller step because gradient was large → G is large → LR shrinks!*

---

### **Method 4: RMSprop**

```
Parameters used: η = 0.1, ρ = 0.9, ε = 1e-8
State variables: E[g²] = 0 (initialized)

Current w: 3
Prediction: 3 × 2 = 6
Error: 6 - 10 = -4
Gradient: 2 × (-4) × 2 = -16

E[g²] Update: E[g²] = ρ × E[g²]_old + (1-ρ) × gradient²
              E[g²] = 0.9 × 0 + 0.1 × (-16)² = 0 + 25.6 = 25.6

Effective LR: η / √(E[g²] + ε) = 0.1 / √25.6 = 0.1 / 5.06 ≈ 0.0198

Weight Update: w = 3 - (0.0198 × -16) = 3 + 0.317 ≈ 3.317
```

✅ **Result**: w = **3.317**, E[g²] = **25.6**

*Bigger step than AdaGrad! E[g²] = 25.6 vs G = 256 (only kept 10% of squared gradient)*

---

### **Method 5: Adam**

```
Parameters used: η = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
State variables: m = 0, v = 0, t = 0 (initialized)

Current w: 3
t = 0 + 1 = 1 (increment timestep)

Prediction: 3 × 2 = 6
Error: 6 - 10 = -4
Gradient: 2 × (-4) × 2 = -16

First Moment: m = β₁ × m_old + (1-β₁) × gradient
              m = 0.9 × 0 + 0.1 × (-16) = -1.6

Second Moment: v = β₂ × v_old + (1-β₂) × gradient²
               v = 0.999 × 0 + 0.001 × (-16)² = 0.256

Bias Correction (crucial at early timesteps!):
  m̂ = m / (1 - β₁ᵗ) = -1.6 / (1 - 0.9¹) = -1.6 / 0.1 = -16
  v̂ = v / (1 - β₂ᵗ) = 0.256 / (1 - 0.999¹) = 0.256 / 0.001 = 256

Weight Update: w = w - η × m̂ / √(v̂ + ε)
               w = 3 - 0.1 × (-16) / √256 
               w = 3 + 1.6 / 16 = 3.1
```

✅ **Result**: w = **3.1**, m = **-1.6**, v = **0.256**, t = **1**

*Bias correction scaled m from -1.6 back to -16 and v from 0.256 to 256!*

---

## 📍 **POINT 2: (x=1, y=5)**

### **Method 1: SGD**

```
Parameters: η = 0.1
Current w: 4.6

Prediction: 4.6 × 1 = 4.6
Error: 4.6 - 5 = -0.4
Gradient: 2 × (-0.4) × 1 = -0.8

Update: w = 4.6 - (0.1 × -0.8) = 4.6 + 0.08 = 4.68
```

✅ **Result**: w = **4.68**

---

### **Method 2: Momentum**

```
Parameters: η = 0.1, β = 0.9
State: v = -16 (from Point 1)
Current w: 4.6

Prediction: 4.6 × 1 = 4.6
Error: 4.6 - 5 = -0.4
Gradient: 2 × (-0.4) × 1 = -0.8

Velocity: v = 0.9 × (-16) + (-0.8) = -14.4 - 0.8 = -15.2
          ↑ 90% of old momentum carried forward!

Weight: w = 4.6 - (0.1 × -15.2) = 4.6 + 1.52 = 6.12
```

✅ **Result**: w = **6.12**, v = **-15.2**

*🚀 OVERSHOOT! The old velocity (-16) dominated the tiny new gradient (-0.8)*

💡 **Why Nesterov Momentum (NAG) exists**: The overshoot you see in Method 2 happens because standard Momentum is "blind"—it calculates the gradient at the current position, adds it to the old velocity, and then leaps forward. By the time it realizes it has passed the target, the momentum is already too high to stop instantly.

Nesterov Accelerated Gradient (NAG) fixes this by changing the order of operations. It performs the "jump" first using the existing velocity, then calculates the gradient at that new "look-ahead" position. If that jump went too far, the gradient at the new spot will immediately point back, acting as an early brake before the weight update is finalized. In code, this is usually just a flag within the SGD optimizer: torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)

---

### **Method 3: AdaGrad**

```
Parameters: η = 0.1, ε = 1e-8
State: G = 256 (from Point 1)
Current w: 3.1

Prediction: 3.1 × 1 = 3.1
Error: 3.1 - 5 = -1.9
Gradient: 2 × (-1.9) × 1 = -3.8

G Update: G = 256 + (-3.8)² = 256 + 14.44 = 270.44
          ↑ G only ever grows!

Effective LR: 0.1 / √270.44 = 0.1 / 16.45 ≈ 0.00608

Weight: w = 3.1 - (0.00608 × -3.8) = 3.1 + 0.023 ≈ 3.123
```

✅ **Result**: w = **3.123**, G = **270.44**

*Learning rate already dying: 0.00608 vs original 0.1*

---

### **Method 4: RMSprop**

```
Parameters: η = 0.1, ρ = 0.9, ε = 1e-8
State: E[g²] = 25.6 (from Point 1)
Current w: 3.317

Prediction: 3.317 × 1 = 3.317
Error: 3.317 - 5 = -1.683
Gradient: 2 × (-1.683) × 1 = -3.366

E[g²] Update: E[g²] = 0.9 × 25.6 + 0.1 × (-3.366)²
                     = 23.04 + 1.133 = 24.173
              ↑ Old value DECAYED by 0.9, only kept 90%

Effective LR: 0.1 / √24.173 ≈ 0.0203

Weight: w = 3.317 + (0.0203 × 3.366) = 3.317 + 0.068 ≈ 3.386
```

✅ **Result**: w = **3.386**, E[g²] = **24.173**

*E[g²] actually decreased (25.6 → 24.17)! LR can recover.*

---

### **Method 5: Adam**

```
Parameters: η = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
State: m = -1.6, v = 0.256, t = 1 (from Point 1)
Current w: 3.1

t = 2 (increment)

Prediction: 3.1 × 1 = 3.1
Error: 3.1 - 5 = -1.9
Gradient: 2 × (-1.9) × 1 = -3.8

First Moment: m = 0.9 × (-1.6) + 0.1 × (-3.8) = -1.44 - 0.38 = -1.82
Second Moment: v = 0.999 × 0.256 + 0.001 × (-3.8)² = 0.256 + 0.014 = 0.270

Bias Correction:
  m̂ = -1.82 / (1 - 0.9²) = -1.82 / 0.19 = -9.58
  v̂ = 0.270 / (1 - 0.999²) = 0.270 / 0.002 = 135

Weight: w = 3.1 - 0.1 × (-9.58) / √135 = 3.1 + 0.958/11.6 = 3.1 + 0.083 ≈ 3.183
```

✅ **Result**: w = **3.183**, m = **-1.82**, v = **0.270**, t = **2**

---

## 📍 **POINT 3: (x=3, y=15)**

### **Method 1: SGD**

```
Current w: 4.68

Prediction: 4.68 × 3 = 14.04
Error: 14.04 - 15 = -0.96
Gradient: 2 × (-0.96) × 3 = -5.76

Update: w = 4.68 + (0.1 × 5.76) = 4.68 + 0.576 = 5.256
```

✅ **Result**: w = **5.256**

---

### **Method 2: Momentum**

```
State: v = -15.2, Current w: 6.12

Prediction: 6.12 × 3 = 18.36
Error: 18.36 - 15 = +3.36 ← POSITIVE! We overshot!
Gradient: 2 × (3.36) × 3 = +20.16

Velocity: v = 0.9 × (-15.2) + 20.16 = -13.68 + 20.16 = +6.48
          ↑ Velocity REVERSED direction!

Weight: w = 6.12 - (0.1 × 6.48) = 6.12 - 0.648 = 5.472
```

✅ **Result**: w = **5.472**, v = **+6.48**

*Momentum is now pulling back toward target*

---

### **Method 3: AdaGrad**

```
State: G = 270.44, Current w: 3.123

Prediction: 3.123 × 3 = 9.369
Error: 9.369 - 15 = -5.631
Gradient: 2 × (-5.631) × 3 = -33.786

G Update: G = 270.44 + 1141.5 = 1411.94
          ↑ HUGE jump from one big gradient!

Effective LR: 0.1 / √1411.94 = 0.1 / 37.58 ≈ 0.00266

Weight: w = 3.123 + (0.00266 × 33.786) = 3.123 + 0.090 ≈ 3.213
```

✅ **Result**: w = **3.213**, G = **1411.94**

*Learning rate is now 2.7% of original! Nearly frozen.*

---

### **Method 4: RMSprop**

```
State: E[g²] = 24.173, Current w: 3.386

Prediction: 3.386 × 3 = 10.158
Error: 10.158 - 15 = -4.842
Gradient: 2 × (-4.842) × 3 = -29.052

E[g²] Update: E[g²] = 0.9 × 24.173 + 0.1 × 844 = 21.76 + 84.4 = 106.16

Effective LR: 0.1 / √106.16 ≈ 0.0097

Weight: w = 3.386 + (0.0097 × 29.052) = 3.386 + 0.282 ≈ 3.668
```

✅ **Result**: w = **3.668**, E[g²] = **106.16**

---

### **Method 5: Adam**

```
State: m = -1.82, v = 0.270, t = 2
Current w: 3.183

t = 3

Prediction: 3.183 × 3 = 9.549
Error: 9.549 - 15 = -5.451
Gradient: 2 × (-5.451) × 3 = -32.706

First Moment: m = 0.9 × (-1.82) + 0.1 × (-32.706) = -1.64 - 3.27 = -4.91
Second Moment: v = 0.999 × 0.270 + 0.001 × 1069.7 = 0.270 + 1.07 = 1.34

Bias Correction:
  m̂ = -4.91 / (1 - 0.729) = -4.91 / 0.271 = -18.12
  v̂ = 1.34 / (1 - 0.997) = 1.34 / 0.003 = 446.7

Weight: w = 3.183 - 0.1 × (-18.12) / √446.7 = 3.183 + 1.81/21.1 = 3.183 + 0.086 ≈ 3.27
```

✅ **Result**: w = **3.27**, m = **-4.91**, v = **1.34**, t = **3**

---

## 📍 **POINT 4: (x=0.5, y=2.5)**

### **Method 1: SGD**

```
Current w: 5.256

Prediction: 5.256 × 0.5 = 2.628
Error: 2.628 - 2.5 = +0.128 ← Slightly above target
Gradient: 2 × (0.128) × 0.5 = +0.128

Update: w = 5.256 - (0.1 × 0.128) = 5.256 - 0.0128 = 5.243
```

✅ **Final w = 5.243** ✨

---

### **Method 2: Momentum**

```
State: v = +6.48, Current w: 5.472

Prediction: 5.472 × 0.5 = 2.736
Error: 2.736 - 2.5 = +0.236
Gradient: 2 × (0.236) × 0.5 = +0.236

Velocity: v = 0.9 × 6.48 + 0.236 = 5.83 + 0.24 = 6.07
          ↑ Still pushing w DOWN (positive v → subtract from w)

Weight: w = 5.472 - (0.1 × 6.07) = 5.472 - 0.607 = 4.865
```

✅ **Final w = 4.865** ✨

*Undershot! Momentum kept pushing past 5*

---

### **Method 3: AdaGrad**

```
State: G = 1411.94, Current w: 3.213

Prediction: 3.213 × 0.5 = 1.607
Error: 1.607 - 2.5 = -0.893
Gradient: 2 × (-0.893) × 0.5 = -0.893

G Update: G = 1411.94 + 0.80 = 1412.74

Effective LR: 0.1 / √1412.74 = 0.00266

Weight: w = 3.213 + (0.00266 × 0.893) = 3.213 + 0.0024 ≈ 3.216
```

✅ **Final w = 3.216** 😢

*Moved only 0.003 - completely frozen!*

---

### **Method 4: RMSprop**

```
State: E[g²] = 106.16, Current w: 3.668

Prediction: 3.668 × 0.5 = 1.834
Error: 1.834 - 2.5 = -0.666
Gradient: 2 × (-0.666) × 0.5 = -0.666

E[g²] Update: E[g²] = 0.9 × 106.16 + 0.1 × 0.44 = 95.5 + 0.04 = 95.59
              ↑ Decayed! LR will recover over time

Effective LR: 0.1 / √95.59 ≈ 0.0102

Weight: w = 3.668 + (0.0102 × 0.666) = 3.668 + 0.0068 ≈ 3.675
```

✅ **Final w = 3.675**

---

### **Method 5: Adam**

```
State: m = -4.91, v = 1.34, t = 3
Current w: 3.27

t = 4

Prediction: 3.27 × 0.5 = 1.635
Error: 1.635 - 2.5 = -0.865
Gradient: 2 × (-0.865) × 0.5 = -0.865

First Moment: m = 0.9 × (-4.91) + 0.1 × (-0.865) = -4.42 - 0.09 = -4.51
Second Moment: v = 0.999 × 1.34 + 0.001 × 0.75 = 1.34 + 0.001 = 1.34

Bias Correction:
  m̂ = -4.51 / (1 - 0.6561) = -4.51 / 0.344 = -13.11
  v̂ = 1.34 / (1 - 0.996) = 1.34 / 0.004 = 335

Weight: w = 3.27 - 0.1 × (-13.11) / √335 = 3.27 + 1.31/18.3 = 3.27 + 0.072 ≈ 3.34
```

✅ **Final w = 3.34**

---

## 🏆 **Final Scoreboard After 1 Epoch**

| Method | Final w | Distance from 5 | State Variable Status | Key Behavior |
|--------|---------|-----------------|----------------------|--------------|
| **SGD** | 5.243 | 0.243 | N/A | Reacted instantly to each point |
| **Momentum** | 4.865 | 0.135 | v = +6.07 | Still "bleeding off" speed from early push |
| **AdaGrad** | 3.216 | 1.784 | G = 1412.74 | The huge early gradient (-16) "broke" the engine |
| **RMSprop** | 3.675 | 1.325 | E[g²] = 95.59 | The "leaky memory" started forgetting old gradients |
| **Adam** | 3.34 | 1.66 | t = 4 | Combination of direction memory and variance safety |

---

## 💡 **Key Insights from Parameter Behavior**

1. **SGD**: η alone controls everything - simple but no adaptation

2. **Momentum**: v accumulates history → can overshoot when gradient reverses. β=0.9 means 90% of old velocity persists

3. **AdaGrad**: G only grows → LR only shrinks → eventually frozen. The big gradient (-16) at Point 1 permanently damaged the learning rate

4. **RMSprop**: E[g²] decays by ρ=0.9 each step → old gradients fade → LR can recover

5. **Adam**: Bias correction is crucial early (t=1,2,3...). Without it, m and v would be too small and updates would be wrong

---

## 🔥 **Learning Rate Warmup: When Bias Correction Isn't Enough**

Even with Adam's bias correction, the first few updates can be unstable. Why? The correction fixes the *magnitude* of m and v, but these early estimates are still based on very few gradients - they're noisy and unreliable.

**Warmup** addresses this by starting with a tiny learning rate and gradually increasing it:

```python
# PyTorch example with linear warmup
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=500)  # 0.0001 → 0.001
main = ConstantLR(optimizer, factor=1.0)

scheduler = SequentialLR(optimizer, [warmup, main], milestones=[500])
```

**When to use warmup:**
- Large models (transformers, deep CNNs)
- Large batch sizes (>512)
- When training explodes in the first few hundred steps

**Typical warmup length**: 1-5% of total training steps, or 500-2000 steps for large models.

---

## 🔄 **What Happens in Epoch 2?**

**Everything carries forward, only the data cycles restart.**

At the start of Epoch 2, each optimizer keeps exactly where it left off:
- **Weights**: SGD continues from w=5.243, Momentum from w=4.865, AdaGrad from w=3.216, etc.
- **State variables**: Momentum keeps its velocity (v=6.07), AdaGrad keeps its accumulated G=1412.74 (this is why it's frozen!), RMSprop keeps E[g²]=95.59, and Adam keeps m, v, and increments t to 5, 6, 7, 8...

**What about data order?** This is a design choice:
- **With reshuffling** (common in practice): The 4 points are randomly reordered, say (3,15), (0.5,2.5), (2,10), (1,5). This adds beneficial noise and helps escape local minima.
- **Without reshuffling**: Same order as Epoch 1. Simpler but can create repetitive update patterns.

**The key insight**: Epoch 2 isn't a "fresh start" - it's a continuation. AdaGrad's G will keep growing (making it more frozen), while Adam's bias correction terms (1-β₁ᵗ) and (1-β₂ᵗ) approach 1 as t increases, meaning the corrections fade and Adam starts taking bigger, more confident steps. This is why Adam typically dominates after several epochs - it was being deliberately cautious early on.

---

## 🔀 **Controlling Data Shuffling Across Epochs**

Shuffling data between epochs helps prevent the model from learning order-dependent patterns and aids escape from local minima. All major ML libraries provide explicit control over this behavior.

---

### **PyTorch (DataLoader)**

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True  # ← Reshuffles at the start of each epoch
)
```

**For reproducible shuffling**, use a seeded generator:

```python
generator = torch.Generator().manual_seed(42)
loader = DataLoader(
    dataset, 
    batch_size=32,
    shuffle=True, 
    generator=generator  # ← Same shuffle order every run
)
```

---

### **scikit-learn (SGDClassifier, SGDRegressor)**

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    shuffle=True,      # ← Default is True, reshuffles each epoch
    random_state=42    # ← Fixes shuffle order for reproducibility
)
```

The `random_state` parameter works across most scikit-learn estimators:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42  # ← Same split every run
)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42  # ← Same trees every run
)
```

---

### **Keras/TensorFlow (model.fit)**

```python
model.fit(
    X, y,
    epochs=10,
    shuffle=True  # ← Default is True
)
```

**For reproducibility**, set a global seed:

```python
import tensorflow as tf
tf.random.set_seed(42)  # ← Fixes all TensorFlow randomness

model.fit(X, y, epochs=10, shuffle=True)
```

---

### **Quick Reference Table**

| Library | Shuffle Parameter | Reproducibility Parameter | Scope |
|---------|-------------------|---------------------------|-------|
| **PyTorch** | `shuffle=True` | `generator=torch.Generator().manual_seed(42)` | Per DataLoader |
| **scikit-learn** | `shuffle=True` | `random_state=42` | Per estimator |
| **TensorFlow/Keras** | `shuffle=True` | `tf.random.set_seed(42)` | Global |

---

### **When to Turn Shuffle Off**

| Scenario | Reason |
|----------|--------|
| Time series forecasting | Temporal order carries predictive information |
| Sequence-to-sequence models | Input sequences must maintain internal order |
| Debugging | Need identical behavior across runs to isolate issues |
| Pre-shuffled data | Already randomized externally, want consistent batches |

---

## 📦 **Mini-Batch vs SGD: How Batch Size Affects Training**

The examples in the previous sections used batch size = 1 (true SGD), where each data point triggers its own update. In practice, we often use **mini-batches** where multiple points are processed together. This fundamentally changes how gradients and state variables evolve.

---

### **Setup**

Using our same 4 data points with w=3, η=0.1:

| Batch Size | Batches per Epoch | Points per Batch |
|------------|-------------------|------------------|
| 1 (SGD) | 4 | Single point each |
| 2 (Mini-batch) | 2 | Batch 1: (2,10), (1,5) / Batch 2: (3,15), (0.5,2.5) |
| 4 (Full batch) | 1 | All points together |

---

### **SGD (Batch Size = 1): Sequential Updates**

Each point updates the weight immediately before the next point is processed:

```
Point 1 (2,10):    gradient = -16   →  w = 3 + 1.6 = 4.6
Point 2 (1,5):     gradient = -0.8  →  w = 4.6 + 0.08 = 4.68
Point 3 (3,15):    gradient = -5.76 →  w = 4.68 + 0.576 = 5.256
Point 4 (0.5,2.5): gradient = +0.128 → w = 5.256 - 0.013 = 5.243
```

**4 updates per epoch**, each computed at a different w

---

### **Mini-Batch (Batch Size = 2): Averaged Updates**

All points in a batch are evaluated at the **same weight**, then gradients are averaged:

```
BATCH 1: Points (2,10) and (1,5) evaluated at w=3
├── Point (2,10): prediction = 3×2 = 6, error = -4, gradient = -16
├── Point (1,5):  prediction = 3×1 = 3, error = -2, gradient = -4
├── Average gradient = (-16 + -4) / 2 = -10
└── Update: w = 3 - (0.1 × -10) = 3 + 1.0 = 4.0

BATCH 2: Points (3,15) and (0.5,2.5) evaluated at w=4.0
├── Point (3,15):    prediction = 4×3 = 12, error = -3, gradient = -18
├── Point (0.5,2.5): prediction = 4×0.5 = 2, error = -0.5, gradient = -0.5
├── Average gradient = (-18 + -0.5) / 2 = -9.25
└── Update: w = 4.0 - (0.1 × -9.25) = 4.0 + 0.925 = 4.925
```

**2 updates per epoch**, each using averaged gradient from 2 points

---

### **Full Batch (Batch Size = 4): Single Update**

All points evaluated at the same weight, one update per epoch:

```
All 4 points evaluated at w=3:
├── (2,10):     gradient = -16
├── (1,5):      gradient = -4
├── (3,15):     gradient = -36
├── (0.5,2.5):  gradient = -2
├── Average = (-16 - 4 - 36 - 2) / 4 = -14.5
└── Update: w = 3 + 1.45 = 4.45
```

**1 update per epoch** - smoothest but slowest learning

---

### **Comparison: Weight Updates**

| | SGD (batch=1) | Mini-batch (batch=2) | Full batch (batch=4) |
|---|---|---|---|
| **Updates per epoch** | 4 | 2 | 1 |
| **Final w after epoch** | 5.243 | 4.925 | 4.45 |
| **Gradient noise** | High | Medium | Low |
| **Parallelization** | None | Partial | Full |

---

### **The Key Insight**

With mini-batch, **all points in the batch are evaluated at the same weight**. Point 2's gradient is computed at w=3, not at w=4.6 like in SGD. This means:

1. **Gradients are "stale"** - computed before the weight could learn from other points in the batch
2. **But they're averaged** - which smooths out noise and gives a more stable direction
3. **Fewer updates** - but each update is more informed (sees more data)

---

## ⚡ **Why Mini-Batch is Faster: Averaging Before Updating**

The mini-batch approach isn't just about noise reduction - it's fundamentally about **parallelization**.

**The key insight**: In a mini-batch, all gradients are computed at the *same* weight value. This means they're **independent calculations** that can run simultaneously.

```
SGD (Batch Size = 1): Sequential - Must Wait
──────────────────────────────────────────────
Time 1: Compute gradient at w=3      → Update to w=4.6
Time 2: Compute gradient at w=4.6    → Update to w=4.68  ← Must wait for Time 1!
Time 3: Compute gradient at w=4.68   → Update to w=5.256 ← Must wait for Time 2!
Time 4: Compute gradient at w=5.256  → Update to w=5.243 ← Must wait for Time 3!

Total: 4 sequential operations
```

```
Mini-Batch (Batch Size = 4): Parallel - No Waiting
──────────────────────────────────────────────
Time 1: Compute ALL gradients at w=3 simultaneously
        ├── GPU Core 1: gradient for (2,10)    = -16
        ├── GPU Core 2: gradient for (1,5)     = -4
        ├── GPU Core 3: gradient for (3,15)    = -36
        └── GPU Core 4: gradient for (0.5,2.5) = -2
        
Time 2: Average gradients → Single update

Total: 2 operations (but Time 1 uses parallel hardware)
```

**Why this matters for GPUs**: A modern GPU has thousands of cores. With batch size = 1, you're using *one* core while thousands sit idle. With batch size = 256, you're using 256 cores simultaneously.

---

### **The Trade-off Equation**

| Factor | Larger Batch | Smaller Batch |
|--------|--------------|---------------|
| GPU utilization | ✅ High | ❌ Low (cores idle) |
| Updates per epoch | ❌ Fewer | ✅ More |
| Gradient noise | ❌ Lower (can hurt generalization) | ✅ Higher (acts as regularizer) |
| Memory required | ❌ More | ✅ Less |

---

### **The "Averaging Before Squaring" Bonus for AdaGrad/Adam**

Beyond parallelization, averaging has a mathematical benefit for adaptive optimizers:

```
Individual gradients:     -16, -4, -36, -2

SGD-style (square then sum):
G = (-16)² + (-4)² + (-36)² + (-2)² = 256 + 16 + 1296 + 4 = 1572

Mini-batch (average then square):
G = ((-16 + -4 + -36 + -2) / 4)² = (-14.5)² = 210.25
```

**G grows 7.5× slower with full-batch!** This is because:

1. Averaging smooths out extreme gradients before they get squared
2. Squaring amplifies differences - a gradient of -36 contributes 1296 individually but only ~210 when averaged with others
3. Mathematically: E[X²] ≥ E[X]² (Jensen's inequality)

This is why AdaGrad can survive longer with larger batches, and why Adam's v (second moment) stays more stable with mini-batches than with pure SGD.

---

## 🔧 **How State Variables Behave in Mini-Batch Mode**

**The rule is simple**: Whatever gradient you use for the weight update, you use that same (averaged) gradient for all state variable updates. State variables update **once per batch**, not once per sample.

---

### **Momentum with Batch Size = 2**

```
BATCH 1: Points (2,10) and (1,5) at w=3
├── Gradients: -16 and -4
├── Averaged gradient: g = -10
│
├── Velocity: v = 0.9 × 0 + (-10) = -10      ← uses averaged g
└── Weight:   w = 3 - 0.1 × (-10) = 4.0

BATCH 2: Points (3,15) and (0.5,2.5) at w=4.0
├── Gradients: -18 and -0.5
├── Averaged gradient: g = -9.25
│
├── Velocity: v = 0.9 × (-10) + (-9.25) = -18.25    ← carries forward
└── Weight:   w = 4.0 - 0.1 × (-18.25) = 5.825
```

| | SGD-style (batch=1) | Mini-batch (batch=2) |
|---|---|---|
| Final velocity | v = 6.07 | v = -18.25 |
| Final weight | w = 4.865 | w = 5.825 |

*Velocity builds differently because it sees averaged gradients less frequently*

---

### **AdaGrad with Batch Size = 2**

```
BATCH 1: Averaged gradient g = -10
├── G = 0 + (-10)² = 100                     ← squared averaged gradient
├── Effective LR = 0.1 / √100 = 0.01
└── w = 3 + 0.01 × 10 = 3.1

BATCH 2: Averaged gradient g = -9.25
├── G = 100 + (-9.25)² = 100 + 85.6 = 185.6
├── Effective LR = 0.1 / √185.6 = 0.0073
└── w = 3.1 + 0.0073 × 9.25 = 3.168
```

| | SGD-style (batch=1) | Mini-batch (batch=2) |
|---|---|---|
| Final G | 1412.74 (frozen!) | 185.6 (healthy) |
| Final weight | w = 3.216 | w = 3.168 |

*AdaGrad survives longer with larger batches!*

---

### **Adam with Batch Size = 2**

```
BATCH 1: g = -10, t = 1
├── m = 0.9 × 0 + 0.1 × (-10) = -1.0
├── v = 0.999 × 0 + 0.001 × 100 = 0.1
├── m̂ = -1.0 / 0.1 = -10
├── v̂ = 0.1 / 0.001 = 100
└── w = 3 - 0.1 × (-10) / √100 = 3 + 0.1 = 3.1

BATCH 2: g = -9.25, t = 2
├── m = 0.9 × (-1.0) + 0.1 × (-9.25) = -1.825
├── v = 0.999 × 0.1 + 0.001 × 85.6 = 0.185
├── m̂ = -1.825 / 0.19 = -9.6
├── v̂ = 0.185 / 0.002 = 92.5
└── w = 3.1 - 0.1 × (-9.6) / √92.5 = 3.1 + 0.1 = 3.2
```

**Key difference**: t increments per batch, not per sample. With batch=1, t reaches 4 after one epoch; with batch=2, t only reaches 2. This means bias correction evolves more slowly with larger batches.

---

### **Summary: State Variables vs Batch Size**

| State Variable | SGD-style (batch=1) | Mini-batch (batch=2) |
|----------------|---------------------|----------------------|
| **Updates per epoch** | 4 | 2 |
| **Momentum v** | Accumulates 4 times | Accumulates 2 times |
| **AdaGrad G** | Sum of 4 squared gradients | Sum of 2 squared *averaged* gradients |
| **Adam t** | Reaches t=4 | Reaches t=2 |

---

## 💡 **Why Batch Size Affects AdaGrad's Survival**

**Averaging happens before squaring**, which has significant mathematical consequences:

```
SGD-style:     G = (-16)² + (-4)² = 256 + 16 = 272
Mini-batch:    G = ((-16 + -4)/2)² = (-10)² = 100
```

The squared averaged gradient (100) is much smaller than the sum of squared individual gradients (272). This is why **AdaGrad survives longer with larger batches** - G grows more slowly, so the learning rate decays more slowly.

This relates to a fundamental statistical property: **Var(mean) < mean(Var)**. Averaging smooths out extremes before they get squared and accumulated.

---

### **The Batch Size Trade-off Spectrum**

| Batch Size | Updates | Noise | Memory | GPU Utilization | Convergence |
|------------|---------|-------|--------|-----------------|-------------|
| **1 (SGD)** | Many | High | Low | Poor | Fast but noisy |
| **32-128 (typical)** | Medium | Medium | Medium | Good | Balanced |
| **Full dataset** | Few | None | High | Excellent | Slow but smooth |

**In practice**, batch sizes of 32-256 offer the best trade-off between noise reduction, computational efficiency, and convergence speed.

---

## 🔧 **Troubleshooting Table: Reading Your Loss Curve**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **Loss oscillates wildly** | Learning rate too high | Decrease η by 2-10× |
| **Loss decreases then explodes** | LR too high + momentum building up | Decrease η, or decrease β to 0.8 |
| **Loss decreases very slowly** | Learning rate too low | Increase η by 2-5× |
| **Loss plateaus early** | Stuck in local minimum or LR decayed too fast | Try SGD+Momentum instead of Adam, or increase η |
| **Loss plateaus, then suddenly drops** | Normal! Escaped a saddle point | Keep training |
| **Training loss drops, validation loss rises** | Overfitting | Add regularization, dropout, or early stopping |
| **Loss is NaN or Inf** | Numerical instability | Decrease η drastically, add gradient clipping, check for data issues |
| **AdaGrad stops learning mid-training** | G accumulated too large | Switch to RMSprop or Adam |
| **Adam converges worse than SGD** | Adam's adaptive LR hurts generalization | Try AdamW, or switch to SGD+Momentum for fine-tuning |

💡 **Note on AdamW**: While Adam is the most famous adaptive optimizer, modern practice—especially for Transformers—almost exclusively uses AdamW. The difference lies in how weight decay (L2 regularization) is handled. In standard Adam, the penalty for large weights is mixed into the gradient, and because Adam then scales that gradient by its moving average, the regularization effect becomes inconsistent. AdamW solves this by decoupling the two: it updates the state variables using only the loss gradient, then subtracts the weight decay penalty directly from the weights at the end of each step. This simple shift is why AdamW generalizes much better on complex datasets. In PyTorch: torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

---

### **Quick Decision Framework**

```
Loss unstable?     ──► Decrease η or β
Loss too slow?     ──► Increase η
Loss plateau?      ──► Try different optimizer or LR schedule
Loss NaN?          ──► Gradient clipping + check data + decrease η
Adam underperforms? ──► Try AdamW or SGD+Momentum for fine-tuning
```

---

### **Gradient Clipping: The Emergency Brake**

When gradients explode, clipping caps their magnitude:

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# TensorFlow/Keras
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

**When to use**: RNNs, transformers, or any model where loss suddenly becomes NaN.

---
