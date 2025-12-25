# Decision Trees (CART): A Mathematical Deep Dive

## Chapter Overview

Decision trees are fundamental supervised learning algorithms that partition the feature space through recursive binary splits. The CART (Classification and Regression Trees) algorithm, introduced by Breiman et al. (1984), provides a principled framework for building these models through greedy optimization of impurity measures.

-----

## 1. Binary Splits: The Foundation of CART

### Concept

Unlike some tree algorithms (e.g., ID3, C4.5) that allow multi-way splits, **CART exclusively uses binary splits**. At each node, the data is partitioned into exactly two child nodes based on a single decision rule.

### Mathematical Formulation

For a node $t$ with dataset $D_t$, a binary split creates:

- Left child: $D_L = {(x, y) \in D_t : x_j \leq \tau}$
- Right child: $D_R = {(x, y) \in D_t : x_j > \tau}$

where $j$ is the feature index and $\tau$ is the threshold value.

### Why Binary?

**Advantages:**

1. **Simplicity**: Easy to interpret and implement
1. **Universality**: Any multi-way split can be represented as sequential binary splits
1. **Computational efficiency**: $O(n \log n)$ per feature vs. $O(2^k)$ for k-way splits
1. **Geometric interpretation**: Creates axis-aligned hyperplane partitions

### Example: Credit Risk Assessment

Suppose you’re building a model to predict loan default using:

- Income (continuous)
- Debt-to-Income ratio (continuous)
- Credit score (continuous)

**Binary Split Example:**

```
Node: All applicants (n=1000)
├─ Credit Score ≤ 650? 
│  ├─ YES → High Risk Node (n=300)
│  └─ NO → Continue splitting (n=700)
```

**Not allowed in CART** (multi-way split):

```
Credit Score Category?
├─ Poor (< 580)
├─ Fair (580-669)
├─ Good (670-739)
└─ Excellent (≥ 740)
```

-----

## 2. Greedy Optimization: Myopic But Effective

### Concept

CART uses a **greedy algorithm**: at each step, it chooses the split that provides the maximum immediate reduction in impurity, without considering future splits.

### Mathematical Framework

At node $t$, CART solves:

$$\arg\max_{j, \tau} \Delta I(t, j, \tau) = I(t) - \left[\frac{|D_L|}{|D_t|} I(D_L) + \frac{|D_R|}{|D_t|} I(D_R)\right]$$

where:

- $I(t)$ is the impurity at node $t$
- $j \in {1, …, p}$ is the feature
- $\tau$ is the threshold
- $|D|$ denotes the number of samples

### Why Greedy?

**Computational Reality:**

- Optimal tree construction is NP-complete
- Greedy approach: $O(n \cdot p \cdot \log n)$ per level
- Optimal approach: $O(2^n)$ - computationally intractable

### Example: Portfolio Classification

You’re classifying strategies as “High Sharpe” (>1.5) or “Low Sharpe” (≤1.5) using volatility and maximum drawdown.

**Dataset at root:**

|Strategy|Volatility|Max DD|Sharpe|Class|
|--------|----------|------|------|-----|
|A       |12%       |-15%  |1.8   |High |
|B       |8%        |-8%   |1.2   |Low  |
|C       |15%       |-22%  |1.6   |High |
|D       |18%       |-25%  |1.1   |Low  |
|E       |10%       |-12%  |2.0   |High |

**Greedy Decision Process:**

*Step 1:* Evaluate all possible splits:

- Volatility ≤ 12%: $\Delta I = 0.013$
- Volatility ≤ 15%: $\Delta I = 0.067$
- Max DD ≤ -15%: $\Delta I = 0.080$ ← **Best immediate gain**
- Max DD ≤ -20%: $\Delta I = 0.040$

*Step 2:* Choose Max DD ≤ -15% because it maximizes immediate impurity reduction

**Key Limitation:** This might not lead to the globally optimal tree. Perhaps splitting first on Volatility, then on Max DD in subsequent levels, might yield better overall performance. But greedy search doesn’t explore this.

-----

## 3. Splitting Criteria: Gini vs Entropy

### Mathematical Definitions

#### Gini Impurity

For a node $t$ with $K$ classes:

$$G(t) = 1 - \sum_{k=1}^{K} p_k^2$$

#### Entropy (Information Gain)

$$H(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

where $p_k$ is the proportion of samples in class $k$ at node $t$.

### Intuition

**Gini Impurity** represents the **expected error rate** of randomly labeling a sample from the node according to the distribution of classes in the node.

**Entropy** measures the **average amount of information** (in bits) needed to identify the class of a sample from the node.

### Derivations

#### Why Gini?

If we randomly pick a sample and randomly assign it a label according to class distribution:

$$P(\text{error}) = \sum_{k=1}^{K} p_k \cdot (1 - p_k) = \sum_{k=1}^{K} p_k - \sum_{k=1}^{K} p_k^2 = 1 - \sum_{k=1}^{K} p_k^2$$

#### Why Entropy?

From information theory, the optimal encoding length for an event with probability $p_k$ is $-\log_2(p_k)$ bits. The expected encoding length across all classes is the entropy.

### Comparative Analysis

**Purity Range:**

- Gini: $[0, 1-1/K]$ for $K$ classes
- Entropy: $[0, \log_2(K)]$ for $K$ classes

For binary classification ($K=2$):

- Gini: $[0, 0.5]$
- Entropy: $[0, 1]$

### Numerical Comparison

Let’s compute both for different class distributions (binary case):

|$p_1$|$p_2$|Gini |Entropy|Difference|
|-----|-----|-----|-------|----------|
|1.0  |0.0  |0.000|0.000  |0.000     |
|0.9  |0.1  |0.180|0.469  |0.289     |
|0.8  |0.2  |0.320|0.722  |0.402     |
|0.7  |0.3  |0.420|0.881  |0.461     |
|0.6  |0.4  |0.480|0.971  |0.491     |
|0.5  |0.5  |0.500|1.000  |0.500     |

**Visualization (conceptual):**

```
Impurity
    │
1.0 ├─────────────────────╱╲ ← Entropy
    │                   ╱    ╲
0.8 │                 ╱        ╲
    │               ╱            ╲
0.6 │             ╱                ╲
    │          ╱─╲ ← Gini           ╲
0.4 │        ╱     ╲                  ╲
    │      ╱         ╲                  ╲
0.2 │    ╱             ╲                  ╲
    │  ╱                 ╲                  ╲
0.0 ├─────────────────────────────────────────
    0.0   0.2   0.4   0.5   0.6   0.8   1.0
                     p₁
```

### When to Use Which?

#### Use Gini When:

1. **Speed matters**: Computationally faster (no logarithm)

- Gini: 2 multiplications per class
- Entropy: 1 log operation per class (expensive)

1. **Default choice**: Works well in 95% of cases
1. **Balanced classes**: Differences are minimal

**Benchmark timing (1M evaluations):**

```python
Gini:    0.23 seconds
Entropy: 0.89 seconds
Speed ratio: 3.9x faster
```

#### Use Entropy When:

1. **Theoretical purity**: Slightly more “pure” splits near 50-50 distributions
1. **Information-theoretic framework**: When you need to explain results using information theory concepts
1. **Legacy compatibility**: Matching C4.5 or ID3 algorithm behavior

### Practical Example: Cryptocurrency Regime Classification

You’re classifying market regimes as “Bull” or “Bear” for BTC.

**Node A:** 60 Bull, 40 Bear (Total: 100)

**Gini:**
$$G(A) = 1 - \left(\frac{60}{100}\right)^2 - \left(\frac{40}{100}\right)^2 = 1 - 0.36 - 0.16 = 0.48$$

**Entropy:**
$$H(A) = -\frac{60}{100}\log_2\left(\frac{60}{100}\right) - \frac{40}{100}\log_2\left(\frac{40}{100}\right)$$
$$H(A) = -0.6 \times (-0.737) - 0.4 \times (-1.322) = 0.442 + 0.529 = 0.971$$

**After split on “RSI ≤ 50”:**

**Left child:** 55 Bull, 10 Bear (Total: 65)

**Gini:**
$$G(L) = 1 - \left(\frac{55}{65}\right)^2 - \left(\frac{10}{65}\right)^2 = 1 - 0.715 - 0.024 = 0.261$$

**Entropy:**
$$H(L) = -\frac{55}{65}\log_2\left(\frac{55}{65}\right) - \frac{10}{65}\log_2\left(\frac{10}{65}\right) = 0.616$$

**Right child:** 5 Bull, 30 Bear (Total: 35)

**Gini:**
$$G(R) = 1 - \left(\frac{5}{35}\right)^2 - \left(\frac{30}{35}\right)^2 = 1 - 0.020 - 0.735 = 0.245$$

**Entropy:**
$$H(R) = -\frac{5}{35}\log_2\left(\frac{5}{35}\right) - \frac{30}{35}\log_2\left(\frac{30}{35}\right) = 0.607$$

**Information Gain:**

**Gini:**
$$\Delta G = 0.48 - \left[\frac{65}{100} \times 0.261 + \frac{35}{100} \times 0.245\right] = 0.48 - 0.256 = 0.224$$

**Entropy:**
$$\Delta H = 0.971 - \left[\frac{65}{100} \times 0.616 + \frac{35}{100} \times 0.607\right] = 0.971 - 0.613 = 0.358$$

**Key Insight:** Both criteria select the same split! The ranking of splits is usually identical or very similar, making the choice largely academic.

### Empirical Evidence

**Research findings:**

- In practice, Gini and Entropy produce trees with <2% difference in accuracy
- Split rankings differ in <5% of cases
- **Recommendation**: Use Gini (default) unless you have specific reasons for Entropy

-----

## 4. Midpoints: Handling Continuous Features

### The Midpoint Strategy

For continuous features, CART considers splits at the **midpoint between consecutive unique sorted values**.

### Algorithm

Given feature $x_j$ with $n$ unique sorted values ${v_1, v_2, …, v_n}$:

1. Compute midpoints: $m_i = \frac{v_i + v_{i+1}}{2}$ for $i = 1, …, n-1$
1. Evaluate splits: $x_j \leq m_i$ for each $i$
1. Choose $m^* = \arg\max_i \Delta I(m_i)$

### Why Midpoints?

**Theorem:** For Gini impurity, the optimal threshold between any two consecutive values lies at their midpoint.

**Proof sketch:** The impurity function is piecewise constant between sorted values. The optimal split point that maximizes $\Delta I$ when transitioning from one value to the next occurs exactly at the midpoint.

### Computational Complexity

- Sorting: $O(n \log n)$
- Evaluating midpoints: $O(n-1) = O(n)$
- **Total per feature:** $O(n \log n)$
- **All features:** $O(p \cdot n \log n)$ where $p$ is number of features

### Example: Interest Rate Prediction

You’re predicting whether a bond trade will be profitable based on yield spread.

**Training data (sorted by spread):**

|Trade|Spread (bps)|Profitable?|
|-----|------------|-----------|
|1    |45          |No         |
|2    |62          |No         |
|3    |78          |Yes        |
|4    |95          |Yes        |
|5    |112         |Yes        |
|6    |138         |No         |

**Midpoint candidates:**

$$m_1 = \frac{45 + 62}{2} = 53.5 \text{ bps}$$

$$m_2 = \frac{62 + 78}{2} = 70.0 \text{ bps}$$

$$m_3 = \frac{78 + 95}{2} = 86.5 \text{ bps}$$

$$m_4 = \frac{95 + 112}{2} = 103.5 \text{ bps}$$

$$m_5 = \frac{112 + 138}{2} = 125.0 \text{ bps}$$

**Evaluating each split:**

Initial Gini: $G = 1 - (3/6)^2 - (3/6)^2 = 0.5$

*Split at 70 bps:*

- Left: 2 No, 0 Yes → $G_L = 0$
- Right: 1 No, 3 Yes → $G_R = 1 - (1/4)^2 - (3/4)^2 = 0.375$
- Weighted: $(2/6) \cdot 0 + (4/6) \cdot 0.375 = 0.25$
- $\Delta G = 0.5 - 0.25 = 0.25$ ← **Best split!**

*Split at 103.5 bps:*

- Left: 2 No, 2 Yes → $G_L = 0.5$
- Right: 1 No, 1 Yes → $G_R = 0.5$
- Weighted: $0.5$
- $\Delta G = 0$ (no improvement)

**Result:** Choose threshold at 70 bps.

-----

## 5. Overfitting and Regularization

### The Problem

Decision trees can perfectly memorize training data by creating a leaf for each sample, achieving 100% training accuracy but poor generalization.

### Mathematical Perspective

Without constraints, CART will grow until:
$$\forall \text{ leaf } t: |D_t| = 1 \text{ or } \forall (x_i, y_i), (x_j, y_j) \in D_t: y_i = y_j$$

This creates a model with **zero bias** but **extremely high variance**.

### Solution 1: Max Depth

**Definition:** Limit tree depth to $d_{max}$.

**Effect on model complexity:**

- Maximum nodes: $2^{d_{max}+1} - 1$
- Maximum leaves: $2^{d_{max}}$

**Example:** Option strategy classification

```
Depth 1 (d_max = 1):
├─ IV Rank ≤ 50?
   ├─ Sell Premium (leaf)
   └─ Buy Options (leaf)

Depth 3 (d_max = 3):
├─ IV Rank ≤ 50?
   ├─ DTE ≤ 30?
   │  ├─ Moneyness ≤ 0.1?
   │  │  ├─ Sell Iron Condor
   │  │  └─ Sell Strangle
   │  └─ Delta ≤ 0.3?
   │     ├─ Sell Put Spread
   │     └─ Sell Call Spread
   └─ (similar structure for right branch)
```

**Choosing $d_{max}$:**

- Use cross-validation
- Typical range: 3-10 for interpretability
- Larger for complex problems (but consider ensemble methods)

### Solution 2: Pruning

**Two approaches:**

#### Pre-pruning (Early Stopping)

Stop splitting if:

- $\Delta I < \epsilon$ (minimum improvement threshold)
- $|D_t| < n_{min}$ (minimum samples per node)
- Depth reached $d_{max}$

**Cost:** May stop too early (greedy myopia)

#### Post-pruning (More Principled)

1. Grow full tree $T_{max}$
1. Define cost-complexity criterion:

$$R_\alpha(T) = R(T) + \alpha|T|$$

where:

- $R(T)$: Classification error
- $|T|$: Number of leaves
- $\alpha \geq 0$: Complexity parameter

1. For increasing $\alpha$, prune subtrees that minimize $R_\alpha(T)$
1. Use cross-validation to select optimal $\alpha^*$

### Mathematical Example: Post-Pruning

**Subtree $T_1$** (3 leaves):

- Misclassifications: 15 out of 100
- $R(T_1) = 0.15$
- $R_\alpha(T_1) = 0.15 + \alpha \cdot 3$

**Subtree $T_2$** (single node replacing $T_1$):

- Misclassifications: 18 out of 100
- $R(T_2) = 0.18$
- $R_\alpha(T_2) = 0.18 + \alpha \cdot 1$

**Pruning decision:**

Prune if $R_\alpha(T_2) < R_\alpha(T_1)$:

$$0.18 + \alpha < 0.15 + 3\alpha$$
$$0.03 < 2\alpha$$
$$\alpha > 0.015$$

**Interpretation:** If we value simplicity at $\alpha > 0.015$ misclassification-equivalents per leaf, we should prune this subtree.

-----

## 6. Gradient Boosting: Making Trees Efficient and Powerful

### The Core Insight

**Traditional CART Problem:** Single trees are unstable and prone to overfitting.

**Gradient Boosting Solution:** Build an ensemble of shallow trees sequentially, where each tree corrects the errors of the previous ensemble.

### Mathematical Framework

#### The Additive Model

$$F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)$$

where:

- $F_M(x)$: Final ensemble prediction
- $h_m(x)$: Individual tree (weak learner)
- $\gamma_m$: Learning rate (shrinkage)
- $M$: Number of trees

#### Gradient Descent in Function Space

At iteration $m$, we fit tree $h_m$ to the **negative gradient** of the loss function:

$$r_{i,m} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$

For squared loss: $r_{i,m} = y_i - F_{m-1}(x_i)$ (the residual)

For logistic loss: $r_{i,m} = y_i - \sigma(F_{m-1}(x_i))$ (residual on probability scale)

### XGBoost Innovations

XGBoost (Extreme Gradient Boosting) improves upon traditional gradient boosting with several key innovations:

#### 1. Regularized Objective

Traditional GB minimizes: $L = \sum_{i=1}^n l(y_i, \hat{y}_i)$

XGBoost minimizes:

$$\mathcal{L} = \sum_{i=1}^n l(y_i, \hat{y}*i) + \sum*{k=1}^K \Omega(f_k)$$

where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

- $T$: Number of leaves
- $w_j$: Leaf weight (prediction value)
- $\gamma$: Penalty for adding leaves
- $\lambda$: L2 regularization on leaf weights

**Effect:** Prevents overfitting by penalizing complex trees.

#### 2. Second-Order Taylor Approximation

XGBoost uses both first and second derivatives:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t)$$

where:

- $g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$ (first derivative)
- $h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$ (second derivative, Hessian)

**Optimal leaf weight:**

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

where $I_j$ is the set of samples in leaf $j$.

**Gain for a split:**

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda}\right] - \gamma$$

**Advantage:** More accurate approximation leads to better splits.

#### 3. Sparsity-Aware Split Finding

**Problem:** Missing values and sparse features are common in real-world data.

**Solution:** XGBoost learns a default direction for each split:

```
For split "Feature j ≤ threshold":
  - Try: Send missing values left
    → Compute gain_left
  - Try: Send missing values right
    → Compute gain_right
  - Choose direction with higher gain
```

**Benefit:** Handles missing data naturally without imputation.

### LightGBM Innovations

LightGBM (Light Gradient Boosting Machine) introduces different optimizations focused on speed and scalability:

#### 1. Leaf-Wise Growth (Best-First)

**Traditional (Level-wise):**

```
Round 1: Split all nodes at depth 0
Round 2: Split all nodes at depth 1
Round 3: Split all nodes at depth 2
...
```

**LightGBM (Leaf-wise):**

```
Round 1: Split the single leaf with highest gain
Round 2: Split the leaf with highest gain (anywhere)
Round 3: Split the leaf with highest gain (anywhere)
...
```

**Mathematical justification:**

At each iteration, select:

$$\text{leaf}^* = \arg\max_{\text{leaf}} \Delta L(\text{leaf})$$

where $\Delta L$ is the loss reduction from splitting that leaf.

**Advantage:** Can achieve lower loss with fewer leaves.

**Risk:** More prone to overfitting (mitigated by `max_depth` constraint).

**Comparison:**

|Metric          |Level-wise|Leaf-wise |
|----------------|----------|----------|
|Loss reduction  |Suboptimal|Optimal   |
|Tree balance    |Balanced  |Unbalanced|
|Overfitting risk|Lower     |Higher    |
|Speed           |Slower    |Faster    |

#### 2. Gradient-Based One-Side Sampling (GOSS)

**Problem:** Large datasets slow down training.

**Traditional sampling:** Randomly select samples → loses information.

**GOSS approach:**

1. Sort samples by gradient magnitude $|g_i|$
1. Keep all samples with large gradients (top $a%$)
1. Randomly sample $b%$ from remaining samples
1. Amplify small-gradient samples by $(1-a)/b$ when computing gain

**Mathematical formulation:**

$$\text{Gain}*{\text{GOSS}} = \frac{1}{n}\left(\frac{(\sum*{i \in A} g_i + \frac{1-a}{b}\sum_{i \in B} g_i)^2}{n_L} - \frac{(\sum_{i \in A} g_i + \frac{1-a}{b}\sum_{i \in B} g_i)^2}{n_R}\right)$$

where:

- $A$: Large gradient samples (kept)
- $B$: Small gradient samples (sampled)

**Benefit:** Keep $a + b \ll 100%$ of data while maintaining accuracy.

**Typical values:** $a = 0.2$, $b = 0.1$ → use only 30% of data per iteration.

#### 3. Exclusive Feature Bundling (EFB)

**Problem:** High-dimensional sparse features slow down training.

**Observation:** Sparse features rarely take non-zero values simultaneously.

**Solution:** Bundle mutually exclusive features into a single feature.

**Example: One-Hot Encoded Categories**

Original features (4 features):

```
Category_A: [1, 0, 0, 0, 1, 0]
Category_B: [0, 1, 0, 0, 0, 0]
Category_C: [0, 0, 1, 0, 0, 1]
Category_D: [0, 0, 0, 1, 0, 0]
```

Bundled (1 feature):

```
Category_Bundled: [1, 2, 3, 4, 1, 3]
```

**Graph-theoretic formulation:**

Construct conflict graph $G = (V, E)$:

- Vertices: Features
- Edges: $(i,j) \in E$ if features $i$ and $j$ are not mutually exclusive

**Problem:** Find minimum number of bundles → Graph coloring (NP-hard)

**Greedy approximation:**

```
1. Sort features by count of non-zero values (descending)
2. For each feature:
   a. Try to add to existing bundle (if low conflict)
   b. If conflict > threshold, create new bundle
```

**Reduction:** 1000 sparse features → 50-100 bundles (10-20x speedup)

### Histogram-Based Algorithms

Both XGBoost and LightGBM use **histogram-based** split finding for efficiency:

#### Traditional Exact Method

For each feature:

1. Sort all $n$ samples: $O(n \log n)$
1. Evaluate all $n-1$ splits: $O(n)$
1. Total: $O(np \log n)$ per tree

#### Histogram Method

**Pre-processing:**

1. Bin continuous features into $k$ discrete bins (e.g., 255 bins)
1. Create feature value → bin mapping

**Split finding:**

1. Build histogram: Count samples and sum gradients per bin
1. Evaluate $k$ candidate splits (instead of $n$)
1. Total: $O(npk)$ where $k \ll n$ (typically $k = 255$)

**Memory optimization:**

Build histogram for left child, compute right child by subtraction:

$$\text{Hist}*{\text{right}} = \text{Hist}*{\text{parent}} - \text{Hist}_{\text{left}}$$

**Speedup example:**

- Dataset: 1M samples, 100 features
- Traditional: $O(10^8 \log 10^6) \approx 2 \times 10^9$ ops
- Histogram ($k=255$): $O(10^8 \times 255) = 2.55 \times 10^{10}$ ops
- But histogram ops are simple (integer arithmetic) vs expensive (float comparisons)
- **Real-world speedup: 10-30x**

### Comparative Example: Trading Signal Generation

**Task:** Predict profitable crypto trades using 500 features (technical indicators, order book data, social sentiment).

**Dataset:** 10M trades, 40% class imbalance

|Method              |Train Time|Trees|Accuracy|Notes               |
|--------------------|----------|-----|--------|--------------------|
|Scikit-learn (exact)|8h 15m    |100  |76.2%   |Baseline            |
|XGBoost (hist)      |45m       |100  |77.8%   |Regularization helps|
|XGBoost (approx)    |28m       |100  |77.6%   |Slightly worse      |
|LightGBM (GOSS)     |12m       |100  |78.1%   |GOSS + EFB          |
|LightGBM (full)     |22m       |100  |78.3%   |Best accuracy       |

**Key takeaways:**

1. **LightGBM**: Best for large datasets with many features
1. **XGBoost**: More stable, better with smaller datasets
1. **Scikit-learn**: Good for small datasets where interpretability matters

### Parameter Guidance for Gradient Boosting

**Training speed priority:**

```python
# LightGBM
params = {
    'boosting_type': 'goss',  # Use GOSS sampling
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_bin': 255,
}
```

**Accuracy priority:**

```python
# XGBoost
params = {
    'tree_method': 'hist',
    'learning_rate': 0.01,  # Smaller for better generalization
    'max_depth': 6,
    'reg_alpha': 0.1,      # L1 regularization
    'reg_lambda': 1.0,     # L2 regularization
}
```

-----

## 7. Scikit-Learn Parameters: A Practical Guide

### DecisionTreeClassifier Key Parameters

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',           # or 'entropy'
    splitter='best',            # or 'random'
    max_depth=None,             # Maximum depth
    min_samples_split=2,        # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples in leaf
    min_weight_fraction_leaf=0.0,
    max_features=None,          # Features to consider
    random_state=42,
    max_leaf_nodes=None,        # Maximum number of leaves
    min_impurity_decrease=0.0,  # Minimum impurity decrease
    class_weight=None,          # or 'balanced'
    ccp_alpha=0.0              # Pruning parameter
)
```

### Parameter Deep Dive

#### 1. `criterion`: Splitting Quality Measure

**Options:** `'gini'` (default), `'entropy'`, `'log_loss'`

**When to change:**

- **Keep default (`'gini'`)** in 95% of cases
- Use `'entropy'` if you need information-theoretic interpretations
- `'log_loss'` is equivalent to `'entropy'` for classification

**Example:**

```python
# Quick experiment
from sklearn.model_selection import cross_val_score

scores_gini = cross_val_score(
    DecisionTreeClassifier(criterion='gini', random_state=42),
    X, y, cv=5
)
scores_entropy = cross_val_score(
    DecisionTreeClassifier(criterion='entropy', random_state=42),
    X, y, cv=5
)

print(f"Gini:    {scores_gini.mean():.4f} ± {scores_gini.std():.4f}")
print(f"Entropy: {scores_entropy.mean():.4f} ± {scores_entropy.std():.4f}")
# Typical result: Difference < 0.01
```

#### 2. `max_depth`: Most Important Regularization

**Default:** `None` (grow until pure leaves)

**Practical ranges:**

- Small datasets (< 1K samples): `3-5`
- Medium datasets (1K-100K): `5-10`
- Large datasets (> 100K): `10-20`

**Impact on complexity:**

|max_depth|Max Leaves|Max Nodes|Use Case             |
|---------|----------|---------|---------------------|
|3        |8         |15       |High interpretability|
|5        |32        |63       |Balanced             |
|10       |1,024     |2,047    |Complex patterns     |
|None     |Unlimited |Unlimited|Overfitting risk     |

**Tuning example:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 7, 10, 15, None]}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted'
)
grid.fit(X_train, y_train)
print(f"Best depth: {grid.best_params_['max_depth']}")
```

**Visualization of depth impact:**

```python
import matplotlib.pyplot as plt

depths = range(1, 21)
train_scores = []
val_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    val_scores.append(clf.score(X_val, y_val))

plt.plot(depths, train_scores, label='Train')
plt.plot(depths, val_scores, label='Validation')
plt.axvline(x=optimal_depth, color='r', linestyle='--')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
```

#### 3. `min_samples_split`: Minimum Samples to Split a Node

**Default:** `2`

**Interpretation:**

- If node has fewer than `min_samples_split` samples, don’t split it (make it a leaf)
- Can be integer (absolute count) or float (fraction of total samples)

**Practical values:**

- Small datasets: `10-20`
- Large datasets: `50-100` or `0.001-0.01` (as fraction)

**Effect:**

```python
# Too low (default = 2): Overfitting
clf_overfit = DecisionTreeClassifier(min_samples_split=2)

# Reasonable
clf_balanced = DecisionTreeClassifier(min_samples_split=20)

# Too high: Underfitting
clf_underfit = DecisionTreeClassifier(min_samples_split=1000)
```

**Mathematical relationship:**

Setting `min_samples_split=s` ensures:

$$\forall \text{ internal node } t: |D_t| \geq s$$

#### 4. `min_samples_leaf`: Minimum Samples in Each Leaf

**Default:** `1`

**Interpretation:**

- Each leaf must have at least `min_samples_leaf` samples
- Smoother decision boundaries than `min_samples_split`

**Practical values:**

- Classification: `5-20`
- Regression: `10-50` (to ensure stable predictions)

**Example:**

```python
# Allow single-sample leaves (overfitting risk)
clf = DecisionTreeClassifier(min_samples_leaf=1)

# Require at least 10 samples per leaf (more robust)
clf = DecisionTreeClassifier(min_samples_leaf=10)
```

**Relationship with `min_samples_split`:**

Must satisfy: `min_samples_leaf ≤ min_samples_split / 2`

Otherwise, no split is possible.

#### 5. `max_features`: Feature Sampling

**Default:** `None` (use all features)

**Options:**

- `None`: Use all $p$ features
- Integer: Use exactly $k$ features
- Float: Use $\lfloor p \times \text{max_features} \rfloor$ features
- `'sqrt'`: Use $\sqrt{p}$ features
- `'log2'`: Use $\log_2(p)$ features

**When to use:**

**Single tree:**

- Usually keep `None` for interpretability
- Use `'sqrt'` or `'log2'` for very high-dimensional data (> 1000 features)

**Random Forest (most important use case):**

- Classification: `'sqrt'` (default and recommended)
- Regression: `p/3` (default)

**Example: High-dimensional crypto data**

```python
# 500 technical indicators
X.shape  # (100000, 500)

# Consider random feature subset
clf = DecisionTreeClassifier(
    max_features='sqrt',  # sqrt(500) ≈ 22 features
    random_state=42
)
# Increases diversity, reduces correlation
```

#### 6. `min_impurity_decrease`: Pruning Threshold

**Default:** `0.0`

**Interpretation:**
A node will be split if this split induces a decrease of impurity ≥ `min_impurity_decrease`:

$$\Delta I \geq \text{min_impurity_decrease}$$

**Calculation:**

$$\Delta I = \frac{|D_t|}{|D|}\left(I(t) - \frac{|D_L|}{|D_t|}I(D_L) - \frac{|D_R|}{|D_t|}I(D_R)\right)$$

**Practical values:**

- Start with `0.0001` for small datasets
- Use `0.001-0.01` for large datasets
- Tune via cross-validation

**Example:**

```python
# Compute weighted impurity decrease
def compute_impurity_decrease(y, left_mask):
    n = len(y)
    n_left = left_mask.sum()
    n_right = n - n_left
    
    # Gini for each split
    gini_parent = gini_impurity(y)
    gini_left = gini_impurity(y[left_mask])
    gini_right = gini_impurity(y[~left_mask])
    
    # Weighted decrease
    weighted = (n_left/n * gini_left + n_right/n * gini_right)
    decrease = (n/total_samples) * (gini_parent - weighted)
    
    return decrease

# Only split if decrease >= threshold
if compute_impurity_decrease(y, split_mask) >= min_impurity_decrease:
    make_split()
else:
    create_leaf()
```

#### 7. `class_weight`: Handling Imbalanced Classes

**Default:** `None`

**Options:**

- `None`: All classes have weight 1
- `'balanced'`: Automatically adjust inversely proportional to class frequencies
- Dictionary: `{0: w_0, 1: w_1, ...}` for custom weights

**Balanced formula:**

$$w_k = \frac{n}{K \cdot n_k}$$

where:

- $n$: Total samples
- $K$: Number of classes
- $n_k$: Samples in class $k$

**Example: Imbalanced crypto fraud detection**

```python
# Dataset: 95% legitimate, 5% fraud
y_train.value_counts()
# 0 (legit): 9500
# 1 (fraud):  500

# Unweighted: Model predicts all as "0" (legit) → 95% accuracy!
clf_unweighted = DecisionTreeClassifier()
clf_unweighted.fit(X_train, y_train)
# Recall for fraud class: ~10% (terrible)

# Balanced weights
clf_balanced = DecisionTreeClassifier(class_weight='balanced')
clf_balanced.fit(X_train, y_train)
# Weights: {0: 0.526, 1: 10.0}
# Recall for fraud class: ~70% (much better)

# Custom weights (even more aggressive on fraud)
clf_custom = DecisionTreeClassifier(
    class_weight={0: 1, 1: 20}
)
```

**Effect on splitting:**

With class weights, Gini becomes:

$$G_{\text{weighted}}(t) = 1 - \sum_{k=1}^K \left(\frac{\sum_{i \in t, y_i=k} w_i}{\sum_{i \in t} w_i}\right)^2$$

#### 8. `ccp_alpha`: Cost-Complexity Pruning

**Default:** `0.0` (no pruning)

**Theory:** This is the $\alpha$ parameter from post-pruning (Section 5):

$$R_\alpha(T) = R(T) + \alpha |T|$$

**How to find optimal `ccp_alpha`:**

```python
from sklearn.tree import DecisionTreeClassifier

# 1. Compute alpha path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# 2. Train trees for different alphas
train_scores = []
val_scores = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    val_scores.append(clf.score(X_val, y_val))

# 3. Plot and select best alpha
import matplotlib.pyplot as plt

plt.plot(ccp_alphas, train_scores, label='Train', marker='o')
plt.plot(ccp_alphas, val_scores, label='Validation', marker='o')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.xscale('log')

# 4. Best alpha
best_idx = np.argmax(val_scores)
best_alpha = ccp_alphas[best_idx]
print(f"Optimal alpha: {best_alpha:.6f}")

# 5. Final model
clf_final = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
clf_final.fit(X_train, y_train)
```

**Typical range:** `0.0001` to `0.1`

### Complete Tuning Recipe

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Step 1: Define comprehensive parameter grid
param_grid = {
    # Core regularization
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    
    # Splitting criterion (usually doesn't matter much)
    'criterion': ['gini', 'entropy'],
    
    # For imbalanced data
    'class_weight': [None, 'balanced'],
    
    # Advanced pruning
    'min_impurity_decrease': [0.0, 0.001, 0.01],
}

# Step 2: Grid search with cross-validation
clf = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(
    clf,
    param_grid,
    cv=5,
    scoring='f1_weighted',  # or 'accuracy', 'roc_auc', etc.
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# Step 3: Best parameters
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# Step 4: Test set evaluation
best_clf = grid.best_estimator_
test_score = best_clf.score(X_test, y_test)
print("Test score:", test_score)
```

### Parameter Priority Ranking

**Most Important (tune first):**

1. `max_depth` - Controls complexity directly
1. `min_samples_split` / `min_samples_leaf` - Prevents overfitting
1. `class_weight` - Essential for imbalanced data

**Secondary (tune if needed):**
4. `min_impurity_decrease` - Alternative to sample-based constraints
5. `max_features` - For high-dimensional data
6. `ccp_alpha` - Post-pruning (advanced)

**Rarely Need to Change:**
7. `criterion` - Gini vs Entropy (< 1% difference)
8. `splitter` - Best vs Random (use “best”)

### Real-World Example: Volatility Regime Classification

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Task: Classify BTC volatility regime (Low/Medium/High)
# Features: Returns, volume, option implied vol, funding rates, etc.

# Parameters based on your domain knowledge
clf = DecisionTreeClassifier(
    # Since we have ~50K daily observations
    max_depth=7,                    # Moderate complexity
    min_samples_split=100,          # Require statistical significance
    min_samples_leaf=50,            # Stable leaf predictions
    
    # We care about all regimes equally
    class_weight='balanced',        # Handle any imbalance
    
    # Use Gini (faster, equally good)
    criterion='gini',
    
    # Reproducibility
    random_state=42
)

# Train and evaluate
clf.fit(X_train, y_train)

print(f"Training accuracy: {clf.score(X_train, y_train):.3f}")
print(f"Validation accuracy: {clf.score(X_val, y_val):.3f}")
print(f"Number of leaves: {clf.get_n_leaves()}")
print(f"Tree depth: {clf.get_depth()}")

# Feature importance for strategy insights
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 important features:")
print(feature_importance.head())
```

-----

## 8. Summary Table: Key Concepts

|Concept            |Key Takeaway                    |Formula/Rule                                          |
|-------------------|--------------------------------|------------------------------------------------------|
|**Split Type**     |Always binary in CART           |$D_L \cup D_R = D_t$, $D_L \cap D_R = \emptyset$      |
|**Logic**          |Greedy: Best split now          |$\arg\max_{j,\tau} \Delta I(j, \tau)$                 |
|**Gini**           |Expected random error           |$G = 1 - \sum_k p_k^2$                                |
|**Entropy**        |Average information needed      |$H = -\sum_k p_k \log_2(p_k)$                         |
|**Gini vs Entropy**|Gini: faster, similar results   |Use Gini (default) 95% of time                        |
|**Midpoints**      |Primary way for continuous      |$m_i = (v_i + v_{i+1})/2$                             |
|**Overfitting**    |Pruning + max depth             |$R_\alpha(T) = R(T) + \alpha                          |
|**XGBoost**        |Regularization + 2nd derivatives|$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$|
|**LightGBM**       |Leaf-wise + GOSS + EFB          |Best-first leaf selection                             |
|**Scikit-learn**   |`max_depth` most important      |Tune with cross-validation                            |

-----

## Practice Problems

**Problem 1:** Calculate Gini impurity and Entropy for a node with 60 samples: 25 class A, 20 class B, 15 class C. Which is larger?

**Problem 2:** Given sorted values [10, 15, 22, 30], what midpoints does CART evaluate?

**Problem 3:** A node has Gini = 0.4. After a split, left child (60 samples) has Gini = 0.3, right child (40 samples) has Gini = 0.2. What is the information gain?

**Problem 4:** You have a dataset with 10,000 samples and 500 features. What sklearn parameters would you start with?

**Problem 5:** In XGBoost, a leaf has $\sum g_i = -15$, $\sum h_i = 10$, and $\lambda = 1$. What is the optimal leaf weight?

**Problem 6:** Explain why LightGBM’s leaf-wise growth can achieve lower loss with fewer leaves than level-wise growth. What’s the tradeoff?

-----

This comprehensive chapter bridges fundamental theory (CART) with modern practice (XGBoost/LightGBM) and provides the mathematical foundations needed to understand and effectively tune decision tree models in production environments.​​​​​​​​​​​​​​​​
