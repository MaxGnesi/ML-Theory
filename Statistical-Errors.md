# Statistical Errors in Machine Learning: A Systematic Guide
## Framework for Systematic Trading and Quantitative Finance

---

## FUNDAMENTAL DISTINCTION: Epistemic vs. Aleatoric Uncertainty

### Epistemic Uncertainty (Reducible - Fixable Errors)
From Greek "episteme" (knowledge) - uncertainty due to **lack of knowledge**.

These are **design errors** you can and should eliminate:
- Specification error (wrong model structure)
- Measurement error (bad data quality)
- Sampling bias (unrepresentative data)
- Selection bias (survivorship, look-ahead)
- Overfitting (insufficient regularization/validation)
- Metric selection (optimizing wrong objective)

**Key characteristic**: Reducible through better data, better models, or better methodology.

**Statistical source**: Model uncertainty, data uncertainty, methodological uncertainty.

### Aleatoric Uncertainty (Irreducible - Consequence Errors)
From Latin "alea" (dice) - uncertainty due to **inherent randomness**.

These are **inherent** to stochastic systems:
- Type I errors (false positives)
- Type II errors (false negatives)
- Confidence interval width
- Prediction intervals
- Market microstructure noise

**Key characteristic**: Persists even with infinite data and perfect model specification.

**Statistical source**: Fundamental randomness in the data-generating process.

### Mathematical Framework
```
Total Prediction Error = Epistemic Uncertainty + Aleatoric Uncertainty
                       = Reducible           + Irreducible

Expected Loss = E[(y - ŷ)²] = Bias² + Variance + σ²
                              \_____Epistemic____/  \Aleatoric/
```

### Trading Application
```
Design phase (Minimize Epistemic):
✓ Fix specification: Use regime-switching model not linear regression
✓ Fix data quality: Validate funding rates across exchanges
✓ Fix sampling: Include bear and bull markets
✓ Fix metric: Use F1-score for imbalanced data, not accuracy

Trading phase (Manage Aleatoric):
◯ Accept: Some trades will lose despite correct signal (market randomness)
◯ Accept: Some signals will be false despite good model (noise > signal temporarily)
◯ Optimize: Choose Type I/II threshold that maximizes E[P&L] given costs
```

### The Critical Insight for Quantitative Trading

**First reduce epistemic uncertainty to near-zero** (fix all design errors), **then optimally manage aleatoric uncertainty** (choose thresholds based on your risk/return profile).

**Common mistake**: Accepting poor performance as "aleatoric" (unavoidable) when it's actually epistemic (fixable). Always verify epistemic sources are eliminated before concluding uncertainty is aleatoric.

**The professional approach**: Treat every apparent aleatoric error as potentially epistemic until proven otherwise through rigorous validation.

---

## 1. THE ML PARADIGM: Train, Validate, Test

Before diving into errors, understand the fundamental ML framework:

```
Historical Data → [Train] → [Validate] → [Test] → Live Trading
                     ↓          ↓          ↓
                  Fit Model  Tune Model  Assess Reality
```

**Key insight**: Different errors manifest at different stages. Some you can catch in validation, others only appear in live trading.

---

## 2. BIAS-VARIANCE DECOMPOSITION: The Master Framework

All prediction error decomposes into three components:

```
Total Error = Bias² + Variance + Irreducible Error
            \___Epistemic____/   \__Aleatoric__/
```

**Bias**: Epistemic - error from wrong model assumptions (underfitting)
**Variance**: Epistemic - error from sensitivity to training data (overfitting)  
**Irreducible Error**: Aleatoric - inherent randomness you can't model

**Key insight**: Bias and Variance are epistemic (you can reduce them through better model selection and regularization). Irreducible error is aleatoric (fundamental limit of prediction).

This framework subsumes most statistical errors we'll discuss.

---

## 3. SPECIFICATION ERROR

### Definition
Your model structure doesn't match the true data-generating process.

### ML Manifestation
- **High bias**: Model too simple (linear when relationship is nonlinear)
- **Wrong features**: Missing important predictors or including irrelevant ones
- **Wrong architecture**: Using random forest when you need LSTM for time series

### Real Example from Your Work
**Ethereum Gas Price Prediction Post-Dencun**:
- Pre-Dencun: Complex models (ARIMA, LSTM) worked because gas prices had complex dynamics
- Post-Dencun: Simple persistence models won because dynamics became simpler
- Specification error: Using LSTM post-Dencun = overspecification → higher variance, no bias improvement

### Practical Implications for Trading
```python
# WRONG: Assuming linear relationship in crypto returns
model = LinearRegression()
model.fit(features, returns)  # Misspecified if true relationship is nonlinear

# BETTER: Let model learn nonlinearity
model = RandomForestRegressor()  # Can capture nonlinear relationships
```

### How to Detect
- Training error is high AND validation error is high → underfitting (too simple)
- Training error is low BUT validation error is high → overfitting (too complex)
- Residual plots show patterns → functional form is wrong
- Domain knowledge contradicts model behavior → wrong features/structure

### For Options Vol Surface
If you fit Black-Scholes implied vol but market shows volatility smile/skew:
- Specification error: Assuming constant volatility across strikes
- Solution: Use SVI model (Stochastic Volatility Inspired) that captures smile

---

## 4. MEASUREMENT ERROR

### Definition
Your input data (X) or output labels (y) contain noise or errors.

### ML Manifestation
- **Features with noise**: Attenuation bias (coefficients pulled toward zero)
- **Labels with noise**: Directly increases irreducible error, can bias model
- **Differential measurement error**: Some observations noisier than others

### Real Example from Your Work
**Basis Trading Monitoring**:
- Exchange API might report incorrect funding rates during high volatility
- Missing trades in order book data during network congestion
- Timestamp errors causing wrong basis calculations
- Wash trading inflating volume metrics

### Practical Implications for Trading
```python
# Example: Funding rate measurement error
# Bad data point: Deribit API returns 0.0001 when actual rate is 0.001
# Result: Your model underestimates basis profitability

# Mitigation strategies:
1. Data validation: Check for outliers, impossible values
2. Multiple sources: Cross-verify Binance, Bybit, Deribit
3. Robust estimation: Use median instead of mean for aggregation
4. Lag windows: Use 15-min average not single snapshot
```

### How to Detect
- Outliers that don't make economic sense (funding rate of 10%/hour)
- Inconsistencies across exchanges for same instrument
- Sudden spikes/drops that reverse immediately
- Missing data patterns (always missing during volatile periods)

### For Smart Contract Vulnerability Detection
Label noise is huge:
- Contract marked "safe" but has undiscovered vulnerability
- Contract marked "risky" but false alarm from static analyzer
- This is why you got 90% F1 score not 100% - some label noise is irreducible

---

## 5. SAMPLING ERROR & DATA Splitting

### Definition
Random variability from using a sample instead of the entire population.

### ML Manifestation
- **Train/test split variability**: Different splits give different results
- **Cross-validation variance**: Each fold produces different metrics
- **Regime sampling**: Your sample period misses important market conditions

### Real Example from Your Work
**Ethereum Post-Dencun Analysis**:
- If you only train on post-Dencun data: limited sample, high sampling error
- If you include pre-Dencun data: wrong regime, specification error
- Trade-off: More data (less sampling error) vs. relevant regime (less specification error)

### Practical Implications for Trading
```python
# WRONG: Random train/test split for time series
X_train, X_test = train_test_split(data, test_size=0.2)  # Breaks time structure!

# CORRECT: Time-aware splitting
split_date = '2024-01-01'
train = data[data.index < split_date]
test = data[data.index >= split_date]

# EVEN BETTER: Walk-forward validation
for i in range(n_windows):
    train = data[start:end]
    test = data[end:end+window]
    # Train model, test, move forward
```

### How to Detect
- High variance in cross-validation scores
- Model performance differs significantly across time periods
- Backtested Sharpe ratio much higher than live Sharpe ratio

### For Regime-Switching Strategies
Your Raydium liquidity provision strategy:
- Must sample both range-bound AND trending regimes
- If you only backtest on 2023 data and it was all trending → sampling error
- Need to ensure training sample covers multiple regime transitions

---

## 6. SELECTION BIAS

### Definition
Your sample is not representative of the population you'll encounter in production.

### Types in Trading

**A. Survivorship Bias**
```
Problem: Only analyzing cryptocurrencies that still exist
Reality: 90% of coins from 2017 are dead
Impact: Your model sees only winners, overestimates returns
```

**B. Look-Ahead Bias**
```
Problem: Using future information in historical analysis
Example: Training on data that includes knowledge of the Dencun upgrade
Impact: Impossibly good backtest, terrible live performance
```

**C. Cherry-Picking**
```
Problem: Testing strategy on Bitcoin during bull market only
Reality: Strategy fails in bear markets
Impact: Overstated performance, blown account in drawdown
```

### Real Example from Your Work
**Smart Contract Vulnerability Detection**:
- If you train only on contracts that were formally audited
- But deploy on unaudited contracts (your true population)
- Selection bias: Audited contracts have different characteristics
- Result: Model underperforms because training distribution ≠ deployment distribution

### Practical Implications for Trading
```python
# Example: Basis trade backtest with survivorship bias

# WRONG: Only backtest on BTC/ETH (survived)
profitable_coins = ['BTC', 'ETH']
backtest(basis_strategy, profitable_coins)  # Survivorship bias!

# CORRECT: Include all coins that existed at each point in time
all_coins_by_date = get_universe_at_date(date)  # Includes dead coins
backtest(basis_strategy, all_coins_by_date)
```

### How to Detect
- Performance degrades dramatically in live trading
- Model works great on test set but fails on new data with slight distribution shift
- Ask: "Am I using any information that wouldn't be available in real-time?"
- Check: "Does my training sample exclude any important subpopulations?"

### For Deribit Options Vol Surface
If you only calibrate SVI model on liquid options:
- Selection bias: Ignoring OTM options with poor liquidity
- But you'll need to price those illiquid strikes too
- Solution: Include all strikes but weight by volume/open interest

---

## 7. OVERFITTING (High Variance)

### Definition
Model learns noise in training data rather than true signal.

### ML Manifestation
This is THE classic ML problem. Related to:
- Too many features relative to samples (p >> n)
- Too complex model (deep neural net when linear would work)
- No regularization
- Training too long

### Real Example from Your Work
**Gas Price Prediction with LSTM**:
- LSTM has thousands of parameters
- Post-Dencun gas prices are simple (almost constant)
- LSTM overfits to random fluctuations in training data
- Simple persistence model generalizes better

### Practical Implications for Trading
```python
# DANGEROUS: 100 technical indicators on 1000 data points
features = calculate_100_indicators(price_data)  # p=100
returns = price_data.pct_change()  # n=1000
model = RandomForestRegressor(n_estimators=500, max_depth=None)
# This WILL overfit!

# SAFER: Regularization and cross-validation
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,  # Limit tree depth
    min_samples_split=20,  # Require minimum samples
    max_features='sqrt'  # Random feature sampling
)
# Cross-validate to detect overfitting
scores = cross_val_score(model, features, returns, cv=5)
print(f"CV scores: {scores}")  # High variance in scores = overfitting
```

### How to Detect
- Large gap between train and validation error
- High variance in cross-validation scores
- Model changes dramatically with small changes to training data
- Feature importance shows random noise is "important"

### The Trading-Specific Problem
Financial data has LOW signal-to-noise ratio:
- Most price movement is noise
- True alpha is tiny (Sharpe 1.5 is amazing)
- Very easy to overfit

**Your basis trading example**:
If your model uses 50 features to predict profitable basis trades:
- With only 200 historical trades, you have p=50, n=200
- Model can easily find spurious correlations
- Solution: Feature selection, regularization, out-of-time validation

---

## 8. TYPE I and TYPE II ERRORS (Aleatoric Uncertainty)

### Definition
Errors arising from making decisions under uncertainty - **CANNOT be eliminated** because they represent aleatoric (irreducible) uncertainty.

### Why These Are Aleatoric, Not Epistemic

Even with:
- Perfect model specification ✓ (no epistemic bias)
- Perfect data quality ✓ (no epistemic measurement error)
- No sampling bias ✓ (no epistemic sampling error)
- No overfitting ✓ (no epistemic variance)
- Optimal metric selection ✓ (no epistemic evaluation error)

You STILL have Type I/II errors because:
1. **Markets are stochastic** - aleatoric randomness in price movements
2. **Signal-to-noise ratio is finite** - can't perfectly separate signal from noise
3. **Feature distributions overlap** - profitable and unprofitable trades share similar characteristics
4. **Fundamental uncertainty** - the future is genuinely uncertain

**This is aleatoric uncertainty** - inherent to the system, not due to lack of knowledge.

### Classical Hypothesis Testing Framework
```
                Reality
              H0 True  |  H0 False
Decision    -------------------------
Accept H0   | Correct  | Type II (β)
            |          | False Negative
Reject H0   | Type I(α)| Correct
            | False    | (Power = 1-β)
            | Positive |
```

**Type I Error (α)**: Reject true null hypothesis (false positive)
**Type II Error (β)**: Fail to reject false null hypothesis (false negative)
**Power (1-β)**: Probability of correctly rejecting false null

### ML Classification Framework
```
                Actual
              Negative | Positive
Predicted   ---------------------
Negative    | TN  | FN (Type II)
Positive    | FP  | TP
            |(Type I)|

Precision = TP/(TP+FP)     ← Minimizes Type I
Recall = TP/(TP+FN)        ← Minimizes Type II
F1 = Harmonic mean of both
```

### Real Example from Your Work
**Smart Contract Vulnerability Detection (90% F1 Score)**:

The 10% error comes from:
- Type I (False Positives): Flagging safe contract as vulnerable
  - Cost: Wasted audit resources, delayed deployment
- Type II (False Negatives): Missing actual vulnerability
  - Cost: Potential exploit, loss of funds

**Why you can't get 100%**: Even with perfect model specification and clean data:
- Some vulnerabilities are subtle (require complex reasoning)
- Some safe patterns look risky (defensive programming)
- Fundamental uncertainty in what constitutes "vulnerable"

### Practical Implications for Trading
```python
# Example: Basis trade signal generation

# Model outputs probability: P(profitable_trade)
probabilities = model.predict_proba(current_market_data)

# You MUST choose a threshold:
threshold = 0.7  # Enter trade if P > 0.7

if probabilities > threshold:
    enter_trade()
else:
    wait()

# This threshold determines your Type I/II tradeoff:
# High threshold (0.9): Few Type I, Many Type II (miss good trades)
# Low threshold (0.5): Many Type I, Few Type II (enter bad trades)
```

### The Fundamental Tradeoff
You can move along the ROC curve but you CANNOT eliminate both errors simultaneously.

**Your Raydium liquidity provision**:
- Type I: Provide liquidity in trending market → get run over (high cost!)
- Type II: Don't provide in range-bound → miss fees (low cost)
- Optimal: High threshold for "range-bound" classification

### Why This is Unavoidable
Even with:
- Perfect model specification ✓
- Perfect data quality ✓
- No sampling bias ✓
- No overfitting ✓

You STILL have Type I/II errors because:
1. Markets are stochastic (irreducible uncertainty)
2. You're making predictions about the future
3. There's genuine overlap in feature distributions between classes

This is a **feature of statistics**, not a bug in your model.

### CRITICAL: The Metric Selection Feedback Loop

**Your key insight**: What looks like an unavoidable Type I/II tradeoff might actually be a **design error** in metric selection!

#### The Problem: Imbalanced Datasets

Consider smart contract vulnerability detection:
- 10,000 contracts total
- 100 are vulnerable (1%)
- 9,900 are safe (99%)

```python
# Dumb model: Always predict "safe"
predictions = ["safe"] * 10000

# Metrics:
accuracy = 9900/10000 = 99.0%  # Looks great!
precision = undefined (no positive predictions)
recall = 0/100 = 0%  # Complete failure!

# The feedback loop:
# 1. You optimize for accuracy (design error - wrong metric)
# 2. Model learns to predict "safe" for everything
# 3. You get 99% accuracy and think "good model, just Type I/II tradeoff"
# 4. But you're missing 100% of vulnerabilities!
# 5. You never learn from failures because metric says you're doing well
```

#### Why This Happens

**Precision vs. Recall for Rare Events**:

For your basis trading with imbalanced classes:
- 1000 market observations
- 50 are profitable trades (5%)
- 950 are unprofitable (95%)

```python
# Model A: Conservative (high precision, low recall)
Predicts: 10 trades (only very confident ones)
Results: 9 correct, 1 wrong
Precision: 9/10 = 90%  ← Looks good!
Recall: 9/50 = 18%     ← Missing most opportunities!

# Model B: Aggressive (low precision, high recall)  
Predicts: 200 trades
Results: 45 correct, 155 wrong
Precision: 45/200 = 22.5%  ← Looks terrible!
Recall: 45/50 = 90%        ← Catching most opportunities!

# Which is better depends on your COSTS:
# - If false positives cost a lot (transaction fees): Choose Model A
# - If false negatives cost a lot (missing alpha): Choose Model B
```

#### The Design Error

**You're optimizing the wrong objective function!**

For imbalanced problems, accuracy and precision can be misleading. You need:

1. **F1-Score** (harmonic mean of precision and recall)
2. **F-beta Score** (weighted toward recall if false negatives are costly)
3. **Precision-Recall AUC** (better than ROC-AUC for imbalanced data)
4. **Cost-sensitive learning** (directly optimize your P&L)

```python
# WRONG: Optimize accuracy on imbalanced data
from sklearn.metrics import accuracy_score
model = optimize_for(accuracy_score)  # Design error!

# CORRECT: Optimize cost-weighted metric
def trading_profit(y_true, y_pred):
    """
    Custom metric that reflects actual trading costs
    """
    tp = sum((y_true == 1) & (y_pred == 1))  # Correct trades
    fp = sum((y_true == 0) & (y_pred == 1))  # Bad trades
    tn = sum((y_true == 0) & (y_pred == 0))  # Correctly avoided
    fn = sum((y_true == 1) & (y_pred == 0))  # Missed opportunities
    
    profit = (
        tp * avg_trade_profit      # ~0.5% per profitable trade
        - fp * transaction_cost     # ~0.1% per bad trade
        - fn * opportunity_cost     # ~0.3% per missed trade
    )
    return profit

model = optimize_for(trading_profit)  # Now optimizing the RIGHT thing
```

#### Real Example: Your Smart Contract Detection

With 90% F1 score, you need to ask:

**What does that 10% error consist of?**
- High precision, low recall? → Missing vulnerabilities (dangerous!)
- Low precision, high recall? → Too many false alarms (expensive)
- Balanced? → True Type I/II tradeoff

```python
# Check the confusion matrix
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

#              precision    recall  f1-score   support
#
#        safe       0.99      0.95      0.97      9900
#  vulnerable       0.50      0.90      0.64       100
#
# Interpretation:
# - High recall (90%) on vulnerabilities: Good! Catching most of them
# - Low precision (50%): Many false alarms
# - This IS a true Type I/II tradeoff (both metrics reasonable)

# Compare to:
#              precision    recall  f1-score   support
#
#        safe       0.99      1.00      0.99      9900
#  vulnerable       0.95      0.10      0.18       100
#
# Interpretation:
# - Recall only 10%: Missing 90% of vulnerabilities!
# - High precision meaningless if you're not finding vulnerabilities
# - This is NOT Type I/II tradeoff, this is DESIGN ERROR in metric choice
```

#### The Feedback Loop Explained

```
1. Choose wrong metric (design error)
        ↓
2. Model optimizes that metric
        ↓
3. Metric says performance is good (99% accuracy!)
        ↓
4. You think: "Just Type I/II tradeoff, unavoidable"
        ↓
5. Never realize model is fundamentally broken
        ↓
6. Deploy to production
        ↓
7. Model fails catastrophically
        ↓
8. Only then discover wrong metric was optimized
```

**Breaking the loop**: Always check confusion matrix, not just aggregate metrics!

#### Practical Guidelines for Imbalanced Data

**For rare events (vulnerabilities, profitable trades, regime changes)**:

1. **Never use accuracy** - it's meaningless
2. **Check precision AND recall** - don't optimize only one
3. **Use F-beta score** where beta > 1 if recall matters more
4. **Plot precision-recall curve** not just ROC curve
5. **Use stratified sampling** in cross-validation
6. **Consider resampling**: SMOTE, undersampling majority class
7. **Cost-sensitive learning**: Directly optimize your objective

```python
# Example: Regime detection for your Raydium strategy
# Imbalanced: 80% range-bound, 20% trending

# WRONG approach
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)  # Will bias toward predicting range-bound (majority)

accuracy = model.score(X_test, y_test)  # Misleading!
# Model might predict range-bound 95% of the time, get 80% accuracy
# But miss most trending periods (disaster for LP strategy)

# CORRECT approach
from sklearn.utils.class_weight import compute_class_weight

# Option 1: Class weights
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y), 
    y=y
)
model = RandomForestClassifier(class_weight='balanced')

# Option 2: Custom loss function
def regime_loss(y_true, y_pred):
    """
    Heavily penalize providing liquidity in trending market
    """
    fp_cost = 100  # Lose money if LP during trend
    fn_cost = 1    # Small opportunity cost if avoid range-bound
    
    fp = sum((y_true == 0) & (y_pred == 1))  # Predict range, actually trend
    fn = sum((y_true == 1) & (y_pred == 0))  # Predict trend, actually range
    
    return fp * fp_cost + fn * fn_cost

# Option 3: Resampling
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# Then evaluate properly
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

#### Summary: Is it Design Error or Consequence Error?

**Ask these questions**:

1. **Is my evaluation metric aligned with my actual objective?**
   - No → Design error (fix metric)
   - Yes → Move to next question

2. **Am I handling class imbalance appropriately?**
   - No → Design error (use class weights, resampling, or proper metric)
   - Yes → Move to next question

3. **Have I examined the confusion matrix, not just aggregate metrics?**
   - No → Check it now! Might reveal design error
   - Yes → Move to next question

4. **Are both precision and recall at reasonable levels given my cost structure?**
   - No → Design error (wrong threshold or wrong model)
   - Yes → True Type I/II tradeoff (consequence error)

**The key insight**: Always validate that what appears to be a Type I/II tradeoff isn't actually a symptom of optimizing the wrong thing. Check your metrics, check your confusion matrix, check your costs. Only after confirming these are correct can you say "this is an unavoidable consequence error I must manage."

---

## 9. ENDOGENEITY & MULTICOLLINEARITY

### Endogeneity
When explanatory variables correlate with the error term, breaking causal interpretation.

**Your basis trading example**: Does funding rate cause volatility or does volatility cause funding rate? Both! This is endogeneity.

**Solution for trading**: Use predictive models (Random Forest, XGBoost) that don't require causal interpretation. You just need to forecast, not explain causality.

### Multicollinearity
When predictor variables are highly correlated with each other.

**Your basis trading**: Funding rates across Binance, Bybit, Deribit are highly correlated.

**Solution**: 
- Use average funding rate instead of individual exchanges
- Apply PCA to extract principal components
- Use Ridge regression (L2 regularization)
- Use tree-based models (naturally robust to multicollinearity)

---

## 10. COMPREHENSIVE ERROR MANAGEMENT FRAMEWORK

### Step 1: Design Phase (Eliminate Ex Ante Errors)

**Checklist**:
- ☐ Model specification matches data-generating process
- ☐ Data quality validated across multiple sources
- ☐ Sample includes all market regimes (bull, bear, sideways)
- ☐ No survivorship or look-ahead bias
- ☐ Regularization and cross-validation implemented
- ☐ Multicollinearity checked (VIF < 10)

### Step 2: Validation Phase (Detect Remaining Issues)

```python
def validate_model(model, data):
    # 1. Cross-validation (detect overfitting)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    if cv_scores.std() > 0.1:
        print("Warning: High variance suggests overfitting")
    
    # 2. Out-of-time validation (detect regime shift)
    train = data[data.index < '2024-01-01']
    test = data[data.index >= '2024-01-01']
    oos_score = test_score(model, test)
    
    # 3. Walk-forward validation (realistic simulation)
    wf_sharpe = walk_forward_test(model, data)
    
    return cv_scores, oos_score, wf_sharpe
```

### Step 3: Deployment Phase (Manage Ex Post Errors)

```python
# 1. Optimize Type I/II tradeoff
cost_false_positive = 0.001  # 10 bps transaction cost
cost_false_negative = 0.0002  # 2 bps opportunity cost

optimal_threshold = optimize_threshold(
    model, validation_data,
    fp_cost=cost_false_positive,
    fn_cost=cost_false_negative
)

# 2. Make decisions with confidence intervals
prob = model.predict_proba(current_data)[0, 1]
if prob > optimal_threshold:
    enter_trade()

# 3. Monitor and adapt
if realized_type_i_rate > expected * 1.5:
    retrain_model()  # Regime has shifted
```

---

## 11. SUMMARY: ERROR TAXONOMY FOR SYSTEMATIC TRADING

### Epistemic Uncertainty (Reducible - Fix These!)
| Error | Cause | Detection | Solution |
|-------|-------|-----------|----------|
| Specification | Wrong model structure | Residual patterns | Domain knowledge, model selection |
| Measurement | Bad data quality | Outliers, inconsistencies | Multiple sources, validation |
| Sampling Bias | Unrepresentative sample | Performance degradation | Include all regimes |
| Selection Bias | Survivorship, look-ahead | Too-good backtest | Point-in-time data |
| Overfitting | Too complex model | Train/val gap | Regularization, simpler model |
| **Metric Selection** | **Wrong evaluation metric** | **Confusion matrix** | **Cost-sensitive learning** |

### Aleatoric Uncertainty (Irreducible - Manage the Tradeoff!)
| Error | Nature | Tradeoff | Optimization |
|-------|--------|----------|--------------|
| Type I | False positive | vs. Type II | ROC curve, cost-benefit |
| Type II | False negative | vs. Type I | Threshold selection |

**CRITICAL**: Before accepting something as aleatoric (irreducible), verify all epistemic sources are eliminated! Check confusion matrix, not just aggregate metrics. What appears as aleatoric might be epistemic metric selection error.

### Decision Framework: Epistemic → Aleatoric
```
Is performance poor? 
├─ YES
│  ├─ EPISTEMIC CHECK 1: Using right metric? (Confusion matrix!)
│  │  └─ NO → Epistemic error: Fix evaluation metric
│  │  └─ YES → Continue to next check
│  │
│  ├─ EPISTEMIC CHECK 2: Model specification
│  │  ├─ Train error high? → Epistemic: Specification error (underfit)
│  │  ├─ Train low, Val high? → Epistemic: Overfitting
│  │  └─ Both reasonable? → Continue to next check
│  │
│  ├─ EPISTEMIC CHECK 3: Data quality
│  │  ├─ Val okay, Test poor? → Epistemic: Sampling/selection bias
│  │  ├─ Test okay, Live poor? → Epistemic: Distribution shift
│  │  └─ Consistent across samples? → Continue to next check
│  │
│  └─ All epistemic checks passed?
│     └─ YES → Remaining error is ALEATORIC
│        └─ Optimize Type I/II tradeoff based on costs
│
└─ NO (Performance good on aggregate metric)
   └─ Still check confusion matrix!
      ├─ One class has terrible recall? → Epistemic (wrong metric)
      └─ Both classes reasonable? → Aleatoric tradeoff is optimized
```

### The Professional Approach

**Always assume epistemic first, aleatoric second:**

1. Systematically eliminate all epistemic sources
2. Only after rigorous validation, conclude error is aleatoric
3. Then optimize the aleatoric tradeoff (Type I/II threshold)

**Red flag**: Claiming "markets are random, nothing works" before checking:
- Model specification ✓
- Data quality ✓  
- Proper sampling ✓
- Correct metrics ✓
- No overfitting ✓

If you haven't checked these, you're likely dealing with epistemic uncertainty, not aleatoric.

### Special Case: Imbalanced Data
For rare events (profitable trades, vulnerabilities, regime changes):

**Epistemic errors that masquerade as aleatoric**:
- Using accuracy instead of F1/F-beta (metric selection error)
- Not using class weights or resampling (specification error)
- Optimizing precision when recall matters (wrong objective)
- Not examining confusion matrix (evaluation error)

**Detection**: 
```python
# Always do this for imbalanced problems:
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# If one class has <50% recall → Likely epistemic (fixable)
# If both classes have 70-90% recall → Likely aleatoric (manage tradeoff)
```

---

## 12. FURTHER READING

### Ex Ante Errors (Design & Methodology)
- **Wooldridge**: "Introductory Econometrics" - Systematic treatment of specification, endogeneity
- **Angrist & Pischke**: "Mostly Harmless Econometrics" - Causal inference, selection bias

### Ex Post Errors (Inference Under Uncertainty)
- **Hastie et al.**: "Elements of Statistical Learning" - Bias-variance, prediction intervals
- **Murphy**: "Probabilistic Machine Learning" - Bayesian uncertainty quantification

### Trading-Specific
- **De Prado**: "Advances in Financial Machine Learning" - Overfitting, backtesting, labels
- **Chan**: "Algorithmic Trading" - Walk-forward, regime detection

### Your Priority
1. **Fix epistemic errors first** (Sections 3-7)
2. **Then optimize aleatoric tradeoff** (Section 8)
3. **Monitor for distribution shift** (regime changes)

The goal: Eliminate epistemic uncertainty, optimally manage aleatoric uncertainty.

---

## 13. AUTOMATION: Diagnostic Script

A Python script (`epistemic_aleatoric_diagnostic.py`) is available that automates this entire framework:

```python
from epistemic_aleatoric_diagnostic import EpistemicAleatoicDiagnostic

# Initialize diagnostic
diagnostic = EpistemicAleatoicDiagnostic(
    model=your_model,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    problem_type='classification',
    is_imbalanced=True,
    cost_fp=0.001,  # Your false positive cost
    cost_fn=0.0003  # Your false negative cost
)

# Run complete diagnostic
result = diagnostic.run_full_diagnostic(verbose=True)

# Visualize
diagnostic.plot_diagnostic_dashboard(save_path='diagnostic.png')
```

**What it does**:
1. Systematically checks all epistemic sources (6 checks)
2. Determines if errors are epistemic (fixable) or aleatoric (manage)
3. Generates prioritized recommendations
4. Creates comprehensive visualization dashboard
5. Optimizes Type I/II threshold if all epistemic checks pass

**Output**: DiagnosticResult object with:
- `error_type`: 'epistemic' or 'aleatoric'
- `epistemic_sources`: Dictionary of which checks failed
- `recommendations`: Ordered list of fixes
- `severity`: 'critical', 'warning', or 'acceptable'
- `metrics`: Full performance metrics

Use this as your standard validation pipeline for every trading strategy.
