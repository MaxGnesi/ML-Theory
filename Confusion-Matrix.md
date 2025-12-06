# Mastering Confusion Matrix Metrics: A Deep Intuition Guide

## Part 1: The Mental Model

### The Fundamental Question Framework

Every confusion matrix metric answers one of four fundamental questions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FOUR FUNDAMENTAL QUESTIONS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Q1: "Of the things that ARE positive, how many did I catch?"           │
│       → TPR (Sensitivity/Recall) = TP / (TP + FN)                       │
│       → Denominator: ACTUAL positives (row in matrix)                   │
│                                                                         │
│  Q2: "Of the things that ARE negative, how many did I correctly leave?" │
│       → TNR (Specificity) = TN / (TN + FP)                              │
│       → Denominator: ACTUAL negatives (row in matrix)                   │
│                                                                         │
│  Q3: "Of the things I CALLED positive, how many really are?"            │
│       → PPV (Precision) = TP / (TP + FP)                                │
│       → Denominator: PREDICTED positives (column in matrix)             │
│                                                                         │
│  Q4: "Of the things I CALLED negative, how many really are?"            │
│       → NPV = TN / (TN + FN)                                            │
│       → Denominator: PREDICTED negatives (column in matrix)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**The critical insight:** Q1 and Q2 condition on REALITY (rows). Q3 and Q4 condition on PREDICTIONS (columns).

### Visual: The Two Perspectives

```
                         PREDICTIONS
                    ┌─────────┬─────────┐
                    │   -     │    +    │
              ┌─────┼─────────┼─────────┤
              │  -  │   TN    │   FP    │ ← TNR = TN/(TN+FP) "How well do I leave negatives alone?"
   REALITY    ├─────┼─────────┼─────────┤
              │  +  │   FN    │   TP    │ ← TPR = TP/(TP+FN) "How well do I catch positives?"
              └─────┴─────────┴─────────┘
                       ↑           ↑
                      NPV         PPV
                      "Of my      "Of my
                      neg calls,  pos calls,
                      how many    how many
                      correct?"   correct?"
```

---

## Part 2: The Metrics with Deep Intuition

### Tier 1: The Four Base Rates

#### True Positive Rate (TPR) — "The Catch Rate"

```
TPR = TP / (TP + FN) = TP / P
```

**Intuition:** You're a security guard. 100 thieves try to sneak in. You catch 85. Your TPR is 85%.

**The FN is your failure:** Every FN is a thief you missed. TPR tells you your hit rate on actual threats.

**When TPR matters most:**
- Medical screening (missing cancer = death)
- Fraud detection (missed fraud = loss)
- Security systems (missed intrusion = breach)
- Any domain where **false negatives are catastrophic**

**The tradeoff:** Maximizing TPR alone → just predict everything positive → TPR = 100% but useless

---

#### True Negative Rate (TNR) — "The Leave-Alone Rate"

```
TNR = TN / (TN + FP) = TN / N
```

**Intuition:** 100 innocent people walk by. You correctly let 90 pass without hassle. Your TNR is 90%.

**The FP is your failure:** Every FP is an innocent person you harassed.

**When TNR matters most:**
- Legal systems (innocent until proven guilty)
- Medical treatments with severe side effects
- Loan approvals (wrongly denying creditworthy applicants)
- Any domain where **false positives are costly**

---

#### The Error Rates (FPR and FNR)

```
FPR = FP / (TN + FP) = 1 - TNR    "False alarm rate"
FNR = FN / (TP + FN) = 1 - TPR    "Miss rate"
```

**Key insight:** These are just the complements. If you know TPR, you know FNR. If you know TNR, you know FPR.

**Why bother with both?** Communication. "5% miss rate" is more intuitive than "95% sensitivity" in some contexts.

---

### Tier 2: The Predictive Values (The Bayesian Perspective)

#### Positive Predictive Value (PPV) — "Trust in Positive Calls"

```
PPV = TP / (TP + FP)
```

**Intuition:** Your model flags 100 transactions as fraudulent. You investigate and find 70 actually are. PPV = 70%.

**Why PPV differs from TPR:**
- TPR asks: "Of actual frauds, how many did I catch?"
- PPV asks: "Of flagged transactions, how many are actually fraud?"

**The prevalence trap:** PPV is heavily influenced by base rate!

```
Example: Disease screening with TPR=99%, TNR=99%

Population 1: 10% prevalence (1000 people: 100 sick, 900 healthy)
- TP = 99, FP = 9
- PPV = 99/108 = 91.7% ✓ Good!

Population 2: 0.1% prevalence (100,000 people: 100 sick, 99,900 healthy)
- TP = 99, FP = 999
- PPV = 99/1098 = 9.0% ✗ Terrible!

Same test, vastly different PPV!
```

**This is why rare disease screening has two stages:** High-sensitivity first screen, then high-specificity confirmation.

---

#### Negative Predictive Value (NPV) — "Trust in Negative Calls"

```
NPV = TN / (TN + FN)
```

**Intuition:** Your model says 1000 patients are healthy. 950 actually are. NPV = 95%.

**Also prevalence-dependent:** In low-prevalence settings, NPV tends to be high naturally (most things are negative anyway).

---

#### False Discovery Rate (FDR) and False Omission Rate (FOR)

```
FDR = FP / (TP + FP) = 1 - PPV   "Of positive calls, what fraction are wrong?"
FOR = FN / (TN + FN) = 1 - NPV   "Of negative calls, what fraction are wrong?"
```

**When to use FDR:** Multiple hypothesis testing in genomics/research. Controlling FDR at 5% means accepting that 5% of your "discoveries" are false.

---

### Tier 3: Composite Metrics

#### F1 Score — "The Compromise"

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2TP / (2TP + FP + FN)
```

**Why harmonic mean?** It punishes extreme imbalance harshly.

```
Example: Precision = 1.0, Recall = 0.01
- Arithmetic mean: (1.0 + 0.01) / 2 = 0.505 (looks okay!)
- Harmonic mean (F1): 2(1.0)(0.01) / (1.0 + 0.01) = 0.0198 (reveals the problem!)
```

**The harmonic mean property:** It's bounded by the smaller value. You can't game F1 by making one metric perfect.

**Limitation:** F1 treats precision and recall as equally important. Often they're not.

---

#### F-Beta Score — "The Weighted Compromise"

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

**Intuition for β:**
- β = 0.5: Precision is 2× more important than recall
- β = 1.0: Equal importance (F1)
- β = 2.0: Recall is 2× more important than precision

**Derivation of the weighting:**
```
When β = 2:
- Numerator weight on Recall: β² = 4
- Numerator weight on Precision: 1
- So Recall counts 4× more in the denominator weighting
```

**Practical guide:**
- F0.5: Use when false positives are costly (spam detection — users hate false positives)
- F1: Use as default when unsure
- F2: Use when false negatives are costly (disease screening — don't miss patients)

---

#### Matthews Correlation Coefficient (MCC) — "The Gold Standard"

```
MCC = (TP × TN - FP × FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Why MCC is often the best single metric:**

1. **Uses all four cells:** Unlike F1 (ignores TN) or accuracy (gets fooled by imbalance)

2. **Symmetric:** Treats positive and negative classes equally

3. **Range [-1, +1]:** Like a correlation coefficient
   - +1: Perfect
   - 0: Random/no skill
   - -1: Perfect inverse (as bad as possible)

4. **Robust to imbalance:** Doesn't inflate with skewed distributions

**The MCC intuition:** It's the correlation between predicted and actual labels. If your predictions are random noise with respect to truth, MCC ≈ 0.

**Edge case warning:** MCC is undefined if any row or column sums to zero (division by zero). This happens when you predict all one class or when a class doesn't exist in data.

---

### Tier 4: Likelihood Ratios — The Clinician's Tools

#### Positive Likelihood Ratio (LR+)

```
LR+ = TPR / FPR = Sensitivity / (1 - Specificity)
```

**Intuition:** "How much more likely is a positive test result in someone with the disease vs. without?"

**Clinical interpretation:**
- LR+ = 10: Positive test is 10× more likely in diseased vs. healthy
- LR+ = 1: Test result is equally likely in both groups (useless test!)
- LR+ > 10: Strong evidence for ruling IN disease
- LR+ > 5: Moderate evidence

**Fagan nomogram application:**
```
Post-test odds = Pre-test odds × LR+
Post-test probability = Post-test odds / (1 + Post-test odds)
```

---

#### Negative Likelihood Ratio (LR−)

```
LR− = FNR / TNR = (1 - Sensitivity) / Specificity
```

**Intuition:** "How much more likely is a negative test result in someone with the disease vs. without?"

**Clinical interpretation:**
- LR− = 0.1: Negative test is 10× more likely in healthy vs. diseased
- LR− = 1: Useless test
- LR− < 0.1: Strong evidence for ruling OUT disease
- LR− < 0.2: Moderate evidence

**The clinical mantra:**
- **SpIN:** High Specificity → Positive test rules IN disease (high LR+)
- **SnNOut:** High Sensitivity → Negative test rules OUT disease (low LR−)

---

#### Diagnostic Odds Ratio (DOR)

```
DOR = LR+ / LR− = (TP × TN) / (FP × FN)
```

**Intuition:** Single number combining both likelihood ratios.

**Properties:**
- DOR = 1: Useless test
- DOR > 1: Test has diagnostic value
- DOR → ∞: Perfect test

**Limitation:** Same DOR can come from different TPR/TNR combinations. A test with TPR=TNR=80% and one with TPR=95%, TNR=50% might have similar DOR but very different clinical utility.

---

### Tier 5: Information-Theoretic Metrics

#### Informedness (Youden's J)

```
Informedness = TPR + TNR - 1 = Sensitivity + Specificity - 1
```

**Intuition:** "How much better than random chance is my classifier?"

**Range interpretation:**
- +1: Perfect (TPR=1, TNR=1)
- 0: Random classifier (TPR + TNR = 1, meaning no better than flipping coins)
- -1: Perfectly wrong (TPR=0, TNR=0)

**Why subtract 1?** A random classifier with TPR=50%, TNR=50% gives Informedness = 0, correctly showing no skill.

**Use case:** Optimal threshold selection. The threshold that maximizes Youden's J is often a good default choice.

---

#### Markedness

```
Markedness = PPV + NPV - 1
```

**Intuition:** "How much does the prediction tell us about reality?"

**Duality with Informedness:**
- Informedness: Conditions on reality, measures prediction quality
- Markedness: Conditions on predictions, measures reality-tracking

**Symmetric relationship:** In the MCC formula:
```
MCC = √(Informedness × Markedness)
```
(when both are positive)

---

### Tier 6: Set-Theoretic Metrics

#### Jaccard Index (IoU)

```
Jaccard = TP / (TP + FN + FP) = |Intersection| / |Union|
```

**Visual intuition:**
```
Actual Positives:    ████████░░░░░░
Predicted Positives: ░░░░████████░░
                         ↑↑↑↑
                      Intersection (TP)
                     
Union = ████████████░░  (TP + FN + FP)

Jaccard = 4/12 in this case
```

**Why ignore TN?** In many applications (object detection, segmentation), negatives are abundant and uninteresting. You care about overlap of positive regions.

**Relation to F1:**
```
Jaccard = F1 / (2 - F1)
F1 = 2 × Jaccard / (1 + Jaccard)
```

---

## Part 3: Prevalence and Its Effects

### The Prevalence Problem Visualized

```
Scenario: Test with 90% Sensitivity, 90% Specificity

HIGH PREVALENCE (50%):
┌────────────────────────────────────────────────────────────────┐
│ Population: 1000 (500 positive, 500 negative)                  │
│                                                                │
│ TP = 450    FP = 50     PPV = 450/500 = 90% ✓                  │
│ FN = 50     TN = 450    NPV = 450/500 = 90% ✓                  │
└────────────────────────────────────────────────────────────────┘

LOW PREVALENCE (1%):
┌────────────────────────────────────────────────────────────────┐
│ Population: 10000 (100 positive, 9900 negative)                │
│                                                                │
│ TP = 90     FP = 990    PPV = 90/1080 = 8.3% ✗                 │
│ FN = 10     TN = 8910   NPV = 8910/8920 = 99.9% ✓              │
└────────────────────────────────────────────────────────────────┘

Same test performance, but PPV collapses in low-prevalence!
```

### Metrics Prevalence Sensitivity Chart

```
┌─────────────────────────────────────────────────────────────────┐
│               PREVALENCE SENSITIVITY                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PREVALENCE-INDEPENDENT          PREVALENCE-DEPENDENT           │
│  (Intrinsic test properties)     (Depends on population)        │
│                                                                 │
│  • TPR (Sensitivity)             • PPV (Precision)              │
│  • TNR (Specificity)             • NPV                          │
│  • FPR                           • Accuracy                     │
│  • FNR                           • F1 Score (moderately)        │
│  • LR+                           • FDR                          │
│  • LR−                           • FOR                          │
│  • DOR                           • Markedness                   │
│  • Informedness (Youden's J)                                    │
│  • MCC                                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 4: The Threshold Game

### Understanding the TPR-FPR Tradeoff

Most classifiers output a probability/score. The threshold determines the confusion matrix.

```
Score Distribution:
                                          Threshold
                                              ↓
Negatives: ████████████▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░
Positives: ░░░░░░░░░░░░░░░░░▓▓▓████████████████████

           0         0.2        0.4        0.6        0.8        1.0

▓ = Overlap region (where errors occur)

Low threshold (0.3):  High TPR, High FPR (catch everything, many false alarms)
High threshold (0.7): Low TPR, Low FPR (miss many, but confident when you call positive)
```

### ROC Curve Intuition

```
        TPR
         1│        ╭───────────● Perfect
          │       ╱
          │      ╱
          │     ╱     ← Better than random
          │    ╱
          │   ╱    Random
        0.5│  ╱   ╱ classifier
          │ ╱  ╱    (diagonal)
          │╱ ╱
         0├─────────────────────→ FPR
          0        0.5          1

Area Under ROC (AUC-ROC):
• 1.0 = Perfect classifier
• 0.5 = Random (no skill)
• <0.5 = Worse than random (invert predictions!)
```

### Precision-Recall Curve

```
      Precision
         1│●
          │ ╲
          │  ╲
          │   ╲___
          │       ╲___
          │           ╲
          │            ╲___
         0├─────────────────→ Recall (TPR)
          0                 1

• Starts high precision, low recall (strict threshold)
• Ends low precision, high recall (lenient threshold)
• Area under PR curve (AUC-PR) better for imbalanced data
```

**When to use which curve:**
- **ROC:** When classes are balanced, or you care equally about both classes
- **PR:** When positive class is rare (imbalanced), or you only care about positive class performance

---

## Part 5: Practical Decision Framework

### Metric Selection Flowchart

```
                         START
                           │
                           ▼
              ┌────────────────────────┐
              │ Are classes balanced?  │
              │   (within 60-40 split) │
              └───────────┬────────────┘
                    │           │
                   YES          NO
                    │           │
                    ▼           ▼
              ┌──────────┐  ┌──────────────────┐
              │ Accuracy │  │ F1, MCC, or      │
              │ is okay  │  │ class-specific   │
              └──────────┘  │ metrics          │
                            └────────┬─────────┘
                                     │
                           ┌─────────┴─────────┐
                           │                   │
                           ▼                   ▼
              ┌───────────────────┐  ┌───────────────────┐
              │ Need single       │  │ Need detailed     │
              │ summary metric?   │  │ breakdown?        │
              │                   │  │                   │
              │ → Use MCC         │  │ → TPR, TNR,       │
              │                   │  │   PPV, NPV        │
              └───────────────────┘  └───────────────────┘
                                     
                           │
                           ▼
              ┌────────────────────────────┐
              │ What's the cost structure? │
              └─────────────┬──────────────┘
                     │      │      │
          ┌──────────┘      │      └──────────┐
          ▼                 ▼                 ▼
   FN costly          Equal           FP costly
   (miss = bad)       cost            (false alarm = bad)
        │                │                  │
        ▼                ▼                  ▼
   Maximize         Balance            Maximize
   TPR/Recall       F1 Score           TNR/Precision
   Use F2           Use F1             Use F0.5
```

### Domain-Specific Recommendations

```
┌─────────────────────────────────────────────────────────────────────────┐
│ DOMAIN                    PRIMARY METRICS         RATIONALE             │
├─────────────────────────────────────────────────────────────────────────┤
│ Medical Screening         Sensitivity (TPR)       Can't miss disease    │
│                          NPV, LR−                 Negative = reassurance│
│                                                                         │
│ Medical Diagnosis         Specificity (TNR)       Confirm before        │
│                          PPV, LR+                 treatment             │
│                                                                         │
│ Fraud Detection          Precision (PPV)         Review capacity        │
│                          Recall (TPR)            limited                │
│                          F1 or F2                                       │
│                                                                         │
│ Spam Filtering           Precision               FP = lost email!       │
│                          F0.5                                           │
│                                                                         │
│ Credit Risk              Depends on bank's       Usually Precision      │
│                          loss function           (avoid bad loans)      │
│                                                                         │
│ Information Retrieval    Precision@K             Top results matter     │
│                          MAP, NDCG                                      │
│                                                                         │
│ Object Detection         IoU/Jaccard             Spatial overlap        │
│                          mAP                                            │
│                                                                         │
│ Academic Research        MCC                     Most rigorous          │
│ (general ML)             AUC-ROC, AUC-PR         single metrics         │
│                                                                         │
│ Trading Signals          Precision               False signals costly   │
│                          MCC                     Want true correlation  │
│                          Informedness            Better than random?    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Edge Cases and Gotchas

### Division by Zero Scenarios

```
┌─────────────────────────────────────────────────────────────────────────┐
│ METRIC          ZERO WHEN                    HANDLING                   │
├─────────────────────────────────────────────────────────────────────────┤
│ TPR             No actual positives (P=0)    Undefined (no positives)   │
│ TNR             No actual negatives (N=0)    Undefined (no negatives)   │
│ PPV             No predicted positives       Undefined or 0 by conv.    │
│ NPV             No predicted negatives       Undefined or 1 by conv.    │
│ F1              PPV or TPR is 0              0                          │
│ MCC             Any row/col sums to 0        Undefined (0 by conv.)     │
│ LR+             FPR = 0                      Infinity (perfect spec.)   │
│ LR−             TNR = 0                      Infinity                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### The "All-One-Class Predictor" Trap

```
Scenario: Predict everything as negative in 99% negative dataset

Confusion Matrix:
                 Pred- | Pred+
           ----|-------|------
   Actual -    |  990  |  0
           ----|-------|------
   Actual +    |  10   |  0

Metrics:
• Accuracy = 990/1000 = 99% ✓ (Looks great!)
• TPR = 0/10 = 0% ✗ (Caught nothing!)
• F1 = 0 ✗ (Reveals the problem)
• MCC = 0 ✗ (Reveals the problem)

Lesson: Always check TPR and F1/MCC, not just accuracy!
```

### Class Imbalance Reality Check

```
Rule of thumb for "imbalanced":
• Mild: 70-30 to 80-20 → Accuracy still somewhat useful
• Moderate: 90-10 to 95-5 → Definitely use F1/MCC
• Severe: 99-1 or worse → Consider specialized techniques:
  - Stratified sampling
  - SMOTE/oversampling
  - Cost-sensitive learning
  - Anomaly detection framing
  - Precision-Recall curves (not ROC)
```

### The Calibration Problem

**Metrics don't tell you about calibration!**

```
Two models, same confusion matrix at threshold 0.5:

Model A (well-calibrated):
• When it says 80% probability, ~80% are actually positive
• Probabilities are meaningful

Model B (poorly calibrated):
• When it says 80% probability, only 50% are actually positive
• Probabilities need adjustment

Same TPR, PPV, F1, etc. — but Model A is more useful!

Use calibration plots and Brier score to assess this separately.
```

---

## Part 7: Multi-Class Extensions

### Confusion Matrix for K Classes

```
                    Predicted Class
                 C1    C2    C3    C4
              ┌─────┬─────┬─────┬─────┐
          C1  │ n11 │ n12 │ n13 │ n14 │
              ├─────┼─────┼─────┼─────┤
Actual    C2  │ n21 │ n22 │ n23 │ n24 │
Class     ├───┼─────┼─────┼─────┼─────┤
          C3  │ n31 │ n32 │ n33 │ n34 │
              ├─────┼─────┼─────┼─────┤
          C4  │ n41 │ n42 │ n43 │ n44 │
              └─────┴─────┴─────┴─────┘

Diagonal = correct predictions
Off-diagonal = errors (shows confusion patterns)
```

### Averaging Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STRATEGY        FORMULA                    USE WHEN                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Macro-average   (Metric_c1 + ... + Metric_ck) / K                       │
│                 Treats all classes equally    Class balance matters     │
│                                              equally                    │
│                                                                         │
│ Micro-average   Compute from aggregated      Sample-level               │
│                 TP, FP, FN across classes    performance matters        │
│                                                                         │
│ Weighted-avg    Σ (n_i × Metric_i) / Σ n_i  Larger classes should       │
│                                              matter more                │
└─────────────────────────────────────────────────────────────────────────┘

Example:
Class A: 100 samples, Precision = 0.9
Class B: 900 samples, Precision = 0.7

Macro precision = (0.9 + 0.7) / 2 = 0.80
Weighted precision = (100×0.9 + 900×0.7) / 1000 = 0.72
```

---

## Part 8: Worked Example with Full Analysis

### Scenario: Crypto Pump Detection Model

```
You build a model to detect pump-and-dump schemes in altcoins.
Test set: 1000 coins over 30 days
• 50 actual pumps (positives)
• 950 non-pumps (negatives)

Confusion Matrix:
                 Pred Normal | Pred Pump
           ----|-------------|----------
   Normal      |    900      |    50    
           ----|-------------|----------
   Pump        |     5       |    45    
```

### Complete Metric Calculation

```
Base values:
TP = 45, TN = 900, FP = 50, FN = 5
P = 50, N = 950, Total = 1000

Tier 1: Base Rates
─────────────────────────────────────────────────────────────────
TPR (Sensitivity) = 45/50 = 0.900 = 90%
  → "We catch 90% of actual pumps"
  
TNR (Specificity) = 900/950 = 0.947 = 94.7%
  → "We correctly ignore 94.7% of non-pumps"
  
FPR (Fall-out) = 50/950 = 0.053 = 5.3%
  → "5.3% false alarm rate"
  
FNR (Miss rate) = 5/50 = 0.100 = 10%
  → "We miss 10% of actual pumps"

Tier 2: Predictive Values
─────────────────────────────────────────────────────────────────
PPV (Precision) = 45/95 = 0.474 = 47.4%
  → "Only 47% of our alerts are real pumps"
  ⚠️ Low! Despite 90% sensitivity and 95% specificity!
  
NPV = 900/905 = 0.994 = 99.4%
  → "99.4% of 'safe' calls are correct"
  
FDR = 50/95 = 0.526 = 52.6%
  → "53% of our alerts are false alarms"
  
FOR = 5/905 = 0.006 = 0.6%
  → "Only 0.6% of 'safe' calls are actually pumps"

Tier 3: Overall Performance
─────────────────────────────────────────────────────────────────
Accuracy = 945/1000 = 0.945 = 94.5%
  → Looks great but misleading (class imbalance!)
  
Balanced Accuracy = (0.900 + 0.947)/2 = 0.924 = 92.4%
  → Better representation

Tier 4: Composite Metrics
─────────────────────────────────────────────────────────────────
F1 = 2×45 / (2×45 + 50 + 5) = 90/145 = 0.621 = 62.1%
  → Balances precision-recall tradeoff
  
F2 = (1+4)×(0.474×0.900) / (4×0.474 + 0.900) = 2.133/2.796 = 0.763
  → If we care more about catching pumps (recall-weighted)
  
MCC = (45×900 - 50×5) / √(95×50×950×905)
    = (40500 - 250) / √4075312500
    = 40250 / 63838 = 0.631
  → Good correlation, robust metric

Tier 5: Likelihood Ratios
─────────────────────────────────────────────────────────────────
LR+ = 0.900/0.053 = 17.1
  → Positive test increases odds 17× (strong evidence)
  
LR- = 0.100/0.947 = 0.106
  → Negative test decreases odds to 1/10 (strong rule-out)
  
DOR = 17.1/0.106 = 161.3
  → High diagnostic odds ratio (good test)

Tier 6: Information Theory
─────────────────────────────────────────────────────────────────
Informedness = 0.900 + 0.947 - 1 = 0.847
  → 85% better than random
  
Markedness = 0.474 + 0.994 - 1 = 0.468
  → Predictions mark reality reasonably well

Tier 7: Set-Based
─────────────────────────────────────────────────────────────────
Jaccard = 45/(45+5+50) = 45/100 = 0.450 = 45%
  → Moderate overlap of predicted and actual pump sets
```

### Interpretation Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│ METRIC           VALUE    ASSESSMENT    IMPLICATION                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Sensitivity      90%      Excellent     Missing only 10% of pumps       │
│ Specificity      94.7%    Excellent     Low false alarm rate            │
│ Precision        47.4%    Moderate      Many alerts are noise           │
│ F1               62.1%    Good          Reasonable balance              │
│ MCC              0.63     Good          Strong correlation              │
│ LR+              17.1     Strong        Positive test is meaningful     │
│ Accuracy         94.5%    Misleading!   Inflated by class imbalance     │
└─────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Despite excellent sensitivity/specificity, precision is only 47%
because pumps are rare (5% prevalence). Half your alerts are false alarms.

PRACTICAL ACTION:
• If reviewing alerts is cheap: Use as-is, accept 53% false alarms
• If reviewing is expensive: Raise threshold to improve precision
• Consider two-stage system: This model for screening, then manual review
```

---

## Part 9: Quick Reference Card

```
+=========================================================================+
|                    CONFUSION MATRIX QUICK REFERENCE                     |
+=========================================================================+
|                                                                         |
|  +-------+-----+-----+        Row-based (Reality):                      |
|  |       |Pred |Pred |        TPR = TP/P    "Catch rate"                |
|  |       |  -  |  +  |        TNR = TN/N    "Leave-alone rate"          |
|  +-------+-----+-----+                                                  |
|  | Act - | TN  | FP  | <-TNR  Column-based (Predictions):               |
|  +-------+-----+-----+        PPV = TP/(TP+FP)  "Alert accuracy"        |
|  | Act + | FN  | TP  | <-TPR  NPV = TN/(TN+FN)  "Clear accuracy"        |
|  +-------+-----+-----+                                                  |
|             ^     ^          PREVALENCE-INDEPENDENT: TPR, TNR, LR+, LR- |
|            NPV   PPV         PREVALENCE-DEPENDENT: PPV, NPV, Accuracy   |
|                                                                         |
|  Complements:                Best single metrics:                       |
|  TPR + FNR = 1               - MCC (imbalanced data)                    |
|  TNR + FPR = 1               - F1 (when you need interpretability)      |
|  PPV + FDR = 1               - Balanced Accuracy (quick check)          |
|  NPV + FOR = 1                                                          |
|                                                                         |
|  F1 = 2*PPV*TPR/(PPV+TPR)    "Harmonic mean, punishes extremes"         |
|  MCC = correlation(pred, actual)  "Gold standard"                       |
|  Informedness = TPR+TNR-1    "Better than random"                       |
|                                                                         |
|  WHEN FN IS COSTLY -> Maximize TPR, use F2                              |
|  WHEN FP IS COSTLY -> Maximize PPV/TNR, use F0.5                        |
|                                                                         |
+=========================================================================+
```

---

## Part 10: Common Mistakes to Avoid

1. **Using accuracy on imbalanced data**
   Always report F1, MCC, or balanced accuracy alongside

2. **Ignoring prevalence when interpreting PPV**
   PPV can be terrible even with excellent sensitivity/specificity if positive class is rare

3. **Comparing F1 scores across different datasets**
   F1 is somewhat prevalence-dependent; MCC is more comparable

4. **Forgetting that threshold affects all metrics**
   Report metrics at specific thresholds, or use AUC for threshold-independent comparison

5. **Optimizing for wrong metric**
   Match metric to business cost function (missed pump vs false alarm costs)

6. **Not checking for degenerate solutions**
   Verify model isn't predicting all one class

7. **Conflating correlation with causation**
   MCC shows correlation, not that predictions are useful for decision-making

8. **Ignoring calibration**
   Good discrimination metrics don't guarantee well-calibrated probabilities
