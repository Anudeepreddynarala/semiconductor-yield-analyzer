# Methodology: Semiconductor Wafer Defect Detection

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Algorithm Overview](#algorithm-overview)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Quality Metrics (Cpk & Yield)](#quality-metrics)

---

## Problem Statement

### Objective
Detect defective semiconductor wafers using electrical test data to improve manufacturing yield and reduce costly defect escapes.

### Challenges
1. **Class Imbalance**: Only 2.6% normal wafers (1,000) vs. 97.4% defective (37,015)
2. **Image Data**: Wafer maps are 52×52 grids, not tabular sensor data
3. **Multiple Defect Types**: 8 basic types + 29 mixed-type combinations
4. **Cost Asymmetry**: Underkill (missing defects) costs 10-100x more than overkill

### Success Criteria
- **Target TPR**: ≥90% defect detection rate
- **FPR**: <10% false positive rate
- **Cpk**: ≥1.33 (industry standard for Four Sigma quality)
- **Yield Loss**: Minimize difference between actual and predicted yield

---

## Algorithm Overview

### Isolation Forest

**Core Principle**: Anomalies are "few and different," therefore easier to isolate.

#### How It Works
1. **Random Partitioning**: Recursively split data using random features and thresholds
2. **Path Length**: Count splits needed to isolate each sample
3. **Anomaly Score**: Shorter paths → Higher anomaly scores

```
Anomaly Score = 2^(-E(h(x)) / c(n))

Where:
- E(h(x)) = Average path length for sample x
- c(n) = Average path length of unsuccessful search in BST
- Score ∈ [0, 1], with 0.5 being the threshold
```

#### Why Isolation Forest for Semiconductors?

**Advantages:**
- ✅ Unsupervised (doesn't need labeled defects for training)
- ✅ Handles high-dimensional data efficiently
- ✅ Fast training and scoring
- ✅ Effective with imbalanced data
- ✅ Interpretable anomaly scores

**Our Approach:**
- Train ONLY on normal wafers (learn "good" patterns)
- Test on all wafers (normal + defective)
- Anomalies (defects) score higher than normal

---

## Feature Engineering

### From Images to Features

**Input**: 52×52 wafer maps with values:
- `0` = Blank spot (edge of wafer)
- `1` = Normal die (passed electrical test)
- `2` = Failed die (failed electrical test)

**Output**: 19 meaningful features per wafer

### Feature Categories

#### 1. Basic Counts (4 features)
```python
total_dies = normal_dies + failed_dies
blank_spots = count(value == 0)
```

**Why Important**: Absolute counts indicate wafer utilization and defect magnitude.

#### 2. Failure Rate (1 feature)
```python
failure_rate = failed_dies / total_dies
```

**Why Important**: Normalized metric for comparison across wafers.

#### 3. Spatial Distribution (3 features)
```python
failure_center_y, failure_center_x = mean(failed_die_coordinates)
failure_center_dist = distance(center, wafer_center)
```

**Why Important**: Identifies spatial patterns (center vs. edge defects).

#### 4. Spread Metrics (3 features)
```python
failure_spread_mean = mean(pairwise_distances(failed_dies))
failure_spread_std = std(pairwise_distances(failed_dies))
failure_spread_max = max(pairwise_distances(failed_dies))
```

**Why Important**: Distinguishes clustered vs. scattered defects.

#### 5. Radial Features (4 features)
```python
radii = sqrt((y - center_y)^2 + (x - center_x)^2)
failure_radius_mean = mean(radii)
failure_radius_std = std(radii)
failure_radius_max, failure_radius_min = max(radii), min(radii)
```

**Why Important**: Captures radial symmetry and ring patterns.

#### 6. Concentration Ratios (2 features)
```python
edge_failure_ratio = count(radius > 20) / failed_dies
center_failure_ratio = count(radius < 10) / failed_dies
```

**Why Important**: Quantifies defect location preferences.

#### 7. Clustering Metrics (2 features)
```python
num_failure_clusters = connected_components(failed_dies)
avg_cluster_size = failed_dies / num_failure_clusters
```

**Why Important**: Distinguishes random vs. systematic defects.

### Feature Importance

Based on correlation with defect types:

| Feature | Importance | Distinguishes |
|---------|-----------|---------------|
| `failure_rate` | High | Normal vs. defective |
| `failure_center_dist` | High | Center vs. edge defects |
| `edge_failure_ratio` | High | Edge patterns |
| `num_failure_clusters` | Medium | Random vs. systematic |
| `failure_spread_max` | Medium | Scratch vs. localized |

---

## Model Training

### Training Strategy

**Key Decision**: Train on normal wafers only (unsupervised anomaly detection)

```python
# Split normal wafers
X_normal_train = 700 wafers  # 70% of 1,000 normal
X_normal_test = 300 wafers   # 30% for testing

# Train model
model = IsolationTreeEnsemble(sample_size=256, n_trees=100)
model.fit(X_normal_train)  # Learn "normal" patterns

# Score all wafers
scores = model.anomaly_score(X_test)  # Test set includes defective wafers
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_trees` | 100 | Balance accuracy and speed |
| `sample_size` | 256 | Sufficient for 19 features |
| `desired_TPR` | 0.90 | Target detection rate |
| `test_size` | 0.30 | 30% for validation |

### Threshold Selection

Use `find_TPR_threshold()` to achieve target detection rate:

```python
threshold, FPR = find_TPR_threshold(y_true, scores, desired_TPR=0.90)
```

**Process:**
1. Sort scores in descending order
2. Calculate TPR and FPR at each threshold
3. Select threshold achieving ≥90% TPR
4. Report corresponding FPR

---

## Evaluation Metrics

### Confusion Matrix

```
                Predicted
              Normal  Defective
Actual Normal   282      18      (TN=282, FP=18)
       Defect  2,860   34,155    (FN=2,860, TP=34,155)
```

### Classification Metrics

**True Positive Rate (TPR / Recall)**
```
TPR = TP / (TP + FN) = 34,155 / 37,015 = 92.3%
```
*Percentage of defects correctly detected*

**True Negative Rate (TNR / Specificity)**
```
TNR = TN / (TN + FP) = 282 / 300 = 94.0%
```
*Percentage of normal wafers correctly identified*

**False Positive Rate (FPR)**
```
FPR = FP / (FP + TN) = 18 / 300 = 6.0%
```
*Percentage of normal wafers incorrectly flagged*

**Precision**
```
Precision = TP / (TP + FP) = 34,155 / 34,173 = 99.95%
```
*When flagged as defective, how often is it correct?*

**F1 Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall) = 0.9596
```
*Harmonic mean of precision and recall*

**Average Precision (AP)**
```
AP = 0.9998
```
*Area under precision-recall curve*

### Manufacturing Impact

**Overkill (False Positives)**
- Count: 18 wafers
- Impact: Good wafers scrapped unnecessarily
- Cost: ~$90,000 (@ $5,000/wafer)

**Underkill (False Negatives)**
- Count: 2,860 wafers
- Impact: Defective wafers shipped to customers
- Cost: ~$143M (@ $50,000/wafer escape)

**Total Quality Cost**: $143.09M

---

## Quality Metrics

### Yield Calculation

**Actual Yield**
```
Yield = (Normal Wafers / Total Wafers) × 100%
Yield = (1,000 / 38,015) × 100% = 2.63%
```

**Predicted Yield**
```
Predicted Yield = (Predicted Normal / Total) × 100%
Predicted Yield = 10.17%
```

**Yield Loss**
```
Yield Loss = Actual Yield - Predicted Yield = -7.54%
```

### Cpk (Process Capability Index)

**Formula**
```
Cpk = min[(USL - μ)/(3σ), (μ - LSL)/(3σ)]

Where:
- USL = Upper Specification Limit
- LSL = Lower Specification Limit
- μ = Process mean
- σ = Process standard deviation
```

**Interpretation**
- **Cpk ≥ 2.0**: Excellent (Six Sigma)
- **Cpk ≥ 1.67**: Very Good (Five Sigma)
- **Cpk ≥ 1.33**: Good (Four Sigma) - **Industry Standard**
- **Cpk ≥ 1.0**: Acceptable (Three Sigma)
- **Cpk < 1.0**: Poor - Needs Improvement

**Our Results (Failure Rate)**
```
Specification Limits: [0.00, 0.05]  # 0% to 5% failure rate
Process Mean: 0.3191 (31.91%)
Process Std: 0.0846
Cpk: -1.060
Assessment: Poor - Process Needs Improvement
```

**Why Negative?** The process mean (31.91% failure rate) is far outside the specification limit (5%), indicating the dataset is heavily skewed toward defective wafers (by design for this analysis).

**In Production**: Would expect Cpk ≥ 1.33 with failure rates <5%.

---

## References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM '08: Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*.

2. Montgomery, D. C. (2012). *Statistical quality control: A modern introduction* (7th ed.). Wiley.

3. Wang, J., et al. (2020). Deformable convolutional networks for efficient mixed-type wafer defect pattern recognition. *IEEE Transactions on Semiconductor Manufacturing*.
