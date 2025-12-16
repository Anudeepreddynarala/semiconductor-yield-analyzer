# Isolation Forest for Semiconductor Wafer Defect Detection

Implementation of the [Isolation Forest](IsolationForestPaper.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou, applied to **semiconductor manufacturing quality control**.

## Overview

This project applies unsupervised anomaly detection to identify defective semiconductor wafers from electrical test data. The Isolation Forest algorithm learns patterns from normal wafers and flags deviations as potential defects.

**Key Features:**
- 🎯 92.3% defect detection rate
- ✅ 94% specificity (low false alarms)
- 📊 19 engineered spatial features from wafer maps
- 🔬 Per-defect-type performance analysis
- 🧪 Comprehensive test suite (95% pass rate)
- 📚 Production-ready with full documentation

## Algorithm Approach

The Isolation Forest uses a unique strategy:
1. **Isolation principle**: Anomalies are few and different, thus easier to isolate
2. **Random partitioning**: Build trees that isolate samples via random splits
3. **Path length**: Anomalies require fewer splits to isolate (shorter paths)
4. **Unsupervised learning**: Train only on normal samples, detect deviations

**Why it works for semiconductors:**
- Defective wafers have distinct spatial failure patterns
- Normal wafers cluster tightly in feature space
- Efficient for high-dimensional manufacturing data

---

## Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd isolation-forest

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (if not already present)
# Place wafer dataset in archive/ directory

# 5. Extract features from wafer maps
python extract_wafer_features.py

# 6. Run defect detection
python wafer_anomaly_detection_v2.py

# 7. Analyze per-defect-type performance
python analyze_defect_types.py

# 8. Run tests
pytest tests/ -v
```

### Dataset

**Mixed-Type Wafer Defect Dataset** ([Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets))
- 38,015 wafer maps (52×52 pixel grids)
- 1,000 normal wafers (2.6%)
- 37,015 defective wafers (97.4%)
- 8 basic defect types: Center, Donut, Edge_Loc, Edge_Ring, Loc, Near_Full, Scratch, Random
- 29 mixed-type defects (combinations of 2-4 basic types)

### Approach

Since wafer maps are images (not direct sensor readings), we **extract 19 spatial features** from each wafer:

**Feature Categories:**
1. **Basic counts**: total dies, failed dies, blank spots
2. **Failure metrics**: failure rate, failure concentration
3. **Spatial distribution**: center of mass, distance from wafer center
4. **Spread metrics**: mean/std/max distance between failures
5. **Radial features**: radius statistics, edge vs center concentration
6. **Clustering**: number of failure clusters, average cluster size

### Training Strategy

**Key insight**: With defects being 97.4% of data, we train on **normal wafers only** to learn what "good" looks like, then detect defects as anomalies.

```python
# Train on 700 normal wafers → Test on 300 normal + 37,015 defective
python wafer_anomaly_detection_v2.py
```

### Results

**Overall Performance** (100 trees, 90% TPR target):
- ✅ **92.3% of defects detected** (34,155 / 37,015)
- ✅ **94% specificity** (282 / 300 normal wafers correctly identified)
- ⚠️ **6% false positive rate** (18 good wafers flagged)
- ❌ **2,860 defects missed** (underkill - high cost risk!)
- **F1 Score**: 0.9596
- **Average Precision**: 0.9998

**Manufacturing Impact:**
- **Overkill**: 18 good wafers scrapped unnecessarily
- **Underkill**: 2,860 bad wafers shipped (⚠️ HIGH RISK - customer failures)
- Cost consideration: Underkill >> Overkill in semiconductor industry

### Per-Defect-Type Performance

| Defect Type | Count | Avg Anomaly Score | Detection @ 0.50 threshold |
|-------------|-------|-------------------|----------------------------|
| Random      | 866   | 0.5894           | 100.0%                     |
| Center      | 13,000| 0.5865           | 98.0%                      |
| Scratch     | 19,000| 0.5845           | 94.7%                      |
| Loc         | 18,000| 0.5838           | 97.0%                      |
| Edge_Loc    | 13,000| 0.5836           | 97.7%                      |
| Edge_Ring   | 12,000| 0.5772           | 96.5%                      |
| Donut       | 12,000| 0.5742           | 94.8%                      |
| Near_Full   | 149   | 0.5663           | 100.0%                     |
| **NORMAL**  | **300**| **0.4314**      | **N/A**                    |

**Key Findings:**
- All defect types score significantly higher than normal (0.43)
- Random and Center defects are most distinguishable
- Near_Full has fewer samples but 100% detection rate
- At threshold 0.50, nearly all defect types achieve 94-100% detection

### Visualization

Run per-defect-type analysis:
```bash
python analyze_defect_types.py
```

Generates:
- Bar chart of average anomaly scores by defect type
- Box plots showing score distributions
- Detection rate tables at different thresholds

### Files

**Data Processing:**
- `extract_wafer_features.py` - Extract spatial features from wafer maps
- `wafer_features.csv` - 19 engineered features for 38,015 wafers
- `wafer_labels.csv` - Defect type labels (one-hot encoded)

**Analysis Scripts:**
- `wafer_anomaly_detection_v2.py` - Main detection script (train on normal only)
- `analyze_defect_types.py` - Per-defect-type performance analysis

**Visualizations:**
- `wafer_trained_on_normal-100trees-90TPR.svg` - Score distributions
- `defect_type_analysis.svg` - Per-defect-type comparison

### Citation

Dataset: Wang et al., "Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition," IEEE Transactions on Semiconductor Manufacturing, 2020.

---

Hooray!
