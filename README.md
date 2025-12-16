# Isolation Forest Implementation


The goal of this project is to implement the original [Isolation Forest](IsolationForestPaper.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.  (A later version of this work is also available: [Isolation-based Anomaly Detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.673.5779&rep=rep1&type=pdf).) 

There are two general approaches to anomaly detection: 

1. model what normal looks like and then look for nonnormal observations
2. focus on the anomalies, which are few and different. This is the interesting and relatively-new approach taken by the authors of isolation forests.

The isolation forest algorithm is original and beautiful in its simplicity; and also seems to work very well, with a few known weaknesses. The academic paper is extremely readable so you should start there.

## Datasets

For this project, we'll use three data sets:

* [Kaggle credit card fraud competition data set](https://www.kaggle.com/mlg-ulb/creditcardfraud); download, unzip to get `creditcard.csv`

* Download the cancer dataset: [cancer.csv](https://github.com/JialiangShi/isolation-forest/blob/master/cancer.csv);

* Download the http dataset: [http.csv](https://github.com/JialiangShi/isolation-forest/blob/master/http.csv); 

My code assumes the data files are in the same directory as the code.

## Visualization of normal versus anomaly separation

Using [plot_anomalies.py](https://github.com/JialiangShi/isolation-forest/blob/master/plot_anomalies.py), you can see the results of the isolation forest trying to detect anomalies. These data sets all have known targets indicating normal versus anomaly, but this information is only used during testing and not during training. In other words, we use this information to discover how well we can separate the distribution of normal versus anomalous observations.  The section provides a number of results, but yours might look different because of the inherent randomness involved in selecting subsets of the data and constructing random trees. (click on the images to enlarge.)

<center>
<table border="0">
<tr><td>http.csv, 200 trees, 99% desired TPR</td></tr>
<tr>
<td border=0>
<a href="images/http-200-99.svg"><img src="images/http-200-99.svg" width="350"></a>
</tr>
</table>
</center>

<table border="0">
<tr><td>creditcard.csv, 200 trees, 80% desired TPR</td><td>creditcard.csv, 200 trees, 90% desired TPR</td></tr>
<tr>
<td border=0>
<a href="images/creditcard-200-80.svg"><img src="images/creditcard-200-80.svg" width="350"></a>
<td border=0>
<a href="images/creditcard-200-90.svg"><img src="images/creditcard-200-90.svg" width="350"></a>
</tr>
</table>

<table border="0">
<tr><td> cancer, 300 trees, 70% desired TPR</td><td> cancer, 300 trees, 80% desired TPR</td></tr>
<tr>
<td border=0>
<a href="images/cancer-300-70.svg"><img src="images/cancer-300-70.svg" width="350"></a>
<td border=0>
<a href="images/cancer-300-80.svg"><img src="images/cancer-300-80.svg" width="350"></a>
</tr>
</table>

## Scoring results

Running [score.py](https://github.com/JialiangShi/isolation-forest/blob/master/score.py), here is a sample run:

```
Running noise=False improved=False
INFO creditcard.csv fit time 0.23s
INFO creditcard.csv 18804 total nodes in 200 trees
INFO creditcard.csv score time 14.54s
SUCCESS creditcard.csv 200 trees at desired TPR 80.0% getting FPR 0.0300%

INFO http.csv fit time 0.28s
INFO http.csv 22430 total nodes in 300 trees
INFO http.csv score time 23.08s
SUCCESS http.csv 300 trees at desired TPR 99.0% getting FPR 0.0053%

INFO cancer.csv fit time 0.08s
INFO cancer.csv 8204 total nodes in 1000 trees
INFO cancer.csv score time 0.73s
SUCCESS cancer.csv 1000 trees at desired TPR 75.0% getting FPR 0.2857%
```

Due to the subsampling of the original data said and the inherent random nature of isolation forest, your results will differ even from run to run.  I'm hoping that the variance is not so high that valid programs fail the scoring, but let me know.

---

## Semiconductor Wafer Defect Detection

This project has been extended to apply Isolation Forest to **semiconductor manufacturing** for detecting defective wafers.

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
