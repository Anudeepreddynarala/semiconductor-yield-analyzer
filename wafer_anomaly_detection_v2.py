import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

from iforest import IsolationTreeEnsemble, find_TPR_threshold


def train_on_normal_detect_defects(X, y, sample_size=256, n_trees=100, desired_TPR=0.90, test_size=0.3, bins=30):
    """
    Train Isolation Forest on NORMAL wafers only, then detect defects as anomalies.

    This is the correct approach for anomaly detection in manufacturing:
    - Learn what "good/normal" looks like (unsupervised on normal wafers)
    - Flag anything different as an anomaly (defects)

    Parameters:
    - X: Feature matrix
    - y: Binary labels (0=normal, 1=defective)
    - sample_size: Subsample size for each tree
    - n_trees: Number of trees
    - desired_TPR: Target detection rate
    - test_size: Fraction of data for testing
    - bins: Histogram bins
    """
    print(f"{'='*70}")
    print("STRATEGY: Train on Normal Wafers, Detect Defects as Anomalies")
    print(f"{'='*70}\n")

    # Separate normal and defective wafers
    X_normal = X[y == 0]
    X_defect = X[y == 1]

    print(f"Dataset composition:")
    print(f"  Normal wafers: {len(X_normal):,}")
    print(f"  Defective wafers: {len(X_defect):,}")
    print(f"  Total: {len(X):,}")

    # Split normal wafers into train/test
    X_normal_train, X_normal_test = train_test_split(X_normal, test_size=test_size, random_state=42)

    print(f"\nTraining set: {len(X_normal_train):,} normal wafers")
    print(f"Test set: {len(X_normal_test):,} normal + {len(X_defect):,} defective = {len(X_normal_test) + len(X_defect):,} total")

    # Train Isolation Forest on NORMAL wafers only
    print(f"\n🔧 Training Isolation Forest on normal wafers only...")
    print(f"   Parameters: sample_size={sample_size}, n_trees={n_trees}")

    it = IsolationTreeEnsemble(sample_size=min(sample_size, len(X_normal_train)), n_trees=n_trees)

    fit_start = time.time()
    it.fit(X_normal_train)
    fit_time = time.time() - fit_start
    print(f"   ✓ Training time: {fit_time:.2f}s")

    # Score all test wafers
    print(f"\n📊 Scoring test wafers...")
    X_test = np.vstack([X_normal_test, X_defect])
    y_test = np.hstack([np.zeros(len(X_normal_test)), np.ones(len(X_defect))])

    score_start = time.time()
    scores_test = it.anomaly_score(X_test)
    score_time = time.time() - score_start
    print(f"   ✓ Scoring time: {score_time:.2f}s")

    # Find threshold for desired TPR on defects
    threshold, FPR = find_TPR_threshold(y_test, scores_test, desired_TPR)
    print(f"   ✓ Threshold for {desired_TPR:.0%} TPR: {threshold:.4f} (FPR: {FPR:.4f})")

    # Make predictions
    y_pred = it.predict_from_anomaly_scores(scores_test, threshold=threshold)

    # Calculate metrics
    confusion = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = confusion.flat

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR_actual = FP / (FP + TN) if (FP + TN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    F1 = f1_score(y_test, y_pred)
    PR = average_precision_score(y_test, scores_test)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Normal  Defective")
    print(f"Actual Normal      {TN:5d}     {FP:5d}")
    print(f"       Defective   {FN:5d}     {TP:5d}")

    print(f"\n📈 Performance Metrics:")
    print(f"  True Positive Rate (Recall):   {TPR:.4f} = {TP:,}/{TP+FN:,} defects detected")
    print(f"  True Negative Rate (Spec.):    {TNR:.4f} = {TN:,}/{TN+FP:,} normals correctly identified")
    print(f"  False Positive Rate:            {FPR_actual:.4f} = {FP:,} false alarms")
    print(f"  Precision:                      {Precision:.4f}")
    print(f"  F1 Score:                       {F1:.4f}")
    print(f"  Average Precision:              {PR:.4f}")

    print(f"\n💡 Manufacturing Impact:")
    print(f"  ✅ Defects caught: {TP:,}/{TP+FN:,} ({TPR:.1%})")
    print(f"  ❌ Defects missed (UNDERKILL): {FN:,} - High risk!")
    print(f"  ⚠️  False alarms (OVERKILL): {FP:,}/{TN+FP:,} good wafers ({FPR_actual:.1%})")
    print(f"  💰 Cost consideration: Underkill >> Overkill in semiconductors")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    normal_scores = scores_test[y_test == 0]
    defect_scores = scores_test[y_test == 1]

    # Normal wafers
    counts0, bins0, _ = axes[0].hist(normal_scores, bins=bins, color='#90EE90', alpha=0.8,
                                      edgecolor='black', label='Normal wafers')
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2.5,
                    label=f'Threshold = {threshold:.3f}')
    axes[0].set_ylabel("Normal Wafer Count", fontsize=13, fontweight='bold')
    axes[0].set_title("Wafer Defect Detection - Trained on Normal Wafers Only",
                     fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Add metrics box
    textstr = f'N = {len(X_test):,} test wafers\n' \
              f'Trees = {n_trees}\n' \
              f'TPR = {TPR:.1%} | FPR = {FPR_actual:.1%}\n' \
              f'F1 = {F1:.3f} | Precision = {Precision:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    axes[0].text(0.02, 0.97, textstr, transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top', bbox=props)

    # Defective wafers
    counts1, bins1, _ = axes[1].hist(defect_scores, bins=bins, color='#FFB6C1', alpha=0.8,
                                      edgecolor='black', label='Defective wafers')
    axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2.5,
                    label=f'Threshold = {threshold:.3f}')
    axes[1].set_xlabel("Anomaly Score", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Defective Wafer Count", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11, loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # Add defect detection info
    textstr = f'Detected: {TP:,}/{TP+FN:,} ({TPR:.1%})\n' \
              f'Missed: {FN:,} defects\n' \
              f'Avg Precision: {PR:.3f}'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.7)
    axes[1].text(0.98, 0.97, textstr, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    output_file = f"wafer_trained_on_normal-{n_trees}trees-{int(desired_TPR*100)}TPR.svg"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\n💾 Visualization saved: {output_file}")
    plt.close()

    return {
        'threshold': threshold,
        'TPR': TPR,
        'FPR': FPR_actual,
        'TNR': TNR,
        'Precision': Precision,
        'F1': F1,
        'avg_precision': PR,
        'confusion_matrix': confusion
    }


if __name__ == '__main__':
    # Load data
    print("Loading wafer data...")
    X = pd.read_csv('wafer_features.csv').values
    y_df = pd.read_csv('wafer_labels.csv')
    y = y_df['is_defective'].values

    print(f"Dataset: {len(X):,} wafers, {X.shape[1]} features\n")

    # Run experiment
    results = train_on_normal_detect_defects(
        X=X,
        y=y,
        sample_size=256,
        n_trees=100,
        desired_TPR=0.90,
        test_size=0.3,
        bins=40
    )

    print(f"\n{'='*70}")
    print("✅ EXPERIMENT COMPLETE")
    print(f"{'='*70}")
