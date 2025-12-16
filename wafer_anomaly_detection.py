import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, average_precision_score, classification_report
import matplotlib.pyplot as plt
import time

from iforest import IsolationTreeEnsemble, find_TPR_threshold


def plot_wafer_anomalies(X, y, sample_size=256, n_trees=100, desired_TPR=0.95, bins=20, output_prefix='wafer'):
    """
    Train Isolation Forest on wafer features and visualize anomaly detection results.

    Parameters:
    - X: Feature matrix (pandas DataFrame or numpy array)
    - y: Binary labels (0=normal, 1=defective)
    - sample_size: Subsample size for each isolation tree
    - n_trees: Number of trees in the ensemble
    - desired_TPR: Desired true positive rate for threshold selection
    - bins: Number of bins for histogram
    - output_prefix: Prefix for output files
    """
    N = len(X)

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    print(f"Training Isolation Forest on {N:,} wafers with {X.shape[1]} features")
    print(f"Parameters: sample_size={sample_size}, n_trees={n_trees}")

    # Train Isolation Forest
    it = IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)

    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    print(f"✓ Fit time: {fit_time:.2f}s")

    # Compute anomaly scores
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    print(f"✓ Score time: {score_time:.2f}s")

    # Find optimal threshold
    threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
    print(f"✓ Threshold for {desired_TPR:.2%} TPR: {threshold:.4f} (FPR: {FPR:.4f})")

    # Make predictions
    y_pred = it.predict_from_anomaly_scores(scores, threshold=threshold)

    # Evaluation metrics
    confusion = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(confusion)

    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR_actual = FP / (FP + TN) if (FP + TN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity

    normal_scores = scores[y == 0]
    defect_scores = scores[y == 1]

    F1 = f1_score(y, y_pred)
    PR = average_precision_score(y, scores)

    print(f"\n📊 Performance Metrics:")
    print(f"  True Positive Rate (Recall):  {TPR:.4f} ({TP}/{TP+FN})")
    print(f"  True Negative Rate (Spec.):   {TNR:.4f} ({TN}/{TN+FP})")
    print(f"  False Positive Rate:           {FPR_actual:.4f}")
    print(f"  F1 Score:                      {F1:.4f}")
    print(f"  Average Precision:             {PR:.4f}")
    print(f"\n💡 Semiconductor Context:")
    print(f"  Defects detected: {TP:,}/{TP+FN:,} ({TPR:.1%})")
    print(f"  False alarms (overkill): {FP:,} good wafers marked as defective")
    print(f"  Missed defects (underkill): {FN:,} bad wafers marked as good")

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot normal wafer scores
    if len(normal_scores) > 0:
        counts0, binlocs0, _ = axes[0].hist(normal_scores, color='#90EE90', bins=bins, alpha=0.7, edgecolor='black')
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
        axes[0].set_ylabel("Normal Wafer Count", fontsize=12, fontweight='bold')
        axes[0].set_title(f"Wafer Anomaly Detection - Isolation Forest ({n_trees} trees)", fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Add performance text
        text_x = 0.98
        text_y_start = 0.95
        axes[0].text(text_x, text_y_start, f"N = {N:,} wafers",
                    transform=axes[0].transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0].text(text_x, text_y_start-0.12, f"TPR: {TPR:.2%} | FPR: {FPR_actual:.2%}",
                    transform=axes[0].transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[0].text(text_x, text_y_start-0.24, f"F1: {F1:.3f} | Avg PR: {PR:.3f}",
                    transform=axes[0].transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot defective wafer scores
    if len(defect_scores) > 0:
        counts1, binlocs1, _ = axes[1].hist(defect_scores, color='#FFB6C1', bins=bins, alpha=0.7, edgecolor='black')
        axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
        axes[1].set_xlabel("Anomaly Score", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Defective Wafer Count", fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Add defect info
        defect_ratio = len(defect_scores) / len(normal_scores) if len(normal_scores) > 0 else 0
        axes[1].text(0.02, 0.95, f"Defect Rate: {len(defect_scores)}/{N} = {len(defect_scores)/N:.1%}",
                    transform=axes[1].transAxes, ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_file = f"{output_prefix}-{n_trees}trees-{int(desired_TPR*100)}TPR.svg"
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"\n💾 Visualization saved: {output_file}")

    plt.close()

    return {
        'threshold': threshold,
        'TPR': TPR,
        'FPR': FPR_actual,
        'TNR': TNR,
        'F1': F1,
        'avg_precision': PR,
        'confusion_matrix': confusion,
        'scores': scores,
        'predictions': y_pred
    }


if __name__ == '__main__':
    # Load features and labels
    print("Loading wafer data...")
    X = pd.read_csv('wafer_features.csv')
    y_df = pd.read_csv('wafer_labels.csv')

    print(f"Loaded {len(X):,} wafers with {X.shape[1]} features")
    print(f"Normal wafers: {y_df['is_normal'].sum():,}")
    print(f"Defective wafers: {y_df['is_defective'].sum():,}")

    # Binary classification: detect defects as anomalies
    y_binary = y_df['is_defective'].values

    # Run anomaly detection
    print("\n" + "="*70)
    print("EXPERIMENT: Detecting Defective Wafers")
    print("="*70)

    results = plot_wafer_anomalies(
        X=X,
        y=y_binary,
        sample_size=256,
        n_trees=100,
        desired_TPR=0.95,
        bins=30,
        output_prefix='wafer_defect_detection'
    )

    print("\n" + "="*70)
    print("✅ Wafer Anomaly Detection Complete!")
    print("="*70)
