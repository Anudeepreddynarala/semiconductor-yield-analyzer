import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

from iforest import IsolationTreeEnsemble


def analyze_defect_type_detection(X, y_df, sample_size=256, n_trees=100, test_size=0.3):
    """
    Analyze Isolation Forest performance for each defect type.

    Shows which defect patterns are easiest/hardest to detect.
    """
    print(f"{'='*80}")
    print("PER-DEFECT-TYPE ANALYSIS")
    print(f"{'='*80}\n")

    # Separate normal and defective
    is_normal = y_df['is_normal'].values == 1
    X_normal = X[is_normal]
    X_defect = X[~is_normal]
    y_defect_df = y_df[~is_normal]

    # Split normal wafers
    X_normal_train, X_normal_test = train_test_split(X_normal, test_size=test_size, random_state=42)

    print(f"Training on {len(X_normal_train):,} normal wafers...")
    it = IsolationTreeEnsemble(sample_size=min(sample_size, len(X_normal_train)), n_trees=n_trees)
    it.fit(X_normal_train)
    print("✓ Training complete\n")

    # Score normal test set
    scores_normal_test = it.anomaly_score(X_normal_test)

    # Score all defective wafers
    print(f"Scoring {len(X_defect):,} defective wafers...")
    scores_defect = it.anomaly_score(X_defect)
    print("✓ Scoring complete\n")

    # Analyze each defect type
    defect_types = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random']

    results = []

    print(f"{'='*80}")
    print(f"{'Defect Type':<15} {'Count':>8} {'Avg Score':>12} {'Min':>8} {'Max':>8} {'Std':>8}")
    print(f"{'='*80}")

    for defect_type in defect_types:
        # Get wafers with this defect type (may be mixed)
        has_defect = y_defect_df[defect_type].values == 1
        count = has_defect.sum()

        if count == 0:
            continue

        defect_scores = scores_defect[has_defect]

        avg_score = defect_scores.mean()
        min_score = defect_scores.min()
        max_score = defect_scores.max()
        std_score = defect_scores.std()

        results.append({
            'defect_type': defect_type,
            'count': count,
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'std_score': std_score
        })

        print(f"{defect_type:<15} {count:>8,} {avg_score:>12.4f} {min_score:>8.4f} {max_score:>8.4f} {std_score:>8.4f}")

    # Normal wafers for comparison
    print(f"{'-'*80}")
    print(f"{'NORMAL':<15} {len(X_normal_test):>8,} {scores_normal_test.mean():>12.4f} "
          f"{scores_normal_test.min():>8.4f} {scores_normal_test.max():>8.4f} {scores_normal_test.std():>8.4f}")
    print(f"{'='*80}\n")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_score', ascending=False)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of average scores
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(results_df)))
    bars = ax1.barh(results_df['defect_type'], results_df['avg_score'], color=colors, edgecolor='black')
    ax1.set_xlabel('Average Anomaly Score', fontsize=12, fontweight='bold')
    ax1.set_title('Defect Type Detectability\n(Higher score = Easier to detect)', fontsize=13, fontweight='bold')
    ax1.axvline(scores_normal_test.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Normal avg = {scores_normal_test.mean():.3f}')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax1.text(row['avg_score'], i, f" {row['avg_score']:.3f}",
                va='center', fontsize=9, fontweight='bold')

    # Box plot of score distributions
    defect_score_lists = []
    defect_labels = []
    for defect_type in results_df['defect_type']:
        has_defect = y_defect_df[defect_type].values == 1
        defect_score_lists.append(scores_defect[has_defect])
        defect_labels.append(f"{defect_type}\n(n={has_defect.sum()})")

    bp = ax2.boxplot(defect_score_lists, labels=defect_labels, patch_artist=True,
                      notch=True, vert=True, showmeans=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(scores_normal_test.mean(), color='green', linestyle='--', linewidth=2,
                label='Normal avg')
    ax2.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax2.set_title('Score Distribution by Defect Type', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_file = "defect_type_analysis.svg"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"💾 Visualization saved: {output_file}\n")
    plt.close()

    # Detection rate at different thresholds
    print(f"{'='*80}")
    print("DETECTION RATES AT DIFFERENT THRESHOLDS")
    print(f"{'='*80}\n")

    thresholds = [0.45, 0.50, 0.55, 0.60]
    print(f"{'Defect Type':<15}", end='')
    for thresh in thresholds:
        print(f"  @{thresh:.2f}", end='')
    print()
    print(f"{'-'*80}")

    for defect_type in results_df['defect_type']:
        has_defect = y_defect_df[defect_type].values == 1
        defect_scores = scores_defect[has_defect]

        print(f"{defect_type:<15}", end='')
        for thresh in thresholds:
            detection_rate = (defect_scores >= thresh).mean()
            print(f"  {detection_rate:>5.1%}", end='')
        print()

    print(f"{'='*80}\n")

    return results_df


if __name__ == '__main__':
    print("Loading data...")
    X = pd.read_csv('wafer_features.csv').values
    y_df = pd.read_csv('wafer_labels.csv')

    results = analyze_defect_type_detection(
        X=X,
        y_df=y_df,
        sample_size=256,
        n_trees=100,
        test_size=0.3
    )

    print("✅ Per-defect-type analysis complete!")
