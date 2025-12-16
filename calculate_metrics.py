"""
Calculate semiconductor manufacturing quality metrics: Cpk and Yield.

Cpk (Process Capability Index) measures how well a process can produce output
within specification limits. It's critical in semiconductor manufacturing for
assessing process quality and consistency.

Yield is the percentage of wafers that pass quality standards.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_yield(y_true: np.ndarray, y_pred: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate semiconductor yield metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels (0=normal, 1=defective)
    y_pred : array-like, optional
        Predicted labels (0=normal, 1=defective)
        If None, calculates actual yield only

    Returns:
    --------
    dict : Dictionary containing yield metrics
        - actual_yield: Percentage of truly normal wafers
        - predicted_yield: Percentage predicted as normal (if y_pred provided)
        - yield_loss: Difference between actual and predicted yield
    """
    metrics = {}

    # Actual yield (ground truth)
    total_wafers = len(y_true)
    normal_wafers = (y_true == 0).sum()
    metrics['actual_yield_pct'] = (normal_wafers / total_wafers) * 100
    metrics['actual_yield_ratio'] = normal_wafers / total_wafers
    metrics['total_wafers'] = total_wafers
    metrics['normal_wafers'] = int(normal_wafers)
    metrics['defective_wafers'] = int(total_wafers - normal_wafers)

    # Predicted yield (if predictions available)
    if y_pred is not None:
        predicted_normal = (y_pred == 0).sum()
        metrics['predicted_yield_pct'] = (predicted_normal / total_wafers) * 100
        metrics['predicted_yield_ratio'] = predicted_normal / total_wafers
        metrics['yield_loss_pct'] = metrics['actual_yield_pct'] - metrics['predicted_yield_pct']

        # Overkill and underkill
        # Overkill: Good wafers marked as defective (False Positives)
        # Underkill: Bad wafers marked as good (False Negatives)
        overkill = ((y_true == 0) & (y_pred == 1)).sum()
        underkill = ((y_true == 1) & (y_pred == 0)).sum()

        metrics['overkill_count'] = int(overkill)
        metrics['underkill_count'] = int(underkill)
        metrics['overkill_rate_pct'] = (overkill / normal_wafers) * 100 if normal_wafers > 0 else 0
        metrics['underkill_rate_pct'] = (underkill / (total_wafers - normal_wafers)) * 100 if (total_wafers - normal_wafers) > 0 else 0

    return metrics


def calculate_cpk(data: np.ndarray,
                  lower_spec_limit: float,
                  upper_spec_limit: float) -> Dict[str, float]:
    """
    Calculate Process Capability Index (Cpk).

    Cpk measures how well a process can produce output within specification limits.

    Cpk Interpretation:
    - Cpk >= 2.0: Excellent (Six Sigma)
    - Cpk >= 1.67: Very good (Five Sigma)
    - Cpk >= 1.33: Good (Four Sigma) - Industry standard
    - Cpk >= 1.0: Acceptable (Three Sigma)
    - Cpk < 1.0: Poor - Process needs improvement

    Parameters:
    -----------
    data : array-like
        Process measurements (e.g., failure rates, defect counts)
    lower_spec_limit : float
        Lower specification limit (LSL)
    upper_spec_limit : float
        Upper specification limit (USL)

    Returns:
    --------
    dict : Dictionary containing Cpk metrics
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation

    # Cp: Process Capability (potential capability)
    cp = (upper_spec_limit - lower_spec_limit) / (6 * std)

    # Cpk: Process Capability Index (actual capability)
    cpu = (upper_spec_limit - mean) / (3 * std)  # Upper capability
    cpl = (mean - lower_spec_limit) / (3 * std)  # Lower capability
    cpk = min(cpu, cpl)

    # Percentage of measurements within spec
    within_spec = ((data >= lower_spec_limit) & (data <= upper_spec_limit)).sum()
    within_spec_pct = (within_spec / len(data)) * 100

    # Process sigma level (approximate)
    sigma_level = cpk * 3

    # Interpretation
    if cpk >= 2.0:
        interpretation = "Excellent (Six Sigma)"
    elif cpk >= 1.67:
        interpretation = "Very Good (Five Sigma)"
    elif cpk >= 1.33:
        interpretation = "Good (Four Sigma) - Industry Standard"
    elif cpk >= 1.0:
        interpretation = "Acceptable (Three Sigma)"
    else:
        interpretation = "Poor - Process Needs Improvement"

    return {
        'cpk': cpk,
        'cp': cp,
        'cpu': cpu,
        'cpl': cpl,
        'mean': mean,
        'std': std,
        'sigma_level': sigma_level,
        'within_spec_pct': within_spec_pct,
        'lower_spec_limit': lower_spec_limit,
        'upper_spec_limit': upper_spec_limit,
        'interpretation': interpretation,
        'total_samples': len(data)
    }


def calculate_wafer_cpk(features_df: pd.DataFrame,
                        feature_name: str = 'failure_rate',
                        lsl: float = 0.0,
                        usl: float = 0.05) -> Dict[str, float]:
    """
    Calculate Cpk for wafer defect features.

    Example: For failure_rate, LSL=0.0 (no failures), USL=0.05 (5% threshold)

    Parameters:
    -----------
    features_df : DataFrame
        Wafer features DataFrame
    feature_name : str
        Feature to calculate Cpk for (default: 'failure_rate')
    lsl : float
        Lower specification limit
    usl : float
        Upper specification limit

    Returns:
    --------
    dict : Cpk metrics for the specified feature
    """
    data = features_df[feature_name].values
    return calculate_cpk(data, lsl, usl)


def generate_metrics_report(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           features_df: pd.DataFrame) -> str:
    """
    Generate comprehensive quality metrics report.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    features_df : DataFrame
        Wafer features

    Returns:
    --------
    str : Formatted metrics report
    """
    # Calculate metrics
    yield_metrics = calculate_yield(y_true, y_pred)
    cpk_failure_rate = calculate_wafer_cpk(features_df, 'failure_rate', lsl=0.0, usl=0.05)
    cpk_failed_dies = calculate_wafer_cpk(features_df, 'failed_dies', lsl=0, usl=100)

    # Format report
    report = f"""
{'='*80}
SEMICONDUCTOR MANUFACTURING QUALITY METRICS REPORT
{'='*80}

YIELD ANALYSIS
{'─'*80}
Total Wafers:              {yield_metrics['total_wafers']:,}
Normal Wafers:             {yield_metrics['normal_wafers']:,}
Defective Wafers:          {yield_metrics['defective_wafers']:,}

Actual Yield:              {yield_metrics['actual_yield_pct']:.2f}%
Predicted Yield:           {yield_metrics['predicted_yield_pct']:.2f}%
Yield Loss:                {yield_metrics['yield_loss_pct']:.2f}%

QUALITY IMPACT
{'─'*80}
Overkill (False Alarms):   {yield_metrics['overkill_count']:,} wafers ({yield_metrics['overkill_rate_pct']:.2f}%)
Underkill (Escapes):       {yield_metrics['underkill_count']:,} wafers ({yield_metrics['underkill_rate_pct']:.2f}%)

💡 Manufacturing Impact:
   • Overkill Cost: ~${yield_metrics['overkill_count'] * 5000:,} (@ $5k/wafer)
   • Underkill Cost: ~${yield_metrics['underkill_count'] * 50000:,} (@ $50k/wafer escape)

PROCESS CAPABILITY (Cpk) - FAILURE RATE
{'─'*80}
Cpk:                       {cpk_failure_rate['cpk']:.3f}
Cp:                        {cpk_failure_rate['cp']:.3f}
Sigma Level:               {cpk_failure_rate['sigma_level']:.2f}σ

Process Mean:              {cpk_failure_rate['mean']:.4f}
Process Std Dev:           {cpk_failure_rate['std']:.4f}

Specification Limits:      [{cpk_failure_rate['lower_spec_limit']:.2f}, {cpk_failure_rate['upper_spec_limit']:.2f}]
Within Spec:               {cpk_failure_rate['within_spec_pct']:.2f}%

Quality Assessment:        {cpk_failure_rate['interpretation']}

PROCESS CAPABILITY (Cpk) - FAILED DIE COUNT
{'─'*80}
Cpk:                       {cpk_failed_dies['cpk']:.3f}
Process Mean:              {cpk_failed_dies['mean']:.1f} dies
Process Std Dev:           {cpk_failed_dies['std']:.1f} dies
Quality Assessment:        {cpk_failed_dies['interpretation']}

{'='*80}
"""
    return report


if __name__ == '__main__':
    # Example usage
    print("Loading wafer data...")

    try:
        X = pd.read_csv('wafer_features.csv')
        y_df = pd.read_csv('wafer_labels.csv')

        # For demonstration, create sample predictions
        # In real use, load actual predictions from model
        y_true = y_df['is_defective'].values

        # Simulate predictions (replace with actual model predictions)
        np.random.seed(42)
        y_pred = y_true.copy()
        # Simulate some errors
        error_indices = np.random.choice(len(y_pred), size=int(len(y_pred) * 0.08), replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]

        # Generate report
        report = generate_metrics_report(y_true, y_pred, X)
        print(report)

        # Save report
        with open('quality_metrics_report.txt', 'w') as f:
            f.write(report)
        print("✅ Report saved to: quality_metrics_report.txt")

    except FileNotFoundError:
        print("⚠️  Data files not found. Run extract_wafer_features.py first.")
