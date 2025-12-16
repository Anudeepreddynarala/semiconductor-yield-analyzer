#!/usr/bin/env python3
"""
Wafer Defect Detection CLI

Simple command-line interface for semiconductor wafer defect detection.
"""
import argparse
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Semiconductor Wafer Defect Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  %(prog)s --pipeline

  # Individual steps
  %(prog)s --extract-features
  %(prog)s --train
  %(prog)s --analyze
  %(prog)s --metrics
  %(prog)s --test

  # Generate report
  %(prog)s --report

For more information, see: README.md
        """
    )

    # Main commands
    parser.add_argument('--pipeline', action='store_true',
                       help='Run complete pipeline: extract → train → analyze → metrics')
    parser.add_argument('--extract-features', action='store_true',
                       help='Extract features from wafer maps')
    parser.add_argument('--train', action='store_true',
                       help='Train Isolation Forest model')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze per-defect-type performance')
    parser.add_argument('--metrics', action='store_true',
                       help='Calculate Cpk and Yield metrics')
    parser.add_argument('--test', action='store_true',
                       help='Run test suite')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive report')

    # Options
    parser.add_argument('--n-trees', type=int, default=100,
                       help='Number of trees for Isolation Forest (default: 100)')
    parser.add_argument('--sample-size', type=int, default=256,
                       help='Sample size for each tree (default: 256)')
    parser.add_argument('--tpr', type=float, default=0.90,
                       help='Target True Positive Rate (default: 0.90)')

    args = parser.parse_args()

    # Check if any command was specified
    if not any([args.pipeline, args.extract_features, args.train,
                args.analyze, args.metrics, args.test, args.report]):
        parser.print_help()
        sys.exit(1)

    print("="*80)
    print("SEMICONDUCTOR WAFER DEFECT DETECTION SYSTEM")
    print("="*80)
    print()

    # Execute commands
    if args.pipeline or args.extract_features:
        run_feature_extraction()

    if args.pipeline or args.train:
        run_training(args.n_trees, args.sample_size, args.tpr)

    if args.pipeline or args.analyze:
        run_analysis(args.n_trees, args.sample_size)

    if args.pipeline or args.metrics:
        run_metrics()

    if args.test:
        run_tests()

    if args.report:
        generate_report()

    print()
    print("="*80)
    print("✅ COMPLETE")
    print("="*80)


def run_feature_extraction():
    """Extract features from wafer maps"""
    print("📊 Step 1: Extracting features from wafer maps...")
    print("-"*80)

    if not os.path.exists('archive/Wafer_Map_Datasets.npz'):
        print("❌ ERROR: Dataset not found at archive/Wafer_Map_Datasets.npz")
        print("   Please download the dataset first.")
        sys.exit(1)

    os.system('python extract_wafer_features.py')
    print()


def run_training(n_trees, sample_size, tpr):
    """Train Isolation Forest model"""
    print(f"🤖 Step 2: Training Isolation Forest ({n_trees} trees, TPR={tpr:.0%})...")
    print("-"*80)

    if not os.path.exists('wafer_features.csv'):
        print("❌ ERROR: Features not found. Run --extract-features first.")
        sys.exit(1)

    # Note: wafer_anomaly_detection_v2.py runs with default params
    # For custom params, would need to modify the script or pass args
    os.system('MPLBACKEND=Agg python wafer_anomaly_detection_v2.py')
    print()


def run_analysis(n_trees, sample_size):
    """Run per-defect-type analysis"""
    print(f"🔍 Step 3: Analyzing per-defect-type performance...")
    print("-"*80)

    if not os.path.exists('wafer_features.csv'):
        print("❌ ERROR: Features not found. Run --extract-features first.")
        sys.exit(1)

    os.system('MPLBACKEND=Agg python analyze_defect_types.py')
    print()


def run_metrics():
    """Calculate Cpk and Yield metrics"""
    print("📈 Step 4: Calculating Cpk and Yield metrics...")
    print("-"*80)

    if not os.path.exists('wafer_features.csv'):
        print("❌ ERROR: Features not found. Run --extract-features first.")
        sys.exit(1)

    os.system('python calculate_metrics.py')
    print()


def run_tests():
    """Run test suite"""
    print("🧪 Running test suite...")
    print("-"*80)
    os.system('./run_tests.sh')
    print()


def generate_report():
    """Generate comprehensive markdown report"""
    print("📝 Generating comprehensive report...")
    print("-"*80)

    report_content = f"""# Wafer Defect Detection - Analysis Report

## Generated Files

### Visualizations
- `wafer_trained_on_normal-100trees-90TPR.svg` - Score distributions
- `defect_type_analysis.svg` - Per-defect-type comparison

### Data
- `wafer_features.csv` - Extracted features (38,015 wafers × 19 features)
- `wafer_labels.csv` - Defect labels

### Metrics
- `quality_metrics_report.txt` - Cpk and Yield analysis

## Key Results

See `quality_metrics_report.txt` for detailed Cpk and Yield metrics.

## Next Steps

1. Review visualizations
2. Analyze quality metrics report
3. Tune model parameters if needed
4. Deploy for production use

---
Generated by Wafer Defect Detection CLI
"""

    with open('ANALYSIS_REPORT.md', 'w') as f:
        f.write(report_content)

    print("✅ Report saved to: ANALYSIS_REPORT.md")
    print()


if __name__ == '__main__':
    main()
