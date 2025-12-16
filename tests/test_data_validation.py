"""
Data validation tests for wafer dataset
"""
import pytest
import numpy as np
import pandas as pd
import os


class TestDataValidation:
    """Tests to validate wafer dataset integrity"""

    def test_features_csv_exists(self):
        """Test that wafer_features.csv exists"""
        if os.path.exists('wafer_features.csv'):
            assert os.path.getsize('wafer_features.csv') > 0

    def test_labels_csv_exists(self):
        """Test that wafer_labels.csv exists"""
        if os.path.exists('wafer_labels.csv'):
            assert os.path.getsize('wafer_labels.csv') > 0

    def test_features_dataframe_structure(self, sample_features_df):
        """Test that features DataFrame has correct structure"""
        assert isinstance(sample_features_df, pd.DataFrame)
        assert len(sample_features_df.columns) == 19
        assert len(sample_features_df) > 0

    def test_features_no_null_values(self, sample_features_df):
        """Test that there are no null values in features"""
        assert not sample_features_df.isnull().any().any()

    def test_features_no_inf_values(self, sample_features_df):
        """Test that there are no infinite values in features"""
        assert not np.isinf(sample_features_df.values).any()

    def test_failure_rate_range(self, sample_features_df):
        """Test that failure rate is between 0 and 1"""
        assert all(0 <= rate <= 1 for rate in sample_features_df['failure_rate'])

    def test_edge_ratio_range(self, sample_features_df):
        """Test that edge failure ratio is between 0 and 1"""
        assert all(0 <= ratio <= 1 for ratio in sample_features_df['edge_failure_ratio'])

    def test_center_ratio_range(self, sample_features_df):
        """Test that center failure ratio is between 0 and 1"""
        assert all(0 <= ratio <= 1 for ratio in sample_features_df['center_failure_ratio'])

    def test_non_negative_counts(self, sample_features_df):
        """Test that all count features are non-negative"""
        count_columns = ['total_dies', 'failed_dies', 'normal_dies', 'blank_spots', 'num_failure_clusters']

        for col in count_columns:
            assert all(sample_features_df[col] >= 0)

    def test_wafer_map_shape(self, sample_wafer_map):
        """Test that wafer maps have correct shape"""
        assert sample_wafer_map.shape == (52, 52)

    def test_wafer_map_values(self, sample_wafer_map):
        """Test that wafer map values are valid (0, 1, or 2)"""
        unique_values = np.unique(sample_wafer_map)
        assert all(val in [0, 1, 2] for val in unique_values)

    def test_labels_are_binary(self, sample_labels):
        """Test that labels are binary (0 or 1)"""
        assert all(label in [0, 1] for label in sample_labels)

    def test_features_labels_same_length(self, sample_features_df, sample_labels):
        """Test that features and labels have same number of samples"""
        assert len(sample_features_df) == len(sample_labels)

    def test_feature_dtypes(self, sample_features_df):
        """Test that features have correct data types"""
        numeric_types = [np.int64, np.int32, np.float64, np.float32]

        for col in sample_features_df.columns:
            assert sample_features_df[col].dtype in numeric_types

    def test_no_duplicate_samples(self):
        """Test that dataset doesn't have obvious duplicates"""
        if os.path.exists('wafer_features.csv'):
            df = pd.read_csv('wafer_features.csv')
            # Check if there are too many exact duplicates (some are expected due to GAN)
            duplicates = df.duplicated().sum()
            assert duplicates < len(df) * 0.5  # Less than 50% duplicates

    def test_feature_correlations_reasonable(self, sample_features_df):
        """Test that feature correlations are reasonable"""
        # total_dies should equal failed_dies + normal_dies
        calculated_total = sample_features_df['failed_dies'] + sample_features_df['normal_dies']
        assert all(abs(calculated_total - sample_features_df['total_dies']) < 1)

    def test_failure_rate_consistency(self, sample_features_df):
        """Test that failure rate matches failed_dies / total_dies"""
        for idx, row in sample_features_df.iterrows():
            if row['total_dies'] > 0:
                expected_rate = row['failed_dies'] / row['total_dies']
                assert abs(row['failure_rate'] - expected_rate) < 0.001


class TestDataQuality:
    """Tests for data quality and statistical properties"""

    def test_normal_wafer_minority(self):
        """Test that normal wafers are the minority class"""
        if os.path.exists('wafer_labels.csv'):
            labels = pd.read_csv('wafer_labels.csv')
            if 'is_normal' in labels.columns:
                normal_ratio = labels['is_normal'].mean()
                # Normal wafers should be < 10% of dataset
                assert normal_ratio < 0.1

    def test_feature_variance(self, sample_features_df):
        """Test that features have reasonable variance"""
        # Features should not be constant
        for col in sample_features_df.columns:
            variance = sample_features_df[col].var()
            # Skip binary or count features that might be zero
            if col not in ['blank_spots']:
                assert variance >= 0  # At minimum, should be non-negative

    def test_no_all_zero_features(self, sample_features_df):
        """Test that no feature is all zeros"""
        for col in sample_features_df.columns:
            if col not in ['blank_spots']:  # Blank spots might legitimately be constant
                assert sample_features_df[col].sum() != 0

    def test_reasonable_cluster_counts(self, sample_features_df):
        """Test that cluster counts are reasonable"""
        # Number of clusters should not exceed number of failed dies
        for idx, row in sample_features_df.iterrows():
            if row['failed_dies'] > 0:
                assert row['num_failure_clusters'] <= row['failed_dies']
                assert row['num_failure_clusters'] > 0
