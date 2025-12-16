"""
Unit tests for wafer feature extraction
"""
import pytest
import numpy as np
from extract_wafer_features import extract_wafer_features


class TestFeatureExtraction:
    """Test suite for wafer feature extraction functions"""

    def test_extract_features_normal_wafer(self, sample_wafer_map_normal):
        """Test feature extraction on a normal wafer with no defects"""
        features = extract_wafer_features(sample_wafer_map_normal)

        assert features['failed_dies'] == 0
        assert features['failure_rate'] == 0.0
        assert features['num_failure_clusters'] == 0
        assert isinstance(features, dict)
        assert len(features) == 19  # Should have 19 features

    def test_extract_features_defective_wafer(self, sample_wafer_map):
        """Test feature extraction on a wafer with center defect"""
        features = extract_wafer_features(sample_wafer_map)

        assert features['failed_dies'] > 0
        assert 0 < features['failure_rate'] <= 1.0
        assert features['num_failure_clusters'] > 0
        assert features['total_dies'] > 0

    def test_feature_keys(self, sample_wafer_map):
        """Test that all expected feature keys are present"""
        features = extract_wafer_features(sample_wafer_map)

        expected_keys = [
            'total_dies', 'failed_dies', 'normal_dies', 'blank_spots',
            'failure_rate', 'failure_center_y', 'failure_center_x',
            'failure_center_dist', 'failure_spread_mean', 'failure_spread_std',
            'failure_spread_max', 'failure_radius_mean', 'failure_radius_std',
            'failure_radius_max', 'failure_radius_min', 'edge_failure_ratio',
            'center_failure_ratio', 'num_failure_clusters', 'avg_cluster_size'
        ]

        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_failure_rate_calculation(self, sample_wafer_map):
        """Test that failure rate is calculated correctly"""
        features = extract_wafer_features(sample_wafer_map)

        expected_rate = features['failed_dies'] / features['total_dies']
        assert abs(features['failure_rate'] - expected_rate) < 1e-10

    def test_center_defect_location(self, sample_wafer_map):
        """Test that center defects are correctly identified as central"""
        features = extract_wafer_features(sample_wafer_map)

        # Center defect should have low distance from wafer center
        assert features['failure_center_dist'] < 10.0
        assert features['center_failure_ratio'] > 0.5  # Most failures in center

    def test_edge_defect_location(self, sample_wafer_map_edge_defect):
        """Test that edge defects are correctly identified"""
        features = extract_wafer_features(sample_wafer_map_edge_defect)

        # Edge defect should have high edge ratio
        assert features['edge_failure_ratio'] > 0.5

    def test_feature_value_ranges(self, sample_wafer_map):
        """Test that feature values are within expected ranges"""
        features = extract_wafer_features(sample_wafer_map)

        # Check non-negative values
        assert features['failed_dies'] >= 0
        assert features['normal_dies'] >= 0
        assert features['total_dies'] >= 0
        assert features['blank_spots'] >= 0

        # Check ratios are between 0 and 1
        assert 0 <= features['failure_rate'] <= 1.0
        assert 0 <= features['edge_failure_ratio'] <= 1.0
        assert 0 <= features['center_failure_ratio'] <= 1.0

        # Check spread values are non-negative
        assert features['failure_spread_mean'] >= 0
        assert features['failure_spread_std'] >= 0
        assert features['failure_spread_max'] >= 0

    def test_empty_wafer(self):
        """Test feature extraction on an empty wafer (all zeros)"""
        empty_wafer = np.zeros((52, 52), dtype=np.int32)
        features = extract_wafer_features(empty_wafer)

        assert features['total_dies'] == 0
        assert features['failed_dies'] == 0
        assert features['failure_rate'] == 0.0

    def test_fully_defective_wafer(self):
        """Test feature extraction on a fully defective wafer"""
        defective_wafer = np.full((52, 52), 2, dtype=np.int32)
        features = extract_wafer_features(defective_wafer)

        assert features['failed_dies'] == 52 * 52
        assert features['failure_rate'] == 1.0
        assert features['num_failure_clusters'] > 0

    def test_single_defect(self):
        """Test feature extraction with a single failed die"""
        wafer = np.ones((52, 52), dtype=np.int32)
        wafer[26, 26] = 2  # Single defect at center

        features = extract_wafer_features(wafer)

        assert features['failed_dies'] == 1
        assert features['num_failure_clusters'] == 1
        assert features['avg_cluster_size'] == 1.0
