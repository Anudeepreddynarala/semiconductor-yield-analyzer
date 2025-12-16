"""
Pytest configuration and fixtures for wafer defect detection tests
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_wafer_map():
    """Create a sample 52x52 wafer map for testing"""
    wafer = np.ones((52, 52), dtype=np.int32)  # All normal dies

    # Add some blank spots (edges)
    wafer[0:5, :] = 0
    wafer[-5:, :] = 0
    wafer[:, 0:5] = 0
    wafer[:, -5:] = 0

    # Add some failed dies (center defect pattern)
    wafer[24:28, 24:28] = 2

    return wafer


@pytest.fixture
def sample_wafer_map_normal():
    """Create a normal wafer with no defects"""
    wafer = np.ones((52, 52), dtype=np.int32)

    # Add blank spots (edges)
    wafer[0:5, :] = 0
    wafer[-5:, :] = 0
    wafer[:, 0:5] = 0
    wafer[:, -5:] = 0

    return wafer


@pytest.fixture
def sample_wafer_map_edge_defect():
    """Create a wafer with edge defect pattern"""
    wafer = np.ones((52, 52), dtype=np.int32)

    # Add blank spots
    wafer[0:5, :] = 0
    wafer[-5:, :] = 0
    wafer[:, 0:5] = 0
    wafer[:, -5:] = 0

    # Add edge failures (ring pattern)
    for i in range(10, 42):
        wafer[10, i] = 2
        wafer[41, i] = 2
        wafer[i, 10] = 2
        wafer[i, 41] = 2

    return wafer


@pytest.fixture
def sample_features_df():
    """Create sample feature DataFrame for testing"""
    data = {
        'total_dies': [2000, 2000, 2000],
        'failed_dies': [100, 50, 200],
        'normal_dies': [1900, 1950, 1800],
        'blank_spots': [704, 704, 704],
        'failure_rate': [0.05, 0.025, 0.1],
        'failure_center_y': [26.0, 26.0, 26.0],
        'failure_center_x': [26.0, 26.0, 26.0],
        'failure_center_dist': [0.5, 0.3, 1.0],
        'failure_spread_mean': [5.0, 3.0, 8.0],
        'failure_spread_std': [2.0, 1.5, 3.0],
        'failure_spread_max': [15.0, 10.0, 20.0],
        'failure_radius_mean': [10.0, 8.0, 15.0],
        'failure_radius_std': [3.0, 2.0, 5.0],
        'failure_radius_max': [20.0, 15.0, 25.0],
        'failure_radius_min': [2.0, 1.0, 3.0],
        'edge_failure_ratio': [0.3, 0.2, 0.5],
        'center_failure_ratio': [0.4, 0.6, 0.3],
        'num_failure_clusters': [10, 5, 15],
        'avg_cluster_size': [10.0, 10.0, 13.3]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels():
    """Create sample labels for testing"""
    return np.array([1, 0, 1])  # defective, normal, defective


@pytest.fixture
def small_dataset():
    """Create a small dataset for integration testing"""
    np.random.seed(42)

    # Generate 100 samples with 5 features
    X_normal = np.random.randn(70, 5) * 0.5 + np.array([0, 0, 0, 0, 0])
    X_defect = np.random.randn(30, 5) * 1.5 + np.array([2, 2, 2, 2, 2])

    X = np.vstack([X_normal, X_defect])
    y = np.hstack([np.zeros(70), np.ones(30)])

    # Shuffle
    indices = np.random.permutation(100)
    X = X[indices]
    y = y[indices]

    return X, y
