import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist

def extract_wafer_features(wafer_map):
    """
    Extract meaningful features from a 52x52 wafer map for anomaly detection.

    wafer_map values:
    - 0: blank spot
    - 1: normal die (passed electrical test)
    - 2: broken die (failed electrical test)

    Returns: dictionary of features
    """
    features = {}

    # Basic counts
    blank_count = (wafer_map == 0).sum()
    normal_count = (wafer_map == 1).sum()
    failed_count = (wafer_map == 2).sum()
    total_dies = normal_count + failed_count

    features['total_dies'] = total_dies
    features['failed_dies'] = failed_count
    features['normal_dies'] = normal_count
    features['blank_spots'] = blank_count

    # Failure rate
    if total_dies > 0:
        features['failure_rate'] = failed_count / total_dies
    else:
        features['failure_rate'] = 0.0

    # Spatial features
    if failed_count > 0:
        # Get coordinates of failed dies
        failed_coords = np.argwhere(wafer_map == 2)

        # Center of mass of failures
        center_y, center_x = failed_coords.mean(axis=0)
        features['failure_center_y'] = center_y
        features['failure_center_x'] = center_x

        # Distance from wafer center (26, 26)
        wafer_center = np.array([26.0, 26.0])
        features['failure_center_dist'] = np.linalg.norm([center_y, center_x] - wafer_center)

        # Spread/dispersion of failures
        if len(failed_coords) > 1:
            distances = cdist(failed_coords, failed_coords, metric='euclidean')
            features['failure_spread_mean'] = distances.mean()
            features['failure_spread_std'] = distances.std()
            features['failure_spread_max'] = distances.max()
        else:
            features['failure_spread_mean'] = 0.0
            features['failure_spread_std'] = 0.0
            features['failure_spread_max'] = 0.0

        # Radial distribution (divide wafer into rings)
        radii = np.sqrt((failed_coords[:, 0] - 26)**2 + (failed_coords[:, 1] - 26)**2)
        features['failure_radius_mean'] = radii.mean()
        features['failure_radius_std'] = radii.std()
        features['failure_radius_max'] = radii.max()
        features['failure_radius_min'] = radii.min()

        # Edge concentration (outer ring)
        edge_threshold = 20  # Distance from center
        edge_failures = (radii > edge_threshold).sum()
        features['edge_failure_ratio'] = edge_failures / failed_count

        # Center concentration
        center_threshold = 10
        center_failures = (radii < center_threshold).sum()
        features['center_failure_ratio'] = center_failures / failed_count

        # Clustering metric using connected components
        failed_binary = (wafer_map == 2).astype(int)
        labeled_array, num_clusters = ndimage.label(failed_binary)
        features['num_failure_clusters'] = num_clusters
        features['avg_cluster_size'] = failed_count / num_clusters if num_clusters > 0 else 0

    else:
        # No failures - set default values
        features['failure_center_y'] = 26.0
        features['failure_center_x'] = 26.0
        features['failure_center_dist'] = 0.0
        features['failure_spread_mean'] = 0.0
        features['failure_spread_std'] = 0.0
        features['failure_spread_max'] = 0.0
        features['failure_radius_mean'] = 0.0
        features['failure_radius_std'] = 0.0
        features['failure_radius_max'] = 0.0
        features['failure_radius_min'] = 0.0
        features['edge_failure_ratio'] = 0.0
        features['center_failure_ratio'] = 0.0
        features['num_failure_clusters'] = 0
        features['avg_cluster_size'] = 0.0

    return features


def process_wafer_dataset(npz_file='archive/Wafer_Map_Datasets.npz'):
    """
    Load wafer dataset and extract features for all samples.

    Returns:
    - X: DataFrame of features
    - y: labels (one-hot encoded defect types)
    - wafer_maps: original wafer map images
    """
    print("Loading wafer dataset...")
    data = np.load(npz_file, allow_pickle=True)
    wafer_maps = data['arr_0']  # (38015, 52, 52)
    labels = data['arr_1']       # (38015, 8)

    print(f"Loaded {len(wafer_maps):,} wafer maps")
    print("Extracting features...")

    # Extract features for all wafers
    features_list = []
    for i, wafer_map in enumerate(wafer_maps):
        if i % 5000 == 0:
            print(f"  Processed {i:,}/{len(wafer_maps):,} wafers...")
        features = extract_wafer_features(wafer_map)
        features_list.append(features)

    print("Creating DataFrame...")
    X = pd.DataFrame(features_list)

    # Add defect type columns to labels
    defect_names = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random']
    y_df = pd.DataFrame(labels, columns=defect_names)
    y_df['is_normal'] = (labels.sum(axis=1) == 0).astype(int)
    y_df['is_defective'] = (labels.sum(axis=1) > 0).astype(int)
    y_df['num_defect_types'] = labels.sum(axis=1)

    print(f"\nFeature extraction complete!")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y_df.shape}")

    return X, y_df, wafer_maps


if __name__ == '__main__':
    # Extract features
    X, y, wafer_maps = process_wafer_dataset()

    # Save to CSV for easy loading
    print("\nSaving to CSV...")
    X.to_csv('wafer_features.csv', index=False)
    y.to_csv('wafer_labels.csv', index=False)

    print("\nFeature statistics:")
    print(X.describe())

    print("\nLabel distribution:")
    print(y[['is_normal', 'is_defective', 'num_defect_types']].describe())
    print("\nDefect type counts:")
    defect_names = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random']
    for col in defect_names:
        print(f"  {col:12s}: {y[col].sum():5d}")

    print("\nFiles saved:")
    print("  - wafer_features.csv")
    print("  - wafer_labels.csv")
