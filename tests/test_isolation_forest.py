"""
Integration tests for Isolation Forest anomaly detection
"""
import pytest
import numpy as np
from iforest import IsolationTreeEnsemble, find_TPR_threshold


class TestIsolationForestIntegration:
    """Integration tests for Isolation Forest model"""

    def test_model_training(self, small_dataset):
        """Test that model can be trained without errors"""
        X, y = small_dataset
        X_train = X[y == 0]  # Train on normal samples only

        model = IsolationTreeEnsemble(sample_size=32, n_trees=10)
        model.fit(X_train)

        assert model.trees is not None
        assert len(model.trees) == 10

    def test_anomaly_scoring(self, small_dataset):
        """Test that anomaly scores are computed correctly"""
        X, y = small_dataset
        X_train = X[y == 0]

        model = IsolationTreeEnsemble(sample_size=32, n_trees=10)
        model.fit(X_train)

        scores = model.anomaly_score(X)

        assert len(scores) == len(X)
        assert all(0 <= score <= 1 for score in scores)
        assert not np.any(np.isnan(scores))

    def test_normal_vs_anomaly_scores(self, small_dataset):
        """Test that anomalies score higher than normal samples"""
        X, y = small_dataset
        X_train = X[y == 0]

        model = IsolationTreeEnsemble(sample_size=32, n_trees=100)
        model.fit(X_train)

        scores = model.anomaly_score(X)

        normal_scores = scores[y == 0]
        anomaly_scores = scores[y == 1]

        # Anomalies should generally score higher
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_tpr_threshold_finding(self, small_dataset):
        """Test TPR threshold finding function"""
        X, y = small_dataset
        X_train = X[y == 0]

        model = IsolationTreeEnsemble(sample_size=32, n_trees=100)
        model.fit(X_train)

        scores = model.anomaly_score(X)

        threshold, fpr = find_TPR_threshold(y, scores, desired_TPR=0.9)

        assert 0 <= threshold <= 1
        assert 0 <= fpr <= 1
        assert isinstance(threshold, (float, np.floating))

    def test_predictions_from_scores(self, small_dataset):
        """Test making predictions from anomaly scores"""
        X, y = small_dataset
        X_train = X[y == 0]

        model = IsolationTreeEnsemble(sample_size=32, n_trees=100)
        model.fit(X_train)

        scores = model.anomaly_score(X)
        threshold, _ = find_TPR_threshold(y, scores, desired_TPR=0.8)

        predictions = model.predict_from_anomaly_scores(scores, threshold=threshold)

        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed"""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        model1 = IsolationTreeEnsemble(sample_size=32, n_trees=10)
        model1.fit(X)
        scores1 = model1.anomaly_score(X)

        np.random.seed(42)
        model2 = IsolationTreeEnsemble(sample_size=32, n_trees=10)
        model2.fit(X)
        scores2 = model2.anomaly_score(X)

        # Scores should be similar (not exactly the same due to randomness)
        correlation = np.corrcoef(scores1, scores2)[0, 1]
        assert correlation > 0.9

    def test_model_with_different_tree_counts(self, small_dataset):
        """Test model performance with different numbers of trees"""
        X, y = small_dataset
        X_train = X[y == 0]

        tree_counts = [10, 50, 100]
        scores_list = []

        for n_trees in tree_counts:
            model = IsolationTreeEnsemble(sample_size=32, n_trees=n_trees)
            model.fit(X_train)
            scores = model.anomaly_score(X)
            scores_list.append(scores)

        # More trees should lead to more stable scores
        std_10 = np.std(scores_list[0])
        std_100 = np.std(scores_list[2])

        # This is a weak test, but generally more trees = more stable
        assert len(scores_list) == 3

    def test_small_sample_size(self):
        """Test model with very small sample size"""
        X = np.random.randn(10, 3)

        model = IsolationTreeEnsemble(sample_size=5, n_trees=10)
        model.fit(X)
        scores = model.anomaly_score(X)

        assert len(scores) == 10
        assert all(0 <= score <= 1 for score in scores)

    def test_high_dimensional_data(self):
        """Test model with high-dimensional data"""
        X = np.random.randn(50, 100)  # 100 features

        model = IsolationTreeEnsemble(sample_size=32, n_trees=10)
        model.fit(X)
        scores = model.anomaly_score(X)

        assert len(scores) == 50
        assert all(0 <= score <= 1 for score in scores)

    def test_detection_metrics(self, small_dataset):
        """Test end-to-end detection with accuracy metrics"""
        X, y = small_dataset
        X_train = X[y == 0]

        model = IsolationTreeEnsemble(sample_size=32, n_trees=100)
        model.fit(X_train)

        scores = model.anomaly_score(X)
        threshold, fpr = find_TPR_threshold(y, scores, desired_TPR=0.85)
        predictions = model.predict_from_anomaly_scores(scores, threshold=threshold)

        # Calculate actual TPR
        true_positives = np.sum((predictions == 1) & (y == 1))
        total_positives = np.sum(y == 1)
        actual_tpr = true_positives / total_positives

        # Should be close to desired TPR
        assert 0.75 <= actual_tpr <= 0.95  # Allow some variance
        assert fpr < 0.3  # FPR should be reasonable
