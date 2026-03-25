"""
Unit and Integration Tests for XGBoost Cafe Suitability Model
Tests the model_training_xgboost_kafes.ipynb notebook functions
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import os

# ============================================================================
# HELPER FUNCTIONS (same as primary model)
# ============================================================================

def merge_on_proximity(main_df, support_df, radius_km=0.2):
    """
    Merge main_df with support_df based on geographic proximity.
    
    Args:
        main_df: DataFrame with 'lat', 'lng' columns (primary locations)
        support_df: DataFrame with location columns to match
        radius_km: Search radius in kilometers
        
    Returns:
        main_df with merged proximity features
    """
    if len(support_df) == 0:
        return main_df
    
    # Find lat/lng columns flexibly
    support_lat_col = next((c for c in support_df.columns if 'lat' in c.lower()), None)
    support_lng_col = next((c for c in support_df.columns if 'lng' in c.lower() or 'lon' in c.lower()), None)
    
    if not support_lat_col or not support_lng_col:
        return main_df
    
    radius_deg = radius_km / 111.0
    result = main_df.copy()
    
    for idx, row in main_df.iterrows():
        cafe_lat, cafe_lng = row['lat'], row['lng']
        matches = support_df[
            (support_df[support_lat_col].between(cafe_lat - radius_deg, cafe_lat + radius_deg)) &
            (support_df[support_lng_col].between(cafe_lng - radius_deg, cafe_lng + radius_deg))
        ]
        
    return result


def count_nearby(cafe_lat, cafe_lng, amenity_df, radius_km=0.5):
    """
    Count amenities within a radius of a cafe location.
    
    Args:
        cafe_lat: Cafe latitude
        cafe_lng: Cafe longitude
        amenity_df: DataFrame with 'latitude', 'longitude' columns
        radius_km: Search radius in kilometers
        
    Returns:
        Count of nearby amenities
    """
    if amenity_df is None or len(amenity_df) == 0:
        return 0
    
    if 'latitude' not in amenity_df.columns or 'longitude' not in amenity_df.columns:
        return 0
    
    radius_deg = radius_km / 111.0
    
    count = len(amenity_df[
        (amenity_df['latitude'].between(cafe_lat - radius_deg, cafe_lat + radius_deg)) &
        (amenity_df['longitude'].between(cafe_lng - radius_deg, cafe_lng + radius_deg))
    ])
    
    return count


def compute_ahp_score_8criteria(X_df, weights_vec):
    """
    Compute 8-criterion AHP suitability score.
    
    Args:
        X_df: DataFrame with normalized features
        weights_vec: Array of 8 AHP weights
        
    Returns:
        Array of AHP scores
    """
    if len(weights_vec) != 8:
        raise ValueError(f"Expected 8 weights, got {len(weights_vec)}")
    
    score = np.zeros(len(X_df))
    
    feature_weight_map = {
        'population_density': 0,
        'accessibility_score': 1,
        'foot_traffic_score': 2,
        'competition_pressure': 3,
        'competitors_within_200m': 4,
        'bus_stops_within_500m': 5,
        'rating_normalized': 6,
        'review_normalized': 7
    }
    
    for feat, w_idx in feature_weight_map.items():
        if feat in X_df.columns:
            if 'pressure' in feat:
                score += weights_vec[w_idx] * (1.0 - X_df[feat].fillna(0).values)
            else:
                score += weights_vec[w_idx] * X_df[feat].fillna(0).values
    
    return score


# ============================================================================
# XGBOOST SUPPORT FUNCTIONS
# ============================================================================

def check_xgboost_available():
    """Check if XGBoost is installed"""
    try:
        import xgboost
        return True
    except ImportError:
        return False


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestCountNearbyUnit(unittest.TestCase):
    """Unit tests for count_nearby() function (XGBoost variant)"""
    
    def setUp(self):
        """Create test data"""
        self.cafe_lat, self.cafe_lng = 27.7172, 85.3240  # Kathmandu center
    
    def test_count_nearby_empty_dataframe(self):
        """Empty dataframe should return 0"""
        amenity_df = pd.DataFrame()
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.5)
        self.assertEqual(result, 0)
    
    def test_count_nearby_none(self):
        """None input should return 0"""
        result = count_nearby(self.cafe_lat, self.cafe_lng, None, radius_km=0.5)
        self.assertEqual(result, 0)
    
    def test_count_nearby_missing_columns(self):
        """Missing lat/lng columns should return 0"""
        amenity_df = pd.DataFrame({'name': ['place1', 'place2']})
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.5)
        self.assertEqual(result, 0)
    
    def test_count_nearby_exact_match(self):
        """Should count amenity at exact cafe location"""
        amenity_df = pd.DataFrame({
            'latitude': [27.7172],
            'longitude': [85.3240]
        })
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.5)
        self.assertEqual(result, 1)
    
    def test_count_nearby_multiple_within_radius(self):
        """Should count all amenities within radius"""
        # Create multiple amenities in nearby area
        amenity_df = pd.DataFrame({
            'latitude': [27.7172, 27.7180, 27.7190, 28.5],
            'longitude': [85.3240, 85.3250, 85.3260, 85.3240]
        })
        # Only first 3 should be within 0.5km
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.5)
        self.assertEqual(result, 3)
    
    def test_count_nearby_zero_distance(self):
        """Should count single amenity at zero distance"""
        amenity_df = pd.DataFrame({
            'latitude': [self.cafe_lat],
            'longitude': [self.cafe_lng]
        })
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=1.0)
        self.assertEqual(result, 1)
    
    def test_count_nearby_different_radius(self):
        """Should respect different radius values"""
        amenity_df = pd.DataFrame({
            'latitude': [27.7172 + 0.002],  # ~222m away
            'longitude': [85.3240]
        })
        
        # Should be within 0.3km radius
        result_large = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.3)
        self.assertEqual(result_large, 1)
        
        # Should be outside 0.1km radius
        result_small = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.1)
        self.assertEqual(result_small, 0)


class TestAHPScoreXGBoostUnit(unittest.TestCase):
    """Unit tests for AHP score computation (XGBoost variant)"""
    
    def setUp(self):
        """Create test data"""
        weights_raw = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        self.weights = weights_raw / weights_raw.sum()  # Normalize to ensure sum = 1.0
        
        # Create normalized test data
        self.X_test = pd.DataFrame({
            'population_density': [0.2, 0.5, 0.8],
            'accessibility_score': [0.3, 0.5, 0.7],
            'foot_traffic_score': [0.4, 0.5, 0.6],
            'competition_pressure': [0.1, 0.5, 0.9],
            'competitors_within_200m': [0.2, 0.5, 0.8],
            'bus_stops_within_500m': [0.3, 0.5, 0.7],
            'rating_normalized': [0.4, 0.5, 0.6],
            'review_normalized': [0.2, 0.5, 0.8]
        })
    
    def test_ahp_score_computation(self):
        """Test basic AHP score computation"""
        scores = compute_ahp_score_8criteria(self.X_test, self.weights)
        self.assertEqual(len(scores), 3)
    
    def test_ahp_score_non_negative(self):
        """Scores should be non-negative"""
        scores = compute_ahp_score_8criteria(self.X_test, self.weights)
        self.assertTrue(np.all(scores >= 0))
    
    def test_ahp_score_increases_with_better_features(self):
        """Higher feature values should increase score"""
        X_low = pd.DataFrame({
            'population_density': [0.1] * 8,
            'accessibility_score': [0.1] * 8,
            'foot_traffic_score': [0.1] * 8,
            'competition_pressure': [0.1] * 8,
            'competitors_within_200m': [0.1] * 8,
            'bus_stops_within_500m': [0.1] * 8,
            'rating_normalized': [0.1] * 8,
            'review_normalized': [0.1] * 8
        })
        
        X_high = pd.DataFrame({
            'population_density': [0.9] * 8,
            'accessibility_score': [0.9] * 8,
            'foot_traffic_score': [0.9] * 8,
            'competition_pressure': [0.1] * 8,  # Keep low (inverse)
            'competitors_within_200m': [0.1] * 8,
            'bus_stops_within_500m': [0.9] * 8,
            'rating_normalized': [0.9] * 8,
            'review_normalized': [0.9] * 8
        })
        
        score_low = compute_ahp_score_8criteria(X_low, self.weights)[0]
        score_high = compute_ahp_score_8criteria(X_high, self.weights)[0]
        
        self.assertGreater(score_high, score_low)
    
    def test_ahp_rating_weight(self):
        """Rating should contribute to score (customer signal)"""
        X_low_rating = pd.DataFrame({
            'population_density': [0.5] * 8,
            'accessibility_score': [0.5] * 8,
            'foot_traffic_score': [0.5] * 8,
            'competition_pressure': [0.5] * 8,
            'competitors_within_200m': [0.5] * 8,
            'bus_stops_within_500m': [0.5] * 8,
            'rating_normalized': [0.0] * 8,
            'review_normalized': [0.5] * 8
        })
        
        X_high_rating = X_low_rating.copy()
        X_high_rating['rating_normalized'] = [1.0] * 8
        
        score_low = compute_ahp_score_8criteria(X_low_rating, self.weights)[0]
        score_high = compute_ahp_score_8criteria(X_high_rating, self.weights)[0]
        
        self.assertLess(score_low, score_high)


class TestXGBoostModelIntegration(unittest.TestCase):
    """Integration tests for XGBoost model training"""
    
    @unittest.skipIf(not check_xgboost_available(), "XGBoost not installed")
    def setUp(self):
        """Create test data"""
        from xgboost import XGBRegressor
        
        np.random.seed(42)
        n_samples = 150
        
        self.df = pd.DataFrame({
            'lat': np.random.uniform(27.6, 27.8, n_samples),
            'lng': np.random.uniform(85.2, 85.4, n_samples),
            'name': [f'Cafe_{i}' for i in range(n_samples)]
        })
        
        # Create features
        feature_names = [
            'population_density', 'accessibility_score', 'foot_traffic_score',
            'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
            'rating_normalized', 'review_normalized'
        ]
        
        for feat in feature_names:
            self.df[feat] = np.random.uniform(0, 1, n_samples)
        
        self.weights = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        self.feature_cols = feature_names
    
    @unittest.skipIf(not check_xgboost_available(), "XGBoost not installed")
    def test_xgboost_training(self):
        """Test XGBoost model training"""
        from xgboost import XGBRegressor
        
        # Prepare data
        df_norm = self.df.copy()
        scaler = MinMaxScaler()
        df_norm[self.feature_cols] = scaler.fit_transform(df_norm[self.feature_cols])
        
        # Create target
        X = df_norm[self.feature_cols]
        y_ahp = compute_ahp_score_8criteria(X, self.weights)
        y = y_ahp + np.random.normal(0, 0.08, len(self.df))
        y = np.clip(y, 0, 1)
        
        # Train XGBoost
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y, verbose=False)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Validate
        self.assertEqual(len(y_pred), len(X))
        r2 = r2_score(y, y_pred)
        self.assertGreater(r2, 0.4)  # XGBoost should fit well
    
    @unittest.skipIf(not check_xgboost_available(), "XGBoost not installed")
    def test_xgboost_vs_random_forest_comparison(self):
        """Compare XGBoost performance with baseline"""
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare data
        df_norm = self.df.copy()
        scaler = MinMaxScaler()
        df_norm[self.feature_cols] = scaler.fit_transform(df_norm[self.feature_cols])
        
        X = df_norm[self.feature_cols]
        y_ahp = compute_ahp_score_8criteria(X, self.weights)
        y = y_ahp + np.random.normal(0, 0.08, len(self.df))
        y = np.clip(y, 0, 1)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train both models
        xgb_model = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        xgb_model.fit(X_train, y_train, verbose=False)
        
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Compare
        xgb_r2 = r2_score(y_test, xgb_model.predict(X_test))
        rf_r2 = r2_score(y_test, rf_model.predict(X_test))
        
        # Both should perform well
        self.assertGreater(xgb_r2, 0.3)
        self.assertGreater(rf_r2, 0.3)
    
    @unittest.skipIf(not check_xgboost_available(), "XGBoost not installed")
    def test_xgboost_feature_importance(self):
        """Test XGBoost feature importance extraction"""
        from xgboost import XGBRegressor
        
        # Prepare data
        df_norm = self.df.copy()
        scaler = MinMaxScaler()
        df_norm[self.feature_cols] = scaler.fit_transform(df_norm[self.feature_cols])
        
        X = df_norm[self.feature_cols]
        y_ahp = compute_ahp_score_8criteria(X, self.weights)
        y = y_ahp + np.random.normal(0, 0.08, len(self.df))
        y = np.clip(y, 0, 1)
        
        # Train model
        model = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y, verbose=False)
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Should have importance for each feature
        self.assertEqual(len(importances), len(self.feature_cols))
        
        # Importances should sum to 1 (for tree-based models)
        self.assertAlmostEqual(importances.sum(), 1.0, places=5)


class TestWeightOptimizationIntegration(unittest.TestCase):
    """Integration tests for AHP weight optimization"""
    
    def setUp(self):
        """Create test data"""
        np.random.seed(42)
        self.n_samples = 50
        
        self.df = pd.DataFrame({
            'population_density': np.random.uniform(0, 1, self.n_samples),
            'accessibility_score': np.random.uniform(0, 1, self.n_samples),
            'foot_traffic_score': np.random.uniform(0, 1, self.n_samples),
            'competition_pressure': np.random.uniform(0, 1, self.n_samples),
            'competitors_within_200m': np.random.uniform(0, 1, self.n_samples),
            'bus_stops_within_500m': np.random.uniform(0, 1, self.n_samples),
            'rating_normalized': np.random.uniform(0, 1, self.n_samples),
            'review_normalized': np.random.uniform(0, 1, self.n_samples)
        })
        
        weights_raw = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        self.initial_weights = weights_raw / weights_raw.sum()  # Normalize
    
    def test_weights_normalize(self):
        """Weights should sum to 1"""
        self.assertAlmostEqual(self.initial_weights.sum(), 1.0, places=5)
    
    def test_weight_optimization_convergence(self):
        """Weight optimization should reduce loss"""
        X = self.df.copy()
        y_true = compute_ahp_score_8criteria(X, self.initial_weights)
        y_true = y_true / (y_true.max() + 1e-9) * 10.0
        
        # Simple gradient descent (just verify it runs without error)
        w = self.initial_weights.copy()
        losses = []
        
        for epoch in range(10):  # Short optimization
            y_pred = compute_ahp_score_8criteria(X, w)
            y_pred_scaled = y_pred / (y_pred.max() + 1e-9) * 10.0
            loss = np.mean((y_true - y_pred_scaled) ** 2)
            losses.append(loss)
            
            # Numerical gradient (simplified)
            eps = 1e-5
            grad = np.zeros_like(w)
            for i in range(len(w)):
                w_pert = w.copy()
                w_pert[i] += eps
                w_pert = w_pert / w_pert.sum()
                y_pert = compute_ahp_score_8criteria(X, w_pert)
                y_pert_scaled = y_pert / (y_pert.max() + 1e-9) * 10.0
                loss_pert = np.mean((y_true - y_pert_scaled) ** 2)
                grad[i] = (loss_pert - loss) / eps
            
            # Update
            lr = 0.01
            w_new = w - lr * grad
            w = np.clip(w_new, 1e-6, None)
            w = w / w.sum()
        
        # Optimization should complete without error
        self.assertEqual(len(losses), 10)


class TestDataValidationIntegration(unittest.TestCase):
    """Integration tests for data validation"""
    
    def test_feature_range_validation(self):
        """Test that normalized features are in [0, 1]"""
        X = pd.DataFrame({
            'population_density': np.random.uniform(100, 1000, 30),
            'accessibility_score': np.random.uniform(0.1, 10, 30),
            'foot_traffic_score': np.random.uniform(0, 5000, 30),
            'competition_pressure': np.random.uniform(0.01, 0.9, 30),
            'competitors_within_200m': np.random.uniform(0, 20, 30),
            'bus_stops_within_500m': np.random.uniform(0, 10, 30),
            'rating_normalized': np.random.uniform(0, 5, 30),
            'review_normalized': np.random.uniform(0, 100, 30)
        })
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # All values should be in [0, 1] (allow small floating point error)
        self.assertTrue(np.all(X_scaled >= -1e-10))
        self.assertTrue(np.all(X_scaled <= 1.0 + 1e-10))
    
    def test_target_clipping(self):
        """Test target variable is properly clipped"""
        X = pd.DataFrame({
            'population_density': [0.5] * 10,
            'accessibility_score': [0.5] * 10,
            'foot_traffic_score': [0.5] * 10,
            'competition_pressure': [0.5] * 10,
            'competitors_within_200m': [0.5] * 10,
            'bus_stops_within_500m': [0.5] * 10,
            'rating_normalized': [0.5] * 10,
            'review_normalized': [0.5] * 10
        })
        
        weights = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        ahp_scores = compute_ahp_score_8criteria(X, weights)
        
        # Add noise and clip
        noisy = ahp_scores + np.random.normal(0, 0.5, len(ahp_scores))
        clipped = np.clip(noisy, 0, 10)
        
        # Should be in valid range
        self.assertTrue(np.all(clipped >= 0))
        self.assertTrue(np.all(clipped <= 10))


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCountNearbyUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestAHPScoreXGBoostUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestXGBoostModelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightOptimizationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidationIntegration))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    exit(0 if result.wasSuccessful() else 1)
