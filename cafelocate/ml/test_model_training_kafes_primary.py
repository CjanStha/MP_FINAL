"""
Unit and Integration Tests for Random Forest Cafe Suitability Model
Tests the model_training_kafes_primary.ipynb notebook functions
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import os
from io import StringIO

# ============================================================================
# HELPER FUNCTIONS (extracted from notebook)
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
# UNIT TESTS
# ============================================================================

class TestCountNearbyUnit(unittest.TestCase):
    """Unit tests for count_nearby() function"""
    
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
    
    def test_count_nearby_within_radius(self):
        """Should count amenities within radius"""
        # Create amenities within ~0.5km (0.0045 degrees ≈ 500m)
        # Latitude change: 0.004 degrees ≈ 440m, 0.006 degrees ≈ 660m
        amenity_df = pd.DataFrame({
            'latitude': [27.7172, 27.7200, 27.7180],
            'longitude': [85.3240, 85.3250, 85.3240]
        })
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.5)
        self.assertEqual(result, 3)
    
    def test_count_nearby_outside_radius(self):
        """Should exclude amenities outside radius"""
        # Far away location (>1 degree away)
        amenity_df = pd.DataFrame({
            'latitude': [28.7172],
            'longitude': [85.3240]
        })
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.5)
        self.assertEqual(result, 0)
    
    def test_count_nearby_boundary(self):
        """Should correctly handle radius boundary"""
        # Create amenity just outside 0.2km radius
        amenity_df = pd.DataFrame({
            'latitude': [27.7172 + 0.003],  # ~333m away
            'longitude': [85.3240]
        })
        result = count_nearby(self.cafe_lat, self.cafe_lng, amenity_df, radius_km=0.2)
        self.assertEqual(result, 0)


class TestAHPScoreUnit(unittest.TestCase):
    """Unit tests for compute_ahp_score_8criteria() function"""
    
    def setUp(self):
        """Create test data with normalized features"""
        weights_raw = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        self.weights = weights_raw / weights_raw.sum()  # Normalize to ensure sum = 1.0
        
        # Create sample normalized data (8 features)
        self.X_test = pd.DataFrame({
            'population_density': [0.5, 1.0, 0.0],
            'accessibility_score': [0.5, 0.5, 0.5],
            'foot_traffic_score': [0.5, 0.5, 0.5],
            'competition_pressure': [0.5, 0.5, 0.5],
            'competitors_within_200m': [0.0, 1.0, 0.5],
            'bus_stops_within_500m': [0.5, 0.5, 0.5],
            'rating_normalized': [0.5, 0.5, 0.5],
            'review_normalized': [0.5, 0.5, 0.5]
        })
    
    def test_score_shape(self):
        """Output should match input rows"""
        scores = compute_ahp_score_8criteria(self.X_test, self.weights)
        self.assertEqual(len(scores), len(self.X_test))
    
    def test_score_range(self):
        """Scores should be in valid range [0, 1]"""
        scores = compute_ahp_score_8criteria(self.X_test, self.weights)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
    
    def test_score_weights_sum_to_one(self):
        """Weights should be normalized"""
        self.assertAlmostEqual(np.sum(self.weights), 1.0, places=3)
    
    def test_score_uniform_features(self):
        """Uniform features should give uniform scores"""
        X_uniform = pd.DataFrame({
            'population_density': [0.5, 0.5],
            'accessibility_score': [0.5, 0.5],
            'foot_traffic_score': [0.5, 0.5],
            'competition_pressure': [0.5, 0.5],
            'competitors_within_200m': [0.5, 0.5],
            'bus_stops_within_500m': [0.5, 0.5],
            'rating_normalized': [0.5, 0.5],
            'review_normalized': [0.5, 0.5]
        })
        scores = compute_ahp_score_8criteria(X_uniform, self.weights)
        # All scores should be approximately equal
        self.assertAlmostEqual(scores[0], scores[1], places=5)
    
    def test_score_invalid_weights(self):
        """Should raise error for incorrect weight count"""
        bad_weights = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            compute_ahp_score_8criteria(self.X_test, bad_weights)
    
    def test_score_inverted_competition_pressure(self):
        """High competition pressure should reduce score (inverse criterion)"""
        X_low_comp = pd.DataFrame({
            'population_density': [1.0] * 8,
            'accessibility_score': [0.0] * 8,
            'foot_traffic_score': [0.0] * 8,
            'competition_pressure': [0.0] * 8,  # Low pressure
            'competitors_within_200m': [0.0] * 8,
            'bus_stops_within_500m': [0.0] * 8,
            'rating_normalized': [0.0] * 8,
            'review_normalized': [0.0] * 8
        })
        
        X_high_comp = pd.DataFrame({
            'population_density': [1.0] * 8,
            'accessibility_score': [0.0] * 8,
            'foot_traffic_score': [0.0] * 8,
            'competition_pressure': [1.0] * 8,  # High pressure
            'competitors_within_200m': [0.0] * 8,
            'bus_stops_within_500m': [0.0] * 8,
            'rating_normalized': [0.0] * 8,
            'review_normalized': [0.0] * 8
        })
        
        score_low_comp = compute_ahp_score_8criteria(X_low_comp, self.weights)[0]
        score_high_comp = compute_ahp_score_8criteria(X_high_comp, self.weights)[0]
        
        # High competition should give lower score
        self.assertGreater(score_low_comp, score_high_comp)
    
    def test_score_customer_signal_integration(self):
        """Higher ratings/reviews should increase score"""
        X_low_rating = pd.DataFrame({
            'population_density': [0.5] * 8,
            'accessibility_score': [0.5] * 8,
            'foot_traffic_score': [0.5] * 8,
            'competition_pressure': [0.5] * 8,
            'competitors_within_200m': [0.5] * 8,
            'bus_stops_within_500m': [0.5] * 8,
            'rating_normalized': [0.0] * 8,  # Low rating
            'review_normalized': [0.0] * 8   # Low reviews
        })
        
        X_high_rating = pd.DataFrame({
            'population_density': [0.5] * 8,
            'accessibility_score': [0.5] * 8,
            'foot_traffic_score': [0.5] * 8,
            'competition_pressure': [0.5] * 8,
            'competitors_within_200m': [0.5] * 8,
            'bus_stops_within_500m': [0.5] * 8,
            'rating_normalized': [1.0] * 8,  # High rating
            'review_normalized': [1.0] * 8   # High reviews
        })
        
        score_low = compute_ahp_score_8criteria(X_low_rating, self.weights)[0]
        score_high = compute_ahp_score_8criteria(X_high_rating, self.weights)[0]
        
        self.assertGreater(score_high, score_low)


class TestNormalizationUnit(unittest.TestCase):
    """Unit tests for feature normalization"""
    
    def test_minmax_scaler_range(self):
        """MinMaxScaler should normalize to [0, 1]"""
        data = pd.DataFrame({
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [100, 200, 300, 400, 500]
        })
        
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
    
    def test_minmax_scaler_extremes(self):
        """MinMaxScaler min/max should be exact"""
        data = pd.DataFrame({
            'feature1': [0, 50, 100]
        })
        
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestModelTrainingIntegration(unittest.TestCase):
    """Integration tests for full model training pipeline"""
    
    def setUp(self):
        """Create realistic test dataset"""
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic cafe data
        self.df = pd.DataFrame({
            'lat': np.random.uniform(27.6, 27.8, n_samples),
            'lng': np.random.uniform(85.2, 85.4, n_samples),
            'name': [f'Cafe_{i}' for i in range(n_samples)],
            'rating': np.random.uniform(0, 5, n_samples),
            'review_count': np.random.randint(0, 100, n_samples)
        })
        
        # Create feature columns
        feature_names = [
            'population_density', 'accessibility_score', 'foot_traffic_score',
            'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
            'osm_amenity_count_500m', 'school_count_750m', 'hospital_count_750m',
            'rating_normalized', 'review_normalized'
        ]
        
        for feat in feature_names:
            self.df[feat] = np.random.uniform(0, 1, n_samples)
        
        self.weights = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
    
    def test_feature_normalization_pipeline(self):
        """Test full feature normalization"""
        df_copy = self.df.copy()
        
        feature_cols = [
            'population_density', 'accessibility_score', 'foot_traffic_score',
            'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
            'rating_normalized', 'review_normalized'
        ]
        
        scaler = MinMaxScaler()
        df_copy[feature_cols] = scaler.fit_transform(df_copy[feature_cols])
        
        # Check all features normalized
        for col in feature_cols:
            self.assertTrue(df_copy[col].min() >= 0)
            self.assertTrue(df_copy[col].max() <= 1)
    
    def test_ahp_score_generation(self):
        """Test AHP score generation from normalized features"""
        # Normalize features
        feature_cols = [
            'population_density', 'accessibility_score', 'foot_traffic_score',
            'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
            'rating_normalized', 'review_normalized'
        ]
        
        df_norm = self.df.copy()
        scaler = MinMaxScaler()
        df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
        
        # Generate AHP scores
        X = df_norm[feature_cols]
        ahp_scores = compute_ahp_score_8criteria(X, self.weights)
        
        # Validate output
        self.assertEqual(len(ahp_scores), len(self.df))
        self.assertTrue(np.all(ahp_scores >= 0))
        self.assertTrue(np.all(ahp_scores <= 1))
    
    def test_random_forest_training(self):
        """Test RandomForest model training end-to-end"""
        # Prepare data
        feature_cols = [
            'population_density', 'accessibility_score', 'foot_traffic_score',
            'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
            'rating_normalized', 'review_normalized'
        ]
        
        df_norm = self.df.copy()
        scaler = MinMaxScaler()
        df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
        
        # Create target
        X = df_norm[feature_cols]
        y_ahp = compute_ahp_score_8criteria(X, self.weights)
        y = y_ahp + np.random.normal(0, 0.1, len(df_norm))
        y = np.clip(y, 0, 1)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Validate
        self.assertEqual(len(y_pred), len(X))
        r2 = r2_score(y, y_pred)
        self.assertGreater(r2, 0.5)  # Should have reasonable fit
    
    def test_model_generalization(self):
        """Test model generalizes to test set"""
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        feature_cols = [
            'population_density', 'accessibility_score', 'foot_traffic_score',
            'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
            'rating_normalized', 'review_normalized'
        ]
        
        df_norm = self.df.copy()
        scaler = MinMaxScaler()
        df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
        
        # Create target
        X = df_norm[feature_cols]
        y_ahp = compute_ahp_score_8criteria(X, self.weights)
        y = y_ahp + np.random.normal(0, 0.1, len(df_norm))
        y = np.clip(y, 0, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Test set should be reasonable
        self.assertGreater(r2_test, 0.3)
        # Overfitting check (train shouldn't be vastly better than test)
        self.assertLess(r2_train - r2_test, 0.7)


class TestDataIntegrityIntegration(unittest.TestCase):
    """Integration tests for data integrity"""
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        X = pd.DataFrame({
            'population_density': [0.5, np.nan, 0.3, 0.8],
            'accessibility_score': [0.4, 0.6, np.nan, 0.7],
            'foot_traffic_score': [0.5, 0.5, 0.5, 0.5],
            'competition_pressure': [0.2, 0.3, 0.4, 0.5],
            'competitors_within_200m': [0.1, 0.2, 0.3, 0.4],
            'bus_stops_within_500m': [0.6, 0.7, 0.8, 0.9],
            'rating_normalized': [1.0, 1.0, 1.0, 1.0],
            'review_normalized': [1.0, 1.0, 1.0, 1.0]
        })
        
        weights = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        
        # Forward fill or fillna should handle NaN
        X_filled = X.fillna(0.0)
        scores = compute_ahp_score_8criteria(X_filled, weights)
        
        # Should complete without errors
        self.assertEqual(len(scores), len(X))
    
    def test_feature_consistency(self):
        """Test consistency of features across pipeline"""
        np.random.seed(42)
        n = 50
        
        X = pd.DataFrame({
            'population_density': np.random.uniform(0, 1, n),
            'accessibility_score': np.random.uniform(0, 1, n),
            'foot_traffic_score': np.random.uniform(0, 1, n),
            'competition_pressure': np.random.uniform(0, 1, n),
            'competitors_within_200m': np.random.uniform(0, 1, n),
            'bus_stops_within_500m': np.random.uniform(0, 1, n),
            'rating_normalized': np.random.uniform(0, 1, n),
            'review_normalized': np.random.uniform(0, 1, n)
        })
        
        weights = np.array([0.286, 0.204, 0.148, 0.048, 0.081, 0.088, 0.089, 0.057])
        
        # Compute score twice (should be idempotent)
        scores1 = compute_ahp_score_8criteria(X, weights)
        scores2 = compute_ahp_score_8criteria(X, weights)
        
        np.testing.assert_array_almost_equal(scores1, scores2)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCountNearbyUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestAHPScoreUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizationUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrityIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)
