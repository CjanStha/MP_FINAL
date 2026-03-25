"""
ML Model Integration for Django API
Loads trained Random Forest and XGBoost models for cafe suitability prediction
Implements ensemble predictions with AHP-weighted features
"""

import os
import json
import pickle
import numpy as np
import warnings
from pathlib import Path
from functools import lru_cache

warnings.filterwarnings('ignore')


class CafeSuitabilityPredictor:
    """
    Unified interface for cafe suitability prediction using trained ML models
    
    Models available:
    - Random Forest v2 (regression): Continuous suitability score [0-10]
    - XGBoost v2 (classification): Discrete tier [Low/Medium/High]
    """
    
    _instance = None  # Singleton pattern
    
    def __new__(cls):
        """Implement singleton to avoid loading models multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load all models and metadata on first instantiation"""
        self.models_dir = Path(__file__).parent.parent / 'ml' / 'models'
        
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        # Load models
        self.rf_model = self._load_pickle('random_forest_v2.pkl')
        self.xgb_model = self._load_pickle('xgboost_v2.pkl')
        self.scaler = self._load_pickle('scaler.pkl')
        
        # Load metadata
        with open(self.models_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_columns = self.metadata['features']
        self.label_encoder = ['Low', 'Medium', 'High']
        
        print(f"✓ Models loaded successfully from {self.models_dir}")
        print(f"  Random Forest v2 R²: {self.metadata['random_forest_v2_r2']:.4f}")
        print(f"  XGBoost v2 Accuracy: {self.metadata['xgboost_v2_accuracy']:.4f}")
    
    def _load_pickle(self, filename):
        """Safely load pickle files with error handling"""
        path = self.models_dir / filename
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {str(e)}")
    
    def validate_features(self, features_dict):
        """
        Validate that all required features are present and valid
        
        Args:
            features_dict: Dictionary with feature values
        
        Returns:
            bool: True if valid
        
        Raises:
            ValueError: If features missing or invalid
        """
        missing = [f for f in self.feature_columns if f not in features_dict]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        invalid = [f for f in self.feature_columns 
                   if not isinstance(features_dict[f], (int, float)) or 
                   not (0 <= features_dict[f] <= 1)]
        if invalid:
            raise ValueError(f"Invalid features (must be [0,1]): {invalid}")
        
        return True
    
    def predict_random_forest(self, features_dict):
        """
        Random Forest regression prediction
        
        Args:
            features_dict: {feature_name: value} with 8 AHP features [0-1]
        
        Returns:
            dict: {
                'score': float [0-10],
                'model': 'random_forest_v2',
                'r2': float,
                'mae': float
            }
        """
        self.validate_features(features_dict)
        
        # Create feature array in correct order
        X = np.array([features_dict[f] for f in self.feature_columns]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        score = self.rf_model.predict(X_scaled)[0]
        score_clipped = np.clip(score, 0, 10)
        
        return {
            'score': float(score_clipped),
            'model': 'random_forest_v2',
            'r2': float(self.metadata['random_forest_v2_r2']),
            'mae': float(self.metadata['random_forest_v2_mae']),
            'explanation': f"Cafe suitable for: {'HIGH' if score_clipped > 6.66 else 'MEDIUM' if score_clipped > 3.33 else 'LOW'} tier market"
        }
    
    def predict_xgboost(self, features_dict):
        """
        XGBoost classification prediction
        
        Args:
            features_dict: {feature_name: value} with 8 AHP features [0-1]
        
        Returns:
            dict: {
                'tier': str ('Low' | 'Medium' | 'High'),
                'confidence': float [0.0-1.0],
                'probabilities': {tier: prob},
                'model': 'xgboost_v2',
                'accuracy': float
            }
        """
        self.validate_features(features_dict)
        
        # Create feature array in correct order
        X = np.array([features_dict[f] for f in self.feature_columns]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict class and probabilities
        tier_pred = self.xgb_model.predict(X_scaled)[0]
        tiers = self.label_encoder
        tier_name = tiers[tier_pred]
        
        # Get confidence from probability
        proba = self.xgb_model.predict_proba(X_scaled)[0]
        confidence = float(proba[tier_pred])
        
        probabilities = {tiers[i]: float(proba[i]) for i in range(3)}
        
        return {
            'tier': tier_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'model': 'xgboost_v2',
            'accuracy': float(self.metadata['xgboost_v2_accuracy']),
            'explanation': f"Cafe classified as {tier_name.upper()} suitability (confidence: {confidence:.1%})"
        }
    
    def predict_ensemble(self, features_dict):
        """
        Ensemble prediction combining both models
        
        Uses: 70% XGBoost confidence + 30% Random Forest score normalization
        
        Args:
            features_dict: {feature_name: value} with 8 AHP features [0-1]
        
        Returns:
            dict: {
                'tier': str,
                'score': float [0-10],
                'confidence': float,
                'random_forest': {...},
                'xgboost': {...},
                'recommendation': str
            }
        """
        # Get predictions from both models
        rf_result = self.predict_random_forest(features_dict)
        xgb_result = self.predict_xgboost(features_dict)
        
        # Normalize RF score to probability [0-1]
        rf_prob = rf_result['score'] / 10.0
        
        # Weighted ensemble confidence
        ensemble_confidence = (0.7 * xgb_result['confidence']) + (0.3 * rf_prob)
        
        # Determine recommended tier
        if xgb_result['tier'] == 'High' or rf_result['score'] > 7:
            recommended_tier = 'High'
        elif xgb_result['tier'] == 'Low' or rf_result['score'] < 3:
            recommended_tier = 'Low'
        else:
            recommended_tier = 'Medium'
        
        # Generate recommendation
        recommendations = {
            'High': "EXCELLENT location for new cafe. High foot traffic, low competition, good demographics.",
            'Medium': "GOOD location with moderate potential. Suitable for established brands or niche concepts.",
            'Low': "CHALLENGING location. Consider competitive differentiation or pricing strategy."
        }
        
        return {
            'tier': recommended_tier,
            'score': rf_result['score'],
            'confidence': float(ensemble_confidence),
            'probabilities': xgb_result['probabilities'],
            'random_forest': rf_result,
            'xgboost': xgb_result,
            'recommendation': recommendations[recommended_tier],
            'model_agreement': "Both models agree" if xgb_result['tier'] == recommended_tier else "Models show some disagreement"
        }
    
    def get_feature_importance(self):
        """
        Get AHP-based feature importance weights
        
        Returns:
            dict: {feature_name: weight, ...}
        """
        weights = {}
        for feature, weight in zip(self.feature_columns, self.metadata['ahp_optimized_weights']):
            weights[feature] = float(weight)
        
        # Sort by descending weight
        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_stats(self):
        """
        Get overall model statistics and performance metrics
        
        Returns:
            dict: Complete model information
        """
        return {
            'models_loaded': True,
            'models_directory': str(self.models_dir),
            'random_forest_v2': {
                'r2_score': float(self.metadata['random_forest_v2_r2']),
                'rmse': float(self.metadata['random_forest_v2_rmse']),
                'mae': float(self.metadata['random_forest_v2_mae']),
                'task': 'Regression (suitability score 0-10)'
            },
            'xgboost_v2': {
                'accuracy': float(self.metadata['xgboost_v2_accuracy']),
                'task': 'Classification (Low/Medium/High)',
                'n_classes': 3
            },
            'ahp_metrics': {
                'consistency_ratio': float(self.metadata['ahp_consistency_ratio']),
                'consistency_status': 'GOOD ✓' if float(self.metadata['ahp_consistency_ratio']) < 0.10 else 'POOR ✗',
                'n_criteria': len(self.feature_columns),
                'criteria': self.feature_columns
            },
            'feature_importance': self.get_feature_importance(),
            'training_data': {
                'n_cafes': 2750,
                'n_wards': 32,
                'n_datasets': 8,
                'test_set_size': 550,
                'label_distribution': {
                    'Low': '16.8%',
                    'Medium': '62.5%',
                    'High': '20.7%'
                }
            }
        }


def get_predictor():
    """
    Factory function to get singleton predictor instance
    
    Usage:
        predictor = get_predictor()
        result = predictor.predict_ensemble({...})
    """
    return CafeSuitabilityPredictor()


# Module-level convenience functions
def predict_suitability(features_dict, method='ensemble'):
    """
    Predict cafe suitability using trained models
    
    Args:
        features_dict: Dictionary with 8 AHP features [0-1]:
            - pop_density
            - accessibility
            - foot_traffic
            - competition_pressure
            - competitor_count
            - transit_access
            - rating
            - review_volume
        method: 'ensemble' (default) | 'random_forest' | 'xgboost'
    
    Returns:
        Prediction result dictionary
    
    Example:
        >>> features = {
        ...     'pop_density': 0.75,
        ...     'accessibility': 0.65,
        ...     'foot_traffic': 0.70,
        ...     'competition_pressure': 0.40,
        ...     'competitor_count': 0.35,
        ...     'transit_access': 0.55,
        ...     'rating': 0.80,
        ...     'review_volume': 0.45
        ... }
        >>> result = predict_suitability(features)
        >>> print(result['tier'])  # 'High', 'Medium', or 'Low'
    """
    predictor = get_predictor()
    
    if method == 'ensemble':
        return predictor.predict_ensemble(features_dict)
    elif method == 'random_forest':
        return predictor.predict_random_forest(features_dict)
    elif method == 'xgboost':
        return predictor.predict_xgboost(features_dict)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == '__main__':
    """
    Test script - Run with: python ml_integration.py
    """
    print("\n" + "="*70)
    print("CAFE SUITABILITY PREDICTOR - TEST SUITE")
    print("="*70)
    
    # Initialize predictor
    predictor = get_predictor()
    
    # Print model stats
    stats = predictor.get_model_stats()
    print("\nModel Statistics:")
    print(f"  RF v2 R²: {stats['random_forest_v2']['r2_score']:.4f}")
    print(f"  XGB v2 Accuracy: {stats['xgboost_v2']['accuracy']:.4f}")
    print(f"  AHP CR: {stats['ahp_metrics']['consistency_ratio']:.4f}")
    
    # Test with sample data
    sample_cafe = {
        'pop_density': 0.75,       # High population density
        'accessibility': 0.65,     # Good accessibility
        'foot_traffic': 0.70,      # High foot traffic
        'competition_pressure': 0.40,  # Moderate competition
        'competitor_count': 0.35,  # Few competitors
        'transit_access': 0.55,    # Good transit
        'rating': 0.80,            # Good rating ⭐
        'review_volume': 0.45      # Decent reviews ⭐
    }
    
    print("\n" + "="*70)
    print("TEST: HIGH SUITABILITY LOCATION")
    print("="*70)
    result = predict_suitability(sample_cafe)
    print(f"Tier: {result['tier']} (Confidence: {result['confidence']:.1%})")
    print(f"Score: {result['score']:.2f}/10")
    print(f"Recommendation: {result['recommendation']}")
    
    # Test with low suitability location
    low_cafe = {
        'pop_density': 0.25,
        'accessibility': 0.30,
        'foot_traffic': 0.20,
        'competition_pressure': 0.85,
        'competitor_count': 0.80,
        'transit_access': 0.15,
        'rating': 0.40,
        'review_volume': 0.10
    }
    
    print("\n" + "="*70)
    print("TEST: LOW SUITABILITY LOCATION")
    print("="*70)
    result = predict_suitability(low_cafe)
    print(f"Tier: {result['tier']} (Confidence: {result['confidence']:.1%})")
    print(f"Score: {result['score']:.2f}/10")
    print(f"Recommendation: {result['recommendation']}")
    
    # Print feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (AHP Weights)")
    print("="*70)
    importance = predictor.get_feature_importance()
    for i, (feat, weight) in enumerate(importance.items(), 1):
        print(f"  {i}. {feat:.<30} {weight:>6.2%}")
    
    print("\n✓ Tests completed successfully!")
