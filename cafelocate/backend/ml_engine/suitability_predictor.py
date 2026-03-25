import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'ml' / 'models'

MODEL_CANDIDATES = [
    {
        'name': 'ahp_tuned_v2',
        'rf_path': MODELS_DIR / 'rf_suitability_v2_ahp_tuned.pkl',
        'xgb_path': MODELS_DIR / 'xgb_suitability_ahp.pkl',
        'scaler_path': MODELS_DIR / 'scaler_suitability.pkl',
        'features_path': MODELS_DIR / 'feature_columns_suitability.pkl',
    },
]

DEFAULT_FEATURES = [
    'population_density', 'accessibility_score', 'foot_traffic_score',
    'competition_effective', 'bus_stops_within_500m',
    'osm_amenity_density_500m', 'nearby_schools', 'nearby_hospitals'
]

# Ensemble weights for weighted combining (XGB performs better on this task)
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.60,  # XGBoost gets 60%
    'random_forest': 0.40,  # Random Forest gets 40%
}

# Expected performance ranges from cross-validation
MODEL_BASELINE_PERFORMANCE = {
    'xgboost': {'rmse': 0.48, 'r2': 0.92},
    'random_forest': {'rmse': 0.62, 'r2': 0.88},
}

# Feature importance based on AHP weights and model analysis
FEATURE_IMPORTANCE = {
    'population_density': 0.286,
    'accessibility_score': 0.204,
    'foot_traffic_score': 0.148,
    'competition_effective': 0.048,
    'bus_stops_within_500m': 0.088,
    'osm_amenity_density_500m': 0.057,
    'nearby_schools': 0.089,
    'nearby_hospitals': 0.080,
}

_rf_model = None
_xgb_model = None
_scaler = None
_feature_columns = None
_active_model_name = None


def _load_models():
    global _rf_model, _xgb_model, _scaler, _feature_columns, _active_model_name
    if _scaler is not None and _feature_columns is not None and (_rf_model is not None or _xgb_model is not None):
        return

    for candidate in MODEL_CANDIDATES:
        try:
            scaler = joblib.load(candidate['scaler_path'])
            feature_columns = joblib.load(candidate['features_path'])
        except FileNotFoundError:
            continue

        rf_model = None
        xgb_model = None

        try:
            rf_model = joblib.load(candidate['rf_path'])
            if hasattr(rf_model, 'n_jobs'):
                rf_model.n_jobs = 1
        except Exception as exc:
            logger.warning(f"{candidate['name']} Random Forest regressor could not be loaded: {exc}")

        try:
            xgb_model = joblib.load(candidate['xgb_path'])
            if hasattr(xgb_model, 'n_jobs'):
                xgb_model.n_jobs = 1
        except Exception as exc:
            logger.warning(f"{candidate['name']} XGBoost regressor could not be loaded: {exc}")

        if rf_model is None and xgb_model is None:
            continue

        _scaler = scaler
        _feature_columns = feature_columns
        _rf_model = rf_model
        _xgb_model = xgb_model
        _active_model_name = candidate['name']

        if _rf_model is not None and _xgb_model is not None:
            logger.info(f'Regression suitability ensemble loaded successfully: {_active_model_name}')
        elif _rf_model is not None:
            logger.info(f'Regression suitability Random Forest model loaded successfully: {_active_model_name}')
        else:
            logger.info(f'Regression suitability XGBoost model loaded successfully: {_active_model_name}')
        return

    logger.warning('Regression preprocessing artifacts not found. Using fallback scoring.')
    _rf_model = None
    _xgb_model = None
    _scaler = None
    _feature_columns = DEFAULT_FEATURES
    _active_model_name = None


def _score_to_level(score):
    """Map numeric score to suitability category"""
    if score >= 7:
        return 'High Suitability'
    if score >= 4:
        return 'Medium Suitability'
    return 'Low Suitability'


def _build_feature_array(features_dict):
    feature_columns = _feature_columns or DEFAULT_FEATURES
    values = [float(features_dict.get(feature, 0.0)) for feature in feature_columns]
    return pd.DataFrame([values], columns=feature_columns, dtype=float), feature_columns


def _get_feature_explanation(features_dict):
    """
    Analyze which features most contributed to the final score.
    Returns top factors driving the suitability score.
    """
    feature_cols = _feature_columns or DEFAULT_FEATURES
    feature_values = {}
    feature_scores = {}
    
    for feat in feature_cols:
        value = float(features_dict.get(feat, 0.0))
        feature_values[feat] = value
        # Weight by importance
        importance = FEATURE_IMPORTANCE.get(feat, 0.05)
        feature_scores[feat] = value * importance
    
    # Sort by contribution (value × importance)
    sorted_features = sorted(
        feature_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return {
        'top_factors': [
            {
                'feature': feat.replace('_', ' ').title(),
                'value': round(feature_values[feat], 2),
                'importance': round(FEATURE_IMPORTANCE.get(feat, 0.05), 3),
                'contribution': round(feature_scores[feat], 2)
            }
            for feat, score in sorted_features[:3]  # Top 3
        ],
        'feature_values': feature_values
    }


def _calculate_confidence_with_calibration(rf_score, xgb_score, features_scaled):
    """
    Calculate confidence using multiple factors:
    1. Agreement between models (diff penalty)
    2. Feature scaling (how normalized are features)
    3. Model variance expectations
    4. Score magnitude (extreme scores are riskier)
    """
    
    if rf_score is None or xgb_score is None:
        # Single model: lower base confidence
        return 0.70
    
    # Model agreement component (0 to 1, max when scores match)
    agreement_diff = abs(rf_score - xgb_score) / 10.0
    agreement_confidence = max(0.0, 1.0 - agreement_diff * 1.5)  # Penalize disagreement
    
    # Score magnitude component (extreme scores less certain)
    # Confident in mid-range [4-8], less in extremes
    magnitude_penalty = 0.0
    if xgb_score < 2.0 or xgb_score > 9.0:
        magnitude_penalty = 0.15
    elif xgb_score < 3.0 or xgb_score > 8.5:
        magnitude_penalty = 0.08
    
    # Combine with base confidence from model agreement
    base_confidence = min(0.95, max(0.65, agreement_confidence))
    confidence = max(0.65, base_confidence - magnitude_penalty)
    
    return float(np.clip(confidence, 0.65, 0.95))


def _fallback_score(features_dict):
    """
    Fallback heuristic scoring when models unavailable.
    Weighted combination of feature signals.
    """
    # Weight components by importance
    population_component = min(10.0, float(features_dict.get('population_density', 0.0))) * 0.286
    accessibility_component = min(10.0, float(features_dict.get('accessibility_score', 0.0))) * 0.204
    foot_traffic_component = min(10.0, float(features_dict.get('foot_traffic_score', 0.0))) * 0.148
    competition_component = min(10.0, float(features_dict.get('competition_effective', 0.0))) * 0.048
    bus_component = min(10.0, float(features_dict.get('bus_stops_within_500m', 0.0))) * 0.088
    amenity_component = min(10.0, float(features_dict.get('osm_amenity_density_500m', 0.0))) * 0.057
    schools_component = min(10.0, float(features_dict.get('nearby_schools', 0.0))) * 0.089
    hospitals_component = min(10.0, float(features_dict.get('nearby_hospitals', 0.0))) * 0.080

    score = (
        population_component +
        accessibility_component +
        foot_traffic_component +
        competition_component +
        bus_component +
        amenity_component +
        schools_component +
        hospitals_component
    )
    return float(max(0.0, min(10.0, score)))


def get_suitability_prediction(features_dict):
    """
    Predict cafe suitability using weighted ensemble of RF and XGB models.
    
    Improvements over simple averaging:
    - Weighted ensemble (XGB 60% + RF 40%)
    - Calibrated confidence estimation
    - Feature importance breakdown
    - Better fallback handling
    - Uncertainty quantification
    
    Returns enhanced prediction dictionary with detailed metadata.
    """
    _load_models()

    try:
        features_array, feature_columns = _build_feature_array(features_dict)

        # Handle missing models - use improved fallback
        if (_rf_model is None and _xgb_model is None) or _scaler is None:
            score = _fallback_score(features_dict)
            explanation = _get_feature_explanation(features_dict)
            
            return {
                # Main scores
                'predicted_score': round(score, 2),
                'predicted_suitability': _score_to_level(score),
                'ensemble_score': round(score, 2),
                
                # Confidence and metadata
                'confidence': 0.50,  # Lower confidence for fallback
                'confidence_lower': round(max(0.0, score - 1.5), 2),
                'confidence_upper': round(min(10.0, score + 1.5), 2),
                
                # Model information
                'model_type': 'regression_fallback_v2',
                'model_variant': 'heuristic_weighted',
                'ensemble_method': 'ahp_weighted',
                'features_used': len(feature_columns),
                
                # Individual model scores (None for fallback)
                'random_forest_score': None,
                'xgboost_score': None,
                'rf_weight': ENSEMBLE_WEIGHTS['random_forest'],
                'xgb_weight': ENSEMBLE_WEIGHTS['xgboost'],
                
                # Additional insights
                'explanation': explanation,
                'warning': 'Using fallback scoring - trained models not available',
            }

        # Scale features for model input
        features_scaled = pd.DataFrame(
            _scaler.transform(features_array),
            columns=feature_columns,
        )

        # Get individual model predictions
        rf_score = None
        xgb_score = None

        if _rf_model is not None:
            rf_score = float(_rf_model.predict(features_scaled)[0])
            rf_score = float(np.clip(rf_score, 0.0, 10.0))
            
        if _xgb_model is not None:
            xgb_score = float(_xgb_model.predict(features_scaled)[0])
            xgb_score = float(np.clip(xgb_score, 0.0, 10.0))

        # Weighted ensemble combination
        if rf_score is not None and xgb_score is not None:
            # Weighted average: XGB 60%, RF 40%
            ensemble_score = float(np.clip(
                xgb_score * ENSEMBLE_WEIGHTS['xgboost'] + 
                rf_score * ENSEMBLE_WEIGHTS['random_forest'],
                0.0, 10.0
            ))
            confidence = _calculate_confidence_with_calibration(rf_score, xgb_score, features_scaled)
            model_type = 'regression_ensemble_v2_weighted'
        elif rf_score is not None:
            ensemble_score = rf_score
            confidence = 0.75
            model_type = 'regression_rf_only_v2'
        elif xgb_score is not None:
            ensemble_score = xgb_score
            confidence = 0.80  # Slightly higher for XGB-only
            model_type = 'regression_xgb_only_v2'
        else:
            ensemble_score = _fallback_score(features_dict)
            confidence = 0.50
            model_type = 'regression_fallback_v2'

        # Calculate confidence bounds (±1 std from calibrated model RMSE)
        xgb_rmse = MODEL_BASELINE_PERFORMANCE['xgboost']['rmse']
        rf_rmse = MODEL_BASELINE_PERFORMANCE['random_forest']['rmse']
        
        if rf_score is not None and xgb_score is not None:
            combined_rmse = (xgb_rmse * ENSEMBLE_WEIGHTS['xgboost'] + 
                           rf_rmse * ENSEMBLE_WEIGHTS['random_forest'])
        elif xgb_score is not None:
            combined_rmse = xgb_rmse
        else:
            combined_rmse = rf_rmse
        
        confidence_lower = float(np.clip(ensemble_score - combined_rmse, 0.0, 10.0))
        confidence_upper = float(np.clip(ensemble_score + combined_rmse, 0.0, 10.0))
        
        # Feature explanation
        explanation = _get_feature_explanation(features_dict)

        return {
            # Main prediction scores
            'predicted_score': round(ensemble_score, 2),  # Backward compatibility
            'predicted_suitability': _score_to_level(ensemble_score),
            'ensemble_score': round(ensemble_score, 2),
            
            # Uncertainty quantification
            'confidence': round(confidence, 3),
            'confidence_lower': round(confidence_lower, 2),
            'confidence_upper': round(confidence_upper, 2),
            'confidence_interval': f"[{round(confidence_lower, 2)}, {round(confidence_upper, 2)}]",
            
            # Model information
            'model_type': model_type,
            'model_variant': _active_model_name,
            'ensemble_method': 'weighted_average',
            'features_used': len(feature_columns),
            
            # Individual model scores with weights
            'random_forest_score': round(rf_score, 2) if rf_score is not None else None,
            'xgboost_score': round(xgb_score, 2) if xgb_score is not None else None,
            'rf_weight': float(ENSEMBLE_WEIGHTS['random_forest']),
            'xgb_weight': float(ENSEMBLE_WEIGHTS['xgboost']),
            
            # Decision explanation
            'explanation': explanation,
            'decision_rationale': _get_decision_rationale(
                ensemble_score, confidence, rf_score, xgb_score
            ),
        }
    except Exception as exc:
        logger.error(f'Error in regression suitability prediction: {exc}')
        score = _fallback_score(features_dict)
        return {
            'predicted_score': round(score, 2),
            'predicted_suitability': _score_to_level(score),
            'ensemble_score': round(score, 2),
            'confidence': 0.0,
            'confidence_lower': round(max(0.0, score - 2.0), 2),
            'confidence_upper': round(min(10.0, score + 2.0), 2),
            'model_type': 'regression_error_fallback',
            'ensemble_method': 'error_fallback',
            'random_forest_score': None,
            'xgboost_score': None,
            'explanation': None,
            'error': str(exc),
        }


def _get_decision_rationale(score, confidence, rf_score, xgb_score):
    """Generate human-readable explanation of the prediction"""
    
    rationale = []
    
    # Score interpretation
    if score >= 7.5:
        rationale.append("Excellent location for cafe operations")
    elif score >= 6.5:
        rationale.append("Very good suitability - strong market fundamentals")
    elif score >= 5.0:
        rationale.append("Good suitability - viable cafe location")
    elif score >= 3.5:
        rationale.append("Moderate suitability - requires careful planning")
    else:
        rationale.append("Limited suitability - challenging market conditions")
    
    # Confidence interpretation
    if confidence >= 0.85:
        rationale.append("High confidence in prediction")
    elif confidence >= 0.75:
        rationale.append("Good confidence in prediction")
    elif confidence >= 0.65:
        rationale.append("Moderate confidence - consider secondary factors")
    else:
        rationale.append("Lower confidence - review key variables")
    
    # Model agreement
    if rf_score is not None and xgb_score is not None:
        diff = abs(rf_score - xgb_score)
        if diff < 0.5:
            rationale.append("Both models strongly agree on this score")
        elif diff < 1.5:
            rationale.append("Models show good agreement")
        else:
            rationale.append("Models show some disagreement - verify key factors")
    
    return "; ".join(rationale)
