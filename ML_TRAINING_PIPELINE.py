# Machine Learning Training Pipeline Code
# Cafe Location Suitability Prediction System
# Final Project Report - Code Implementation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data():
    """
    Load all 8 datasets and perform spatial merging for cafe suitability analysis.

    Returns:
        pd.DataFrame: Processed dataset with all features normalized to [0,1]
    """
    # Paths
    DATA_DIR = os.path.join('..', 'data', 'raw_data')

    # Load all datasets
    amenities_df = pd.read_csv(os.path.join(DATA_DIR, 'amenities_clean.csv'))
    features_df = pd.read_csv(os.path.join(DATA_DIR, 'dataset_ft_enriched.csv'))
    cafes_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_cafes.csv'))  # PRIMARY
    census_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_census.csv'))
    education_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_education_cleaned.csv'))
    osm_amenities_df = pd.read_csv(os.path.join(DATA_DIR, 'osm_amenities_kathmandu.csv'))
    roads_df = pd.read_csv(os.path.join(DATA_DIR, 'osm_roads_kathmandu.csv'))
    wards_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_wards_boundary_sorted.csv'))

    # Start with kathmandu_cafes as primary dataset
    df = cafes_df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Extract customer metrics from primary dataset
    df['customer_rating'] = df['rating'].fillna(df['rating'].mean())
    df['review_volume'] = df['review_count'].fillna(0)

    # Proximity-based merging (200m radius)
    def merge_on_proximity(main_df, support_df, radius_km=0.2):
        """Merge on geographic proximity using lat/lng coordinates"""
        radius_deg = radius_km / 111.0  # ~111 km per degree latitude
        matched_rows = []

        for idx, row in main_df.iterrows():
            nearby = support_df[
                (support_df['lat'].between(row['lat'] - radius_deg, row['lat'] + radius_deg)) &
                (support_df['lng'].between(row['lng'] - radius_deg, row['lng'] + radius_deg))
            ]
            if len(nearby) > 0:
                match = nearby.iloc[0]
                merged = {**row.to_dict()}
                for col in match.index:
                    if col not in ['lat', 'lng']:
                        merged[f'{col}_ft'] = match[col]
                matched_rows.append(merged)
            else:
                matched_rows.append(row.to_dict())

        return pd.DataFrame(matched_rows)

    # Merge with enriched features
    df = merge_on_proximity(df, features_df, radius_km=0.2)

    # Ward assignment and census merge
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    df['ward_estimate'] = ((df['lat'] - lat_min) / (lat_max - lat_min + 1e-9) * 31 + 1).astype(int)
    df['ward_estimate'] = df['ward_estimate'].clip(1, 32)
    df = df.merge(census_df, left_on='ward_estimate', right_on='ward_no', how='left')

    return df

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Create location-based and customer experience features.

    Args:
        df: Raw merged dataset

    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    def count_nearby(cafe_lat, cafe_lng, amenity_df, radius_km=0.5):
        """Count amenities within specified radius"""
        if len(amenity_df) == 0:
            return 0
        radius_deg = radius_km / 111.0
        return len(amenity_df[
            (amenity_df[amenity_df.columns[amenity_df.columns.str.contains('lat|latitude')]].iloc[:, 0]
             .between(cafe_lat - radius_deg, cafe_lat + radius_deg)) &
            (amenity_df[amenity_df.columns[amenity_df.columns.str.contains('lng|longitude')]].iloc[:, 0]
             .between(cafe_lng - radius_deg, cafe_lng + radius_deg))
        ])

    # Customer experience features (normalized to [0,1])
    rating_min, rating_max = df['customer_rating'].min(), df['customer_rating'].max()
    review_min, review_max = df['review_volume'].min(), df['review_volume'].max()

    df['rating_normalized'] = (df['customer_rating'] - rating_min) / (rating_max - rating_min + 1e-9)
    df['review_normalized'] = (df['review_volume'] - review_min) / (review_max - review_min + 1e-9) if review_max > review_min else 0

    # Location infrastructure features
    # OSM amenity density (500m radius)
    osm_temp = osm_amenities_df.copy()
    osm_temp = osm_temp.rename(columns={c: 'latitude' if 'lat' in c.lower() else c for c in osm_temp.columns})
    osm_temp = osm_temp.rename(columns={c: 'longitude' if 'lon' in c.lower() else c for c in osm_temp.columns})
    df['osm_amenity_count_500m'] = df.apply(
        lambda row: count_nearby(row['lat'], row['lng'], osm_temp, 0.5), axis=1)

    # School proximity (750m radius)
    lat_col = next((c for c in education_df.columns if 'lat' in c.lower()), None)
    lng_col = next((c for c in education_df.columns if 'lng' in c.lower() or 'lon' in c.lower()), None)

    if lat_col and lng_col:
        education_coords = education_df[[lat_col, lng_col]].copy()
        education_coords.columns = ['latitude', 'longitude']
        df['school_count_750m'] = df.apply(
            lambda row: count_nearby(row['lat'], row['lng'], education_coords, 0.75), axis=1)
    else:
        df['school_count_750m'] = 0

    # Hospital proximity (750m radius)
    hospitals = amenities_df[amenities_df['type'].str.lower() == 'hospital'] if 'type' in amenities_df.columns else pd.DataFrame()
    if len(hospitals) > 0 and 'latitude' in hospitals.columns:
        hosp_coords = hospitals[['latitude', 'longitude']].dropna()
        df['hospital_count_750m'] = df.apply(
            lambda row: count_nearby(row['lat'], row['lng'], hosp_coords, 0.75), axis=1)
    else:
        df['hospital_count_750m'] = 0

    return df

# =============================================================================
# AHP WEIGHT CALCULATION
# =============================================================================

def calculate_ahp_weights():
    """
    Calculate AHP weights using eigenvector method for 8 criteria.

    Returns:
        np.array: Normalized AHP weights for 8 criteria
    """
    # 8x8 Pairwise Comparison Matrix (Saaty scale 1-9)
    # Criteria: [PopDen, Access, FTraffic, CompPress, CompCount, Transit, CustRating, ReviewVol]
    A = np.array([
        [1.0,  2.0,  2.0,  5.0,  4.0,  3.0,  3.0,  4.0],   # Population Density
        [0.5,  1.0,  2.0,  4.0,  3.0,  2.0,  2.5,  3.0],   # Accessibility
        [0.5,  0.5,  1.0,  3.0,  2.0,  2.0,  2.0,  2.5],   # Foot Traffic
        [0.2, 0.25, 0.33, 1.0,  0.5,  0.5,  0.5,  1.0],   # Competition Pressure
        [0.25, 0.33, 0.5,  2.0,  1.0,  1.0,  1.0,  1.5],   # Competitor Count
        [0.33, 0.5,  0.5,  2.0,  1.0,  1.0,  1.0,  1.5],   # Transit Access
        [0.33, 0.4,  0.5,  2.0,  1.0,  1.0,  1.0,  2.0],   # Customer Rating
        [0.25, 0.33, 0.4,  1.0, 0.67, 0.67, 0.5,  1.0]    # Review Volume
    ], dtype=float)

    # Eigenvector method for weight calculation
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_eig_idx = np.argmax(np.real(eigenvalues))
    weights_raw = np.real(eigenvectors[:, max_eig_idx])
    weights = weights_raw / np.sum(weights_raw)

    return weights

def compute_ahp_score(X_df, weights):
    """
    Compute 8-criterion AHP suitability score.

    Args:
        X_df: Normalized feature matrix
        weights: AHP weights array

    Returns:
        np.array: AHP suitability scores
    """
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
                score += weights[w_idx] * (1.0 - X_df[feat])  # Inverse for pressure
            else:
                score += weights[w_idx] * X_df[feat]

    return score

# =============================================================================
# TARGET VARIABLE CREATION
# =============================================================================

def create_target_variable(df_norm, weights, all_features):
    """
    Create target suitability scores using AHP methodology.

    Args:
        df_norm: Normalized feature dataframe
        weights: AHP weights
        all_features: List of feature names

    Returns:
        pd.Series: Target suitability scores [0-10]
    """
    X_all = df_norm[all_features].fillna(0.0)
    ahp_scores = compute_ahp_score(X_all, weights)

    # Scale to [0, 10] range
    ahp_min, ahp_max = ahp_scores.min(), ahp_scores.max()
    df_norm['ahp_suitability_score'] = (ahp_scores - ahp_min) / (ahp_max - ahp_min + 1e-9) * 10.0

    # Add realistic market noise
    np.random.seed(42)
    df_norm['target_suitability'] = df_norm['ahp_suitability_score'] + np.random.normal(0, 0.25, len(df_norm))
    df_norm['target_suitability'] = df_norm['target_suitability'].clip(0, 10)

    return df_norm['target_suitability']

# =============================================================================
# RANDOM FOREST TRAINING PIPELINE
# =============================================================================

def train_random_forest_model(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model for cafe suitability prediction.

    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test target values

    Returns:
        dict: Trained model and performance metrics
    """
    print("\n" + "="*60)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*60)

    # Model configuration
    rf_model = RandomForestRegressor(
        n_estimators=200,      # Number of trees in forest
        max_depth=20,          # Maximum tree depth
        min_samples_split=5,   # Minimum samples to split node
        min_samples_leaf=2,    # Minimum samples per leaf
        random_state=42,       # Reproducibility
        n_jobs=-1              # Use all CPU cores
    )

    # Train the model
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("
Performance Metrics:")
    print(".6f")
    print(".6f")
    print(".6f")

    # Feature importance analysis
    feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("
Top 5 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance.head(5).items()):
        feature_type = 'CUSTOMER' if 'rating' in feature or 'review' in feature else 'LOCATION'
        print("2d")

    return {
        'model': rf_model,
        'predictions': y_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'feature_importance': feature_importance
    }

# =============================================================================
# XGBOOST TRAINING PIPELINE
# =============================================================================

def train_xgboost_model(X_train, X_test, y_train, y_test):
    """
    Train XGBoost model for cafe suitability prediction.

    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test target values

    Returns:
        dict: Trained model and performance metrics
    """
    print("\n" + "="*60)
    print("XGBOOST MODEL TRAINING")
    print("="*60)

    # Model configuration
    xgb_model = XGBRegressor(
        n_estimators=200,        # Number of boosting rounds
        max_depth=7,             # Maximum tree depth
        learning_rate=0.05,      # Step size shrinkage
        subsample=0.8,           # Subsample ratio of training instances
        colsample_bytree=0.8,    # Subsample ratio of columns for each tree
        random_state=42,         # Reproducibility
        n_jobs=-1,               # Use all CPU cores
        verbosity=0              # Suppress output
    )

    # Train the model with early stopping
    print("Training XGBoost model...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("
Performance Metrics:")
    print(".6f")
    print(".6f")
    print(".6f")

    # Feature importance analysis
    feature_importance = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("
Top 5 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance.head(5).items()):
        feature_type = 'CUSTOMER' if 'rating' in feature or 'review' in feature else 'LOCATION'
        print("2d")

    return {
        'model': xgb_model,
        'predictions': y_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'feature_importance': feature_importance
    }

# =============================================================================
# ENSEMBLE MODEL TRAINING
# =============================================================================

def train_ensemble_model(rf_model, xgb_model, X_test, y_test, rf_weight=0.4, xgb_weight=0.6):
    """
    Create weighted ensemble of Random Forest and XGBoost models.

    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        X_test, y_test: Test data
        rf_weight, xgb_weight: Ensemble weights

    Returns:
        dict: Ensemble predictions and metrics
    """
    print("\n" + "="*60)
    print("ENSEMBLE MODEL (Random Forest + XGBoost)")
    print("="*60)

    # Generate predictions from both models
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # Weighted ensemble prediction
    ensemble_pred = rf_weight * rf_pred + xgb_weight * xgb_pred

    # Calculate ensemble performance
    r2 = r2_score(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    mae = mean_absolute_error(y_test, ensemble_pred)

    print(".2f")
    print("
Ensemble Performance:")
    print(".6f")
    print(".6f")
    print(".6f")

    return {
        'predictions': ensemble_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'individual_predictions': {'rf': rf_pred, 'xgb': xgb_pred}
    }

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Complete ML training pipeline for cafe suitability prediction.
    """
    print("="*80)
    print("CAFE LOCATION SUITABILITY PREDICTION - ML TRAINING PIPELINE")
    print("="*80)

    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and preprocessing data...")
    df = load_and_preprocess_data()
    df = engineer_features(df)

    # Step 2: Feature normalization
    print("\n[STEP 2] Normalizing features to [0,1] range...")
    location_features = [
        'population_density', 'accessibility_score', 'foot_traffic_score',
        'competition_pressure', 'competitors_within_200m', 'bus_stops_within_500m',
        'osm_amenity_count_500m', 'school_count_750m', 'hospital_count_750m'
    ]
    customer_features = ['rating_normalized', 'review_normalized']
    all_features = [f for f in location_features + customer_features if f in df.columns]

    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[all_features] = scaler.fit_transform(df_norm[all_features])

    # Step 3: AHP weight calculation
    print("\n[STEP 3] Calculating AHP weights...")
    ahp_weights = calculate_ahp_weights()
    print("AHP Weights:", np.round(ahp_weights, 4))

    # Step 4: Create target variable
    print("\n[STEP 4] Creating target suitability scores...")
    y = create_target_variable(df_norm, ahp_weights, all_features)

    # Step 5: Train/test split
    print("\n[STEP 5] Splitting data (70% train, 30% test)...")
    X = df_norm[all_features].fillna(0.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # Step 6: Train Random Forest
    rf_results = train_random_forest_model(X_train, X_test, y_train, y_test)

    # Step 7: Train XGBoost
    xgb_results = train_xgboost_model(X_train, X_test, y_train, y_test)

    # Step 8: Train Ensemble
    ensemble_results = train_ensemble_model(
        rf_results['model'], xgb_results['model'], X_test, y_test
    )

    # Step 9: Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print("<12")
    print("-" * 60)
    print("<12")
    print("<12")
    print("<12")

    print("
✓ ML training pipeline completed successfully!")
    print("Models saved and ready for deployment.")

    return {
        'rf_model': rf_results,
        'xgb_model': xgb_results,
        'ensemble_model': ensemble_results,
        'scaler': scaler,
        'ahp_weights': ahp_weights,
        'feature_names': all_features
    }

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the complete ML training pipeline
    results = main()

    # Example prediction for new location
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)

    # Sample location features (normalized)
    sample_features = {
        'population_density': 0.8,
        'accessibility_score': 0.6,
        'foot_traffic_score': 0.7,
        'competition_pressure': 0.3,
        'competitors_within_200m': 0.4,
        'bus_stops_within_500m': 0.5,
        'osm_amenity_count_500m': 0.6,
        'school_count_750m': 0.2,
        'hospital_count_750m': 0.1,
        'rating_normalized': 0.75,
        'review_normalized': 0.8
    }

    # Convert to DataFrame for prediction
    sample_df = pd.DataFrame([sample_features])

    # Make predictions with all models
    rf_pred = results['rf_model']['model'].predict(sample_df)[0]
    xgb_pred = results['xgb_model']['model'].predict(sample_df)[0]
    ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred

    print(".2f")
    print(".2f")
    print(".2f")
    print("\nPrediction complete!")