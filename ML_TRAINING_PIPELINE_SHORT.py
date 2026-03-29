import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data():
    """Load all 8 datasets and perform spatial merging"""
    DATA_DIR = os.path.join('..', 'data', 'raw_data')

    # Load datasets
    amenities_df = pd.read_csv(os.path.join(DATA_DIR, 'amenities_clean.csv'))
    features_df = pd.read_csv(os.path.join(DATA_DIR, 'dataset_ft_enriched.csv'))
    cafes_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_cafes.csv'))
    census_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_census.csv'))
    education_df = pd.read_csv(os.path.join(DATA_DIR, 'kathmandu_education_cleaned.csv'))
    osm_amenities_df = pd.read_csv(os.path.join(DATA_DIR, 'osm_amenities_kathmandu.csv'))

    # Start with primary dataset
    df = cafes_df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['customer_rating'] = df['rating'].fillna(df['rating'].mean())
    df['review_volume'] = df['review_count'].fillna(0)

    # Proximity-based merging (200m radius)
    def merge_on_proximity(main_df, support_df, radius_km=0.2):
        radius_deg = radius_km / 111.0
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

    df = merge_on_proximity(df, features_df, radius_km=0.2)

    # Ward assignment and census merge
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    df['ward_estimate'] = ((df['lat'] - lat_min) / (lat_max - lat_min + 1e-9) * 31 + 1).astype(int)
    df['ward_estimate'] = df['ward_estimate'].clip(1, 32)
    df = df.merge(census_df, left_on='ward_estimate', right_on='ward_no', how='left')

    return df

def engineer_features(df):
    """Create location-based and customer experience features"""
    def count_nearby(cafe_lat, cafe_lng, amenity_df, radius_km=0.5):
        if len(amenity_df) == 0:
            return 0
        radius_deg = radius_km / 111.0
        return len(amenity_df[
            (amenity_df[amenity_df.columns[amenity_df.columns.str.contains('lat|latitude')]].iloc[:, 0]
             .between(cafe_lat - radius_deg, cafe_lat + radius_deg)) &
            (amenity_df[amenity_df.columns[amenity_df.columns.str.contains('lng|longitude')]].iloc[:, 0]
             .between(cafe_lng - radius_deg, cafe_lng + radius_deg))
        ])

    # Customer features (normalized)
    rating_min, rating_max = df['customer_rating'].min(), df['customer_rating'].max()
    review_min, review_max = df['review_volume'].min(), df['review_volume'].max()

    df['rating_normalized'] = (df['customer_rating'] - rating_min) / (rating_max - rating_min + 1e-9)
    df['review_normalized'] = (df['review_volume'] - review_min) / (review_max - review_min + 1e-9) if review_max > review_min else 0

    # Location features
    osm_temp = osm_amenities_df.copy()
    osm_temp = osm_temp.rename(columns={c: 'latitude' if 'lat' in c.lower() else c for c in osm_temp.columns})
    osm_temp = osm_temp.rename(columns={c: 'longitude' if 'lon' in c.lower() else c for c in osm_temp.columns})
    df['osm_amenity_count_500m'] = df.apply(
        lambda row: count_nearby(row['lat'], row['lng'], osm_temp, 0.5), axis=1)

    # School proximity
    lat_col = next((c for c in education_df.columns if 'lat' in c.lower()), None)
    lng_col = next((c for c in education_df.columns if 'lng' in c.lower() or 'lon' in c.lower()), None)

    if lat_col and lng_col:
        education_coords = education_df[[lat_col, lng_col]].copy()
        education_coords.columns = ['latitude', 'longitude']
        df['school_count_750m'] = df.apply(
            lambda row: count_nearby(row['lat'], row['lng'], education_coords, 0.75), axis=1)
    else:
        df['school_count_750m'] = 0

    # Hospital proximity
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
    """Calculate AHP weights using eigenvector method"""
    A = np.array([
        [1.0,  2.0,  2.0,  5.0,  4.0,  3.0,  3.0,  4.0],
        [0.5,  1.0,  2.0,  4.0,  3.0,  2.0,  2.5,  3.0],
        [0.5,  0.5,  1.0,  3.0,  2.0,  2.0,  2.0,  2.5],
        [0.2, 0.25, 0.33, 1.0,  0.5,  0.5,  0.5,  1.0],
        [0.25, 0.33, 0.5,  2.0,  1.0,  1.0,  1.0,  1.5],
        [0.33, 0.5,  0.5,  2.0,  1.0,  1.0,  1.0,  1.5],
        [0.33, 0.4,  0.5,  2.0,  1.0,  1.0,  1.0,  2.0],
        [0.25, 0.33, 0.4,  1.0, 0.67, 0.67, 0.5,  1.0]
    ], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_eig_idx = np.argmax(np.real(eigenvalues))
    weights_raw = np.real(eigenvectors[:, max_eig_idx])
    weights = weights_raw / np.sum(weights_raw)

    return weights

def compute_ahp_score(X_df, weights):
    """Compute 8-criterion AHP suitability score"""
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
                score += weights[w_idx] * (1.0 - X_df[feat])
            else:
                score += weights[w_idx] * X_df[feat]

    return score

def create_target_variable(df_norm, weights, all_features):
    """Create target suitability scores using AHP"""
    X_all = df_norm[all_features].fillna(0.0)
    ahp_scores = compute_ahp_score(X_all, weights)

    ahp_min, ahp_max = ahp_scores.min(), ahp_scores.max()
    df_norm['ahp_suitability_score'] = (ahp_scores - ahp_min) / (ahp_max - ahp_min + 1e-9) * 10.0

    np.random.seed(42)
    df_norm['target_suitability'] = df_norm['ahp_suitability_score'] + np.random.normal(0, 0.25, len(df_norm))
    df_norm['target_suitability'] = df_norm['target_suitability'].clip(0, 10)

    return df_norm['target_suitability']

# =============================================================================
# RANDOM FOREST TRAINING
# =============================================================================

def train_random_forest_model(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\nRANDOM FOREST MODEL TRAINING")

    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(".6f")
    print(".6f")
    print(".6f")

    feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("\nTop 5 Features:")
    for i, (feature, importance) in enumerate(feature_importance.head(5).items()):
        print("2d")

    return {
        'model': rf_model, 'predictions': y_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'feature_importance': feature_importance
    }

# =============================================================================
# XGBOOST TRAINING
# =============================================================================

def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Train XGBoost model"""
    print("\nXGBOOST MODEL TRAINING")

    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(".6f")
    print(".6f")
    print(".6f")

    feature_importance = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("\nTop 5 Features:")
    for i, (feature, importance) in enumerate(feature_importance.head(5).items()):
        print("2d")

    return {
        'model': xgb_model, 'predictions': y_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'feature_importance': feature_importance
    }

# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

def train_ensemble_model(rf_model, xgb_model, X_test, y_test, rf_weight=0.4, xgb_weight=0.6):
    """Create weighted ensemble"""
    print("\nENSEMBLE MODEL (RF + XGBoost)")

    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = rf_weight * rf_pred + xgb_weight * xgb_pred

    r2 = r2_score(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    mae = mean_absolute_error(y_test, ensemble_pred)

    print(".2f")
    print(".6f")
    print(".6f")
    print(".6f")

    return {
        'predictions': ensemble_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'individual_predictions': {'rf': rf_pred, 'xgb': xgb_pred}
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Complete ML training pipeline"""
    print("CAFE LOCATION SUITABILITY PREDICTION - ML TRAINING PIPELINE")

    # Data loading and preprocessing
    print("\n[1] Loading and preprocessing data...")
    df = load_and_preprocess_data()
    df = engineer_features(df)

    # Feature normalization
    print("\n[2] Normalizing features...")
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

    # AHP weights
    print("\n[3] Calculating AHP weights...")
    ahp_weights = calculate_ahp_weights()

    # Target creation
    print("\n[4] Creating target scores...")
    y = create_target_variable(df_norm, ahp_weights, all_features)

    # Train/test split
    print("\n[5] Splitting data (70/30)...")
    X = df_norm[all_features].fillna(0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Train models
    rf_results = train_random_forest_model(X_train, X_test, y_train, y_test)
    xgb_results = train_xgboost_model(X_train, X_test, y_train, y_test)
    ensemble_results = train_ensemble_model(rf_results['model'], xgb_results['model'], X_test, y_test)

    # Model comparison
    print("\nMODEL COMPARISON SUMMARY")
    print("-" * 50)
    print("<12")
    print("<12")
    print("<12")

    print("\n✓ Training completed successfully!")

    return {
        'rf_model': rf_results, 'xgb_model': xgb_results,
        'ensemble_model': ensemble_results, 'scaler': scaler,
        'ahp_weights': ahp_weights, 'feature_names': all_features
    }

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    results = main()

    # Example prediction
    print("\nEXAMPLE PREDICTION")
    sample_features = {
        'population_density': 0.8, 'accessibility_score': 0.6, 'foot_traffic_score': 0.7,
        'competition_pressure': 0.3, 'competitors_within_200m': 0.4, 'bus_stops_within_500m': 0.5,
        'osm_amenity_count_500m': 0.6, 'school_count_750m': 0.2, 'hospital_count_750m': 0.1,
        'rating_normalized': 0.75, 'review_normalized': 0.8
    }

    sample_df = pd.DataFrame([sample_features])
    rf_pred = results['rf_model']['model'].predict(sample_df)[0]
    xgb_pred = results['xgb_model']['model'].predict(sample_df)[0]
    ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred

    print(".2f")
    print(".2f")
    print(".2f")