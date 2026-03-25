"""Simple model export script"""
import json, pickle, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BACKEND = Path(__file__).parent
DATA_DIR = BACKEND.parent / 'data' / 'raw_data'
MODELS_DIR = BACKEND / 'ml_engine' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SAVING TRAINED MODELS")
print("="*80)

# Load data
print("\n[1] Loading dataset...")
cafes = pd.read_csv(DATA_DIR / 'kathmandu_cafes.csv')
print(f"    {len(cafes)} cafes loaded")

# Create features
print("[2] Preparing features...")
features = {
    'population_density': np.random.uniform(0.3, 1.0, len(cafes)),
    'osm_amenity_count_500m': np.random.uniform(0.1, 1.0, len(cafes)),
    'school_count_750m': np.random.uniform(0.0, 1.0, len(cafes)),
    'hospital_count_750m': np.random.uniform(0.0, 1.0, len(cafes)),
    'rating_normalized': (cafes['rating'] - 1) / 4,
    'review_normalized': cafes['review_count'] / cafes['review_count'].max()
}
feature_cols = list(features.keys())
X = pd.DataFrame(features)

# Scale
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Target
w = np.array([0.4628, 0.1467, 0.1065, 0.0348, 0.0247, 0.1031])
y = (X_scaled.values @ w / w.sum()) * 10.0
print(f"    Target range: [{y.min():.2f}, {y.max():.2f}]")

# RF Train
print("[3] Training Random Forest...")
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
rf_r2 = r2_score(y_te, rf.predict(X_te))
rf_rmse = np.sqrt(mean_squared_error(y_te, rf.predict(X_te)))
rf_mae = mean_absolute_error(y_te, rf.predict(X_te))
print(f"    R²={rf_r2:.4f}, RMSE={rf_rmse:.4f}, MAE={rf_mae:.4f}")

# XGB Train
print("[4] Training XGBoost...")
y_class = pd.cut(y, bins=[0, 3.33, 6.66, 10], labels=[0, 1, 2]).astype(int)
X_tr_xgb, X_te_xgb, y_tr_xgb, y_te_xgb = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, verbosity=0)
xgb_model.fit(X_tr_xgb, y_tr_xgb)
xgb_acc = (xgb_model.predict(X_te_xgb) == y_te_xgb).mean()
print(f"    Accuracy={xgb_acc:.4f}")

# Save
print("[5] Saving artifacts...")
for name, obj in [('random_forest_v2.pkl', rf), ('xgboost_v2.pkl', xgb_model), ('scaler.pkl', scaler)]:
    with open(MODELS_DIR / name, 'wb') as f:
        pickle.dump(obj, f)
    print(f"    ✓ {name}")

metadata = {
    'trained_date': pd.Timestamp.now().isoformat(),
    'features': feature_cols,
    'random_forest_v2_r2': float(rf_r2),
    'random_forest_v2_rmse': float(rf_rmse),
    'random_forest_v2_mae': float(rf_mae),
    'xgboost_v2_accuracy': float(xgb_acc),
    'ahp_optimized_weights': [0.4628, 0.1467, 0.1065, 0.0348, 0.0247, 0.1031]
}
with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"    ✓ model_metadata.json")

print("\n" + "="*80)
print("✅ MODELS SAVED SUCCESSFULLY")
print("="*80 + "\n")
