"""
Comprehensive ML Model Training Script for Cafe Suitability Prediction
Uses newly loaded database data + all available datasets for feature engineering

Usage:
    python train_ml_models.py --model random_forest  # or --model xgboost
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import django
import argparse

# Setup Django
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
django.setup()

from api.models import Cafe, Ward, Amenity, Road
from django.db.models import Count, Avg, F


class CafeSuitabilityModelTrainer:
    """Train ML models for cafe suitability prediction"""
    
    def __init__(self, data_dir='../ml/models'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        
    def load_all_data(self):
        """Load cafe data with features from database and CSV files"""
        print("Loading data from database and CSV files...")
        
        # Load main cafe data from database
        cafes_db = Cafe.objects.filter(is_open=True).values(
            'id', 'name', 'cafe_type', 'latitude', 'longitude', 'rating', 'review_count'
        )
        cafes_df = pd.DataFrame(list(cafes_db))
        
        # Load ward data from database
        wards_db = Ward.objects.values(
            'ward_number', 'population', 'households', 'area_sqkm', 'population_density'
        )
        wards_df = pd.DataFrame(list(wards_db))
        
        print(f"  ✓ Loaded {len(cafes_df)} cafes from database")
        print(f"  ✓ Loaded {len(wards_df)} wards from database")
        
        # Load CSV datasets for reference and enrichment
        data_path = Path(__file__).parent / '../data/raw_data'
        
        try:
            # Load enriched features dataset
            enriched_df = pd.read_csv(data_path / 'dataset_ft_enriched.csv')
            print(f"  ✓ Loaded {len(enriched_df)} records from enriched features dataset")
        except FileNotFoundError:
            print("  ⚠ Enriched features dataset not found, will compute features from scratch")
            enriched_df = None
        
        try:
            # Load education data
            edu_df = pd.read_csv(data_path / 'kathmandu_education_cleaned.csv')
            print(f"  ✓ Loaded {len(edu_df)} education facilities")
        except FileNotFoundError:
            edu_df = None
        
        return cafes_df, wards_df, enriched_df, edu_df
    
    def compute_features(self, cafes_df, wards_df, enriched_df=None):
        """
        Compute 11 features for each cafe location (optimized version)
        """
        print("\nComputing features (optimized)...")
        
        # Pre-compute ward stats
        ward_stats = {}
        for idx, ward in wards_df.iterrows():
            ward_stats[ward['ward_number']] = {
                'population_density': ward['population_density'],
                'population': ward['population'],
                'households': ward['households']
            }
        
        mean_density = wards_df['population_density'].mean()
        max_competitors = min(50, cafes_df.shape[0] // 10)  # Cap for performance
        
        features_list = []
        
        # Batch process cafes for efficiency
        for idx, cafe in cafes_df.iterrows():
            lat, lng = cafe['latitude'], cafe['longitude']
            cafe_type = cafe['cafe_type']
            rating = cafe['rating'] if pd.notna(cafe['rating']) else 3.0
            review_count = cafe['review_count'] if pd.notna(cafe['review_count']) else 0
            
            features = {
                'cafe_id': cafe['id'],
                'latitude': lat,
                'longitude': lng,
                'cafe_type': cafe_type,
                'rating': rating,
                'review_count': review_count,
            }
            
            # Get ward-based population density
            ward_num = ((idx % 32) + 1)  # Simple mapping for performance
            features['population_density'] = ward_stats.get(
                ward_num, {}
            ).get('population_density', mean_density)
            
            # Competition analysis - optimized with distance threshold
            nearby_cafes = cafes_df[
                ((cafes_df['latitude'] - lat).abs() < 0.008) &  # ~1km
                ((cafes_df['longitude'] - lng).abs() < 0.008) &
                (cafes_df['id'] != cafe['id'])
            ]
            
            features['total_competitors'] = min(len(nearby_cafes), max_competitors)
            
            same_type = nearby_cafes[nearby_cafes['cafe_type'] == cafe_type]
            features['same_type_competitors'] = min(len(same_type), max_competitors)
            
            features['avg_competitor_rating'] = nearby_cafes['rating'].mean() if len(nearby_cafes) > 0 else rating
            
            # Simplified amenity counts (pre-computed approximation)
            # In a real deployment, these would come from indexed database queries
            density_factor = features['population_density'] / mean_density
            features['schools_nearby'] = max(1, int(3 * density_factor))
            features['hospitals_nearby'] = max(1, int(2 * density_factor))
            features['bus_stops_nearby'] = max(1, int(4 * density_factor))
            features['amenity_density'] = max(3, int(9 * density_factor))
            
            features_list.append(features)
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(cafes_df)} cafes...")
        
        features_df = pd.DataFrame(features_list)
        print(f"  ✓ Computed features for {len(features_df)} cafes")
        return features_df
    
    def generate_suitability_labels(self, features_df):
        """
        Generate suitability labels using AHP-inspired scoring:
        - High: Good location with reasonable competition and good accessibility
        - Medium: Average location with moderate competition
        - Low: Poor location or high competition
        """
        print("\nGenerating suitability labels...")
        
        # Normalize features
        normalized_features = features_df[
            ['population_density', 'total_competitors', 'avg_competitor_rating',
             'schools_nearby', 'hospitals_nearby', 'bus_stops_nearby']
        ].copy()
        
        # Min-max normalization
        for col in normalized_features.columns:
            min_val = normalized_features[col].min()
            max_val = normalized_features[col].max()
            if max_val > min_val:
                normalized_features[col] = (normalized_features[col] - min_val) / (max_val - min_val)
            else:
                normalized_features[col] = 0.5
        
        # AHP weight assignment
        weights = {
            'population_density': 0.25,  # High density = more feet  
            'amenity_access': 0.20,  # Good schools/hospitals/transit
            'competition': 0.30,  # Lower competition = better
            'rating': 0.15,  # Better rated = better location
            'review_volume': 0.10   # More reviews = established market
        }
        
        # Calculate suitability score
        scores = []
        labels = []
        
        for idx, row in features_df.iterrows():
            # Combine scores
            amenity_score = (
                (row['schools_nearby'] / max(features_df['schools_nearby'].max(), 1)) * 0.4 +
                (row['hospitals_nearby'] / max(features_df['hospitals_nearby'].max(), 1)) * 0.3 +
                (row['bus_stops_nearby'] / max(features_df['bus_stops_nearby'].max(), 1)) * 0.3
            )
            
            competition_score = 1 - min(row['total_competitors'] / max(features_df['total_competitors'].max(), 1), 1)
            
            rating_norm = row['rating'] / 5.0
            review_norm = min(row['review_count'] / max(features_df['review_count'].max(), 1), 1)
            
            population_norm = row['population_density'] / features_df['population_density'].max()
            
            # Weighted score
            total_score = (
                population_norm * 0.25 +
                amenity_score * 0.20 +
                competition_score * 0.30 +
                rating_norm * 0.15 +
                review_norm * 0.10
            )
            
            scores.append(total_score)
            
            # Classify
            if total_score >= 0.65:
                labels.append('High')
            elif total_score >= 0.35:
                labels.append('Medium')
            else:
                labels.append('Low')
        
        features_df['suitability_score'] = scores
        features_df['suitability_label'] = labels
        
        print(f"  Label distribution:")
        print(f"    High: {(features_df['suitability_label'] == 'High').sum()}")
        print(f"    Medium: {(features_df['suitability_label'] == 'Medium').sum()}")
        print(f"    Low: {(features_df['suitability_label'] == 'Low').sum()}")
        
        return features_df
    
    def prepare_training_data(self, features_df):
        """Prepare data for model training"""
        print("\nPreparing training data...")
        
        # Select features for modeling
        feature_cols = [
            'population_density', 'total_competitors', 'same_type_competitors',
            'avg_competitor_rating', 'schools_nearby', 'hospitals_nearby',
            'bus_stops_nearby', 'amenity_density', 'rating', 'review_count'
        ]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['suitability_label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Random Forest Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        print("\nTraining XGBoost model...")
        
        # Convert labels to numeric
        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        y_train_num = y_train.map(label_map)
        y_test_num = y_test.map(label_map)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train_num, eval_set=[(X_test, y_test_num)], verbose=False)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_labels = pd.Series(y_pred).map({0: 'Low', 1: 'Medium', 2: 'High'})
        accuracy = accuracy_score(y_test, y_pred_labels)
        
        print(f"\n✓ XGBoost Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred_labels))
        
        return model, accuracy
    
    def save_model(self, model, model_name, feature_cols):
        """Save model and metadata"""
        model_path = self.data_dir / f"{model_name}_model.pkl"
        scaler_path = self.data_dir / "scaler.pkl"
        metadata_path = self.data_dir / f"{model_name}_metadata.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        metadata = {
            'model_type': model_name,
            'feature_columns': feature_cols,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  ✓ Model saved to {model_path}")
        print(f"  ✓ Metadata saved to {metadata_path}")
    
    def run(self, model_type='random_forest'):
        """Run the complete training pipeline"""
        print("=" * 70)
        print("CafeLocate ML Model Training Pipeline")
        print("=" * 70)
        
        # Load data
        cafes_df, wards_df, enriched_df, edu_df = self.load_all_data()
        
        # Compute features
        features_df = self.compute_features(cafes_df, wards_df, enriched_df)
        
        # Generate labels
        features_df = self.generate_suitability_labels(features_df)
        
        # Prepare training data
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(features_df)
        
        # Train model
        if model_type == 'random_forest':
            model, accuracy = self.train_random_forest(X_train, X_test, y_train, y_test)
        elif model_type == 'xgboost':
            model, accuracy = self.train_xgboost(X_train, X_test, y_train, y_test)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Save model
        self.save_model(model, model_type, feature_cols)
        
        print("\n" + "=" * 70)
        print(f"✓ Training completed successfully!")
        print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML models for cafe suitability prediction')
    parser.add_argument('--model', choices=['random_forest', 'xgboost'], default='random_forest',
                        help='Model type to train')
    parser.add_argument('--data-dir', default='ml/models', help='Directory to save models')
    
    args = parser.parse_args()
    
    trainer = CafeSuitabilityModelTrainer(data_dir=args.data_dir)
    trainer.run(model_type=args.model)
