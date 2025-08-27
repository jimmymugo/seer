#!/usr/bin/env python3
"""
Advanced Prediction Model - Ensemble models with confidence intervals and multiple algorithms
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import joblib
import os

# ML imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    from scipy import stats
except ImportError as e:
    logging.warning(f"Some ML libraries not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEPLPredictor:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_weights = {}
        
    def load_advanced_processed_data(self):
        """Load advanced processed data"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM processed_players_advanced", conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error loading advanced data: {e}")
            # Fallback to basic data
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM processed_players", conn)
            conn.close()
            return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling"""
        # Select numerical features for modeling
        feature_columns = [
            'price', 'form', 'total_points', 'goals_scored', 'assists', 
            'clean_sheets', 'goals_conceded', 'influence', 'creativity', 
            'threat', 'ict_index', 'expected_goals', 'expected_assists',
            'shots_on_target', 'key_passes', 'big_chances_created',
            'tackles', 'interceptions', 'rotation_risk', 'likely_to_start',
            'recent_form_weighted', 'team_form', 'fixture_congestion',
            'goals_per_90', 'assists_per_90', 'xg_per_90', 'xa_per_90',
            'value_for_money', 'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward'
        ]
        
        # Filter columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Fill missing values
        for col in available_features:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, available_features
    
    def create_synthetic_target(self, df: pd.DataFrame) -> pd.Series:
        """Create synthetic target variable for training"""
        # Since we don't have actual future points, create a synthetic target
        # based on current performance and form
        target = (
            df['total_points'] * 0.3 +  # Historical performance
            df['form'] * 0.4 +          # Recent form
            df['expected_goals'] * 4 +   # Expected goals (4 points each)
            df['expected_assists'] * 3 + # Expected assists (3 points each)
            df['clean_sheets'] * 0.5 +   # Clean sheet bonus
            df['influence'] * 0.1 +      # Influence bonus
            df['creativity'] * 0.05 +    # Creativity bonus
            df['threat'] * 0.05          # Threat bonus
        )
        
        # Add some randomness to simulate real-world uncertainty
        noise = np.random.normal(0, 0.5, len(target))
        target = target + noise
        
        return target
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series):
        """Train multiple models for ensemble prediction"""
        logger.info("Training ensemble models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Model 1: XGBoost
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            self.models['xgboost'] = xgb_model
            
            # Evaluate
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
            logger.info(f"XGBoost MAE: {mae_xgb:.3f}")
            
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")
        
        # Model 2: Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            self.models['random_forest'] = rf_model
            
            # Evaluate
            y_pred_rf = rf_model.predict(X_test_scaled)
            mae_rf = mean_absolute_error(y_test, y_pred_rf)
            logger.info(f"Random Forest MAE: {mae_rf:.3f}")
            
        except Exception as e:
            logger.warning(f"Random Forest training failed: {e}")
        
        # Model 3: Gradient Boosting
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            self.models['gradient_boosting'] = gb_model
            
            # Evaluate
            y_pred_gb = gb_model.predict(X_test_scaled)
            mae_gb = mean_absolute_error(y_test, y_pred_gb)
            logger.info(f"Gradient Boosting MAE: {mae_gb:.3f}")
            
        except Exception as e:
            logger.warning(f"Gradient Boosting training failed: {e}")
        
        # Model 4: Bayesian Ridge (for uncertainty estimation)
        try:
            bayesian_model = BayesianRidge()
            bayesian_model.fit(X_train_scaled, y_train)
            self.models['bayesian'] = bayesian_model
            
            # Evaluate
            y_pred_bayes = bayesian_model.predict(X_test_scaled)
            mae_bayes = mean_absolute_error(y_test, y_pred_bayes)
            logger.info(f"Bayesian Ridge MAE: {mae_bayes:.3f}")
            
        except Exception as e:
            logger.warning(f"Bayesian Ridge training failed: {e}")
        
        # Set model weights based on performance (inverse of MAE)
        mae_scores = {
            'xgboost': mae_xgb if 'xgboost' in self.models else float('inf'),
            'random_forest': mae_rf if 'random_forest' in self.models else float('inf'),
            'gradient_boosting': mae_gb if 'gradient_boosting' in self.models else float('inf'),
            'bayesian': mae_bayes if 'bayesian' in self.models else float('inf')
        }
        
        # Calculate weights (inverse of MAE)
        total_inverse_mae = sum(1/mae for mae in mae_scores.values() if mae != float('inf'))
        self.model_weights = {
            model: (1/mae_scores[model]) / total_inverse_mae 
            for model in mae_scores.keys() 
            if mae_scores[model] != float('inf')
        }
        
        logger.info(f"Model weights: {self.model_weights}")
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals"""
        if not self.models:
            logger.error("No models trained. Please train models first.")
            return np.array([]), np.array([])
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = []
        confidence_intervals = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name == 'bayesian':
                # Bayesian model provides uncertainty
                pred, std = model.predict(X_scaled, return_std=True)
                predictions.append(pred)
                confidence_intervals.append(std)
            else:
                # Other models - use cross-validation for uncertainty estimation
                pred = model.predict(X_scaled)
                predictions.append(pred)
                
                # Estimate uncertainty using model variance
                if hasattr(model, 'estimators_'):
                    # For ensemble models, use variance across estimators
                    estimator_predictions = []
                    for estimator in model.estimators_:
                        estimator_predictions.append(estimator.predict(X_scaled))
                    std = np.std(estimator_predictions, axis=0)
                else:
                    # For single models, use a fixed uncertainty
                    std = np.ones(len(pred)) * 0.5
                
                confidence_intervals.append(std)
        
        # Weighted ensemble prediction
        weighted_pred = np.zeros(len(X))
        weighted_std = np.zeros(len(X))
        
        for i, model_name in enumerate(self.models.keys()):
            weight = self.model_weights.get(model_name, 1.0 / len(self.models))
            weighted_pred += weight * predictions[i]
            weighted_std += weight * confidence_intervals[i]
        
        return weighted_pred, weighted_std
    
    def rule_based_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """Rule-based prediction as baseline"""
        predictions = []
        
        for _, player in df.iterrows():
            # Base prediction from form and expected performance
            base_pred = (
                player.get('form', 0) * 0.4 +
                player.get('expected_goals', 0) * 4 +
                player.get('expected_assists', 0) * 3 +
                player.get('clean_sheets', 0) * 0.5
            )
            
            # Adjust for rotation risk
            rotation_risk = player.get('rotation_risk', 0.5)
            base_pred *= (1 - rotation_risk * 0.3)
            
            # Adjust for team form
            team_form = player.get('team_form', 0.5)
            base_pred *= (0.8 + team_form * 0.4)
            
            # Adjust for fixture difficulty
            fixture_difficulty = player.get('next_opponent_difficulty', 3.0)
            difficulty_factor = 1.0 - (fixture_difficulty - 3.0) * 0.1
            base_pred *= difficulty_factor
            
            predictions.append(max(0, base_pred))
        
        return np.array(predictions)
    
    def train_and_predict_advanced(self) -> pd.DataFrame:
        """Main method to train models and generate predictions"""
        logger.info("Starting advanced model training and prediction...")
        
        # Load data
        df = self.load_advanced_processed_data()
        if df.empty:
            logger.error("No data available for training")
            return pd.DataFrame()
        
        # Prepare features
        X, self.feature_columns = self.prepare_features(df)
        
        # Create synthetic target
        y = self.create_synthetic_target(df)
        
        # Train ensemble models
        self.train_ensemble_models(X, y)
        
        # Make predictions
        ml_predictions, confidence_intervals = self.predict_with_confidence(X)
        rule_predictions = self.rule_based_prediction(df)
        
        # Combine predictions (70% ML, 30% rule-based)
        final_predictions = 0.7 * ml_predictions + 0.3 * rule_predictions
        
        # Create results dataframe
        results_df = df.copy()
        results_df['predicted_points'] = final_predictions
        results_df['confidence_interval'] = confidence_intervals
        results_df['rule_based_prediction'] = rule_predictions
        results_df['ml_prediction'] = ml_predictions
        
        # Calculate confidence scores (inverse of uncertainty)
        max_confidence = np.max(confidence_intervals)
        results_df['confidence_score'] = 1.0 - (confidence_intervals / max_confidence)
        results_df['confidence_score'] = results_df['confidence_score'].clip(0.1, 0.95)
        
        # Add prediction metadata
        results_df['prediction_timestamp'] = datetime.now()
        results_df['model_version'] = 'advanced_ensemble_v1'
        
        # Save predictions
        conn = sqlite3.connect(self.db_path)
        results_df.to_sql('predictions_advanced', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"Advanced predictions generated for {len(results_df)} players")
        return results_df
    
    def get_top_predictions_advanced(self, top_n: int = 10, position: Optional[str] = None) -> pd.DataFrame:
        """Get top predictions with advanced metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM predictions_advanced ORDER BY predicted_points DESC"
            if position:
                query += f" WHERE position_name = '{position}'"
            query += f" LIMIT {top_n}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            return pd.DataFrame()
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        joblib.dump(self.scalers['standard'], scaler_path)
        
        # Save model weights
        weights_path = os.path.join(model_dir, "model_weights.json")
        with open(weights_path, 'w') as f:
            import json
            json.dump(self.model_weights, f)
        
        logger.info("All models saved successfully")

if __name__ == "__main__":
    predictor = AdvancedEPLPredictor()
    predictions = predictor.train_and_predict_advanced()
    print(f"Generated predictions for {len(predictions)} players")
