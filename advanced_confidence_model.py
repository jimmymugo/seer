#!/usr/bin/env python3
"""
Advanced Confidence Model - Calculates prediction intervals and uncertainty
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib

class AdvancedConfidenceModel:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        
    def calculate_prediction_intervals(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence intervals for predictions"""
        enhanced_df = predictions_df.copy()
        
        # Calculate prediction intervals based on multiple factors
        for idx, row in enhanced_df.iterrows():
            # Base uncertainty from confidence score
            base_uncertainty = (1 - row['confidence_score']) * 5.0
            
            # Form uncertainty (more variable form = higher uncertainty)
            form_uncertainty = abs(row['recent_form_weighted'] - row['team_form']) * 2.0
            
            # Price uncertainty (expensive players have higher variance)
            price_uncertainty = row['price'] * 0.1
            
            # Rotation risk uncertainty
            rotation_uncertainty = row['rotation_risk'] * 3.0
            
            # Injury uncertainty
            injury_uncertainty = (1 - row['likely_to_start']) * 4.0
            
            # Total uncertainty
            total_uncertainty = base_uncertainty + form_uncertainty + price_uncertainty + rotation_uncertainty + injury_uncertainty
            
            # Calculate confidence interval
            predicted_points = row['predicted_points']
            lower_bound = max(0, predicted_points - total_uncertainty)
            upper_bound = predicted_points + total_uncertainty
            
            enhanced_df.loc[idx, 'confidence_lower'] = round(lower_bound, 1)
            enhanced_df.loc[idx, 'confidence_upper'] = round(upper_bound, 1)
            enhanced_df.loc[idx, 'uncertainty_range'] = round(total_uncertainty, 1)
            enhanced_df.loc[idx, 'confidence_interval'] = f"{predicted_points:.1f} ± {total_uncertainty:.1f}"
            
        return enhanced_df
    
    def calculate_ensemble_confidence(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence based on ensemble model agreement"""
        enhanced_df = predictions_df.copy()
        
        # Simulate ensemble predictions (in real implementation, this would come from multiple models)
        for idx, row in enhanced_df.iterrows():
            # Create synthetic ensemble predictions
            base_prediction = row['predicted_points']
            
            # Model 1: XGBoost-like prediction
            model1_pred = base_prediction * (0.9 + np.random.normal(0, 0.1))
            
            # Model 2: Random Forest-like prediction  
            model2_pred = base_prediction * (0.95 + np.random.normal(0, 0.08))
            
            # Model 3: Linear model prediction
            model3_pred = base_prediction * (0.85 + np.random.normal(0, 0.12))
            
            # Calculate ensemble statistics
            ensemble_predictions = [model1_pred, model2_pred, model3_pred]
            ensemble_mean = np.mean(ensemble_predictions)
            ensemble_std = np.std(ensemble_predictions)
            
            # Confidence based on ensemble agreement
            ensemble_confidence = max(0.1, 1.0 - (ensemble_std / ensemble_mean))
            
            enhanced_df.loc[idx, 'ensemble_mean'] = round(ensemble_mean, 1)
            enhanced_df.loc[idx, 'ensemble_std'] = round(ensemble_std, 1)
            enhanced_df.loc[idx, 'ensemble_confidence'] = round(ensemble_confidence, 2)
            
        return enhanced_df
    
    def update_predictions_with_confidence(self):
        """Update predictions table with confidence intervals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load current predictions
            predictions_df = pd.read_sql_query("SELECT * FROM predictions_advanced", conn)
            
            if len(predictions_df) == 0:
                print("No predictions found to update")
                return
            
            # Calculate confidence intervals
            enhanced_df = self.calculate_prediction_intervals(predictions_df)
            enhanced_df = self.calculate_ensemble_confidence(enhanced_df)
            
            # Add confidence columns to database
            try:
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN confidence_lower REAL")
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN confidence_upper REAL")
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN uncertainty_range REAL")
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN confidence_interval TEXT")
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN ensemble_mean REAL")
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN ensemble_std REAL")
                conn.execute("ALTER TABLE predictions_advanced ADD COLUMN ensemble_confidence REAL")
            except:
                pass  # Columns might already exist
            
            # Update predictions with confidence data
            for idx, row in enhanced_df.iterrows():
                conn.execute("""
                    UPDATE predictions_advanced 
                    SET confidence_lower = ?, confidence_upper = ?, uncertainty_range = ?,
                        confidence_interval = ?, ensemble_mean = ?, ensemble_std = ?, ensemble_confidence = ?
                    WHERE id = ?
                """, (
                    row['confidence_lower'], row['confidence_upper'], row['uncertainty_range'],
                    row['confidence_interval'], row['ensemble_mean'], row['ensemble_std'], 
                    row['ensemble_confidence'], row['id']
                ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Updated {len(enhanced_df)} predictions with confidence intervals")
            
        except Exception as e:
            print(f"❌ Error updating confidence intervals: {e}")

if __name__ == "__main__":
    confidence_model = AdvancedConfidenceModel()
    confidence_model.update_predictions_with_confidence()
