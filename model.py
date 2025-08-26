"""
Prediction Model - Rule-based and ML models for EPL player performance prediction
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EPLPredictor:
    def __init__(self, db_path: str = "data/epl_data.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'recent_form', 'form_momentum', 'next_fixture_difficulty',
            'team_strength', 'points_per_game', 'goals_per_game', 'assists_per_game',
            'value_for_money', 'injury_status', 'chance_of_playing',
            'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward',
            'influence', 'creativity', 'threat', 'ict_index'
        ]
    
    def load_processed_data(self):
        """Load processed player data"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM processed_players", conn)
        conn.close()
        return df
    
    def rule_based_prediction(self, df):
        """Rule-based prediction using simple scoring"""
        df = df.copy()
        df['rule_score'] = df['recent_form'] * 2
        df['rule_score'] += (6 - df['next_fixture_difficulty']) * 0.5
        df['rule_score'] += (df['team_strength'] - 1000) / 1000 * 2
        df.loc[df['is_goalkeeper'], 'rule_score'] += df['clean_sheets'] * 0.5
        df.loc[df['is_defender'], 'rule_score'] += df['clean_sheets'] * 0.3
        df.loc[df['is_midfielder'], 'rule_score'] += df['assists_per_game'] * 10
        df.loc[df['is_forward'], 'rule_score'] += df['goals_per_game'] * 15
        df['rule_score'] *= df['injury_status'] * df['chance_of_playing']
        df['rule_score'] += df['value_for_money'] * 0.1
        return df
    
    def train_ml_model(self, df):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        X = df[self.feature_columns].fillna(0)
        y = df['total_points']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        logger.info(f"Model trained - MSE: {mse:.2f}")
        return mse
    
    def predict_points(self, df):
        """Predict points for all players"""
        df = df.copy()
        df = self.rule_based_prediction(df)
        
        if self.model is not None:
            X = df[self.feature_columns].fillna(0)
            df['ml_predicted_points'] = self.model.predict(X)
        else:
            df['ml_predicted_points'] = df['rule_score']
        
        df['predicted_points'] = 0.3 * df['rule_score'] + 0.7 * df['ml_predicted_points']
        df['confidence_score'] = (
            (df['recent_form'] > 0) * 0.3 +
            (df['injury_status'] == 1) * 0.3 +
            (df['chance_of_playing'] > 0.8) * 0.2 +
            (df['next_fixture_difficulty'] < 4) * 0.2
        )
        return df
    
    def get_top_predictions(self, df, top_n=10):
        """Get top N predicted players"""
        return df.nlargest(top_n, 'predicted_points')[
            ['name', 'position_name', 'team_id', 'price', 'recent_form', 
             'next_fixture_difficulty', 'predicted_points', 'confidence_score']
        ]
    
    def save_model(self, filepath="epl_model.pkl"):
        """Save trained model"""
        if self.model is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler, 'feature_columns': self.feature_columns}, f)
            logger.info(f"Model saved to {filepath}")
    
    def train_and_predict(self):
        """Main method to train model and generate predictions"""
        logger.info("Starting prediction pipeline...")
        
        df = self.load_processed_data()
        
        try:
            self.train_ml_model(df)
            self.save_model()
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.info("Using rule-based predictions only")
        
        predictions_df = self.predict_points(df)
        top_predictions = self.get_top_predictions(predictions_df)
        
        conn = sqlite3.connect(self.db_path)
        predictions_df.to_sql('predictions', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info("Prediction pipeline completed!")
        return top_predictions

if __name__ == "__main__":
    predictor = EPLPredictor()
    top_players = predictor.train_and_predict()
    print("\nTop 10 Predicted Players:")
    print(top_players.to_string(index=False))
