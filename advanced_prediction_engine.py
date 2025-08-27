#!/usr/bin/env python3
"""
Advanced EPL Prediction Engine
Implements comprehensive prediction framework with core FPL inputs, advanced stats, and contextual factors
"""

import sqlite3
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictionEngine:
    def __init__(self):
        self.db_path = "epl_data.db"
        self.fpl_api_base = "https://fantasy.premierleague.com/api"
        self.models = {}
        self.feature_columns = []
        
    def fetch_fpl_bootstrap_data(self):
        """Fetch current FPL bootstrap data"""
        try:
            response = requests.get(f"{self.fpl_api_base}/bootstrap-static/")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to fetch FPL bootstrap data: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error fetching FPL bootstrap data: {e}")
            return None
    
    def fetch_player_history(self, player_id):
        """Fetch historical data for a specific player"""
        try:
            url = f"{self.fpl_api_base}/element-summary/{player_id}/"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            return None
    
    def fetch_fixtures_data(self):
        """Fetch fixtures data for difficulty ratings"""
        try:
            response = requests.get(f"{self.fpl_api_base}/fixtures/")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            return None
    
    def calculate_player_form(self, history_data, matches=5):
        """Calculate player form based on last N matches"""
        if not history_data or 'history' not in history_data:
            return 0.0
        
        recent_matches = history_data['history'][-matches:]
        if not recent_matches:
            return 0.0
        
        total_points = sum(match.get('total_points', 0) for match in recent_matches)
        return total_points / len(recent_matches)
    
    def calculate_minutes_probability(self, history_data, matches=5):
        """Calculate probability of starting based on recent minutes"""
        if not history_data or 'history' not in history_data:
            return 0.5
        
        recent_matches = history_data['history'][-matches:]
        if not recent_matches:
            return 0.5
        
        total_minutes = sum(match.get('minutes', 0) for match in recent_matches)
        avg_minutes = total_minutes / len(recent_matches)
        
        # Convert to probability (90 minutes = 1.0, 0 minutes = 0.0)
        return min(avg_minutes / 90.0, 1.0)
    
    def calculate_rotation_risk(self, history_data, matches=5):
        """Calculate rotation risk based on minutes consistency"""
        if not history_data or 'history' not in history_data:
            return 0.5
        
        recent_matches = history_data['history'][-matches:]
        if not recent_matches:
            return 0.5
        
        minutes_list = [match.get('minutes', 0) for match in recent_matches]
        variance = np.var(minutes_list)
        
        # Higher variance = higher rotation risk
        return min(variance / 1000.0, 1.0)
    
    def calculate_advanced_stats(self, history_data, matches=5):
        """Calculate advanced stats (xG, xA, shots, etc.)"""
        if not history_data or 'history' not in history_data:
            return {
                'avg_goals': 0.0,
                'avg_assists': 0.0,
                'avg_clean_sheets': 0.0,
                'avg_shots': 0.0,
                'avg_key_passes': 0.0,
                'avg_big_chances': 0.0,
                'avg_tackles': 0.0,
                'avg_interceptions': 0.0
            }
        
        recent_matches = history_data['history'][-matches:]
        if not recent_matches:
            return {
                'avg_goals': 0.0,
                'avg_assists': 0.0,
                'avg_clean_sheets': 0.0,
                'avg_shots': 0.0,
                'avg_key_passes': 0.0,
                'avg_big_chances': 0.0,
                'avg_tackles': 0.0,
                'avg_interceptions': 0.0
            }
        
        stats = {
            'avg_goals': np.mean([match.get('goals_scored', 0) for match in recent_matches]),
            'avg_assists': np.mean([match.get('assists', 0) for match in recent_matches]),
            'avg_clean_sheets': np.mean([match.get('clean_sheets', 0) for match in recent_matches]),
            'avg_shots': np.mean([match.get('shots', 0) for match in recent_matches]),
            'avg_key_passes': np.mean([match.get('key_passes', 0) for match in recent_matches]),
            'avg_big_chances': np.mean([match.get('big_chances_created', 0) for match in recent_matches]),
            'avg_tackles': np.mean([match.get('tackles', 0) for match in recent_matches]),
            'avg_interceptions': np.mean([match.get('interceptions', 0) for match in recent_matches])
        }
        
        return stats
    
    def calculate_fixture_difficulty(self, player_team_id, opponent_team_id, is_home, fixtures_data):
        """Calculate fixture difficulty based on opponent strength"""
        if not fixtures_data:
            return 3.0  # Default medium difficulty
        
        # Find the specific fixture
        fixture = None
        for f in fixtures_data:
            if (f['team_h'] == player_team_id and f['team_a'] == opponent_team_id) or \
               (f['team_a'] == player_team_id and f['team_h'] == opponent_team_id):
                fixture = f
                break
        
        if fixture:
            # Use FPL's difficulty rating (1-5, where 1 is easiest)
            difficulty = fixture.get('difficulty', 3)
            return difficulty
        
        return 3.0  # Default medium difficulty
    
    def calculate_home_away_factor(self, is_home):
        """Calculate home/away performance factor"""
        return 1.1 if is_home else 0.9  # 10% boost at home, 10% penalty away
    
    def calculate_team_form(self, team_id, history_data, matches=5):
        """Calculate team form based on recent results"""
        # This would require team-level data, simplified for now
        return 1.0  # Neutral factor
    
    def calculate_injury_risk(self, player_data):
        """Calculate injury/suspension risk"""
        status = player_data.get('status', 'a')  # a=available, i=injured, s=suspended, u=unavailable
        
        if status == 'a':
            return 0.0
        elif status == 'i':
            return 0.8
        elif status == 's':
            return 0.6
        elif status == 'u':
            return 0.4
        else:
            return 0.2
    
    def calculate_set_piece_role(self, player_data):
        """Calculate set-piece involvement factor"""
        # Check if player is likely to take penalties, free kicks, corners
        # This would require additional data, simplified for now
        return 1.0  # Neutral factor
    
    def calculate_fixture_congestion_factor(self, player_team_id, fixtures_data):
        """Calculate fixture congestion factor"""
        if not fixtures_data:
            return 1.0
        
        # Count upcoming fixtures in next 7 days
        upcoming_fixtures = 0
        current_time = datetime.now()
        
        for fixture in fixtures_data:
            if fixture.get('team_h') == player_team_id or fixture.get('team_a') == player_team_id:
                # Check if fixture is in next 7 days
                fixture_date = datetime.fromisoformat(fixture.get('kickoff_time', '').replace('Z', '+00:00'))
                if 0 < (fixture_date - current_time).days <= 7:
                    upcoming_fixtures += 1
        
        # More fixtures = higher rotation risk
        if upcoming_fixtures >= 3:
            return 0.8  # 20% penalty for heavy congestion
        elif upcoming_fixtures == 2:
            return 0.9  # 10% penalty for moderate congestion
        else:
            return 1.0  # No penalty
    
    def extract_features_for_player(self, player_data, history_data, fixtures_data, opponent_team_id, is_home):
        """Extract all features for a player"""
        features = {}
        
        # Core FPL inputs
        features['form'] = self.calculate_player_form(history_data, matches=5)
        features['minutes_probability'] = self.calculate_minutes_probability(history_data, matches=5)
        features['rotation_risk'] = self.calculate_rotation_risk(history_data, matches=5)
        
        # Advanced stats
        advanced_stats = self.calculate_advanced_stats(history_data, matches=5)
        features.update(advanced_stats)
        
        # Contextual factors
        features['fixture_difficulty'] = self.calculate_fixture_difficulty(
            player_data.get('team'), opponent_team_id, is_home, fixtures_data
        )
        features['home_away_factor'] = self.calculate_home_away_factor(is_home)
        features['team_form'] = self.calculate_team_form(player_data.get('team'), history_data)
        features['injury_risk'] = self.calculate_injury_risk(player_data)
        features['set_piece_role'] = self.calculate_set_piece_role(player_data)
        features['fixture_congestion'] = self.calculate_fixture_congestion_factor(
            player_data.get('team'), fixtures_data
        )
        
        # Position-specific features
        position_type = player_data.get('element_type', 1)
        features['is_goalkeeper'] = 1 if position_type == 1 else 0
        features['is_defender'] = 1 if position_type == 2 else 0
        features['is_midfielder'] = 1 if position_type == 3 else 0
        features['is_forward'] = 1 if position_type == 4 else 0
        
        # Price factor
        features['price'] = player_data.get('now_cost', 0) / 10.0
        
        return features
    
    def build_training_dataset(self):
        """Build training dataset from historical data"""
        print("🔄 Building training dataset...")
        
        # Fetch FPL data
        fpl_data = self.fetch_fpl_bootstrap_data()
        if not fpl_data:
            print("❌ Failed to fetch FPL data")
            return None
        
        fixtures_data = self.fetch_fixtures_data()
        
        players = fpl_data.get('elements', [])
        teams = fpl_data.get('teams', [])
        
        training_data = []
        
        for player in players:
            print(f"📊 Processing {player.get('web_name', 'Unknown')}...")
            
            # Fetch player history
            history_data = self.fetch_player_history(player['id'])
            if not history_data:
                continue
            
            # Process each historical match
            for match in history_data.get('history', []):
                # Extract features for this match
                features = self.extract_features_for_player(
                    player, history_data, fixtures_data, 
                    match.get('opponent_team'), match.get('was_home', False)
                )
                
                # Add target variable (actual points)
                features['actual_points'] = match.get('total_points', 0)
                features['gameweek'] = match.get('round', 0)
                
                training_data.append(features)
            
            # Rate limiting
            time.sleep(0.1)
        
        if not training_data:
            print("❌ No training data generated")
            return None
        
        df = pd.DataFrame(training_data)
        print(f"✅ Generated {len(df)} training samples")
        
        return df
    
    def train_models(self, training_data):
        """Train multiple prediction models"""
        print("🔄 Training prediction models...")
        
        if training_data is None or training_data.empty:
            print("❌ No training data available")
            return
        
        # Prepare features and target
        feature_columns = [col for col in training_data.columns 
                         if col not in ['actual_points', 'gameweek']]
        self.feature_columns = feature_columns
        
        X = training_data[feature_columns]
        y = training_data['actual_points']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"✅ {name} - MSE: {mse:.3f}, MAE: {mae:.3f}")
            
            self.models[name] = model
        
        print("✅ All models trained successfully!")
    
    def predict_player_points(self, player_data, history_data, fixtures_data, opponent_team_id, is_home):
        """Predict points for a specific player"""
        # Extract features
        features = self.extract_features_for_player(
            player_data, history_data, fixtures_data, opponent_team_id, is_home
        )
        
        # Create feature vector
        feature_vector = pd.DataFrame([features])
        feature_vector = feature_vector[self.feature_columns]
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(feature_vector)[0]
            predictions[name] = max(0, pred)  # Ensure non-negative
        
        # Ensemble prediction (weighted average)
        weights = {
            'linear_regression': 0.2,
            'random_forest': 0.25,
            'gradient_boosting': 0.25,
            'xgboost': 0.3
        }
        
        ensemble_prediction = sum(predictions[name] * weights[name] for name in predictions)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'features': features
        }
    
    def generate_advanced_predictions(self):
        """Generate advanced predictions for all players"""
        print("🔄 Generating advanced predictions...")
        
        # Fetch current data
        fpl_data = self.fetch_fpl_bootstrap_data()
        fixtures_data = self.fetch_fixtures_data()
        
        if not fpl_data:
            print("❌ Failed to fetch FPL data")
            return
        
        players = fpl_data.get('elements', [])
        teams = fpl_data.get('teams', [])
        
        predictions_data = []
        
        for player in players:
            print(f"📊 Predicting for {player.get('web_name', 'Unknown')}...")
            
            # Fetch player history
            history_data = self.fetch_player_history(player['id'])
            if not history_data:
                continue
            
            # Get next fixture
            next_fixture = self.get_next_fixture(player.get('team'), fixtures_data)
            if not next_fixture:
                continue
            
            # Determine if home or away
            is_home = next_fixture['team_h'] == player.get('team')
            opponent_team_id = next_fixture['team_a'] if is_home else next_fixture['team_h']
            
            # Get prediction
            prediction_result = self.predict_player_points(
                player, history_data, fixtures_data, opponent_team_id, is_home
            )
            
            # Store prediction data
            pred_data = {
                'player_id': player['id'],
                'name': player.get('web_name', ''),
                'team_id': player.get('team'),
                'team_name': self.get_team_name(player.get('team'), teams),
                'position_type': player.get('element_type'),
                'position_name': self.get_position_name(player.get('element_type')),
                'price': player.get('now_cost', 0) / 10.0,
                'predicted_points': prediction_result['ensemble_prediction'],
                'linear_prediction': prediction_result['individual_predictions']['linear_regression'],
                'rf_prediction': prediction_result['individual_predictions']['random_forest'],
                'gb_prediction': prediction_result['individual_predictions']['gradient_boosting'],
                'xgb_prediction': prediction_result['individual_predictions']['xgboost'],
                'form': prediction_result['features']['form'],
                'minutes_probability': prediction_result['features']['minutes_probability'],
                'rotation_risk': prediction_result['features']['rotation_risk'],
                'fixture_difficulty': prediction_result['features']['fixture_difficulty'],
                'home_away_factor': prediction_result['features']['home_away_factor'],
                'injury_risk': prediction_result['features']['injury_risk'],
                'fixture_congestion': prediction_result['features']['fixture_congestion'],
                'avg_goals': prediction_result['features']['avg_goals'],
                'avg_assists': prediction_result['features']['avg_assists'],
                'avg_clean_sheets': prediction_result['features']['avg_clean_sheets'],
                'avg_shots': prediction_result['features']['avg_shots'],
                'avg_key_passes': prediction_result['features']['avg_key_passes'],
                'avg_big_chances': prediction_result['features']['avg_big_chances'],
                'avg_tackles': prediction_result['features']['avg_tackles'],
                'avg_interceptions': prediction_result['features']['avg_interceptions']
            }
            
            predictions_data.append(pred_data)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Store predictions in database
        self.store_advanced_predictions(predictions_data)
        
        print(f"✅ Generated {len(predictions_data)} advanced predictions!")
    
    def get_next_fixture(self, team_id, fixtures_data):
        """Get next fixture for a team"""
        if not fixtures_data:
            return None
        
        current_time = datetime.now()
        
        for fixture in fixtures_data:
            if fixture.get('team_h') == team_id or fixture.get('team_a') == team_id:
                fixture_date = datetime.fromisoformat(fixture.get('kickoff_time', '').replace('Z', '+00:00'))
                if fixture_date > current_time:
                    return fixture
        
        return None
    
    def get_team_name(self, team_id, teams_data):
        """Get team name from team ID"""
        for team in teams_data:
            if team['id'] == team_id:
                return team['name']
        return 'Unknown'
    
    def get_position_name(self, position_type):
        """Convert position type to name"""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_type, 'UNK')
    
    def store_advanced_predictions(self, predictions_data):
        """Store advanced predictions in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create advanced predictions table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advanced_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                name TEXT,
                team_id INTEGER,
                team_name TEXT,
                position_type INTEGER,
                position_name TEXT,
                price REAL,
                predicted_points REAL,
                linear_prediction REAL,
                rf_prediction REAL,
                gb_prediction REAL,
                xgb_prediction REAL,
                form REAL,
                minutes_probability REAL,
                rotation_risk REAL,
                fixture_difficulty REAL,
                home_away_factor REAL,
                injury_risk REAL,
                fixture_congestion REAL,
                avg_goals REAL,
                avg_assists REAL,
                avg_clean_sheets REAL,
                avg_shots REAL,
                avg_key_passes REAL,
                avg_big_chances REAL,
                avg_tackles REAL,
                avg_interceptions REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Clear existing predictions
        cursor.execute("DELETE FROM advanced_predictions")
        
        # Insert new predictions
        for pred in predictions_data:
            cursor.execute("""
                INSERT INTO advanced_predictions (
                    player_id, name, team_id, team_name, position_type, position_name, price,
                    predicted_points, linear_prediction, rf_prediction, gb_prediction, xgb_prediction,
                    form, minutes_probability, rotation_risk, fixture_difficulty, home_away_factor,
                    injury_risk, fixture_congestion, avg_goals, avg_assists, avg_clean_sheets,
                    avg_shots, avg_key_passes, avg_big_chances, avg_tackles, avg_interceptions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred['player_id'], pred['name'], pred['team_id'], pred['team_name'],
                pred['position_type'], pred['position_name'], pred['price'],
                pred['predicted_points'], pred['linear_prediction'], pred['rf_prediction'],
                pred['gb_prediction'], pred['xgb_prediction'], pred['form'],
                pred['minutes_probability'], pred['rotation_risk'], pred['fixture_difficulty'],
                pred['home_away_factor'], pred['injury_risk'], pred['fixture_congestion'],
                pred['avg_goals'], pred['avg_assists'], pred['avg_clean_sheets'],
                pred['avg_shots'], pred['avg_key_passes'], pred['avg_big_chances'],
                pred['avg_tackles'], pred['avg_interceptions']
            ))
        
        conn.commit()
        conn.close()
        print("✅ Advanced predictions stored in database")

def main():
    """Main function to run the advanced prediction engine"""
    print("🚀 Starting Advanced EPL Prediction Engine...")
    
    engine = AdvancedPredictionEngine()
    
    # Build training dataset
    training_data = engine.build_training_dataset()
    
    if training_data is not None:
        # Train models
        engine.train_models(training_data)
        
        # Generate predictions
        engine.generate_advanced_predictions()
        
        print("✅ Advanced prediction engine completed successfully!")
    else:
        print("❌ Failed to build training dataset")

if __name__ == "__main__":
    main()
