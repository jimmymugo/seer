"""
Feature Engineering - Process raw EPL data and create features for prediction
"""

import pandas as pd
import numpy as np
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        
    def load_data(self):
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        players_df = pd.read_sql_query("SELECT * FROM players", conn)
        teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
        fixtures_df = pd.read_sql_query("SELECT * FROM fixtures", conn)
        conn.close()
        return players_df, teams_df, fixtures_df
    
    def engineer_features(self, players_df, teams_df, fixtures_df):
        """Engineer features for prediction"""
        # Position features
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        players_df['position_name'] = players_df['position'].map(position_map)
        players_df['is_goalkeeper'] = (players_df['position'] == 1).astype(int)
        players_df['is_defender'] = (players_df['position'] == 2).astype(int)
        players_df['is_midfielder'] = (players_df['position'] == 3).astype(int)
        players_df['is_forward'] = (players_df['position'] == 4).astype(int)
        
        # Form features
        players_df['recent_form'] = players_df['form'].astype(float)
        players_df['form_momentum'] = np.where(players_df['recent_form'] > 5, 1, 0)
        
        # Team features
        players_df = players_df.merge(
            teams_df[['id', 'strength', 'strength_overall_home', 'strength_overall_away']],
            left_on='team_id', right_on='id', suffixes=('', '_team')
        )
        players_df['team_strength'] = players_df['strength']
        
        # Fixture difficulty - simplified version
        try:
            next_fixtures = fixtures_df[fixtures_df['finished'] == False]
            if len(next_fixtures) > 0:
                team_difficulty = {}
                for _, fixture in next_fixtures.iterrows():
                    team_h, team_a = fixture['team_h'], fixture['team_a']
                    if team_h not in team_difficulty: team_difficulty[team_h] = []
                    if team_a not in team_difficulty: team_difficulty[team_a] = []
                    team_difficulty[team_h].append(fixture['team_h_difficulty'])
                    team_difficulty[team_a].append(fixture['team_a_difficulty'])
                
                avg_difficulty = {team: np.mean(diffs) for team, diffs in team_difficulty.items()}
                players_df['next_fixture_difficulty'] = players_df['team_id'].map(avg_difficulty).fillna(3)
            else:
                # No fixtures available, use default difficulty
                players_df['next_fixture_difficulty'] = 3.0
        except Exception as e:
            # If any error occurs with fixtures, use default difficulty
            logger.warning(f"Error processing fixtures: {e}. Using default difficulty.")
            players_df['next_fixture_difficulty'] = 3.0
        
        # Statistical features
        players_df['points_per_game'] = players_df['total_points'] / 38
        players_df['goals_per_game'] = players_df['goals_scored'] / 38
        players_df['assists_per_game'] = players_df['assists'] / 38
        players_df['value_for_money'] = players_df['total_points'] / players_df['price']
        
        # Injury features
        players_df['injury_status'] = np.where(
            players_df['chance_of_playing_next_round'].isna(), 1,
            np.where(players_df['chance_of_playing_next_round'] == 0, 0, 1)
        )
        players_df['chance_of_playing'] = players_df['chance_of_playing_next_round'].fillna(100) / 100
        
        return players_df
    
    def engineer_all_features(self):
        """Main method to engineer all features"""
        logger.info("Starting feature engineering...")
        
        players_df, teams_df, fixtures_df = self.load_data()
        processed_df = self.engineer_features(players_df, teams_df, fixtures_df)
        
        # Save processed data
        conn = sqlite3.connect(self.db_path)
        processed_df.to_sql('processed_players', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"Feature engineering completed. Processed {len(processed_df)} players.")
        return processed_df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    processed_data = engineer.engineer_all_features()
