#!/usr/bin/env python3
"""
Advanced Feature Engineering - Enhanced features with exponential weighting and advanced metrics
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        
    def load_advanced_data(self):
        """Load advanced data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load advanced tables
        players_df = pd.read_sql_query("SELECT * FROM players_advanced", conn)
        teams_df = pd.read_sql_query("SELECT * FROM teams_advanced", conn)
        fixtures_df = pd.read_sql_query("SELECT * FROM fixtures_advanced", conn)
        
        # Load performance history if available
        try:
            history_df = pd.read_sql_query("SELECT * FROM player_performance_history", conn)
        except:
            history_df = pd.DataFrame()
        
        # Load news if available
        try:
            news_df = pd.read_sql_query("SELECT * FROM player_news", conn)
        except:
            news_df = pd.DataFrame()
        
        conn.close()
        return players_df, teams_df, fixtures_df, history_df, news_df
    
    def calculate_exponential_weighted_form(self, history_df: pd.DataFrame, player_id: int, 
                                          decay_factor: float = 0.8) -> float:
        """Calculate exponentially weighted recent form"""
        if history_df.empty:
            return 0.0
        
        player_history = history_df[history_df['player_id'] == player_id].copy()
        if player_history.empty:
            return 0.0
        
        # Sort by match date (most recent first)
        player_history = player_history.sort_values('match_date', ascending=False)
        
        # Calculate exponential weights
        weights = np.array([decay_factor ** i for i in range(len(player_history))])
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average of total points
        weighted_form = np.average(player_history['total_points'].values, weights=weights)
        
        return weighted_form
    
    def calculate_opponent_adjusted_stats(self, player_data: Dict, fixtures_df: pd.DataFrame, 
                                        teams_df: pd.DataFrame) -> Dict:
        """Calculate opponent-adjusted statistics"""
        adjusted_stats = {}
        
        # Get team defensive strength
        team_id = player_data.get('team_id')
        if team_id is None:
            return adjusted_stats
        
        # Find upcoming fixtures for this team
        upcoming_fixtures = fixtures_df[
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
            (fixtures_df['finished'] == False)
        ]
        
        if upcoming_fixtures.empty:
            adjusted_stats['next_opponent_difficulty'] = 3.0
            adjusted_stats['opponent_adjusted_goals'] = player_data.get('goals_scored', 0)
            adjusted_stats['opponent_adjusted_assists'] = player_data.get('assists', 0)
            return adjusted_stats
        
        # Calculate average opponent difficulty
        difficulties = []
        for _, fixture in upcoming_fixtures.iterrows():
            if fixture['team_h'] == team_id:
                difficulties.append(fixture['team_a_difficulty'])
            else:
                difficulties.append(fixture['team_h_difficulty'])
        
        avg_difficulty = np.mean(difficulties) if difficulties else 3.0
        adjusted_stats['next_opponent_difficulty'] = avg_difficulty
        
        # Adjust goals and assists based on opponent difficulty
        # Higher difficulty = lower expected output
        difficulty_factor = 1.0 - (avg_difficulty - 3.0) * 0.1  # 10% reduction per difficulty level
        
        adjusted_stats['opponent_adjusted_goals'] = player_data.get('goals_scored', 0) * difficulty_factor
        adjusted_stats['opponent_adjusted_assists'] = player_data.get('assists', 0) * difficulty_factor
        
        return adjusted_stats
    
    def calculate_rotation_risk_features(self, player_data: Dict, history_df: pd.DataFrame) -> Dict:
        """Calculate rotation risk based on recent minutes and team context"""
        rotation_features = {}
        
        player_id = player_data.get('id')
        if player_id is None or history_df.empty:
            rotation_features['rotation_risk'] = 0.5
            rotation_features['minutes_consistency'] = 0.5
            return rotation_features
        
        # Get recent performance history
        recent_history = history_df[history_df['player_id'] == player_id].copy()
        if recent_history.empty:
            rotation_features['rotation_risk'] = 0.5
            rotation_features['minutes_consistency'] = 0.5
            return rotation_features
        
        # Sort by date and get last 5 matches
        recent_history = recent_history.sort_values('match_date', ascending=False).head(5)
        
        # Calculate minutes consistency
        minutes_played = recent_history['minutes_played'].values
        avg_minutes = np.mean(minutes_played)
        minutes_std = np.std(minutes_played)
        
        # Rotation risk: higher if inconsistent minutes or low average
        minutes_consistency = 1.0 - (minutes_std / 90.0)  # Normalize by full match
        rotation_risk = 1.0 - (avg_minutes / 90.0)  # Higher risk if low minutes
        
        # Combine factors
        final_rotation_risk = (rotation_risk + (1.0 - minutes_consistency)) / 2
        
        rotation_features['rotation_risk'] = max(0.0, min(1.0, final_rotation_risk))
        rotation_features['minutes_consistency'] = max(0.0, min(1.0, minutes_consistency))
        
        return rotation_features
    
    def calculate_team_context_features(self, player_data: Dict, teams_df: pd.DataFrame, 
                                      fixtures_df: pd.DataFrame) -> Dict:
        """Calculate team-level context features"""
        team_features = {}
        
        team_id = player_data.get('team_id')
        if team_id is None:
            return team_features
        
        # Get team data
        team_data = teams_df[teams_df['id'] == team_id]
        if team_data.empty:
            return team_features
        
        team_data = team_data.iloc[0]
        
        # Team form and strength
        team_features['team_form'] = team_data.get('form_last_5', 0.5)
        team_features['team_attacking_strength'] = team_data.get('attacking_strength', 3.0) / 5.0
        team_features['team_defensive_strength'] = team_data.get('defensive_strength', 3.0) / 5.0
        
        # Fixture congestion
        upcoming_fixtures = fixtures_df[
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
            (fixtures_df['finished'] == False)
        ]
        
        # Count fixtures in next 7 days
        next_week = datetime.now() + timedelta(days=7)
        congested_fixtures = upcoming_fixtures[
            pd.to_datetime(upcoming_fixtures['kickoff_time']) <= next_week
        ]
        
        fixture_congestion = len(congested_fixtures)
        team_features['fixture_congestion'] = fixture_congestion
        team_features['congestion_risk'] = min(1.0, fixture_congestion / 3.0)  # Normalize
        
        return team_features
    
    def calculate_advanced_performance_metrics(self, player_data: Dict) -> Dict:
        """Calculate advanced performance metrics"""
        advanced_metrics = {}
        
        # Expected goals and assists efficiency
        goals = player_data.get('goals_scored', 0)
        assists = player_data.get('assists', 0)
        xg = player_data.get('expected_goals', 0)
        xa = player_data.get('expected_assists', 0)
        minutes = player_data.get('minutes_played_last_5', 0)
        
        # Efficiency metrics
        if xg > 0:
            advanced_metrics['goals_efficiency'] = goals / xg
        else:
            advanced_metrics['goals_efficiency'] = 1.0
        
        if xa > 0:
            advanced_metrics['assists_efficiency'] = assists / xa
        else:
            advanced_metrics['assists_efficiency'] = 1.0
        
        # Per 90 metrics
        if minutes > 0:
            advanced_metrics['goals_per_90'] = (goals * 90) / minutes
            advanced_metrics['assists_per_90'] = (assists * 90) / minutes
            advanced_metrics['xg_per_90'] = (xg * 90) / minutes
            advanced_metrics['xa_per_90'] = (xa * 90) / minutes
        else:
            advanced_metrics['goals_per_90'] = 0
            advanced_metrics['assists_per_90'] = 0
            advanced_metrics['xg_per_90'] = 0
            advanced_metrics['xa_per_90'] = 0
        
        # Defensive metrics for defenders and midfielders
        position = player_data.get('position')
        if position in [2, 3]:  # DEF or MID
            tackles = player_data.get('tackles', 0)
            interceptions = player_data.get('interceptions', 0)
            
            if minutes > 0:
                advanced_metrics['tackles_per_90'] = (tackles * 90) / minutes
                advanced_metrics['interceptions_per_90'] = (interceptions * 90) / minutes
            else:
                advanced_metrics['tackles_per_90'] = 0
                advanced_metrics['interceptions_per_90'] = 0
        
        return advanced_metrics
    
    def engineer_all_advanced_features(self):
        """Main method to engineer all advanced features"""
        logger.info("Starting advanced feature engineering...")
        
        # Load data
        players_df, teams_df, fixtures_df, history_df, news_df = self.load_advanced_data()
        
        if players_df.empty:
            logger.error("No players data available")
            return pd.DataFrame()
        
        # Process each player
        enhanced_players = []
        
        for _, player in players_df.iterrows():
            player_dict = player.to_dict()
            
            # Calculate exponential weighted form
            weighted_form = self.calculate_exponential_weighted_form(history_df, player['id'])
            player_dict['exponential_weighted_form'] = weighted_form
            
            # Calculate opponent-adjusted stats
            opponent_stats = self.calculate_opponent_adjusted_stats(player_dict, fixtures_df, teams_df)
            player_dict.update(opponent_stats)
            
            # Calculate rotation risk features
            rotation_features = self.calculate_rotation_risk_features(player_dict, history_df)
            player_dict.update(rotation_features)
            
            # Calculate team context features
            team_features = self.calculate_team_context_features(player_dict, teams_df, fixtures_df)
            player_dict.update(team_features)
            
            # Calculate advanced performance metrics
            performance_metrics = self.calculate_advanced_performance_metrics(player_dict)
            player_dict.update(performance_metrics)
            
            # Add position-specific features
            position = player.get('position')
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            player_dict['position_name'] = position_map.get(position, 'UNK')
            
            # Create position flags
            player_dict['is_goalkeeper'] = 1 if position == 1 else 0
            player_dict['is_defender'] = 1 if position == 2 else 0
            player_dict['is_midfielder'] = 1 if position == 3 else 0
            player_dict['is_forward'] = 1 if position == 4 else 0
            
            # Value for money
            price = player.get('price', 0)
            total_points = player.get('total_points', 0)
            if price > 0:
                player_dict['value_for_money'] = total_points / price
            else:
                player_dict['value_for_money'] = 0
            
            enhanced_players.append(player_dict)
        
        # Convert to DataFrame
        enhanced_df = pd.DataFrame(enhanced_players)
        
        # Save processed data
        conn = sqlite3.connect(self.db_path)
        enhanced_df.to_sql('processed_players_advanced', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"Advanced feature engineering completed. Processed {len(enhanced_df)} players.")
        return enhanced_df

if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    processed_data = engineer.engineer_all_advanced_features()
