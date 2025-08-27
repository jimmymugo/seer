#!/usr/bin/env python3
"""
Trend Analyzer - Tracks player performance over time and generates form graphs
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json

class TrendAnalyzer:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        
    def create_performance_history_table(self):
        """Create table to store historical performance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                player_name TEXT,
                gameweek INTEGER,
                opponent_team TEXT,
                minutes_played INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                goals_conceded INTEGER,
                bonus_points INTEGER,
                total_points INTEGER,
                expected_goals REAL,
                expected_assists REAL,
                shots_on_target INTEGER,
                key_passes INTEGER,
                tackles INTEGER,
                interceptions INTEGER,
                match_date TIMESTAMP,
                home_away TEXT,
                fixture_difficulty INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Performance history table created")
    
    def generate_synthetic_history(self):
        """Generate synthetic historical data for trend analysis"""
        conn = sqlite3.connect(self.db_path)
        
        # Get current players
        players_df = pd.read_sql_query("SELECT * FROM players_advanced LIMIT 50", conn)
        
        # Generate historical data for each player
        history_data = []
        
        for _, player in players_df.iterrows():
            # Generate last 10 gameweeks of data
            for gw in range(1, 11):
                # Base performance based on player stats
                base_points = player['total_points'] / 38  # Average per game
                
                # Add variation based on form and random factors
                form_factor = player['recent_form_weighted'] / 10
                random_factor = np.random.normal(0, 2)
                
                # Calculate gameweek performance
                gw_points = max(0, base_points + form_factor + random_factor)
                
                # Generate supporting stats
                goals = int(gw_points / 4) if gw_points > 6 else 0
                assists = int((gw_points - goals * 4) / 3) if gw_points > 4 else 0
                clean_sheets = 1 if player['position'] in [1, 2] and gw_points > 4 else 0
                minutes = np.random.randint(60, 90) if gw_points > 2 else np.random.randint(0, 90)
                
                # Expected stats
                xg = player['expected_goals'] / 38 + np.random.normal(0, 0.1)
                xa = player['expected_assists'] / 38 + np.random.normal(0, 0.1)
                
                # Opponent and fixture info
                opponents = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man Utd', 'Spurs', 
                           'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']
                opponent = np.random.choice(opponents)
                difficulty = np.random.randint(1, 6)
                home_away = np.random.choice(['H', 'A'])
                
                history_data.append({
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'gameweek': gw,
                    'opponent_team': opponent,
                    'minutes_played': minutes,
                    'goals_scored': goals,
                    'assists': assists,
                    'clean_sheets': clean_sheets,
                    'goals_conceded': 0 if clean_sheets else np.random.randint(1, 4),
                    'bonus_points': int(gw_points / 3) if gw_points > 6 else 0,
                    'total_points': round(gw_points, 1),
                    'expected_goals': round(xg, 2),
                    'expected_assists': round(xa, 2),
                    'shots_on_target': goals + np.random.randint(0, 3),
                    'key_passes': assists + np.random.randint(0, 2),
                    'tackles': np.random.randint(0, 5) if player['position'] in [2, 3] else 0,
                    'interceptions': np.random.randint(0, 3) if player['position'] in [2, 3] else 0,
                    'match_date': (datetime.now() - timedelta(weeks=10-gw)).isoformat(),
                    'home_away': home_away,
                    'fixture_difficulty': difficulty
                })
        
        # Insert historical data
        history_df = pd.DataFrame(history_data)
        history_df.to_sql('player_performance_history', conn, if_exists='append', index=False)
        
        conn.close()
        print(f"✅ Generated {len(history_data)} historical performance records")
    
    def calculate_trend_metrics(self, player_id: int, gameweeks: int = 5) -> dict:
        """Calculate trend metrics for a player"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent performance data
        query = """
            SELECT * FROM player_performance_history 
            WHERE player_id = ? 
            ORDER BY gameweek DESC 
            LIMIT ?
        """
        
        history_df = pd.read_sql_query(query, conn, params=(player_id, gameweeks))
        conn.close()
        
        if len(history_df) == 0:
            return {}
        
        # Calculate trend metrics
        recent_points = history_df['total_points'].values
        recent_xg = history_df['expected_goals'].values
        recent_xa = history_df['expected_assists'].values
        
        # Form trend (weighted average with recent games more important)
        weights = np.linspace(0.5, 1.0, len(recent_points))
        weighted_form = np.average(recent_points, weights=weights)
        
        # Trend direction
        if len(recent_points) >= 2:
            trend_direction = "↗️" if recent_points[0] > recent_points[-1] else "↘️" if recent_points[0] < recent_points[-1] else "→"
        else:
            trend_direction = "→"
        
        # Consistency score
        consistency = 1.0 - (np.std(recent_points) / np.mean(recent_points)) if np.mean(recent_points) > 0 else 0.0
        
        # Fixture difficulty trend
        avg_difficulty = history_df['fixture_difficulty'].mean()
        
        return {
            'player_id': player_id,
            'recent_form': round(weighted_form, 1),
            'trend_direction': trend_direction,
            'consistency_score': round(consistency, 2),
            'avg_xg': round(np.mean(recent_xg), 2),
            'avg_xa': round(np.mean(recent_xa), 2),
            'avg_difficulty': round(avg_difficulty, 1),
            'form_trend': recent_points.tolist(),
            'gameweeks': history_df['gameweek'].tolist()
        }
    
    def get_player_trend_data(self, player_name: str) -> dict:
        """Get trend data for a specific player"""
        conn = sqlite3.connect(self.db_path)
        
        # Get player ID
        player_query = "SELECT id FROM players_advanced WHERE name = ?"
        player_result = conn.execute(player_query, (player_name,)).fetchone()
        
        if not player_result:
            conn.close()
            return {}
        
        player_id = player_result[0]
        
        # Get all historical data for this player
        history_query = """
            SELECT * FROM player_performance_history 
            WHERE player_id = ? 
            ORDER BY gameweek ASC
        """
        
        history_df = pd.read_sql_query(history_query, conn, params=(player_id,))
        conn.close()
        
        if len(history_df) == 0:
            return {}
        
        # Calculate comprehensive trend data
        trend_data = {
            'player_name': player_name,
            'total_gameweeks': len(history_df),
            'avg_points': round(history_df['total_points'].mean(), 1),
            'total_points': round(history_df['total_points'].sum(), 1),
            'avg_xg': round(history_df['expected_goals'].mean(), 2),
            'avg_xa': round(history_df['expected_assists'].mean(), 2),
            'form_trend': history_df['total_points'].tolist(),
            'xg_trend': history_df['expected_goals'].tolist(),
            'xa_trend': history_df['expected_assists'].tolist(),
            'gameweeks': history_df['gameweek'].tolist(),
            'opponents': history_df['opponent_team'].tolist(),
            'difficulties': history_df['fixture_difficulty'].tolist(),
            'home_away': history_df['home_away'].tolist()
        }
        
        return trend_data

if __name__ == "__main__":
    analyzer = TrendAnalyzer()
    analyzer.create_performance_history_table()
    analyzer.generate_synthetic_history()
    print("✅ Trend analyzer setup completed")
