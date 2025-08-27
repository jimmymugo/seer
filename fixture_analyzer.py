#!/usr/bin/env python3
"""
Fixture Analyzer - Generates fixture difficulty heatmaps and upcoming match analysis
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json

class FixtureAnalyzer:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        
    def create_fixtures_table(self):
        """Create table to store upcoming fixtures"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS upcoming_fixtures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                team_name TEXT,
                opponent_id INTEGER,
                opponent_name TEXT,
                gameweek INTEGER,
                fixture_date TIMESTAMP,
                home_away TEXT,
                difficulty_rating INTEGER,
                venue TEXT,
                predicted_goals_for REAL,
                predicted_goals_against REAL,
                clean_sheet_probability REAL,
                win_probability REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Upcoming fixtures table created")
    
    def generate_upcoming_fixtures(self):
        """Generate upcoming fixtures for the next 5 gameweeks"""
        conn = sqlite3.connect(self.db_path)
        
        # Get teams
        teams_df = pd.read_sql_query("SELECT * FROM teams_advanced", conn)
        
        fixtures_data = []
        
        # Generate fixtures for next 5 gameweeks
        for gw in range(1, 6):
            # Create fixture pairs (each team plays once per gameweek)
            team_ids = teams_df['id'].tolist()
            np.random.shuffle(team_ids)
            
            for i in range(0, len(team_ids), 2):
                if i + 1 < len(team_ids):
                    home_team_id = team_ids[i]
                    away_team_id = team_ids[i + 1]
                    
                    home_team = teams_df[teams_df['id'] == home_team_id].iloc[0]
                    away_team = teams_df[teams_df['id'] == away_team_id].iloc[0]
                    
                    # Calculate difficulty based on team strengths
                    home_strength = home_team['strength_overall_home']
                    away_strength = away_team['strength_overall_away']
                    
                    # Difficulty for home team (lower = easier)
                    home_difficulty = max(1, min(5, int(6 - (home_strength / 20))))
                    away_difficulty = max(1, min(5, int(away_strength / 20 + 1)))
                    
                    # Predict match outcomes
                    home_attack = home_team['attacking_strength']
                    away_defense = away_team['defensive_strength']
                    away_attack = away_team['attacking_strength']
                    home_defense = home_team['defensive_strength']
                    
                    # Predicted goals
                    home_goals = max(0, (home_attack - away_defense + 50) / 100 + np.random.normal(0, 0.5))
                    away_goals = max(0, (away_attack - home_defense + 30) / 100 + np.random.normal(0, 0.5))
                    
                    # Probabilities
                    home_win_prob = 0.4 + (home_attack - away_attack) / 200
                    clean_sheet_prob = 0.2 + (home_defense - away_attack) / 200
                    
                    # Home team fixture
                    fixtures_data.append({
                        'team_id': home_team_id,
                        'team_name': home_team['name'],
                        'opponent_id': away_team_id,
                        'opponent_name': away_team['name'],
                        'gameweek': gw,
                        'fixture_date': (datetime.now() + timedelta(weeks=gw)).isoformat(),
                        'home_away': 'H',
                        'difficulty_rating': home_difficulty,
                        'venue': home_team['name'],
                        'predicted_goals_for': round(home_goals, 1),
                        'predicted_goals_against': round(away_goals, 1),
                        'clean_sheet_probability': round(clean_sheet_prob, 2),
                        'win_probability': round(home_win_prob, 2)
                    })
                    
                    # Away team fixture
                    fixtures_data.append({
                        'team_id': away_team_id,
                        'team_name': away_team['name'],
                        'opponent_id': home_team_id,
                        'opponent_name': home_team['name'],
                        'gameweek': gw,
                        'fixture_date': (datetime.now() + timedelta(weeks=gw)).isoformat(),
                        'home_away': 'A',
                        'difficulty_rating': away_difficulty,
                        'venue': home_team['name'],
                        'predicted_goals_for': round(away_goals, 1),
                        'predicted_goals_against': round(home_goals, 1),
                        'clean_sheet_probability': round(0.1 + (away_team['defensive_strength'] - home_attack) / 200, 2),
                        'win_probability': round(1 - home_win_prob, 2)
                    })
        
        # Insert fixtures
        fixtures_df = pd.DataFrame(fixtures_data)
        fixtures_df.to_sql('upcoming_fixtures', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"✅ Generated {len(fixtures_data)} upcoming fixtures")
    
    def get_team_fixture_difficulty(self, team_name: str, gameweeks: int = 5) -> dict:
        """Get fixture difficulty analysis for a specific team"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM upcoming_fixtures 
            WHERE team_name = ? 
            ORDER BY gameweek ASC 
            LIMIT ?
        """
        
        fixtures_df = pd.read_sql_query(query, conn, params=(team_name, gameweeks))
        conn.close()
        
        if len(fixtures_df) == 0:
            return {}
        
        # Calculate difficulty metrics
        avg_difficulty = fixtures_df['difficulty_rating'].mean()
        total_difficulty = fixtures_df['difficulty_rating'].sum()
        
        # Difficulty trend
        difficulties = fixtures_df['difficulty_rating'].tolist()
        if len(difficulties) >= 2:
            difficulty_trend = "Easier" if difficulties[-1] < difficulties[0] else "Harder" if difficulties[-1] > difficulties[0] else "Similar"
        else:
            difficulty_trend = "Single fixture"
        
        # Fixture details
        fixture_details = []
        for _, fixture in fixtures_df.iterrows():
            difficulty_color = {
                1: "🟢 Very Easy",
                2: "🟡 Easy", 
                3: "🟠 Medium",
                4: "🔴 Hard",
                5: "⚫ Very Hard"
            }.get(fixture['difficulty_rating'], "⚪ Unknown")
            
            fixture_details.append({
                'gameweek': fixture['gameweek'],
                'opponent': fixture['opponent_name'],
                'home_away': fixture['home_away'],
                'difficulty': fixture['difficulty_rating'],
                'difficulty_label': difficulty_color,
                'predicted_goals_for': fixture['predicted_goals_for'],
                'predicted_goals_against': fixture['predicted_goals_against'],
                'win_probability': fixture['win_probability'],
                'clean_sheet_probability': fixture['clean_sheet_probability']
            })
        
        return {
            'team_name': team_name,
            'avg_difficulty': round(avg_difficulty, 1),
            'total_difficulty': total_difficulty,
            'difficulty_trend': difficulty_trend,
            'fixtures': fixture_details,
            'difficulty_breakdown': {
                'very_easy': len(fixtures_df[fixtures_df['difficulty_rating'] == 1]),
                'easy': len(fixtures_df[fixtures_df['difficulty_rating'] == 2]),
                'medium': len(fixtures_df[fixtures_df['difficulty_rating'] == 3]),
                'hard': len(fixtures_df[fixtures_df['difficulty_rating'] == 4]),
                'very_hard': len(fixtures_df[fixtures_df['difficulty_rating'] == 5])
            }
        }
    
    def get_fixture_heatmap_data(self) -> dict:
        """Get data for fixture difficulty heatmap"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all fixtures
        fixtures_df = pd.read_sql_query("SELECT * FROM upcoming_fixtures ORDER BY gameweek, team_name", conn)
        conn.close()
        
        if len(fixtures_df) == 0:
            return {}
        
        # Create heatmap data
        teams = fixtures_df['team_name'].unique()
        gameweeks = fixtures_df['gameweek'].unique()
        
        heatmap_data = []
        for team in teams:
            team_fixtures = fixtures_df[fixtures_df['team_name'] == team]
            for gw in gameweeks:
                gw_fixture = team_fixtures[team_fixtures['gameweek'] == gw]
                if len(gw_fixture) > 0:
                    fixture = gw_fixture.iloc[0]
                    heatmap_data.append({
                        'team': team,
                        'gameweek': gw,
                        'difficulty': fixture['difficulty_rating'],
                        'opponent': fixture['opponent_name'],
                        'home_away': fixture['home_away']
                    })
                else:
                    heatmap_data.append({
                        'team': team,
                        'gameweek': gw,
                        'difficulty': 0,
                        'opponent': 'No fixture',
                        'home_away': 'N/A'
                    })
        
        return {
            'teams': teams.tolist(),
            'gameweeks': gameweeks.tolist(),
            'heatmap_data': heatmap_data
        }
    
    def get_player_fixture_impact(self, player_name: str) -> dict:
        """Get fixture impact analysis for a specific player"""
        conn = sqlite3.connect(self.db_path)
        
        # Get player info
        player_query = "SELECT * FROM players_advanced WHERE name = ?"
        player_df = pd.read_sql_query(player_query, conn, params=(player_name,))
        
        if len(player_df) == 0:
            conn.close()
            return {}
        
        player = player_df.iloc[0]
        team_id = player['team_id']
        
        # Get team fixtures
        fixtures_query = """
            SELECT * FROM upcoming_fixtures 
            WHERE team_id = ? 
            ORDER BY gameweek ASC 
            LIMIT 5
        """
        
        fixtures_df = pd.read_sql_query(fixtures_query, conn, params=(team_id,))
        conn.close()
        
        if len(fixtures_df) == 0:
            return {}
        
        # Calculate fixture impact on player performance
        fixture_impacts = []
        for _, fixture in fixtures_df.iterrows():
            base_prediction = player['predicted_points']
            
            # Adjust based on fixture difficulty
            difficulty_factor = (6 - fixture['difficulty_rating']) / 5  # Easier = higher factor
            
            # Adjust based on home/away
            venue_factor = 1.1 if fixture['home_away'] == 'H' else 0.9
            
            # Adjust based on predicted goals
            if player['position'] in [3, 4]:  # MID, FWD
                goals_factor = 1.0 + (fixture['predicted_goals_for'] - 1.5) * 0.2
            elif player['position'] in [1, 2]:  # GK, DEF
                goals_factor = 1.0 + (1.5 - fixture['predicted_goals_against']) * 0.3
            else:
                goals_factor = 1.0
            
            # Calculate adjusted prediction
            adjusted_prediction = base_prediction * difficulty_factor * venue_factor * goals_factor
            
            fixture_impacts.append({
                'gameweek': fixture['gameweek'],
                'opponent': fixture['opponent_name'],
                'home_away': fixture['home_away'],
                'difficulty': fixture['difficulty_rating'],
                'base_prediction': round(base_prediction, 1),
                'adjusted_prediction': round(adjusted_prediction, 1),
                'impact': round(adjusted_prediction - base_prediction, 1),
                'impact_percentage': round(((adjusted_prediction - base_prediction) / base_prediction) * 100, 1)
            })
        
        return {
            'player_name': player_name,
            'position': ['GK', 'DEF', 'MID', 'FWD'][player['position'] - 1],
            'team': fixtures_df.iloc[0]['team_name'] if len(fixtures_df) > 0 else 'Unknown',
            'fixture_impacts': fixture_impacts,
            'avg_impact': round(np.mean([f['impact'] for f in fixture_impacts]), 1) if fixture_impacts else 0
        }

if __name__ == "__main__":
    analyzer = FixtureAnalyzer()
    analyzer.create_fixtures_table()
    analyzer.generate_upcoming_fixtures()
    print("✅ Fixture analyzer setup completed")
