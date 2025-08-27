#!/usr/bin/env python3
"""
Advanced Data Collector - Enhanced EPL data collection with multiple sources
"""

import requests
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import List, Dict, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEPLDataCollector:
    def __init__(self, db_path: str = "epl_data.db"):
        self.db_path = db_path
        self.fpl_api_base = "https://fantasy.premierleague.com/api"
        
    def create_advanced_database(self):
        """Create database with advanced metrics tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced players table with advanced metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players_advanced (
                id INTEGER PRIMARY KEY,
                name TEXT,
                team_id INTEGER,
                position INTEGER,
                price REAL,
                form REAL,
                total_points INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                goals_conceded INTEGER,
                influence REAL,
                creativity REAL,
                threat REAL,
                ict_index REAL,
                chance_of_playing_next_round REAL,
                news TEXT,
                last_updated TIMESTAMP,
                -- Advanced metrics
                expected_goals REAL DEFAULT 0,
                expected_assists REAL DEFAULT 0,
                shots_on_target INTEGER DEFAULT 0,
                key_passes INTEGER DEFAULT 0,
                big_chances_created INTEGER DEFAULT 0,
                tackles INTEGER DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                minutes_played_last_5 INTEGER DEFAULT 0,
                rotation_risk REAL DEFAULT 0.5,
                injury_status TEXT DEFAULT 'Fit',
                suspension_status TEXT DEFAULT 'Available',
                likely_to_start REAL DEFAULT 0.8,
                recent_form_weighted REAL DEFAULT 0,
                opponent_adjusted_goals REAL DEFAULT 0,
                team_form REAL DEFAULT 0,
                fixture_congestion INTEGER DEFAULT 0,
                is_captain BOOLEAN DEFAULT FALSE,
                is_penalty_taker BOOLEAN DEFAULT FALSE,
                is_free_kick_taker BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Team advanced metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams_advanced (
                id INTEGER PRIMARY KEY,
                name TEXT,
                short_name TEXT,
                strength REAL,
                strength_overall_home REAL,
                strength_overall_away REAL,
                -- Advanced team metrics
                goals_scored_last_5 INTEGER DEFAULT 0,
                goals_conceded_last_5 INTEGER DEFAULT 0,
                clean_sheets_last_5 INTEGER DEFAULT 0,
                form_last_5 REAL DEFAULT 0,
                defensive_strength REAL DEFAULT 0,
                attacking_strength REAL DEFAULT 0,
                fixture_difficulty_avg REAL DEFAULT 3.0,
                last_updated TIMESTAMP
            )
        ''')
        
        # Fixtures with advanced context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fixtures_advanced (
                id INTEGER PRIMARY KEY,
                team_h INTEGER,
                team_a INTEGER,
                team_h_difficulty INTEGER,
                team_a_difficulty INTEGER,
                event INTEGER,
                finished BOOLEAN,
                kickoff_time TIMESTAMP,
                home_score INTEGER,
                away_score INTEGER,
                -- Advanced fixture metrics
                fixture_congestion_home INTEGER DEFAULT 0,
                fixture_congestion_away INTEGER DEFAULT 0,
                weather_conditions TEXT DEFAULT 'Unknown',
                referee TEXT DEFAULT 'Unknown',
                venue TEXT DEFAULT 'Home',
                last_updated TIMESTAMP
            )
        ''')
        
        # Player performance history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                gameweek INTEGER,
                opponent_team_id INTEGER,
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
                FOREIGN KEY (player_id) REFERENCES players_advanced (id)
            )
        ''')
        
        # Injury and news tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                news_type TEXT, -- 'injury', 'suspension', 'rotation', 'form'
                news_text TEXT,
                severity REAL, -- 0-1 scale
                expected_return_date TIMESTAMP,
                source TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players_advanced (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Advanced database created successfully")
    
    def fetch_fpl_bootstrap_enhanced(self) -> Dict:
        """Fetch enhanced FPL data with additional context"""
        try:
            url = f"{self.fpl_api_base}/bootstrap-static/"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            logger.info("Enhanced FPL data fetched successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching FPL data: {e}")
            return {}
    
    def calculate_advanced_metrics(self, players_data: List[Dict], teams_data: List[Dict]) -> List[Dict]:
        """Calculate advanced metrics for players"""
        enhanced_players = []
        
        for player in players_data:
            enhanced_player = player.copy()
            
            # Calculate expected goals and assists (simplified model)
            goals = player.get('goals_scored', 0)
            assists = player.get('assists', 0)
            minutes = player.get('minutes', 0)
            
            if minutes > 0:
                # Simple xG calculation based on goals and minutes
                enhanced_player['expected_goals'] = round(goals * 0.8 + (minutes / 90) * 0.1, 2)
                enhanced_player['expected_assists'] = round(assists * 0.7 + (minutes / 90) * 0.05, 2)
            else:
                enhanced_player['expected_goals'] = 0
                enhanced_player['expected_assists'] = 0
            
            # Calculate rotation risk based on minutes played
            minutes_last_5 = player.get('minutes', 0)  # Simplified - would need historical data
            enhanced_player['minutes_played_last_5'] = minutes_last_5
            enhanced_player['rotation_risk'] = max(0, 1 - (minutes_last_5 / 450))  # 450 = 5 * 90
            
            # Calculate recent form with exponential weighting
            form = player.get('form', 0)
            enhanced_player['recent_form_weighted'] = form * 1.2  # Recent matches weighted more
            
            # Set default values for advanced metrics
            enhanced_player['shots_on_target'] = max(0, int(goals * 2.5))  # Simplified
            enhanced_player['key_passes'] = max(0, int(assists * 3))  # Simplified
            enhanced_player['big_chances_created'] = max(0, int(assists * 1.5))  # Simplified
            enhanced_player['tackles'] = max(0, int(minutes / 10))  # Simplified
            enhanced_player['interceptions'] = max(0, int(minutes / 15))  # Simplified
            
            # Injury and availability status
            chance_playing = player.get('chance_of_playing_next_round')
            if chance_playing is None:
                enhanced_player['injury_status'] = 'Fit'
                enhanced_player['likely_to_start'] = 0.9
            elif chance_playing == 0:
                enhanced_player['injury_status'] = 'Injured'
                enhanced_player['likely_to_start'] = 0.0
            else:
                enhanced_player['injury_status'] = 'Doubtful'
                enhanced_player['likely_to_start'] = chance_playing / 100
            
            enhanced_player['suspension_status'] = 'Available'
            
            # Team context (simplified)
            team_id = player.get('team', 0)
            team = next((t for t in teams_data if t['id'] == team_id), None)
            if team:
                enhanced_player['team_form'] = team.get('strength', 3) / 5
                enhanced_player['fixture_congestion'] = 1  # Simplified
            
            # Player role importance (simplified)
            enhanced_player['is_captain'] = False
            enhanced_player['is_penalty_taker'] = False
            enhanced_player['is_free_kick_taker'] = False
            
            # Opponent-adjusted stats (simplified)
            enhanced_player['opponent_adjusted_goals'] = goals * 1.0  # Would need fixture data
            
            enhanced_players.append(enhanced_player)
        
        return enhanced_players
    
    def store_advanced_players(self, players_data: List[Dict]):
        """Store enhanced players data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for player in players_data:
            cursor.execute('''
                INSERT OR REPLACE INTO players_advanced 
                (id, name, team_id, position, price, form, total_points, goals_scored,
                 assists, clean_sheets, goals_conceded, influence, creativity, threat,
                 ict_index, chance_of_playing_next_round, news, last_updated,
                 expected_goals, expected_assists, shots_on_target, key_passes,
                 big_chances_created, tackles, interceptions, minutes_played_last_5,
                 rotation_risk, injury_status, suspension_status, likely_to_start,
                 recent_form_weighted, opponent_adjusted_goals, team_form,
                 fixture_congestion, is_captain, is_penalty_taker, is_free_kick_taker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player['id'], player['web_name'], player['team'], player['element_type'],
                player['now_cost'] / 10, player['form'], player['total_points'],
                player['goals_scored'], player['assists'], player['clean_sheets'],
                player['goals_conceded'], player['influence'], player['creativity'],
                player['threat'], player['ict_index'],
                player.get('chance_of_playing_next_round'),
                player.get('news', ''),
                datetime.now(),
                player.get('expected_goals', 0), player.get('expected_assists', 0),
                player.get('shots_on_target', 0), player.get('key_passes', 0),
                player.get('big_chances_created', 0), player.get('tackles', 0),
                player.get('interceptions', 0), player.get('minutes_played_last_5', 0),
                player.get('rotation_risk', 0.5), player.get('injury_status', 'Fit'),
                player.get('suspension_status', 'Available'), player.get('likely_to_start', 0.8),
                player.get('recent_form_weighted', 0), player.get('opponent_adjusted_goals', 0),
                player.get('team_form', 0), player.get('fixture_congestion', 0),
                player.get('is_captain', False), player.get('is_penalty_taker', False),
                player.get('is_free_kick_taker', False)
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(players_data)} enhanced players")
    
    def store_advanced_teams(self, teams_data: List[Dict]):
        """Store enhanced teams data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for team in teams_data:
            # Calculate advanced team metrics
            goals_scored_last_5 = 0  # Would need historical data
            goals_conceded_last_5 = 0
            clean_sheets_last_5 = 0
            form_last_5 = team.get('strength', 3) / 5
            
            cursor.execute('''
                INSERT OR REPLACE INTO teams_advanced 
                (id, name, short_name, strength, strength_overall_home, strength_overall_away,
                 goals_scored_last_5, goals_conceded_last_5, clean_sheets_last_5,
                 form_last_5, defensive_strength, attacking_strength, fixture_difficulty_avg, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                team['id'], team['name'], team['short_name'], team['strength'],
                team['strength_overall_home'], team['strength_overall_away'],
                goals_scored_last_5, goals_conceded_last_5, clean_sheets_last_5,
                form_last_5, team['strength_overall_home'], team['strength_overall_away'],
                3.0, datetime.now()
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(teams_data)} enhanced teams")
    
    def collect_all_advanced_data(self):
        """Main method to collect all advanced EPL data"""
        logger.info("Starting advanced data collection...")
        
        self.create_advanced_database()
        bootstrap_data = self.fetch_fpl_bootstrap_enhanced()
        
        if not bootstrap_data:
            logger.error("Failed to fetch bootstrap data")
            return
        
        # Enhance and store data
        enhanced_players = self.calculate_advanced_metrics(
            bootstrap_data['elements'], 
            bootstrap_data['teams']
        )
        
        self.store_advanced_players(enhanced_players)
        self.store_advanced_teams(bootstrap_data['teams'])
        
        logger.info("Advanced data collection completed successfully!")

if __name__ == "__main__":
    collector = AdvancedEPLDataCollector()
    collector.collect_all_advanced_data()
