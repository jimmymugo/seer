"""
EPL Data Collector - Fetches player and fixture data from Fantasy Premier League API
"""

import requests
import sqlite3
from datetime import datetime
import logging
from typing import Dict, List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EPLDataCollector:
    def __init__(self, db_path: str = "data/epl_data.db"):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_bootstrap_static(self) -> Dict:
        """Fetch all static data from FPL API"""
        url = f"{self.base_url}/bootstrap-static/"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def create_database(self):
        """Create SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                team_id INTEGER,
                position TEXT,
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
                last_updated TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                short_name TEXT,
                strength INTEGER,
                strength_overall_home INTEGER,
                strength_overall_away INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fixtures (
                id INTEGER PRIMARY KEY,
                team_h INTEGER,
                team_a INTEGER,
                team_h_difficulty INTEGER,
                team_a_difficulty INTEGER,
                event INTEGER,
                finished BOOLEAN,
                kickoff_time TIMESTAMP,
                home_score INTEGER,
                away_score INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database created successfully")
    
    def store_teams(self, teams_data: List[Dict]):
        """Store teams data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for team in teams_data:
            cursor.execute('''
                INSERT OR REPLACE INTO teams 
                (id, name, short_name, strength, strength_overall_home, strength_overall_away)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                team['id'], team['name'], team['short_name'], team['strength'],
                team['strength_overall_home'], team['strength_overall_away']
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(teams_data)} teams")
    
    def store_players(self, players_data: List[Dict]):
        """Store players data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for player in players_data:
            cursor.execute('''
                INSERT OR REPLACE INTO players 
                (id, name, team_id, position, price, form, total_points, goals_scored,
                 assists, clean_sheets, goals_conceded, influence, creativity, threat,
                 ict_index, chance_of_playing_next_round, news, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player['id'], player['web_name'], player['team'], player['element_type'],
                player['now_cost'] / 10, player['form'], player['total_points'],
                player['goals_scored'], player['assists'], player['clean_sheets'],
                player['goals_conceded'], player['influence'], player['creativity'],
                player['threat'], player['ict_index'],
                player.get('chance_of_playing_next_round'), player.get('news', ''),
                datetime.now()
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(players_data)} players")
    
    def store_fixtures(self, fixtures_data: List[Dict]):
        """Store fixtures data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for fixture in fixtures_data:
            cursor.execute('''
                INSERT OR REPLACE INTO fixtures 
                (id, team_h, team_a, team_h_difficulty, team_a_difficulty, event,
                 finished, kickoff_time, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fixture['id'], fixture['team_h'], fixture['team_a'],
                fixture['team_h_difficulty'], fixture['team_a_difficulty'],
                fixture['event'], fixture['finished'], fixture['kickoff_time'],
                fixture.get('team_h_score'), fixture.get('team_a_score')
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(fixtures_data)} fixtures")
    
    def collect_all_data(self):
        """Main method to collect all EPL data"""
        logger.info("Starting data collection...")
        
        self.create_database()
        bootstrap_data = self.fetch_bootstrap_static()
        
        self.store_teams(bootstrap_data['teams'])
        self.store_players(bootstrap_data['elements'])
        self.store_fixtures(bootstrap_data['events'])
        
        logger.info("Data collection completed successfully!")

if __name__ == "__main__":
    collector = EPLDataCollector()
    collector.collect_all_data()
