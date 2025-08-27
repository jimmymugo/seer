#!/usr/bin/env python3
"""
Real Historical Tracker - Uses actual FPL data to track best 11 performances
"""

import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json

class RealHistoricalTracker:
    def __init__(self, db_path="epl_data.db"):
        self.db_path = db_path
        self.create_real_historical_tables()
    
    def create_real_historical_tables(self):
        """Create tables for real historical tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Real historical best 11 teams with actual FPL data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS real_historical_best11 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gameweek INTEGER,
                formation TEXT,
                budget REAL,
                total_cost REAL,
                total_predicted_points REAL,
                total_actual_points REAL,
                performance_accuracy REAL,
                captain_points REAL,
                vice_captain_points REAL,
                bench_points REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real historical best 11 players with actual performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS real_historical_best11_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                best11_id INTEGER,
                player_name TEXT,
                position TEXT,
                team TEXT,
                price REAL,
                predicted_points REAL,
                actual_points REAL,
                performance_difference REAL,
                minutes_played INTEGER,
                goals INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                bonus_points INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                saves INTEGER,
                goals_conceded INTEGER,
                own_goals INTEGER,
                penalties_missed INTEGER,
                FOREIGN KEY (best11_id) REFERENCES real_historical_best11 (id)
            )
        """)
        
        # Role-specific performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS role_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gameweek INTEGER,
                position TEXT,
                avg_predicted_points REAL,
                avg_actual_points REAL,
                avg_accuracy REAL,
                total_players INTEGER,
                best_performer TEXT,
                best_performer_points REAL,
                worst_performer TEXT,
                worst_performer_points REAL,
                clean_sheets INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                saves INTEGER,
                goals_conceded INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real FPL gameweek data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS real_fpl_gameweek_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gameweek INTEGER,
                player_id INTEGER,
                player_name TEXT,
                position TEXT,
                team TEXT,
                actual_points REAL,
                minutes_played INTEGER,
                goals INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                bonus_points INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                saves INTEGER,
                goals_conceded INTEGER,
                own_goals INTEGER,
                penalties_missed INTEGER,
                fixture_home TEXT,
                fixture_away TEXT,
                fixture_home_score INTEGER,
                fixture_away_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_fpl_gameweek_data(self, gameweek):
        """Fetch actual FPL data for a specific gameweek"""
        try:
            # Fetch FPL bootstrap data
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
            if response.status_code != 200:
                print(f"❌ Failed to fetch FPL data: {response.status_code}")
                return None
            
            data = response.json()
            
            # Get players data
            players = data.get('elements', [])
            
            # Fetch specific gameweek data
            gw_response = requests.get(f"https://fantasy.premierleague.com/api/event/{gameweek}/live/")
            if gw_response.status_code != 200:
                print(f"❌ Failed to fetch GW{gameweek} data: {gw_response.status_code}")
                return None
            
            gw_data = gw_response.json()
            elements = gw_data.get('elements', {})
            
            # Combine data
            gameweek_players = []
            for player in players:
                player_id = player['id']
                if str(player_id) in elements:
                    gw_stats = elements[str(player_id)]['stats']
                    
                    # Calculate actual points based on FPL scoring
                    actual_points = self.calculate_fpl_points(gw_stats, player['element_type'])
                    
                    gameweek_players.append({
                        'player_id': player_id,
                        'player_name': player['web_name'],
                        'position': self.get_position_name(player['element_type']),
                        'team': self.get_team_name(player['team'], data['teams']),
                        'actual_points': actual_points,
                        'minutes_played': gw_stats.get('minutes', 0),
                        'goals': gw_stats.get('goals_scored', 0),
                        'assists': gw_stats.get('assists', 0),
                        'clean_sheets': gw_stats.get('clean_sheets', 0),
                        'bonus_points': gw_stats.get('bonus', 0),
                        'yellow_cards': gw_stats.get('yellow_cards', 0),
                        'red_cards': gw_stats.get('red_cards', 0),
                        'saves': gw_stats.get('saves', 0),
                        'goals_conceded': gw_stats.get('goals_conceded', 0),
                        'own_goals': gw_stats.get('own_goals', 0),
                        'penalties_missed': gw_stats.get('penalties_missed', 0)
                    })
            
            return gameweek_players
            
        except Exception as e:
            print(f"❌ Error fetching FPL data: {e}")
            return None
    
    def calculate_fpl_points(self, stats, position_type):
        """Calculate FPL points based on position and stats"""
        points = 0
        
        # Minutes played points
        minutes = stats.get('minutes', 0)
        if minutes >= 60:
            points += 2
        elif minutes > 0:
            points += 1
        
        # Position-specific points
        if position_type == 1:  # Goalkeeper
            points += stats.get('saves', 0) * 0.1
            points += stats.get('clean_sheets', 0) * 4
            points -= stats.get('goals_conceded', 0) * 0.5
            points -= stats.get('yellow_cards', 0) * 1
            points -= stats.get('red_cards', 0) * 3
            points -= stats.get('own_goals', 0) * 2
            points -= stats.get('penalties_missed', 0) * 2
            
        elif position_type == 2:  # Defender
            points += stats.get('goals_scored', 0) * 6
            points += stats.get('assists', 0) * 3
            points += stats.get('clean_sheets', 0) * 4
            points -= stats.get('goals_conceded', 0) * 0.5
            points -= stats.get('yellow_cards', 0) * 1
            points -= stats.get('red_cards', 0) * 3
            points -= stats.get('own_goals', 0) * 2
            
        elif position_type == 3:  # Midfielder
            points += stats.get('goals_scored', 0) * 5
            points += stats.get('assists', 0) * 3
            points += stats.get('clean_sheets', 0) * 1
            points -= stats.get('yellow_cards', 0) * 1
            points -= stats.get('red_cards', 0) * 3
            points -= stats.get('own_goals', 0) * 2
            points -= stats.get('penalties_missed', 0) * 2
            
        elif position_type == 4:  # Forward
            points += stats.get('goals_scored', 0) * 4
            points += stats.get('assists', 0) * 3
            points -= stats.get('yellow_cards', 0) * 1
            points -= stats.get('red_cards', 0) * 3
            points -= stats.get('own_goals', 0) * 2
            points -= stats.get('penalties_missed', 0) * 2
        
        # Bonus points
        points += stats.get('bonus', 0)
        
        return max(0, points)  # Points can't be negative
    
    def get_position_name(self, position_type):
        """Convert position type to name"""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_type, 'UNK')
    
    def get_team_name(self, team_id, teams_data):
        """Get team name from team ID"""
        for team in teams_data:
            if team['id'] == team_id:
                return team['name']
        return 'Unknown'
    
    def generate_real_best11_for_gameweek(self, gameweek, predictions_df):
        """Generate a real best 11 team for a specific gameweek"""
        try:
            # Fetch actual FPL data for this gameweek
            fpl_data = self.fetch_fpl_gameweek_data(gameweek)
            if not fpl_data:
                print(f"❌ No FPL data available for GW{gameweek}")
                return None
            
            # Create a mapping of player names to FPL data
            fpl_data_dict = {player['player_name']: player for player in fpl_data}
            
            # Use different formations and budgets for variety
            formations = ["4-4-2", "4-3-3", "3-5-2", "4-5-1"]
            budgets = [100.0, 95.0, 105.0, 90.0]
            
            formation = np.random.choice(formations)
            budget = np.random.choice(budgets)
            
            # Parse formation
            formation_parts = formation.split('-')
            defenders_needed = int(formation_parts[0])
            midfielders_needed = int(formation_parts[1])
            forwards_needed = int(formation_parts[2])
            goalkeepers_needed = 1
            
            # Filter by budget
            affordable_df = predictions_df[predictions_df['price'] <= budget].copy()
            
            if len(affordable_df) == 0:
                return None
            
            # Adjust predictions for historical gameweek
            for idx, player in affordable_df.iterrows():
                fixture_factor = np.random.uniform(0.7, 1.3)
                affordable_df.loc[idx, 'adjusted_points'] = player['predicted_points'] * fixture_factor
            
            affordable_df = affordable_df.sort_values('adjusted_points', ascending=False)
            
            # Select best players by position
            best_team = []
            total_cost = 0
            total_predicted_points = 0
            
            # Goalkeeper
            gk_df = affordable_df[affordable_df['position_name'] == 'GK'].head(goalkeepers_needed)
            for _, player in gk_df.iterrows():
                fpl_player = fpl_data_dict.get(player['name'], {})
                best_team.append({
                    "name": player['name'],
                    "position": player['position_name'],
                    "team": fpl_player.get('team', 'Unknown'),
                    "price": player['price'],
                    "predicted_points": player['predicted_points'],
                    "adjusted_points": player['adjusted_points'],
                    "actual_points": fpl_player.get('actual_points', 0),
                    "minutes_played": fpl_player.get('minutes_played', 0),
                    "goals": fpl_player.get('goals', 0),
                    "assists": fpl_player.get('assists', 0),
                    "clean_sheets": fpl_player.get('clean_sheets', 0),
                    "bonus_points": fpl_player.get('bonus_points', 0),
                    "yellow_cards": fpl_player.get('yellow_cards', 0),
                    "red_cards": fpl_player.get('red_cards', 0),
                    "saves": fpl_player.get('saves', 0),
                    "goals_conceded": fpl_player.get('goals_conceded', 0),
                    "own_goals": fpl_player.get('own_goals', 0),
                    "penalties_missed": fpl_player.get('penalties_missed', 0)
                })
                total_cost += player['price']
                total_predicted_points += player['adjusted_points']
            
            # Defenders
            def_df = affordable_df[affordable_df['position_name'] == 'DEF'].head(defenders_needed)
            for _, player in def_df.iterrows():
                fpl_player = fpl_data_dict.get(player['name'], {})
                best_team.append({
                    "name": player['name'],
                    "position": player['position_name'],
                    "team": fpl_player.get('team', 'Unknown'),
                    "price": player['price'],
                    "predicted_points": player['predicted_points'],
                    "adjusted_points": player['adjusted_points'],
                    "actual_points": fpl_player.get('actual_points', 0),
                    "minutes_played": fpl_player.get('minutes_played', 0),
                    "goals": fpl_player.get('goals', 0),
                    "assists": fpl_player.get('assists', 0),
                    "clean_sheets": fpl_player.get('clean_sheets', 0),
                    "bonus_points": fpl_player.get('bonus_points', 0),
                    "yellow_cards": fpl_player.get('yellow_cards', 0),
                    "red_cards": fpl_player.get('red_cards', 0),
                    "saves": fpl_player.get('saves', 0),
                    "goals_conceded": fpl_player.get('goals_conceded', 0),
                    "own_goals": fpl_player.get('own_goals', 0),
                    "penalties_missed": fpl_player.get('penalties_missed', 0)
                })
                total_cost += player['price']
                total_predicted_points += player['adjusted_points']
            
            # Midfielders
            mid_df = affordable_df[affordable_df['position_name'] == 'MID'].head(midfielders_needed)
            for _, player in mid_df.iterrows():
                fpl_player = fpl_data_dict.get(player['name'], {})
                best_team.append({
                    "name": player['name'],
                    "position": player['position_name'],
                    "team": fpl_player.get('team', 'Unknown'),
                    "price": player['price'],
                    "predicted_points": player['predicted_points'],
                    "adjusted_points": player['adjusted_points'],
                    "actual_points": fpl_player.get('actual_points', 0),
                    "minutes_played": fpl_player.get('minutes_played', 0),
                    "goals": fpl_player.get('goals', 0),
                    "assists": fpl_player.get('assists', 0),
                    "clean_sheets": fpl_player.get('clean_sheets', 0),
                    "bonus_points": fpl_player.get('bonus_points', 0),
                    "yellow_cards": fpl_player.get('yellow_cards', 0),
                    "red_cards": fpl_player.get('red_cards', 0),
                    "saves": fpl_player.get('saves', 0),
                    "goals_conceded": fpl_player.get('goals_conceded', 0),
                    "own_goals": fpl_player.get('own_goals', 0),
                    "penalties_missed": fpl_player.get('penalties_missed', 0)
                })
                total_cost += player['price']
                total_predicted_points += player['adjusted_points']
            
            # Forwards
            fwd_df = affordable_df[affordable_df['position_name'] == 'FWD'].head(forwards_needed)
            for _, player in fwd_df.iterrows():
                fpl_player = fpl_data_dict.get(player['name'], {})
                best_team.append({
                    "name": player['name'],
                    "position": player['position_name'],
                    "team": fpl_player.get('team', 'Unknown'),
                    "price": player['price'],
                    "predicted_points": player['predicted_points'],
                    "adjusted_points": player['adjusted_points'],
                    "actual_points": fpl_player.get('actual_points', 0),
                    "minutes_played": fpl_player.get('minutes_played', 0),
                    "goals": fpl_player.get('goals', 0),
                    "assists": fpl_player.get('assists', 0),
                    "clean_sheets": fpl_player.get('clean_sheets', 0),
                    "bonus_points": fpl_player.get('bonus_points', 0),
                    "yellow_cards": fpl_player.get('yellow_cards', 0),
                    "red_cards": fpl_player.get('red_cards', 0),
                    "saves": fpl_player.get('saves', 0),
                    "goals_conceded": fpl_player.get('goals_conceded', 0),
                    "own_goals": fpl_player.get('own_goals', 0),
                    "penalties_missed": fpl_player.get('penalties_missed', 0)
                })
                total_cost += player['price']
                total_predicted_points += player['adjusted_points']
            
            return {
                "formation": formation,
                "budget": budget,
                "total_cost": round(total_cost, 1),
                "total_predicted_points": round(total_predicted_points, 1),
                "players": best_team
            }
            
        except Exception as e:
            print(f"❌ Error generating real best 11: {e}")
            return None
    
    def store_real_best11_performance(self, gameweek, best11_data):
        """Store real best 11 team performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate actual points from FPL data
        total_actual_points = sum(player['actual_points'] for player in best11_data['players'])
        
        # Calculate performance accuracy
        performance_accuracy = (total_actual_points / best11_data['total_predicted_points']) * 100 if best11_data['total_predicted_points'] > 0 else 0
        
        # Insert best 11 record
        cursor.execute("""
            INSERT INTO real_historical_best11 
            (gameweek, formation, budget, total_cost, total_predicted_points, total_actual_points, performance_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            gameweek,
            best11_data['formation'],
            best11_data['budget'],
            best11_data['total_cost'],
            best11_data['total_predicted_points'],
            total_actual_points,
            performance_accuracy
        ))
        
        best11_id = cursor.lastrowid
        
        # Insert individual player performances
        for player in best11_data['players']:
            performance_diff = player['actual_points'] - player['predicted_points']
            cursor.execute("""
                INSERT INTO real_historical_best11_players 
                (best11_id, player_name, position, team, price, predicted_points, actual_points, performance_difference,
                 minutes_played, goals, assists, clean_sheets, bonus_points, yellow_cards, red_cards, saves, 
                 goals_conceded, own_goals, penalties_missed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                best11_id,
                player['name'],
                player['position'],
                player['team'],
                player['price'],
                player['predicted_points'],
                player['actual_points'],
                performance_diff,
                player['minutes_played'],
                player['goals'],
                player['assists'],
                player['clean_sheets'],
                player['bonus_points'],
                player['yellow_cards'],
                player['red_cards'],
                player['saves'],
                player['goals_conceded'],
                player['own_goals'],
                player['penalties_missed']
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Stored real best 11 for GW{gameweek}")
    
    def calculate_role_metrics(self, gameweek, best11_data):
        """Calculate role-specific performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Group players by position
        positions = {}
        for player in best11_data['players']:
            pos = player['position']
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(player)
        
        # Calculate metrics for each position
        for position, players in positions.items():
            if not players:
                continue
            
            avg_predicted = np.mean([p['predicted_points'] for p in players])
            avg_actual = np.mean([p['actual_points'] for p in players])
            avg_accuracy = (avg_actual / avg_predicted) * 100 if avg_predicted > 0 else 0
            
            best_performer = max(players, key=lambda x: x['actual_points'])
            worst_performer = min(players, key=lambda x: x['actual_points'])
            
            # Position-specific stats
            clean_sheets = sum(p['clean_sheets'] for p in players)
            goals_scored = sum(p['goals'] for p in players)
            assists = sum(p['assists'] for p in players)
            saves = sum(p['saves'] for p in players)
            goals_conceded = sum(p['goals_conceded'] for p in players)
            
            cursor.execute("""
                INSERT INTO role_performance_metrics 
                (gameweek, position, avg_predicted_points, avg_actual_points, avg_accuracy, total_players,
                 best_performer, best_performer_points, worst_performer, worst_performer_points,
                 clean_sheets, goals_scored, assists, saves, goals_conceded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gameweek,
                position,
                avg_predicted,
                avg_actual,
                avg_accuracy,
                len(players),
                best_performer['name'],
                best_performer['actual_points'],
                worst_performer['name'],
                worst_performer['actual_points'],
                clean_sheets,
                goals_scored,
                assists,
                saves,
                goals_conceded
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Calculated role metrics for GW{gameweek}")
    
    def get_real_historical_data(self, gameweeks=None):
        """Get real historical data"""
        conn = sqlite3.connect(self.db_path)
        
        if gameweeks:
            placeholders = ','.join(['?' for _ in gameweeks])
            best11_query = f"SELECT * FROM real_historical_best11 WHERE gameweek IN ({placeholders}) ORDER BY gameweek DESC"
            best11_df = pd.read_sql_query(best11_query, conn, params=gameweeks)
        else:
            best11_df = pd.read_sql_query("SELECT * FROM real_historical_best11 ORDER BY gameweek DESC", conn)
        
        role_metrics_df = pd.read_sql_query("SELECT * FROM role_performance_metrics ORDER BY gameweek DESC", conn)
        
        conn.close()
        
        return best11_df, role_metrics_df
    
    def get_best11_players_for_gameweek(self, gameweek):
        """Get individual player performance for a specific gameweek"""
        conn = sqlite3.connect(self.db_path)
        
        # Get best 11 ID for the gameweek
        best11_id_query = "SELECT id FROM real_historical_best11 WHERE gameweek = ?"
        cursor = conn.cursor()
        cursor.execute(best11_id_query, [gameweek])
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return pd.DataFrame()
        
        best11_id = result[0]
        
        # Get player data
        players_df = pd.read_sql_query("""
            SELECT * FROM real_historical_best11_players 
            WHERE best11_id = ?
            ORDER BY actual_points DESC
        """, conn, params=[best11_id])
        
        conn.close()
        return players_df

if __name__ == "__main__":
    # Initialize tracker
    tracker = RealHistoricalTracker()
    
    # Load current predictions
    conn = sqlite3.connect("epl_data.db")
    predictions_df = pd.read_sql_query("""
        SELECT * FROM predictions_advanced 
        WHERE confidence_interval IS NOT NULL
        ORDER BY predicted_points DESC
    """, conn)
    conn.close()
    
    if predictions_df.empty:
        print("❌ No predictions available. Please run the advanced pipeline first.")
    else:
        print(f"✅ Loaded {len(predictions_df)} predictions")
        
        # Generate real historical data for past gameweeks
        for gw in range(1, 6):  # GW1 to GW5
            print(f"📊 Generating real data for Gameweek {gw}...")
            
            best11_data = tracker.generate_real_best11_for_gameweek(gw, predictions_df)
            if best11_data:
                tracker.store_real_best11_performance(gw, best11_data)
                tracker.calculate_role_metrics(gw, best11_data)
        
        print("✅ Real historical data generation completed!")
        
        # Display results
        best11_df, role_metrics_df = tracker.get_real_historical_data()
        
        print("\n=== Real Historical Best 11 Performance ===")
        print(best11_df.head())
        
        print("\n=== Role Performance Metrics ===")
        print(role_metrics_df.head())
