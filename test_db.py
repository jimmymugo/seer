#!/usr/bin/env python3

import sqlite3
import pandas as pd

def test_database():
    print("Testing database...")
    
    # Connect to database
    conn = sqlite3.connect("epl_data.db")
    
    # Check existing tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"Existing tables: {tables['name'].tolist()}")
    
    # Check players data
    players_df = pd.read_sql_query("SELECT * FROM players LIMIT 5", conn)
    print(f"Players columns: {players_df.columns.tolist()}")
    print(f"Number of players: {len(pd.read_sql_query('SELECT * FROM players', conn))}")
    
    # Create processed_players table manually
    print("Creating processed_players table...")
    
    # Load all data
    players_df = pd.read_sql_query("SELECT * FROM players", conn)
    teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
    fixtures_df = pd.read_sql_query("SELECT * FROM fixtures", conn)
    
    print(f"Loaded {len(players_df)} players, {len(teams_df)} teams, {len(fixtures_df)} fixtures")
    
    # Simple feature engineering
    processed_df = players_df.copy()
    
    # Add position names
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    processed_df['position_name'] = processed_df['position'].map(position_map)
    
    # Add basic features
    processed_df['recent_form'] = processed_df['form'].astype(float)
    processed_df['points_per_game'] = processed_df['total_points'] / 38
    processed_df['value_for_money'] = processed_df['total_points'] / processed_df['price']
    
    # Save to database
    processed_df.to_sql('processed_players', conn, if_exists='replace', index=False)
    print(f"✅ Created processed_players table with {len(processed_df)} players")
    
    # Create simple predictions
    predictions_df = processed_df.copy()
    predictions_df['predicted_points'] = predictions_df['recent_form'] * 2 + predictions_df['points_per_game'] * 10
    predictions_df['confidence_score'] = 0.7  # Default confidence
    
    # Select only needed columns
    predictions_df = predictions_df[['name', 'position_name', 'team_id', 'price', 'recent_form', 'predicted_points', 'confidence_score']]
    
    predictions_df.to_sql('predictions', conn, if_exists='replace', index=False)
    print(f"✅ Created predictions table with {len(predictions_df)} predictions")
    
    conn.close()
    
    print("✅ Database setup completed!")

if __name__ == "__main__":
    test_database()
