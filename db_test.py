#!/usr/bin/env python3

import sqlite3
import pandas as pd

def test_database():
    try:
        print("Testing database connection...")
        
        # Test main database
        conn = sqlite3.connect("epl_data.db")
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print(f"Tables in main database: {tables['name'].tolist()}")
        
        player_count = pd.read_sql_query("SELECT COUNT(*) as count FROM players", conn)
        print(f"Player count: {player_count['count'].iloc[0]}")
        conn.close()
        
        # Test data directory database
        conn = sqlite3.connect("data/epl_data.db")
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print(f"Tables in data database: {tables['name'].tolist()}")
        
        player_count = pd.read_sql_query("SELECT COUNT(*) as count FROM players", conn)
        print(f"Player count: {player_count['count'].iloc[0]}")
        conn.close()
        
        print("✅ Database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_database()
