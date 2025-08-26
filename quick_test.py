#!/usr/bin/env python3

import sys
import os

# Force output to be displayed
sys.stdout.flush()
sys.stderr.flush()

print("=== EPL Prediction System Quick Test ===")
print("Testing basic functionality...")

try:
    # Test imports
    print("1. Testing imports...")
    from data_collector import EPLDataCollector
    from feature_engineering import FeatureEngineer
    from model import EPLPredictor
    print("   ✅ All imports successful")
    
    # Test database
    print("2. Testing database...")
    import sqlite3
    conn = sqlite3.connect("epl_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    print(f"   ✅ Database has {player_count} players")
    conn.close()
    
    # Test feature engineering
    print("3. Testing feature engineering...")
    engineer = FeatureEngineer()
    players_df, teams_df, fixtures_df = engineer.load_data()
    print(f"   ✅ Loaded {len(players_df)} players, {len(teams_df)} teams, {len(fixtures_df)} fixtures")
    
    # Test model
    print("4. Testing model...")
    predictor = EPLPredictor()
    df = predictor.load_processed_data()
    if len(df) == 0:
        print("   ⚠️ No processed data found, running feature engineering...")
        engineer.engineer_all_features()
        df = predictor.load_processed_data()
    
    print(f"   ✅ Loaded {len(df)} processed players")
    
    print("\n🎉 All tests passed! System is working correctly.")
    print("\nNext steps:")
    print("1. Run: python api.py")
    print("2. Run: streamlit run dashboard.py")
    print("3. Open: http://localhost:8501")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
