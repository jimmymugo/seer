#!/usr/bin/env python3

import sys
import traceback

def test_feature_engineering():
    try:
        print("Testing feature engineering...")
        
        from feature_engineering import FeatureEngineer
        print("✅ FeatureEngineer imported")
        
        engineer = FeatureEngineer()
        print("✅ FeatureEngineer instantiated")
        
        # Load data
        players_df, teams_df, fixtures_df = engineer.load_data()
        print(f"✅ Data loaded - {len(players_df)} players, {len(teams_df)} teams, {len(fixtures_df)} fixtures")
        
        # Engineer features
        processed_df = engineer.engineer_features(players_df, teams_df, fixtures_df)
        print(f"✅ Features engineered - {len(processed_df.columns)} features created")
        
        # Save to database
        import sqlite3
        conn = sqlite3.connect("data/epl_data.db")
        processed_df.to_sql('processed_players', conn, if_exists='replace', index=False)
        conn.close()
        print("✅ Processed data saved to database")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_engineering()
    if success:
        print("🎉 Feature engineering completed successfully!")
    else:
        print("💥 Feature engineering failed!")
        sys.exit(1)
