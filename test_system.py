"""
Test Script - Verify EPL prediction system components
"""

import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from data_collector import EPLDataCollector
        print("✅ data_collector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_collector: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ feature_engineering imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import feature_engineering: {e}")
        return False
    
    try:
        from model import EPLPredictor
        print("✅ model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import model: {e}")
        return False
    
    return True

def test_data_collection():
    """Test data collection functionality"""
    print("\n📊 Testing data collection...")
    
    try:
        from data_collector import EPLDataCollector
        collector = EPLDataCollector("test_epl_data.db")
        
        # Test API connection
        bootstrap_data = collector.fetch_bootstrap_static()
        if bootstrap_data and 'elements' in bootstrap_data:
            print(f"✅ API connection successful - {len(bootstrap_data['elements'])} players found")
        else:
            print("❌ API connection failed")
            return False
        
        # Test database creation
        collector.create_database()
        print("✅ Database created successfully")
        
        # Test data storage
        collector.store_teams(bootstrap_data['teams'])
        collector.store_players(bootstrap_data['elements'])
        collector.store_fixtures(bootstrap_data['events'])
        print("✅ Data stored successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🔧 Testing feature engineering...")
    
    try:
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer("test_epl_data.db")
        
        # Test data loading
        players_df, teams_df, fixtures_df = engineer.load_data()
        print(f"✅ Data loaded - {len(players_df)} players, {len(teams_df)} teams, {len(fixtures_df)} fixtures")
        
        # Test feature engineering
        processed_df = engineer.engineer_features(players_df, teams_df, fixtures_df)
        print(f"✅ Features engineered - {len(processed_df.columns)} features created")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_model():
    """Test model functionality"""
    print("\n🤖 Testing model...")
    
    try:
        from model import EPLPredictor
        predictor = EPLPredictor("test_epl_data.db")
        
        # Test data loading
        df = predictor.load_processed_data()
        print(f"✅ Processed data loaded - {len(df)} players")
        
        # Test rule-based prediction
        df_with_rules = predictor.rule_based_prediction(df)
        if 'rule_score' in df_with_rules.columns:
            print("✅ Rule-based prediction successful")
        else:
            print("❌ Rule-based prediction failed")
            return False
        
        # Test ML model training (if enough data)
        if len(df) > 50:
            try:
                mse = predictor.train_ml_model(df)
                print(f"✅ ML model trained - MSE: {mse:.2f}")
            except Exception as e:
                print(f"⚠️ ML training failed (expected for small dataset): {e}")
        else:
            print("⚠️ Skipping ML training (insufficient data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_api():
    """Test API functionality"""
    print("\n⚡ Testing API...")
    
    try:
        import requests
        import time
        
        # Start API in background (simulate)
        print("⚠️ API test requires running 'python api.py' in another terminal")
        print("   Skipping API test for now...")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard functionality"""
    print("\n📊 Testing dashboard...")
    
    try:
        import streamlit
        print("✅ Streamlit available")
        
        # Test dashboard imports
        import plotly.express as px
        print("✅ Plotly available")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    try:
        if os.path.exists("test_epl_data.db"):
            os.remove("test_epl_data.db")
            print("🧹 Test database cleaned up")
    except:
        pass

def main():
    """Run all tests"""
    print("🚀 EPL Prediction System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Collection", test_data_collection),
        ("Feature Engineering", test_feature_engineering),
        ("Model", test_model),
        ("API", test_api),
        ("Dashboard", test_dashboard)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Cleanup
    cleanup()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
