#!/usr/bin/env python3

import subprocess
import time
import sys
import os

def run_command(cmd, description):
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    print("🚀 Starting EPL Prediction System...")
    
    # Step 1: Run feature engineering
    success = run_command(
        "python -c \"from feature_engineering import FeatureEngineer; engineer = FeatureEngineer(); engineer.engineer_all_features()\"",
        "Feature Engineering"
    )
    
    if not success:
        print("❌ Feature engineering failed. Stopping.")
        return False
    
    # Step 2: Run model training
    success = run_command(
        "python -c \"from model import EPLPredictor; predictor = EPLPredictor(); predictor.train_and_predict()\"",
        "Model Training"
    )
    
    if not success:
        print("❌ Model training failed. Stopping.")
        return False
    
    # Step 3: Start API
    print("\n🌐 Starting FastAPI...")
    api_process = subprocess.Popen(
        ["python", "api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Step 4: Start Dashboard
    print("\n📊 Starting Streamlit Dashboard...")
    dashboard_process = subprocess.Popen(
        ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("\n🎉 EPL Prediction System is running!")
    print("=" * 50)
    print("📊 Dashboard: http://localhost:8501")
    print("⚡ API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        api_process.terminate()
        dashboard_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main()
