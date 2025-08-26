#!/usr/bin/env python3

import sys
import traceback

def main():
    print("🚀 Running EPL Prediction Pipeline...")
    
    try:
        # Step 1: Feature Engineering
        print("Step 1: Feature Engineering...")
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        processed_data = engineer.engineer_all_features()
        print(f"✅ Feature engineering completed. Processed {len(processed_data)} players.")
        
        # Step 2: Model Training and Prediction
        print("Step 2: Model Training and Prediction...")
        from model import EPLPredictor
        predictor = EPLPredictor()
        top_predictions = predictor.train_and_predict()
        print(f"✅ Model training and prediction completed.")
        print(f"✅ Generated predictions for {len(top_predictions)} top players.")
        
        print("\n🎉 Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
