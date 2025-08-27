#!/usr/bin/env python3
"""
Advanced Pipeline Runner - Complete advanced EPL prediction pipeline
"""

import logging
import sys
from datetime import datetime

# Import advanced modules
from advanced_data_collector import AdvancedEPLDataCollector
from advanced_feature_engineering import AdvancedFeatureEngineer
from advanced_model import AdvancedEPLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_advanced_pipeline():
    """Run the complete advanced pipeline"""
    logger.info("🚀 Starting Advanced EPL Prediction Pipeline")
    
    try:
        # Step 1: Advanced Data Collection
        logger.info("📊 Step 1: Collecting advanced data...")
        collector = AdvancedEPLDataCollector()
        collector.collect_all_advanced_data()
        logger.info("✅ Advanced data collection completed")
        
        # Step 2: Advanced Feature Engineering
        logger.info("🔧 Step 2: Engineering advanced features...")
        engineer = AdvancedFeatureEngineer()
        processed_data = engineer.engineer_all_advanced_features()
        logger.info(f"✅ Advanced feature engineering completed. Processed {len(processed_data)} players")
        
        # Step 3: Advanced Model Training and Prediction
        logger.info("🤖 Step 3: Training ensemble models and generating predictions...")
        predictor = AdvancedEPLPredictor()
        predictions = predictor.train_and_predict_advanced()
        logger.info(f"✅ Advanced predictions generated for {len(predictions)} players")
        
        # Step 4: Save models
        logger.info("💾 Step 4: Saving trained models...")
        predictor.save_models()
        logger.info("✅ Models saved successfully")
        
        # Step 5: Display top predictions
        logger.info("🏆 Step 5: Top 10 Advanced Predictions:")
        top_predictions = predictor.get_top_predictions_advanced(top_n=10)
        
        if not top_predictions.empty:
            for i, (_, player) in enumerate(top_predictions.iterrows(), 1):
                logger.info(f"{i:2d}. {player['name']:20s} ({player['position_name']:3s}) - "
                           f"{player['predicted_points']:5.1f} pts (Conf: {player['confidence_score']:.2f})")
        
        logger.info("🎉 Advanced pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("⚽ EPL Advanced Prediction Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = run_advanced_pipeline()
    
    print()
    print("=" * 60)
    if success:
        print("✅ Pipeline completed successfully!")
        print("🌐 Access the advanced dashboard at: http://localhost:8501")
        print("📊 Access the API at: http://localhost:8000")
    else:
        print("❌ Pipeline failed!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
