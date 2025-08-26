"""
Main Pipeline - Orchestrates the entire EPL prediction system
"""

import logging
import time
from datetime import datetime
from data_collector import EPLDataCollector
from feature_engineering import FeatureEngineer
from model import EPLPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the complete EPL prediction pipeline"""
    logger.info("🚀 Starting EPL Player Prediction Pipeline")
    
    try:
        # Step 1: Data Collection
        logger.info("📊 Step 1: Collecting EPL data...")
        collector = EPLDataCollector()
        collector.collect_all_data()
        logger.info("✅ Data collection completed")
        
        # Step 2: Feature Engineering
        logger.info("🔧 Step 2: Engineering features...")
        engineer = FeatureEngineer()
        processed_data = engineer.engineer_all_features()
        logger.info(f"✅ Feature engineering completed. Processed {len(processed_data)} players")
        
        # Step 3: Model Training and Prediction
        logger.info("🤖 Step 3: Training model and generating predictions...")
        predictor = EPLPredictor()
        top_predictions = predictor.train_and_predict()
        logger.info(f"✅ Model training and prediction completed")
        
        # Display results
        logger.info("🏆 Top 10 Predicted Players:")
        for i, (_, player) in enumerate(top_predictions.iterrows(), 1):
            logger.info(f"{i:2d}. {player['name']} ({player['position_name']}) - "
                       f"Predicted: {player['predicted_points']:.1f} pts, "
                       f"Confidence: {player['confidence_score']:.2f}")
        
        logger.info("🎉 Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False

def run_scheduled_pipeline(interval_hours=24):
    """Run pipeline on a schedule"""
    logger.info(f"🕐 Starting scheduled pipeline (every {interval_hours} hours)")
    
    while True:
        try:
            success = run_pipeline()
            if success:
                logger.info(f"⏰ Next run in {interval_hours} hours")
            else:
                logger.warning("⚠️ Pipeline failed, retrying in 1 hour")
                time.sleep(3600)  # Wait 1 hour on failure
                continue
                
            time.sleep(interval_hours * 3600)  # Convert hours to seconds
            
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline stopped by user")
            break
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
            time.sleep(3600)  # Wait 1 hour on error

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EPL Player Prediction Pipeline")
    parser.add_argument("--scheduled", action="store_true", 
                       help="Run pipeline on a schedule (every 24 hours)")
    parser.add_argument("--interval", type=int, default=24,
                       help="Interval in hours for scheduled runs (default: 24)")
    
    args = parser.parse_args()
    
    if args.scheduled:
        run_scheduled_pipeline(args.interval)
    else:
        run_pipeline()
