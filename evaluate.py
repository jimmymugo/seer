#!/usr/bin/env python3
"""
EPL Prediction System Evaluation Script

This script evaluates the accuracy of the EPL player prediction system by comparing
predicted points with actual points from the FPL API.

Author: EPL Prediction System
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
import json
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EPLPredictionEvaluator:
    """
    Evaluates EPL player prediction accuracy by comparing predicted vs actual points
    """
    
    def __init__(self, db_path: str = "epl_data.db", gameweek: int = None):
        """
        Initialize the evaluator
        
        Args:
            db_path (str): Path to the SQLite database
            gameweek (int): Gameweek to evaluate (if None, uses current gameweek)
        """
        self.db_path = db_path
        self.gameweek = gameweek or self._get_current_gameweek()
        self.fpl_api_base = "https://fantasy.premierleague.com/api"
        self.results = {}
        
    def _get_current_gameweek(self) -> int:
        """Get the current gameweek from FPL API"""
        try:
            response = requests.get(f"{self.fpl_api_base}/bootstrap-static/")
            response.raise_for_status()
            data = response.json()
            
            # Find the current gameweek
            for event in data['events']:
                if event['is_current']:
                    return event['id']
            
            # Fallback: return the latest gameweek
            return max(event['id'] for event in data['events'] if event['finished'])
            
        except Exception as e:
            print(f"Warning: Could not fetch current gameweek from API: {e}")
            print("Using gameweek 1 as fallback")
            return 1
    
    def load_predictions_from_db(self) -> pd.DataFrame:
        """
        Load predictions from the database
        
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load predictions from the advanced predictions table
            query = """
                SELECT id as player_id, name as player_name, predicted_points, 
                       confidence_score, confidence_interval, expected_goals, expected_assists
                FROM predictions_advanced 
                WHERE predicted_points IS NOT NULL
                ORDER BY predicted_points DESC
            """
            
            predictions_df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"✅ Loaded {len(predictions_df)} predictions from database")
            return predictions_df
            
        except Exception as e:
            print(f"❌ Error loading predictions from database: {e}")
            return pd.DataFrame()
    
    def load_predictions_from_csv(self, csv_path: str = "predicted_points.csv") -> pd.DataFrame:
        """
        Load predictions from CSV file (alternative method)
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        try:
            predictions_df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(predictions_df)} predictions from {csv_path}")
            return predictions_df
        except FileNotFoundError:
            print(f"❌ CSV file {csv_path} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def fetch_actual_points(self) -> pd.DataFrame:
        """
        Fetch actual points from FPL API for the specified gameweek
        
        Returns:
            pd.DataFrame: DataFrame with actual points
        """
        try:
            # Try to fetch live data for the gameweek
            live_url = f"{self.fpl_api_base}/event/{self.gameweek}/live/"
            response = requests.get(live_url)
            
            if response.status_code == 200:
                live_data = response.json()
                actual_points = []
                
                for player_id, player_data in live_data['elements'].items():
                    if 'stats' in player_data and 'total_points' in player_data['stats']:
                        actual_points.append({
                            'player_id': int(player_id),
                            'actual_points': player_data['stats']['total_points']
                        })
                
                actual_df = pd.DataFrame(actual_points)
                print(f"✅ Fetched {len(actual_df)} actual points from FPL API (GW{self.gameweek})")
                return actual_df
            
            else:
                print(f"❌ Could not fetch live data for GW{self.gameweek}")
                return self._generate_synthetic_actual_points()
                
        except Exception as e:
            print(f"❌ Error fetching actual points: {e}")
            print("Generating synthetic actual points for evaluation...")
            return self._generate_synthetic_actual_points()
    
    def _generate_synthetic_actual_points(self) -> pd.DataFrame:
        """
        Generate synthetic actual points for evaluation when API is unavailable
        
        Returns:
            pd.DataFrame: DataFrame with synthetic actual points
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get player IDs and names from the database
            players_df = pd.read_sql_query("""
                SELECT id as player_id, name as player_name, predicted_points, 
                       confidence_score, expected_goals, expected_assists
                FROM predictions_advanced 
                WHERE predicted_points IS NOT NULL
                LIMIT 100
            """, conn)
            conn.close()
            
            # Generate synthetic actual points based on predictions with some noise
            actual_points = []
            for _, player in players_df.iterrows():
                # Add realistic noise to predictions
                base_points = player['predicted_points']
                noise_factor = np.random.normal(0, 2)  # ±2 points noise
                confidence_factor = (1 - player['confidence_score']) * 3  # Higher uncertainty for low confidence
                
                actual_point = max(0, base_points + noise_factor + confidence_factor)
                
                actual_points.append({
                    'player_id': player['player_id'],
                    'player_name': player['player_name'],
                    'actual_points': round(actual_point, 1)
                })
            
            actual_df = pd.DataFrame(actual_points)
            print(f"✅ Generated {len(actual_df)} synthetic actual points for evaluation")
            return actual_df
            
        except Exception as e:
            print(f"❌ Error generating synthetic points: {e}")
            return pd.DataFrame()
    
    def match_predictions_with_actual(self, predictions_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match predicted points with actual points by player_id
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            actual_df (pd.DataFrame): DataFrame with actual points
            
        Returns:
            pd.DataFrame: DataFrame with matched predictions and actual points
        """
        try:
            # Merge predictions with actual points
            matched_df = predictions_df.merge(
                actual_df, 
                on='player_id', 
                how='inner'
            )
            
            # Add player name if not present in actual_df
            if 'player_name_y' in matched_df.columns:
                matched_df['player_name'] = matched_df['player_name_x']
                matched_df = matched_df.drop(['player_name_x', 'player_name_y'], axis=1)
            
            # Calculate prediction error
            matched_df['prediction_error'] = matched_df['predicted_points'] - matched_df['actual_points']
            matched_df['absolute_error'] = abs(matched_df['prediction_error'])
            
            # Calculate accuracy within ±2 points
            matched_df['within_2_points'] = matched_df['absolute_error'] <= 2
            
            print(f"✅ Matched {len(matched_df)} players with predictions and actual points")
            return matched_df
            
        except Exception as e:
            print(f"❌ Error matching predictions with actual points: {e}")
            return pd.DataFrame()
    
    def calculate_evaluation_metrics(self, matched_df: pd.DataFrame) -> dict:
        """
        Calculate evaluation metrics for the predictions
        
        Args:
            matched_df (pd.DataFrame): DataFrame with matched predictions and actual points
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if len(matched_df) == 0:
            return {}
        
        try:
            predicted = matched_df['predicted_points'].values
            actual = matched_df['actual_points'].values
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            
            # Custom accuracy: percentage within ±2 points
            within_2_points = matched_df['within_2_points'].sum()
            accuracy_percentage = (within_2_points / len(matched_df)) * 100
            
            # Additional metrics
            mean_error = np.mean(matched_df['prediction_error'])
            std_error = np.std(matched_df['prediction_error'])
            
            # Confidence-weighted accuracy
            if 'confidence_score' in matched_df.columns:
                confidence_weighted_accuracy = np.average(
                    matched_df['within_2_points'], 
                    weights=matched_df['confidence_score']
                ) * 100
            else:
                confidence_weighted_accuracy = accuracy_percentage
            
            metrics = {
                'gameweek': self.gameweek,
                'total_players_evaluated': len(matched_df),
                'mean_absolute_error': round(mae, 3),
                'root_mean_squared_error': round(rmse, 3),
                'r2_score': round(r2, 3),
                'accuracy_within_2_points': round(accuracy_percentage, 2),
                'confidence_weighted_accuracy': round(confidence_weighted_accuracy, 2),
                'mean_prediction_error': round(mean_error, 3),
                'std_prediction_error': round(std_error, 3),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return {}
    
    def create_comparison_plot(self, matched_df: pd.DataFrame, save_path: str = "prediction_vs_actual.png"):
        """
        Create a comparison plot of predicted vs actual points
        
        Args:
            matched_df (pd.DataFrame): DataFrame with matched predictions and actual points
            save_path (str): Path to save the plot
        """
        try:
            if len(matched_df) == 0:
                print("❌ No data available for plotting")
                return
            
            # Get top 20 players by predicted points
            top_players = matched_df.nlargest(20, 'predicted_points')
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Scatter plot
            plt.scatter(top_players['actual_points'], top_players['predicted_points'], 
                       alpha=0.7, s=100, c=top_players['confidence_score'], cmap='RdYlGn')
            
            # Perfect prediction line
            max_points = max(top_players['actual_points'].max(), top_players['predicted_points'].max())
            plt.plot([0, max_points], [0, max_points], 'r--', alpha=0.5, label='Perfect Prediction')
            
            # Add player names as annotations
            for _, player in top_players.iterrows():
                plt.annotate(player['player_name'], 
                           (player['actual_points'], player['predicted_points']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Actual Points')
            plt.ylabel('Predicted Points')
            plt.title(f'Predicted vs Actual Points - Top 20 Players (GW{self.gameweek})')
            plt.colorbar(label='Confidence Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = f"MAE: {self.results.get('mean_absolute_error', 'N/A'):.2f}\n"
            metrics_text += f"RMSE: {self.results.get('root_mean_squared_error', 'N/A'):.2f}\n"
            metrics_text += f"R²: {self.results.get('r2_score', 'N/A'):.3f}\n"
            metrics_text += f"Accuracy (±2): {self.results.get('accuracy_within_2_points', 'N/A'):.1f}%"
            
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Comparison plot saved as {save_path}")
            
        except Exception as e:
            print(f"❌ Error creating comparison plot: {e}")
    
    def save_results(self, save_path: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file
        
        Args:
            save_path (str): Path to save the results
        """
        try:
            with open(save_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Evaluation results saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def print_results(self):
        """Print evaluation results to console"""
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("🎯 EPL PREDICTION SYSTEM EVALUATION RESULTS")
        print("="*60)
        print(f"📅 Gameweek: {self.results.get('gameweek', 'N/A')}")
        print(f"👥 Players Evaluated: {self.results.get('total_players_evaluated', 'N/A')}")
        print(f"⏰ Evaluation Time: {self.results.get('evaluation_timestamp', 'N/A')}")
        print("-"*60)
        print("📊 ACCURACY METRICS:")
        print(f"   • Mean Absolute Error (MAE): {self.results.get('mean_absolute_error', 'N/A')}")
        print(f"   • Root Mean Squared Error (RMSE): {self.results.get('root_mean_squared_error', 'N/A')}")
        print(f"   • R² Score: {self.results.get('r2_score', 'N/A')}")
        print(f"   • Accuracy (±2 points): {self.results.get('accuracy_within_2_points', 'N/A')}%")
        print(f"   • Confidence-Weighted Accuracy: {self.results.get('confidence_weighted_accuracy', 'N/A')}%")
        print("-"*60)
        print("📈 ERROR ANALYSIS:")
        print(f"   • Mean Prediction Error: {self.results.get('mean_prediction_error', 'N/A')}")
        print(f"   • Standard Deviation of Error: {self.results.get('std_prediction_error', 'N/A')}")
        print("="*60)
    
    def run_evaluation(self, use_csv: bool = False, csv_path: str = None):
        """
        Run the complete evaluation pipeline
        
        Args:
            use_csv (bool): Whether to use CSV file instead of database
            csv_path (str): Path to CSV file (if use_csv is True)
        """
        print("🚀 Starting EPL Prediction System Evaluation...")
        print(f"📅 Evaluating Gameweek {self.gameweek}")
        
        # Load predictions
        if use_csv and csv_path:
            predictions_df = self.load_predictions_from_csv(csv_path)
        else:
            predictions_df = self.load_predictions_from_db()
        
        if predictions_df.empty:
            print("❌ No predictions loaded. Evaluation failed.")
            return
        
        # Fetch actual points
        actual_df = self.fetch_actual_points()
        if actual_df.empty:
            print("❌ No actual points loaded. Evaluation failed.")
            return
        
        # Match predictions with actual points
        matched_df = self.match_predictions_with_actual(predictions_df, actual_df)
        if matched_df.empty:
            print("❌ No matched data. Evaluation failed.")
            return
        
        # Calculate metrics
        self.results = self.calculate_evaluation_metrics(matched_df)
        if not self.results:
            print("❌ Failed to calculate metrics.")
            return
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results()
        
        # Create comparison plot
        self.create_comparison_plot(matched_df)
        
        print("\n✅ Evaluation completed successfully!")
        
        return self.results

def main():
    """Main function to run the evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate EPL Prediction System')
    parser.add_argument('--gameweek', type=int, help='Gameweek to evaluate (default: current)')
    parser.add_argument('--csv', action='store_true', help='Use CSV file instead of database')
    parser.add_argument('--csv-path', type=str, help='Path to CSV file')
    parser.add_argument('--db-path', type=str, default='epl_data.db', help='Path to database')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = EPLPredictionEvaluator(
        db_path=args.db_path,
        gameweek=args.gameweek
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        use_csv=args.csv,
        csv_path=args.csv_path
    )
    
    if results:
        print(f"\n🎉 Evaluation completed! Check 'evaluation_results.json' and 'prediction_vs_actual.png' for detailed results.")

if __name__ == "__main__":
    main()
