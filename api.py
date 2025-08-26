"""
FastAPI Backend - REST API for EPL player predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sqlite3
import logging
from datetime import datetime
import uvicorn

from data_collector import EPLDataCollector
from feature_engineering import FeatureEngineer
from model import EPLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EPL Player Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlayerPrediction(BaseModel):
    name: str
    position: str
    team_id: int
    price: float
    recent_form: float
    fixture_difficulty: float
    predicted_points: float
    confidence_score: float

class PredictionResponse(BaseModel):
    predictions: List[PlayerPrediction]
    last_updated: str
    total_players: int

data_collector = EPLDataCollector("epl_data.db")
feature_engineer = FeatureEngineer("epl_data.db")
predictor = EPLPredictor("epl_data.db")

@app.get("/")
async def root():
    return {"message": "EPL Player Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/predictions", response_model=PredictionResponse)
async def get_predictions(
    top_n: int = 10,
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    max_price: Optional[float] = None
):
    try:
        conn = sqlite3.connect("epl_data.db")
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if position:
            query += " AND position_name = ?"
            params.append(position)
        if team_id:
            query += " AND team_id = ?"
            params.append(team_id)
        if max_price:
            query += " AND price <= ?"
            params.append(max_price)
        
        query += " ORDER BY predicted_points DESC LIMIT ?"
        params.append(top_n)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        predictions = []
        for _, row in df.iterrows():
            predictions.append(PlayerPrediction(
                name=row['name'],
                position=row['position_name'],
                team_id=row['team_id'],
                price=row['price'],
                recent_form=row['recent_form'],
                fixture_difficulty=row['next_fixture_difficulty'],
                predicted_points=row['predicted_points'],
                confidence_score=row['confidence_score']
            ))
        
        return PredictionResponse(
            predictions=predictions,
            last_updated=datetime.now().isoformat(),
            total_players=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/refresh")
async def refresh_predictions():
    try:
        logger.info("Refreshing predictions...")
        
        # Simple refresh without complex feature engineering
        import sqlite3
        import pandas as pd
        
        # Connect to database
        conn = sqlite3.connect("epl_data.db")
        
        # Get players data
        players_df = pd.read_sql_query("SELECT * FROM players", conn)
        
        # Simple feature engineering
        processed_df = players_df.copy()
        
        # Add position names
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        processed_df['position_name'] = processed_df['position'].map(position_map)
        
        # Add basic features
        processed_df['recent_form'] = processed_df['form'].astype(float)
        processed_df['points_per_game'] = processed_df['total_points'] / 38
        processed_df['value_for_money'] = processed_df['total_points'] / processed_df['price']
        processed_df['next_fixture_difficulty'] = 3.0  # Default difficulty
        
        # Save processed data
        processed_df.to_sql('processed_players', conn, if_exists='replace', index=False)
        
        # Create simple predictions
        predictions_df = processed_df.copy()
        predictions_df['predicted_points'] = predictions_df['recent_form'] * 2 + predictions_df['points_per_game'] * 10
        predictions_df['confidence_score'] = 0.7  # Default confidence
        
        # Select only needed columns
        predictions_df = predictions_df[['name', 'position_name', 'team_id', 'price', 'recent_form', 'next_fixture_difficulty', 'predicted_points', 'confidence_score']]
        
        # Save predictions
        predictions_df.to_sql('predictions', conn, if_exists='replace', index=False)
        
        conn.close()
        
        return {
            "status": "success",
            "message": "Predictions refreshed successfully",
            "players_processed": len(processed_df),
            "predictions_generated": len(predictions_df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players")
async def get_players(position: Optional[str] = None, limit: int = 50):
    try:
        conn = sqlite3.connect("epl_data.db")
        query = "SELECT id, name, position_name, team_id, price, form, total_points FROM processed_players"
        params = []
        
        if position:
            query += " WHERE position_name = ?"
            params.append(position)
        
        query += " ORDER BY total_points DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return {"players": df.to_dict('records'), "count": len(df)}
        
    except Exception as e:
        logger.error(f"Error getting players: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams")
async def get_teams():
    try:
        conn = sqlite3.connect("epl_data.db")
        df = pd.read_sql_query("SELECT * FROM teams", conn)
        conn.close()
        return {"teams": df.to_dict('records'), "count": len(df)}
        
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/best11")
async def get_best11(
    budget: float = 100.0,
    formation: str = "4-4-2"
):
    """Get the best 11 players for the next gameweek"""
    try:
        conn = sqlite3.connect("epl_data.db")
        
        # Get predictions
        predictions_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_points DESC", conn)
        
        if len(predictions_df) == 0:
            raise HTTPException(status_code=404, detail="No predictions available. Please refresh data first.")
        
        # Parse formation (e.g., "4-4-2" -> [4, 4, 2])
        formation_parts = formation.split('-')
        if len(formation_parts) != 3:
            formation = "4-4-2"
            formation_parts = [4, 4, 2]
        
        defenders_needed = int(formation_parts[0])
        midfielders_needed = int(formation_parts[1])
        forwards_needed = int(formation_parts[2])
        goalkeepers_needed = 1
        
        # Filter by budget
        affordable_df = predictions_df[predictions_df['price'] <= budget]
        
        # Select best players by position
        best_team = []
        total_cost = 0
        total_predicted_points = 0
        
        # Goalkeeper
        gk_df = affordable_df[affordable_df['position_name'] == 'GK'].head(goalkeepers_needed)
        for _, player in gk_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "team_id": player['team_id'],
                "price": player['price'],
                "predicted_points": player['predicted_points'],
                "confidence_score": player['confidence_score']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        # Defenders
        def_df = affordable_df[affordable_df['position_name'] == 'DEF'].head(defenders_needed)
        for _, player in def_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "team_id": player['team_id'],
                "price": player['price'],
                "predicted_points": player['predicted_points'],
                "confidence_score": player['confidence_score']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        # Midfielders
        mid_df = affordable_df[affordable_df['position_name'] == 'MID'].head(midfielders_needed)
        for _, player in mid_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "team_id": player['team_id'],
                "price": player['price'],
                "predicted_points": player['predicted_points'],
                "confidence_score": player['confidence_score']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        # Forwards
        fwd_df = affordable_df[affordable_df['position_name'] == 'FWD'].head(forwards_needed)
        for _, player in fwd_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "team_id": player['team_id'],
                "price": player['price'],
                "predicted_points": player['predicted_points'],
                "confidence_score": player['confidence_score']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        conn.close()
        
        return {
            "formation": formation,
            "budget": budget,
            "total_cost": round(total_cost, 1),
            "budget_remaining": round(budget - total_cost, 1),
            "total_predicted_points": round(total_predicted_points, 1),
            "players": best_team,
            "count": len(best_team),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting best 11: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize-team")
async def optimize_team(
    budget: float = 100.0,
    formation: str = "4-4-2",
    max_players_per_team: int = 3
):
    """Get optimized team considering FPL rules"""
    try:
        conn = sqlite3.connect("epl_data.db")
        
        # Get predictions
        predictions_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_points DESC", conn)
        
        if len(predictions_df) == 0:
            raise HTTPException(status_code=404, detail="No predictions available. Please refresh data first.")
        
        # Parse formation
        formation_parts = formation.split('-')
        if len(formation_parts) != 3:
            formation = "4-4-2"
            formation_parts = [4, 4, 2]
        
        defenders_needed = int(formation_parts[0])
        midfielders_needed = int(formation_parts[1])
        forwards_needed = int(formation_parts[2])
        goalkeepers_needed = 1
        
        # Filter by budget
        affordable_df = predictions_df[predictions_df['price'] <= budget].copy()
        
        # Simple optimization algorithm
        selected_players = []
        total_cost = 0
        team_counts = {}
        
        # Function to select best player for position
        def select_best_player(position, count_needed):
            nonlocal total_cost, team_counts
            players = []
            
            for _, player in affordable_df[affordable_df['position_name'] == position].iterrows():
                # Check if we already have max players from this team
                team_id = player['team_id']
                if team_counts.get(team_id, 0) >= max_players_per_team:
                    continue
                
                # Check if we can afford this player
                if total_cost + player['price'] > budget:
                    continue
                
                players.append(player)
            
            # Sort by predicted points and take the best
            players.sort(key=lambda x: x['predicted_points'], reverse=True)
            return players[:count_needed]
        
        # Select players for each position
        for position, count in [('GK', goalkeepers_needed), ('DEF', defenders_needed), 
                               ('MID', midfielders_needed), ('FWD', forwards_needed)]:
            best_players = select_best_player(position, count)
            
            for player in best_players:
                selected_players.append({
                    "name": player['name'],
                    "position": player['position_name'],
                    "team_id": player['team_id'],
                    "price": player['price'],
                    "predicted_points": player['predicted_points'],
                    "confidence_score": player['confidence_score']
                })
                total_cost += player['price']
                team_counts[player['team_id']] = team_counts.get(player['team_id'], 0) + 1
        
        total_predicted_points = sum(player['predicted_points'] for player in selected_players)
        
        conn.close()
        
        return {
            "formation": formation,
            "budget": budget,
            "total_cost": round(total_cost, 1),
            "budget_remaining": round(budget - total_cost, 1),
            "total_predicted_points": round(total_predicted_points, 1),
            "max_players_per_team": max_players_per_team,
            "players": selected_players,
            "count": len(selected_players),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error optimizing team: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
