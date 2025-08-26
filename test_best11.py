#!/usr/bin/env python3

import sqlite3
import pandas as pd

def test_best11():
    print("Testing Best 11 feature...")
    
    try:
        # Connect to database
        conn = sqlite3.connect("epl_data.db")
        
        # Check if predictions exist
        predictions_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_points DESC", conn)
        print(f"✅ Found {len(predictions_df)} predictions")
        
        if len(predictions_df) == 0:
            print("❌ No predictions available")
            return False
        
        # Test best 11 generation
        budget = 100.0
        formation = "4-4-2"
        
        # Parse formation
        formation_parts = formation.split('-')
        defenders_needed = int(formation_parts[0])
        midfielders_needed = int(formation_parts[1])
        forwards_needed = int(formation_parts[2])
        goalkeepers_needed = 1
        
        # Filter by budget
        affordable_df = predictions_df[predictions_df['price'] <= budget]
        print(f"✅ Found {len(affordable_df)} affordable players")
        
        # Select best players by position
        best_team = []
        total_cost = 0
        total_predicted_points = 0
        
        # Goalkeeper
        gk_df = affordable_df[affordable_df['position_name'] == 'GK'].head(goalkeepers_needed)
        print(f"✅ Selected {len(gk_df)} goalkeeper(s)")
        for _, player in gk_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "price": player['price'],
                "predicted_points": player['predicted_points']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        # Defenders
        def_df = affordable_df[affordable_df['position_name'] == 'DEF'].head(defenders_needed)
        print(f"✅ Selected {len(def_df)} defender(s)")
        for _, player in def_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "price": player['price'],
                "predicted_points": player['predicted_points']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        # Midfielders
        mid_df = affordable_df[affordable_df['position_name'] == 'MID'].head(midfielders_needed)
        print(f"✅ Selected {len(mid_df)} midfielder(s)")
        for _, player in mid_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "price": player['price'],
                "predicted_points": player['predicted_points']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        # Forwards
        fwd_df = affordable_df[affordable_df['position_name'] == 'FWD'].head(forwards_needed)
        print(f"✅ Selected {len(fwd_df)} forward(s)")
        for _, player in fwd_df.iterrows():
            best_team.append({
                "name": player['name'],
                "position": player['position_name'],
                "price": player['price'],
                "predicted_points": player['predicted_points']
            })
            total_cost += player['price']
            total_predicted_points += player['predicted_points']
        
        conn.close()
        
        print(f"\n🏆 Best 11 Team Generated!")
        print(f"Formation: {formation}")
        print(f"Total Cost: £{total_cost:.1f}M")
        print(f"Budget Remaining: £{budget - total_cost:.1f}M")
        print(f"Total Predicted Points: {total_predicted_points:.1f}")
        print(f"Players: {len(best_team)}")
        
        print("\n📋 Team:")
        for player in best_team:
            print(f"• {player['name']} ({player['position']}) - £{player['price']}M - {player['predicted_points']:.1f} pts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_best11()
