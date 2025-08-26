"""
Streamlit Dashboard - Interactive visualization for EPL player predictions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sqlite3
from datetime import datetime

st.set_page_config(page_title="EPL Player Predictor", page_icon="⚽", layout="wide")

@st.cache_data(ttl=300)
def load_predictions():
    try:
        response = requests.get("http://localhost:8000/predictions?top_n=50")
        return response.json()['predictions'] if response.status_code == 200 else []
    except:
        return []

@st.cache_data(ttl=300)
def load_teams():
    try:
        response = requests.get("http://localhost:8000/teams")
        return response.json()['teams'] if response.status_code == 200 else []
    except:
        return []

def load_from_database():
    try:
        conn = sqlite3.connect("data/epl_data.db")
        predictions_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_points DESC LIMIT 50", conn)
        teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
        conn.close()
        return predictions_df, teams_df
    except:
        return pd.DataFrame(), pd.DataFrame()

def generate_best11_local(budget, formation):
    """Generate best 11 locally"""
    try:
        conn = sqlite3.connect("epl_data.db")
        predictions_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_points DESC", conn)
        conn.close()
        
        if len(predictions_df) == 0:
            return {"error": "No predictions available"}
        
        # Parse formation
        formation_parts = formation.split('-')
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
        
        return {
            "formation": formation,
            "budget": budget,
            "total_cost": round(total_cost, 1),
            "budget_remaining": round(budget - total_cost, 1),
            "total_predicted_points": round(total_predicted_points, 1),
            "players": best_team,
            "count": len(best_team)
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("⚽ EPL Player Predictor")
    st.markdown("### AI-powered predictions for Fantasy Premier League performance")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Load data
    predictions = load_predictions()
    teams = load_teams()
    
    if not predictions:
        st.warning("⚠️ API not available. Loading from database...")
        predictions_df, teams_df = load_from_database()
        if predictions_df.empty:
            st.error("❌ No data available. Please run the data collection first.")
            return
        predictions = predictions_df.to_dict('records')
        teams = teams_df.to_dict('records')
    
    df = pd.DataFrame(predictions)
    
    if df.empty:
        st.error("❌ No predictions available")
        return
    
    # Filters
    positions = ['All'] + list(df['position'].unique()) if 'position' in df.columns else ['All']
    selected_position = st.sidebar.selectbox("Position", positions)
    
    team_names = {team['id']: team['name'] for team in teams} if teams else {}
    if 'team_id' in df.columns:
        df['team_name'] = df['team_id'].map(team_names)
        teams_list = ['All'] + list(df['team_name'].dropna().unique())
        selected_team = st.sidebar.selectbox("Team", teams_list)
    else:
        selected_team = 'All'
    
    max_price = st.sidebar.slider("Max Price (£M)", 
                                 float(df['price'].min()) if 'price' in df.columns else 5.0,
                                 float(df['price'].max()) if 'price' in df.columns else 15.0,
                                 float(df['price'].max()) if 'price' in df.columns else 15.0)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_position != 'All':
        filtered_df = filtered_df[filtered_df['position'] == selected_position]
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['team_name'] == selected_team]
    if 'price' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['price'] <= max_price]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(filtered_df))
    with col2:
        avg_points = filtered_df['predicted_points'].mean() if 'predicted_points' in filtered_df.columns else 0
        st.metric("Avg Predicted Points", f"{avg_points:.1f}")
    with col3:
        avg_confidence = filtered_df['confidence_score'].mean() if 'confidence_score' in filtered_df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    with col4:
        if st.button("🔄 Refresh Data"):
            try:
                response = requests.get("http://localhost:8000/predictions/refresh")
                if response.status_code == 200:
                    st.success("✅ Data refreshed!")
                    st.rerun()
                else:
                    st.error("❌ Failed to refresh")
            except:
                st.error("❌ API not available")
    
    # Top predictions
    st.header("🏆 Top Predicted Players")
    
    if not filtered_df.empty:
        top_10 = filtered_df.head(10)
        
        for idx, player in top_10.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{player['name']}** ({player.get('position', 'N/A')})")
                    if 'team_name' in player:
                        st.markdown(f"*{player['team_name']}*")
                
                with col2:
                    if 'price' in player:
                        st.metric("Price", f"£{player['price']:.1f}M")
                
                with col3:
                    if 'recent_form' in player:
                        st.metric("Form", f"{player['recent_form']:.1f}")
                
                with col4:
                    if 'predicted_points' in player:
                        st.metric("Predicted", f"{player['predicted_points']:.1f}")
                
                with col5:
                    if 'confidence_score' in player:
                        confidence_color = "🟢" if player['confidence_score'] > 0.7 else "🟡" if player['confidence_score'] > 0.4 else "🔴"
                        st.markdown(f"{confidence_color} {player['confidence_score']:.2f}")
                
                st.divider()
    
    # Charts
    st.header("📊 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'predicted_points' in filtered_df.columns and 'position' in filtered_df.columns:
            fig = px.box(filtered_df, x='position', y='predicted_points', 
                        title="Predicted Points by Position")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'confidence_score' in filtered_df.columns and 'predicted_points' in filtered_df.columns:
            fig = px.scatter(filtered_df, x='predicted_points', y='confidence_score',
                           hover_data=['name'], title="Confidence vs Predicted Points")
            st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Performance
    if 'price' in filtered_df.columns and 'predicted_points' in filtered_df.columns:
        fig = px.scatter(filtered_df, x='price', y='predicted_points',
                        hover_data=['name'], title="Price vs Predicted Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Best 11 Section
    st.header("🏆 Best 11 for Next Gameweek")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.slider("Budget (£M)", 80.0, 120.0, 100.0, 0.5)
    
    with col2:
        formation = st.selectbox("Formation", ["4-4-2", "4-3-3", "3-5-2", "5-3-2", "4-5-1"])
    
    with col3:
        if st.button("Generate Best 11"):
            try:
                # Try API first
                response = requests.get(f"http://localhost:8000/best11?budget={budget}&formation={formation}")
                if response.status_code == 200:
                    team_data = response.json()
                else:
                    # Fallback: generate locally
                    team_data = generate_best11_local(budget, formation)
                
                st.success(f"✅ Best 11 Generated!")
                st.metric("Total Cost", f"£{team_data['total_cost']}M")
                st.metric("Budget Remaining", f"£{team_data['budget_remaining']}M")
                st.metric("Predicted Points", f"{team_data['total_predicted_points']:.1f}")
                
                # Display team
                st.subheader(f"Formation: {team_data['formation']}")
                
                # Group players by position
                gk_players = [p for p in team_data['players'] if p['position'] == 'GK']
                def_players = [p for p in team_data['players'] if p['position'] == 'DEF']
                mid_players = [p for p in team_data['players'] if p['position'] == 'MID']
                fwd_players = [p for p in team_data['players'] if p['position'] == 'FWD']
                
                # Display in formation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Goalkeeper**")
                    for player in gk_players:
                        st.write(f"• {player['name']} (£{player['price']}M) - {player['predicted_points']:.1f} pts")
                
                with col2:
                    st.write("**Defenders**")
                    for player in def_players:
                        st.write(f"• {player['name']} (£{player['price']}M) - {player['predicted_points']:.1f} pts")
                
                with col3:
                    st.write("**Midfielders**")
                    for player in mid_players:
                        st.write(f"• {player['name']} (£{player['price']}M) - {player['predicted_points']:.1f} pts")
                
                st.write("**Forwards**")
                for player in fwd_players:
                    st.write(f"• {player['name']} (£{player['price']}M) - {player['predicted_points']:.1f} pts")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"⚽ EPL Player Predictor | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
