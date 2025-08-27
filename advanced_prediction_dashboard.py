#!/usr/bin/env python3
"""
Advanced Prediction Dashboard
Displays comprehensive prediction features including core FPL inputs, advanced stats, and contextual factors
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def load_advanced_predictions():
    """Load advanced predictions from database"""
    conn = sqlite3.connect("epl_data.db")
    query = """
        SELECT * FROM advanced_predictions 
        ORDER BY predicted_points DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_historical_data():
    """Load historical best 11 data"""
    conn = sqlite3.connect("epl_data.db")
    
    # Load best 11 performance
    best11_query = "SELECT * FROM real_historical_best11 ORDER BY gameweek"
    best11_df = pd.read_sql_query(best11_query, conn)
    
    # Load role metrics
    role_query = "SELECT * FROM role_performance_metrics ORDER BY gameweek, position"
    role_df = pd.read_sql_query(role_query, conn)
    
    conn.close()
    return best11_df, role_df

def main():
    st.set_page_config(
        page_title="Advanced EPL Prediction Dashboard",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚽ Advanced EPL Prediction Dashboard")
    st.markdown("### Comprehensive Prediction Analysis with Core FPL Inputs, Advanced Stats & Contextual Factors")
    
    # Load data
    try:
        predictions_df = load_advanced_predictions()
        best11_df, role_df = load_historical_data()
        
        if predictions_df.empty:
            st.error("❌ No advanced predictions available. Please run the advanced prediction engine first.")
            return
            
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Position filter
    positions = ['All'] + list(predictions_df['position_name'].unique())
    selected_position = st.sidebar.selectbox("Position", positions)
    
    # Team filter
    teams = ['All'] + list(predictions_df['team_name'].unique())
    selected_team = st.sidebar.selectbox("Team", teams)
    
    # Price range filter
    min_price = float(predictions_df['price'].min())
    max_price = float(predictions_df['price'].max())
    price_range = st.sidebar.slider(
        "Price Range (£M)", 
        min_value=min_price, 
        max_value=max_price, 
        value=(min_price, max_price)
    )
    
    # Apply filters
    filtered_df = predictions_df.copy()
    
    if selected_position != 'All':
        filtered_df = filtered_df[filtered_df['position_name'] == selected_position]
    
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['team_name'] == selected_team]
    
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(filtered_df))
    
    with col2:
        avg_prediction = filtered_df['predicted_points'].mean()
        st.metric("Avg Predicted Points", f"{avg_prediction:.2f}")
    
    with col3:
        top_prediction = filtered_df['predicted_points'].max()
        st.metric("Top Prediction", f"{top_prediction:.2f}")
    
    with col4:
        avg_form = filtered_df['form'].mean()
        st.metric("Avg Form (Last 5)", f"{avg_form:.2f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Top Predictions", 
        "🎯 Core FPL Inputs", 
        "📈 Advanced Stats", 
        "🏟️ Contextual Factors", 
        "🤖 Model Comparison", 
        "📋 Historical Performance"
    ])
    
    with tab1:
        st.header("📊 Top Predictions")
        
        # Top 20 predictions
        top_20 = filtered_df.head(20)
        
        # Create bar chart
        fig = px.bar(
            top_20, 
            x='name', 
            y='predicted_points',
            color='position_name',
            title="Top 20 Predicted Performers",
            labels={'predicted_points': 'Predicted Points', 'name': 'Player Name'},
            color_discrete_map={'GK': '#1f77b4', 'DEF': '#ff7f0e', 'MID': '#2ca02c', 'FWD': '#d62728'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Predictions")
        display_columns = [
            'name', 'team_name', 'position_name', 'price', 'predicted_points',
            'form', 'minutes_probability', 'rotation_risk', 'fixture_difficulty'
        ]
        st.dataframe(
            top_20[display_columns].round(2),
            use_container_width=True
        )
    
    with tab2:
        st.header("🎯 Core FPL Inputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Player Form Analysis
            st.subheader("Player Form (Last 5 Matches)")
            
            form_fig = px.scatter(
                filtered_df,
                x='form',
                y='predicted_points',
                color='position_name',
                size='price',
                hover_data=['name', 'team_name'],
                title="Form vs Predicted Points"
            )
            st.plotly_chart(form_fig, use_container_width=True)
            
            # Minutes Probability
            st.subheader("Minutes Probability (Rotation Risk)")
            
            minutes_fig = px.histogram(
                filtered_df,
                x='minutes_probability',
                color='position_name',
                title="Distribution of Minutes Probability",
                nbins=20
            )
            st.plotly_chart(minutes_fig, use_container_width=True)
        
        with col2:
            # Rotation Risk Analysis
            st.subheader("Rotation Risk Analysis")
            
            rotation_fig = px.scatter(
                filtered_df,
                x='rotation_risk',
                y='predicted_points',
                color='position_name',
                size='minutes_probability',
                hover_data=['name', 'team_name'],
                title="Rotation Risk vs Predicted Points"
            )
            st.plotly_chart(rotation_fig, use_container_width=True)
            
            # Core Stats Summary
            st.subheader("Core Stats Summary")
            
            core_stats = filtered_df.groupby('position_name').agg({
                'form': 'mean',
                'minutes_probability': 'mean',
                'rotation_risk': 'mean',
                'predicted_points': 'mean'
            }).round(3)
            
            st.dataframe(core_stats, use_container_width=True)
    
    with tab3:
        st.header("📈 Advanced Stats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Attacking Stats
            st.subheader("Attacking Performance")
            
            attacking_stats = ['avg_goals', 'avg_assists', 'avg_shots', 'avg_key_passes', 'avg_big_chances']
            
            # Create subplot for attacking stats
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Goals per Game', 'Assists per Game', 'Shots per Game', 'Key Passes per Game'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=filtered_df['avg_goals'], y=filtered_df['predicted_points'], 
                          mode='markers', name='Goals', marker=dict(color='red')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=filtered_df['avg_assists'], y=filtered_df['predicted_points'], 
                          mode='markers', name='Assists', marker=dict(color='blue')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=filtered_df['avg_shots'], y=filtered_df['predicted_points'], 
                          mode='markers', name='Shots', marker=dict(color='green')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=filtered_df['avg_key_passes'], y=filtered_df['predicted_points'], 
                          mode='markers', name='Key Passes', marker=dict(color='orange')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Attacking Stats vs Predicted Points")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Defensive Stats
            st.subheader("Defensive Performance")
            
            defensive_stats = ['avg_clean_sheets', 'avg_tackles', 'avg_interceptions']
            
            def_fig = px.scatter_3d(
                filtered_df,
                x='avg_clean_sheets',
                y='avg_tackles', 
                z='avg_interceptions',
                color='predicted_points',
                size='price',
                hover_data=['name', 'team_name', 'position_name'],
                title="Defensive Stats 3D Analysis"
            )
            st.plotly_chart(def_fig, use_container_width=True)
            
            # Advanced Stats Summary
            st.subheader("Advanced Stats Summary")
            
            advanced_summary = filtered_df.groupby('position_name').agg({
                'avg_goals': 'mean',
                'avg_assists': 'mean',
                'avg_clean_sheets': 'mean',
                'avg_shots': 'mean',
                'avg_key_passes': 'mean',
                'avg_tackles': 'mean',
                'avg_interceptions': 'mean'
            }).round(3)
            
            st.dataframe(advanced_summary, use_container_width=True)
    
    with tab4:
        st.header("🏟️ Contextual Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fixture Difficulty Analysis
            st.subheader("Fixture Difficulty Impact")
            
            fixture_fig = px.box(
                filtered_df,
                x='fixture_difficulty',
                y='predicted_points',
                color='position_name',
                title="Predicted Points by Fixture Difficulty"
            )
            st.plotly_chart(fixture_fig, use_container_width=True)
            
            # Home/Away Factor
            st.subheader("Home/Away Performance Factor")
            
            home_away_fig = px.histogram(
                filtered_df,
                x='home_away_factor',
                color='position_name',
                title="Distribution of Home/Away Factors",
                nbins=15
            )
            st.plotly_chart(home_away_fig, use_container_width=True)
        
        with col2:
            # Injury Risk Analysis
            st.subheader("Injury Risk Analysis")
            
            injury_fig = px.scatter(
                filtered_df,
                x='injury_risk',
                y='predicted_points',
                color='position_name',
                size='minutes_probability',
                hover_data=['name', 'team_name'],
                title="Injury Risk vs Predicted Points"
            )
            st.plotly_chart(injury_fig, use_container_width=True)
            
            # Fixture Congestion
            st.subheader("Fixture Congestion Factor")
            
            congestion_fig = px.scatter(
                filtered_df,
                x='fixture_congestion',
                y='rotation_risk',
                color='predicted_points',
                size='price',
                hover_data=['name', 'team_name'],
                title="Fixture Congestion vs Rotation Risk"
            )
            st.plotly_chart(congestion_fig, use_container_width=True)
        
        # Contextual Factors Summary
        st.subheader("Contextual Factors Summary")
        
        contextual_summary = filtered_df.groupby('position_name').agg({
            'fixture_difficulty': 'mean',
            'home_away_factor': 'mean',
            'injury_risk': 'mean',
            'fixture_congestion': 'mean'
        }).round(3)
        
        st.dataframe(contextual_summary, use_container_width=True)
    
    with tab5:
        st.header("🤖 Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model Predictions Comparison
            st.subheader("Model Predictions Comparison")
            
            models = ['linear_prediction', 'rf_prediction', 'gb_prediction', 'xgb_prediction']
            model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
            
            # Create comparison chart
            comparison_data = []
            for i, model in enumerate(models):
                comparison_data.append({
                    'Model': model_names[i],
                    'Mean Prediction': filtered_df[model].mean(),
                    'Std Prediction': filtered_df[model].std()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            comp_fig = px.bar(
                comparison_df,
                x='Model',
                y='Mean Prediction',
                error_y='Std Prediction',
                title="Model Predictions Comparison"
            )
            st.plotly_chart(comp_fig, use_container_width=True)
            
            # Model correlation matrix
            st.subheader("Model Correlation Matrix")
            
            model_corr = filtered_df[models].corr()
            corr_fig = px.imshow(
                model_corr,
                title="Model Predictions Correlation",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with col2:
            # Ensemble vs Individual Models
            st.subheader("Ensemble vs Individual Models")
            
            # Scatter plot comparing ensemble to best individual model
            best_individual = filtered_df[models].max(axis=1)
            
            ensemble_fig = px.scatter(
                x=best_individual,
                y=filtered_df['predicted_points'],
                color=filtered_df['position_name'],
                hover_data=[filtered_df['name'], filtered_df['team_name']],
                title="Ensemble vs Best Individual Model",
                labels={'x': 'Best Individual Prediction', 'y': 'Ensemble Prediction'}
            )
            
            # Add diagonal line
            max_val = max(best_individual.max(), filtered_df['predicted_points'].max())
            ensemble_fig.add_trace(
                go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                          name='Perfect Correlation', line=dict(dash='dash'))
            )
            
            st.plotly_chart(ensemble_fig, use_container_width=True)
            
            # Model performance metrics
            st.subheader("Model Performance Metrics")
            
            model_metrics = []
            for i, model in enumerate(models):
                mae = np.mean(np.abs(filtered_df[model] - filtered_df['predicted_points']))
                mse = np.mean((filtered_df[model] - filtered_df['predicted_points']) ** 2)
                model_metrics.append({
                    'Model': model_names[i],
                    'MAE': mae,
                    'MSE': mse,
                    'Correlation': filtered_df[model].corr(filtered_df['predicted_points'])
                })
            
            metrics_df = pd.DataFrame(model_metrics).round(4)
            st.dataframe(metrics_df, use_container_width=True)
    
    with tab6:
        st.header("📋 Historical Performance")
        
        if not best11_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical Best 11 Performance
                st.subheader("Historical Best 11 Performance")
                
                hist_fig = px.line(
                    best11_df,
                    x='gameweek',
                    y=['total_predicted_points', 'total_actual_points'],
                    title="Predicted vs Actual Points (Historical Best 11)",
                    labels={'value': 'Points', 'variable': 'Type'}
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # Performance Accuracy
                st.subheader("Prediction Accuracy Over Time")
                
                accuracy_fig = px.bar(
                    best11_df,
                    x='gameweek',
                    y='performance_accuracy',
                    title="Prediction Accuracy by Gameweek",
                    labels={'performance_accuracy': 'Accuracy (%)'}
                )
                st.plotly_chart(accuracy_fig, use_container_width=True)
            
            with col2:
                # Role Performance Analysis
                if not role_df.empty:
                    st.subheader("Role Performance Analysis")
                    
                    role_fig = px.scatter(
                        role_df,
                        x='avg_predicted_points',
                        y='avg_actual_points',
                        color='position',
                        size='total_players',
                        hover_data=['gameweek', 'best_performer', 'worst_performer'],
                        title="Predicted vs Actual Points by Position"
                    )
                    st.plotly_chart(role_fig, use_container_width=True)
                    
                    # Role Metrics Summary
                    st.subheader("Role Performance Summary")
                    
                    role_summary = role_df.groupby('position').agg({
                        'avg_predicted_points': 'mean',
                        'avg_actual_points': 'mean',
                        'avg_accuracy': 'mean',
                        'clean_sheets': 'sum',
                        'goals_scored': 'sum',
                        'assists': 'sum'
                    }).round(3)
                    
                    st.dataframe(role_summary, use_container_width=True)
        else:
            st.info("📊 No historical performance data available. Run the historical data generation first.")
        
        # Feature Importance Analysis
        st.subheader("Feature Importance Analysis")
        
        # Calculate correlation with predicted points
        feature_columns = [
            'form', 'minutes_probability', 'rotation_risk', 'fixture_difficulty',
            'home_away_factor', 'injury_risk', 'fixture_congestion',
            'avg_goals', 'avg_assists', 'avg_clean_sheets', 'avg_shots',
            'avg_key_passes', 'avg_big_chances', 'avg_tackles', 'avg_interceptions'
        ]
        
        correlations = []
        for feature in feature_columns:
            if feature in filtered_df.columns:
                corr = abs(filtered_df[feature].corr(filtered_df['predicted_points']))
                correlations.append({'Feature': feature, 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        
        importance_fig = px.bar(
            corr_df,
            x='Feature',
            y='Correlation',
            title="Feature Importance (Correlation with Predicted Points)"
        )
        importance_fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(importance_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📊 Prediction Framework Summary
    
    **Core FPL Inputs:**
    - Player Form (last 3-5 matches average points)
    - Minutes Probability (rotation risk assessment)
    - Goals, Assists, Clean Sheets (core scoring drivers)
    - Fixture Difficulty (opponent strength rating)
    - Injury/Suspension Status (availability flags)
    
    **Advanced Stats:**
    - Expected Goals (xG) and Expected Assists (xA)
    - Shots on Target, Big Chances Created, Key Passes
    - Defensive Actions (tackles, interceptions, clearances)
    - Set-piece Involvement (penalties, free kicks, corners)
    
    **Contextual Factors:**
    - Opponent Adjustment (defensive/attacking strength)
    - Home vs Away Performance
    - Fixture Congestion (UCL/Europa rotation risk)
    - Team Form and Manager Tendencies
    
    **Prediction Models:**
    - Linear Regression (baseline)
    - Random Forest (ensemble)
    - Gradient Boosting (advanced)
    - XGBoost (optimized)
    - Ensemble (weighted combination)
    """)

if __name__ == "__main__":
    main()
