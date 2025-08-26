# ⚽ EPL Player Prediction MVP

An AI-powered system that predicts the **best performing English Premier League (EPL) players** for upcoming gameweeks using **real-time Fantasy Premier League (FPL) data, machine learning models, and a visualization dashboard**.

---

## 📌 Features
- 🔄 **Real-time Data Collection** from the Fantasy Premier League API  
- 🧹 **Data Processing & Feature Engineering** (form, fixture difficulty, home/away factor, injury status)  
- 🤖 **Prediction Engine**  
  - Rule-based baseline scoring  
  - Machine learning model (XGBoost regression) for expected points  
- 📊 **Streamlit Dashboard**  
  - Top 10 predicted players  
  - Filters: by team, position, price  
  - Fixture difficulty display & confidence scores  
- ⚡ **FastAPI Backend** with `/predictions` endpoint  
- 🐳 **Dockerized Setup** for easy deployment  

---

## 🏗️ Project Structure
```plaintext
epl-prediction-mvp/
├── data_collector.py        # Fetch & store EPL data
├── feature_engineering.py   # Process & generate features
├── model.py                 # ML training & prediction
├── api.py                   # FastAPI backend
├── dashboard.py             # Streamlit dashboard
├── requirements.txt         # Dependencies
├── docker-compose.yml       # Container orchestration
└── README.md                # Project documentation
