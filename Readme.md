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

⚙️ Installation
1. Clone Repository
git clone https://github.com/yourusername/epl-prediction-mvp.git
cd epl-prediction-mvp

2. Setup Environment
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

3. Run Services (Local)

Run FastAPI API:

uvicorn api:app --reload


Run Streamlit Dashboard:

streamlit run dashboard.py

4. Run with Docker
docker-compose up --build

🔮 Usage
API

Predictions available at:

GET http://localhost:8000/predictions

Dashboard

Open in browser:

http://localhost:8501


You’ll see:

📈 Top 10 predicted players for next week

🔍 Filters (position, team, budget)

⚽ Fixture difficulty

📊 Data Source

Fantasy Premier League API:
https://fantasy.premierleague.com/api/bootstrap-static/

🚀 Roadmap

 Real-time data fetcher

 Rule-based baseline predictor

 ML model refinement (XGBoost + time-series)

 Player injury/rotation news integration

 Deployment on AWS/GCP

 Mobile app version

🛠️ Tech Stack

Python 3.10+

FastAPI – backend API

Streamlit – dashboard

Scikit-learn / XGBoost – ML predictions

PostgreSQL / SQLite – database

Docker & Docker Compose – deployment

👨‍💻 Contributors

Built by Trix

⚠️ Disclaimer

This project is for educational and research purposes only.
It is not affiliated with the Premier League or Fantasy Premier League.
Predictions are estimates and not guarantees of actual player performance.
