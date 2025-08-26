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
├── main.py                  # Main pipeline orchestrator
├── requirements.txt         # Dependencies
├── docker-compose.yml       # Container orchestration
├── Dockerfile              # Container definition
└── README.md               # Project documentation
```

---

## ⚙️ Installation

### Option 1: Local Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/epl-prediction-mvp.git
cd epl-prediction-mvp
```

2. **Setup Environment**
```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
# or
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
```

3. **Run Data Pipeline**
```bash
python main.py
```

4. **Start Services**

**FastAPI Backend:**
```bash
python api.py
# API available at: http://localhost:8000
```

**Streamlit Dashboard:**
```bash
streamlit run dashboard.py
# Dashboard available at: http://localhost:8501
```

### Option 2: Docker Setup

1. **Build and Run with Docker Compose**
```bash
docker-compose up --build
```

2. **Access Services**
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

---

## 🔮 Usage

### API Endpoints

**Get Predictions:**
```bash
GET http://localhost:8000/predictions?top_n=10&position=MID&max_price=10.0
```

**Refresh Data:**
```bash
GET http://localhost:8000/predictions/refresh
```

**Get Players:**
```bash
GET http://localhost:8000/players?position=FWD&limit=20
```

**Get Teams:**
```bash
GET http://localhost:8000/teams
```

### Dashboard Features

1. **Top Predictions**: View the top 10 predicted players
2. **Filters**: Filter by position, team, and price
3. **Analytics**: Interactive charts showing performance metrics
4. **Real-time Updates**: Refresh data with one click

### Command Line Usage

**Run once:**
```bash
python main.py
```

**Run scheduled (every 24 hours):**
```bash
python main.py --scheduled
```

**Custom interval:**
```bash
python main.py --scheduled --interval 12  # Every 12 hours
```

---

## 📊 Data Source

**Fantasy Premier League API:**
- Base URL: `https://fantasy.premierleague.com/api/bootstrap-static/`
- Provides: Players, teams, fixtures, and performance data
- Updates: Real-time during the season

**Features Collected:**
- Player stats (goals, assists, form, ICT index)
- Team strength and fixture difficulty
- Injury status and availability
- Historical performance data

---

## 🤖 Prediction Model

### Rule-based Scoring
- Recent form × fixture weight
- Team strength bonus
- Position-specific bonuses
- Injury penalties

### Machine Learning (XGBoost)
- Features: 18 engineered features
- Target: Total FPL points
- Training: Historical player data
- Evaluation: MSE and R² metrics

### Combined Prediction
- Weighted average: 30% rule-based + 70% ML
- Confidence scoring based on feature consistency

---

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: API
python api.py

# Terminal 2: Dashboard  
streamlit run dashboard.py

# Terminal 3: Data Pipeline
python main.py --scheduled
```

### Production with Docker
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables
```bash
DATABASE_URL=sqlite:///data/epl_data.db
PYTHONPATH=/app
PYTHONUNBUFFERED=1
```

---

## 🔧 Configuration

### Database
- **Default**: SQLite (MVP)
- **Production**: PostgreSQL (recommended)
- **Location**: `./data/epl_data.db`

### API Settings
- **Port**: 8000
- **Host**: 0.0.0.0
- **CORS**: Enabled for all origins

### Dashboard Settings
- **Port**: 8501
- **Theme**: Light mode
- **Cache**: 5 minutes

---

## 📈 Performance

### Model Metrics
- **MSE**: ~15-20 points
- **R² Score**: ~0.6-0.7
- **Training Time**: ~30 seconds
- **Prediction Time**: <1 second

### System Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+
- **Storage**: 1GB+
- **Network**: Internet connection for API calls

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: Streamlit
- **ML**: Scikit-learn, XGBoost
- **Database**: SQLite (MVP), PostgreSQL (production)
- **Deployment**: Docker, Docker Compose
- **Data**: Fantasy Premier League API

---

## 🔄 Data Pipeline

1. **Data Collection** (Every 24 hours)
   - Fetch from FPL API
   - Store in database
   - Handle rate limiting

2. **Feature Engineering**
   - Clean and normalize data
   - Create 18 features
   - Handle missing values

3. **Model Training**
   - Train XGBoost model
   - Evaluate performance
   - Save model artifacts

4. **Prediction Generation**
   - Generate predictions for all players
   - Calculate confidence scores
   - Store results

---

## 🚀 Roadmap

- [ ] **Real-time data fetcher** with websockets
- [ ] **Advanced ML models** (LSTM, Transformer)
- [ ] **Player injury/rotation news** integration
- [ ] **Team selection optimizer**
- [ ] **Mobile app** version
- [ ] **AWS/GCP deployment**
- [ ] **Email notifications**
- [ ] **Historical performance tracking**

---

## 👨‍💻 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=.
```

### Code Quality
```bash
# Install linting tools
pip install black flake8 mypy

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not affiliated with the Premier League or Fantasy Premier League. Predictions are estimates and not guarantees of actual player performance. Use at your own risk.

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👨‍💻 Contributors

Built by **Trix**

---

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **API Docs**: http://localhost:8000/docs (when running)

---

## 🎯 Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd epl-prediction-mvp
pip install -r requirements.txt

# 2. Run pipeline
python main.py

# 3. Start services
python api.py &
streamlit run dashboard.py

# 4. Open dashboard
# http://localhost:8501
```

**That's it!** 🎉 Your EPL prediction system is now running.
