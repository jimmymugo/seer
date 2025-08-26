# 🚀 Quick Setup Guide

## Prerequisites
- Python 3.10+
- Git
- Docker (optional)

## Quick Start (3 minutes)

### Option 1: One-Command Setup
```bash
# Clone and run
git clone <your-repo-url>
cd epl-prediction-mvp
./start.sh
```

### Option 2: Manual Setup
```bash
# 1. Clone repository
git clone <your-repo-url>
cd epl-prediction-mvp

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline
python main.py

# 5. Start services
python api.py &          # Terminal 1
streamlit run dashboard.py  # Terminal 2
```

### Option 3: Docker Setup
```bash
# Build and run everything
docker-compose up --build
```

## Access Points
- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Test the System
```bash
python test_system.py
```

## Troubleshooting

### Common Issues
1. **Port already in use**: Change ports in `api.py` or `dashboard.py`
2. **API connection failed**: Check internet connection
3. **Import errors**: Ensure all dependencies are installed

### Logs
- API logs: Check terminal running `api.py`
- Dashboard logs: Check terminal running `dashboard.py`
- Data pipeline logs: Check terminal running `main.py`

## Next Steps
1. Open dashboard at http://localhost:8501
2. Explore predictions and filters
3. Try the API endpoints
4. Customize the model parameters

## Support
- Check the main README.md for detailed documentation
- Run `python test_system.py` to diagnose issues
- Check API docs at http://localhost:8000/docs
