#!/bin/bash

# EPL Player Prediction System Startup Script

echo "⚽ Starting EPL Player Prediction System..."
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Run initial data collection and prediction
echo "🚀 Running initial data pipeline..."
python main.py

# Start services
echo "🌐 Starting services..."

# Start API in background
echo "⚡ Starting FastAPI backend..."
python api.py &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start dashboard
echo "📊 Starting Streamlit dashboard..."
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 &
DASHBOARD_PID=$!

echo ""
echo "🎉 EPL Prediction System is running!"
echo "===================================="
echo "📊 Dashboard: http://localhost:8501"
echo "⚡ API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null
    kill $DASHBOARD_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep script running
wait
