#!/bin/bash

echo "========================================"
echo "🧠 Mental Health Chatbot - Activate Environment"
echo "========================================"

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "📦 Please run setup_env.sh first"
    exit 1
fi

echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "✅ Environment activated!"
echo ""
echo "🚀 Available commands:"
echo "  python run.py web    - Start web server"
echo "  python run.py cli    - Start CLI"
echo "  python run.py test   - Run tests"
echo "  python run.py check  - Check dependencies"
echo ""
echo "🛑 To deactivate: deactivate"
echo ""

exec $SHELL 