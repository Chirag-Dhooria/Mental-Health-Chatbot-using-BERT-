#!/bin/bash

echo "========================================"
echo "🧠 Mental Health Chatbot - Environment Setup"
echo "========================================"

echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🔧 Running dependency fix (if needed)..."
python fix_dependencies.py

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "🚀 To start the chatbot:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run web server: python run.py web"
echo "  3. Or run CLI: python run.py cli"
echo ""
echo "🧪 To run tests:"
echo "  python run.py test"
echo ""
echo "🔧 If you have dependency issues:"
echo "  python fix_dependencies.py"
echo "" 