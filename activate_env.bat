@echo off
echo ========================================
echo 🧠 Mental Health Chatbot - Activate Environment
echo ========================================

if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo 📦 Please run setup_env.bat first
    pause
    exit /b 1
)

echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo ✅ Environment activated!
echo.
echo 🚀 Available commands:
echo   python run.py web    - Start web server
echo   python run.py cli    - Start CLI
echo   python run.py test   - Run tests
echo   python run.py check  - Check dependencies
echo.
echo 🛑 To deactivate: deactivate
echo.

cmd /k 