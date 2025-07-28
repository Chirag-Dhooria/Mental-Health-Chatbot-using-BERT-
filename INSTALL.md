# 🚀 Installation Guide - Mental Health Chatbot

This guide will help you set up the Mental Health Chatbot MVP with proper virtual environment isolation.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **4GB+ RAM** (for ML models)
- **Internet connection** (for model download)
- **Git** (optional, for version control)

## 🛠️ Quick Setup (Recommended)

### Windows Users

1. **Download and extract** the project files
2. **Open Command Prompt** in the project directory
3. **Run the setup script:**
   ```cmd
   setup_env.bat
   ```
4. **Activate the environment:**
   ```cmd
   activate_env.bat
   ```
5. **Start the chatbot:**
   ```cmd
   python run.py web
   ```

### Linux/Mac Users

1. **Open Terminal** in the project directory
2. **Make scripts executable:**
   ```bash
   chmod +x setup_env.sh
   chmod +x activate_env.sh
   ```
3. **Run the setup script:**
   ```bash
   ./setup_env.sh
   ```
4. **Activate the environment:**
   ```bash
   ./activate_env.sh
   ```
5. **Start the chatbot:**
   ```bash
   python run.py web
   ```

## 📦 Manual Installation

If the automated scripts don't work, follow these manual steps:

### Step 1: Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python run.py check
```

You should see all packages marked as ✅.

### Step 4: Run the Application

**Web Interface:**
```bash
python run.py web
```
Open http://localhost:5000 in your browser

**CLI Interface:**
```bash
python run.py cli
```

## 🧪 Testing the Installation

### Run Automated Tests

```bash
python run.py test
```

### Manual Testing

1. **Start the web server:**
   ```bash
   python run.py web
   ```

2. **Open browser** to http://localhost:5000

3. **Test basic functionality:**
   - Type "Hello" → Should get greeting response
   - Type "I feel sad" → Should get supportive response
   - Type "I'm happy" → Should get positive response

## 🔧 Troubleshooting

### Common Issues

#### 1. "Python not found" Error
**Solution:**
- Ensure Python is installed and in PATH
- Try `python3` instead of `python`
- Install Python from https://python.org

#### 2. "pip not found" Error
**Solution:**
```bash
python -m ensurepip --upgrade
```

#### 3. "Permission denied" Error (Linux/Mac)
**Solution:**
```bash
chmod +x setup_env.sh
chmod +x activate_env.sh
```

#### 4. "Module not found" Errors
**Solution:**
```bash
# Ensure virtual environment is activated
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 5. "Out of memory" Error
**Solution:**
- Close other applications
- Ensure 4GB+ RAM available
- Consider using CPU-only PyTorch:
  ```bash
  pip uninstall torch
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

#### 6. "Port 5000 already in use" Error
**Solution:**
- Change port in `app.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)
  ```
- Or kill the process using port 5000

#### 7. "Model download failed" Error
**Solution:**
- Check internet connection
- Try with VPN if needed
- Check firewall settings
- Manual download may be required

### Dependency-Specific Issues

#### PyTorch Issues
```bash
# CPU-only version (smaller, faster)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Transformers Issues
```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Reinstall
pip uninstall transformers
pip install transformers==4.38.2
```

#### NLTK Issues
```python
# Run this in Python to download required data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📊 System Requirements

### Minimum Requirements
- **RAM:** 4GB
- **Storage:** 2GB free space
- **Python:** 3.8+
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **RAM:** 8GB+
- **Storage:** 5GB free space
- **CPU:** Multi-core processor
- **GPU:** Optional (CUDA-compatible for faster inference)

## 🔄 Updating the Application

### Update Dependencies
```bash
# Activate environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Update Code
```bash
# If using git
git pull origin main

# Reinstall if needed
pip install -r requirements.txt
```

## 🚀 Production Deployment

### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📞 Getting Help

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Run the diagnostic command:**
   ```bash
   python run.py check
   ```
3. **Check the logs** in the terminal output
4. **Review the manual test cases** in `tests/manual_test_cases.md`
5. **Create an issue** with detailed error information

## ✅ Verification Checklist

After installation, verify:

- [ ] Virtual environment is activated
- [ ] All dependencies are installed (`python run.py check`)
- [ ] Web server starts without errors (`python run.py web`)
- [ ] CLI interface works (`python run.py cli`)
- [ ] Tests pass (`python run.py test`)
- [ ] Web interface loads at http://localhost:5000
- [ ] Chat functionality works in web interface
- [ ] Emotion detection works (try "I feel sad")

---

**🎉 Congratulations! Your Mental Health Chatbot is ready to use!** 