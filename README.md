# 🧠 AI Mental Health Chatbot - BERT-powered MVP

A privacy-respecting AI chatbot designed to provide emotional support through **BERT-based intent classification** and therapeutic responses.

## 🎯 Project Overview

This MVP demonstrates a complete mental health chatbot system with:

- **BERT-based Intent Detection**: Advanced NLP using BERT for accurate intent recognition
- **Emotion Classification**: Keyword-based emotion detection as supplementary analysis
- **Therapeutic Responses**: Contextual, supportive responses based on intent
- **Privacy-First**: No personal data logging, anonymized sessions only
- **Multiple Interfaces**: Web UI, CLI, and API endpoints
- **Comprehensive Testing**: Manual test cases + PyTest automation
- **Model Training**: Automatic BERT model training on intent data

## 🏗️ Architecture

```
User Input → BERT Tokenization → Intent Classification → Response Generation → User
     ↓              ↓                    ↓                    ↓
  Web UI/CLI → BERT Model → Intent Prediction → Template Responses
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 4GB+ RAM (for BERT models)
- Internet connection (for model download)

### 🛠️ Environment Setup (Recommended)

**Windows:**
```bash
# Setup virtual environment
setup_env.bat

# Activate environment
activate_env.bat
```

**Linux/Mac:**
```bash
# Setup virtual environment
chmod +x setup_env.sh
./setup_env.sh

# Activate environment
chmod +x activate_env.sh
./activate_env.sh
```

### 📦 Manual Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python run.py check
   ```

### 🚀 Running the Application

**Web Interface:**
```bash
python run.py web
# or
python app.py
```
Open http://localhost:5000 in your browser

**CLI Interface:**
```bash
python run.py cli
# or
python cli_chatbot.py
```

**Run Tests:**
```bash
python run.py test
# or
pytest tests/ -v
```

## 📁 Project Structure

```
mental-health-chatbot/
├── app.py                  # Flask API server
├── chatbot.py              # BERT-based chatbot engine
├── cli_chatbot.py          # Command-line interface
├── run.py                  # Launcher script
├── requirements.txt         # Python dependencies
├── setup_env.bat          # Windows environment setup
├── setup_env.sh           # Unix environment setup
├── activate_env.bat       # Windows environment activation
├── activate_env.sh        # Unix environment activation
├── fix_dependencies.py    # Dependency fixer
├── test_scikit_learn.py  # ML dependency tester
├── README.md              # This file
├── INSTALL.md             # Detailed installation guide
├── templates/
│   └── index.html         # BERT-powered web interface
├── tests/
│   ├── test_chatbot.py    # PyTest unit tests
│   ├── test_api.py        # API endpoint tests
│   └── manual_test_cases.md # Manual testing guide
├── data/
│   └── sample_queries.json # Sample test data
├── models/                # Trained BERT models (auto-created)
└── Mental-Health-Chatbot-using-BERT-/
    └── intents.json       # Intent patterns and responses
```

## 🔧 Core Components

### 1. BERT Chatbot Engine (`chatbot.py`)

The core AI engine that handles:
- **BERT Model Training**: Automatic training on intent data
- **Intent Classification**: BERT-based intent recognition
- **Response Generation**: Contextual therapeutic responses
- **Model Persistence**: Save/load trained models

```python
from chatbot import MentalHealthChatbot

chatbot = MentalHealthChatbot()
result = chatbot.process_message("I feel sad today")
print(result['response'])  # Supportive response
print(result['intent'])    # 'sad' (BERT prediction)
print(result['intent_confidence'])  # 0.95 (BERT confidence)
```

### 2. Flask API (`app.py`)

RESTful API endpoints:
- `POST /chat` - Main chat endpoint
- `GET /health` - Health check with model info
- `GET /stats` - Anonymized statistics
- `GET /model` - BERT model information
- `POST /retrain` - Retrain BERT model
- `POST /test` - Testing endpoint

### 3. Web Interface (`templates/index.html`)

Modern, responsive web UI with:
- Real-time BERT-powered chat interface
- Intent/emotion visualization
- High-confidence prediction indicators
- Mobile-responsive design

### 4. CLI Interface (`cli_chatbot.py`)

Command-line version with:
- Interactive BERT-powered chat session
- Model information display
- Model retraining capability
- Session statistics

## 🧪 Testing

### Automated Tests

Run the full test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
# Unit tests
pytest tests/test_chatbot.py -v

# API tests (requires server running)
pytest tests/test_api.py -v
```

### Manual Testing

Follow the comprehensive manual test cases in `tests/manual_test_cases.md`:

1. **BERT Intent Detection Tests**
   - Greeting, sad, stressed, happy intents
   - High-confidence predictions

2. **Emotion Classification Tests**
   - Sad, joy, anger, fear emotions

3. **Response Quality Tests**
   - Supportive, positive, calming responses

4. **Edge Case Tests**
   - Empty messages, long text, special characters

5. **API Endpoint Tests**
   - Health, chat, stats, model endpoints

6. **Privacy Tests**
   - No personal data logging

7. **Performance Tests**
   - Response time, concurrent users

8. **UI Tests**
   - Responsive design, accessibility

## 🔒 Privacy Features

- **No Personal Data Storage**: Messages processed in memory only
- **Anonymized Statistics**: Only aggregate usage data
- **Local BERT Processing**: No data sent to external services
- **Session Isolation**: No cross-session data leakage
- **Privacy Indicators**: Clear privacy notices in UI

## 📊 API Documentation

### Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
  "message": "I feel sad today"
}
```

**Response:**
```json
{
  "status": "success",
  "response": "I'm really sorry you're feeling this way. I'm here to listen.",
  "intent": "sad",
  "emotion": "sad",
  "intent_confidence": 0.95,
  "emotion_confidence": 0.85
}
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "chatbot_loaded": true,
  "total_requests": 42,
  "model_info": {
    "model_path": "models/mental_health_chatbot",
    "device": "cpu",
    "max_length": 128,
    "num_intents": 25,
    "intents": ["greeting", "sad", "happy", ...]
  }
}
```

### Model Information

```http
GET /model
```

**Response:**
```json
{
  "model_info": {
    "model_path": "models/mental_health_chatbot",
    "device": "cpu",
    "max_length": 128,
    "num_intents": 25,
    "intents": ["greeting", "sad", "happy", "stressed", ...]
  },
  "status": "success"
}
```

### Retrain Model

```http
POST /retrain
```

**Response:**
```json
{
  "message": "Model retraining completed successfully",
  "status": "success"
}
```

## 🎨 Features

### ✅ Implemented Features

- [x] BERT-based intent detection with high accuracy
- [x] Automatic model training on intent data
- [x] Model persistence and loading
- [x] Keyword-based emotion classification
- [x] Therapeutic response generation
- [x] Web interface with real-time BERT predictions
- [x] CLI interface for terminal users
- [x] RESTful API endpoints
- [x] Privacy-focused design
- [x] Comprehensive testing suite
- [x] Error handling and validation
- [x] Session statistics (anonymized)
- [x] Responsive web design
- [x] Accessibility features
- [x] Virtual environment setup scripts
- [x] Model retraining capability
- [x] High-confidence prediction indicators

### 🔮 Future Enhancements

- [ ] Fine-tuned BERT models for specific mental health domains
- [ ] Multi-language BERT support
- [ ] Advanced emotion classification using BERT
- [ ] Conversation memory and context
- [ ] Crisis detection and escalation
- [ ] Integration with mental health resources
- [ ] Voice interface
- [ ] Mobile app
- [ ] Admin dashboard
- [ ] Analytics and insights
- [ ] Customizable responses

## 🛠️ Development

### Adding New Intents

1. Edit `Mental-Health-Chatbot-using-BERT-/intents.json`
2. Add new intent with patterns and responses
3. Retrain the BERT model: `python cli_chatbot.py` → type `retrain`
4. Test with manual test cases
5. Update automated tests

### Model Training

The BERT model automatically trains on the intent data:

```python
# Automatic training on initialization
chatbot = MentalHealthChatbot()

# Manual retraining
chatbot.retrain_model()
```

### Customizing Responses

1. Edit the response templates in `Mental-Health-Chatbot-using-BERT-/intents.json`
2. Add new intent-based responses
3. Retrain the model
4. Test response quality
5. Update documentation

## 🚀 Deployment

### Local Development

```bash
# Setup environment
setup_env.bat  # Windows
./setup_env.sh # Linux/Mac

# Activate environment
activate_env.bat  # Windows
./activate_env.sh # Linux/Mac

# Run web server
python run.py web

# Run CLI version
python run.py cli

# Run tests
python run.py test
```

### Production Deployment

1. **Environment Setup**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

2. **WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Docker Deployment**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 5000
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Run the test suite
6. Submit a pull request

## 📝 License

This project is for educational and research purposes. Please ensure compliance with local regulations regarding mental health services.

## ⚠️ Disclaimer

This chatbot is designed for educational and research purposes. It is not a substitute for professional mental health care. If you're experiencing a mental health crisis, please contact a mental health professional or emergency services.

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Review the test documentation
- Check the API documentation
- Run the test suite for debugging

## 🔧 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Ensure virtual environment is activated
   python run.py check
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **BERT Model Training Issues**
   - Ensure sufficient RAM (4GB+)
   - Check internet connection for model download
   - Try CPU-only PyTorch if GPU issues occur

3. **Model Download Issues**
   - Ensure internet connection
   - Check firewall settings
   - Try running with VPN if needed

4. **Memory Issues**
   - Close other applications
   - Ensure 4GB+ RAM available
   - Consider using CPU-only PyTorch

5. **Port Conflicts**
   - Change port in `app.py` if 5000 is busy
   - Check if another service is using the port

---

**🧠 Built with ❤️ for mental health awareness and support** 