from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from datetime import datetime
from chatbot import MentalHealthChatbot
import os
import numpy as np

# Configure logging to exclude personal data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize chatbot
print("🧠 Initializing BERT-based Mental Health Chatbot...")
chatbot = MentalHealthChatbot()

# Privacy-focused session storage (no personal data)
session_stats = {
    "total_requests": 0,
    "intents_detected": {},
    "emotions_detected": {},
    "last_request_time": None
}

@app.route('/')
def home():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint - processes user messages and returns responses.
    Privacy-focused: no personal data is logged or stored.
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message is required',
                'status': 'error'
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty',
                'status': 'error'
            }), 400
        
        # Process the message using BERT model
        result = chatbot.process_message(user_message)
        
        # Update anonymized session stats (no personal data)
        session_stats['total_requests'] += 1
        session_stats['last_request_time'] = datetime.now().isoformat()
        
        intent = result['intent']
        emotion = result['emotion']
        
        session_stats['intents_detected'][intent] = session_stats['intents_detected'].get(intent, 0) + 1
        session_stats['emotions_detected'][emotion] = session_stats['emotions_detected'].get(emotion, 0) + 1
        
        # Convert numpy values to Python floats before rounding
        intent_confidence = float(result['intent_confidence']) if isinstance(result['intent_confidence'], (np.ndarray, np.integer, np.floating)) else result['intent_confidence']
        emotion_confidence = float(result['emotion_confidence']) if isinstance(result['emotion_confidence'], (np.ndarray, np.integer, np.floating)) else result['emotion_confidence']
        
        # Log anonymized interaction (no personal data)
        logging.info(f"Chat interaction - Intent: {intent}, Emotion: {emotion}, Intent Confidence: {intent_confidence:.2f}")
        
        return jsonify({
            'response': result['response'],
            'intent': intent,
            'emotion': emotion,
            'intent_confidence': round(intent_confidence, 3),
            'emotion_confidence': round(emotion_confidence, 3),
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_info = chatbot.get_model_info()
    return jsonify({
        'status': 'healthy',
        'chatbot_loaded': chatbot is not None,
        'total_requests': session_stats['total_requests'],
        'model_info': model_info
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get anonymized usage statistics.
    No personal data is included.
    """
    return jsonify({
        'total_requests': session_stats['total_requests'],
        'intents_detected': session_stats['intents_detected'],
        'emotions_detected': session_stats['emotions_detected'],
        'last_request_time': session_stats['last_request_time']
    })

@app.route('/model', methods=['GET'])
def get_model_info():
    """
    Get information about the BERT model.
    """
    model_info = chatbot.get_model_info()
    return jsonify({
        'model_info': model_info,
        'status': 'success'
    })

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain the BERT model endpoint.
    """
    try:
        chatbot.retrain_model()
        return jsonify({
            'message': 'Model retraining completed successfully',
            'status': 'success'
        })
    except Exception as e:
        logging.error(f"Error retraining model: {str(e)}")
        return jsonify({
            'error': 'Model retraining failed',
            'status': 'error'
        }), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """
    Test endpoint for automated testing.
    """
    data = request.get_json()
    test_message = data.get('message', '')
    
    if not test_message:
        return jsonify({'error': 'Test message required'}), 400
    
    result = chatbot.process_message(test_message)
    
    # Convert numpy values to Python floats for JSON serialization
    intent_confidence = float(result['intent_confidence']) if isinstance(result['intent_confidence'], (np.ndarray, np.integer, np.floating)) else result['intent_confidence']
    emotion_confidence = float(result['emotion_confidence']) if isinstance(result['emotion_confidence'], (np.ndarray, np.integer, np.floating)) else result['emotion_confidence']
    
    return jsonify({
        'test_result': {
            'input': test_message,
            'intent': result['intent'],
            'emotion': result['emotion'],
            'response': result['response'],
            'intent_confidence': intent_confidence,
            'emotion_confidence': emotion_confidence
        }
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🧠 Mental Health Chatbot API Starting...")
    print("📊 Privacy-focused: No personal data is logged or stored")
    print("🤖 BERT-based intent classification")
    print("🔗 API endpoints:")
    print("   - POST /chat - Main chat endpoint")
    print("   - GET  /health - Health check")
    print("   - GET  /stats - Anonymized statistics")
    print("   - GET  /model - Model information")
    print("   - POST /retrain - Retrain model")
    print("   - POST /test - Testing endpoint")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 