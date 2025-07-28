import json
import re
import random
import numpy as np
import torch
import pickle
import os
from typing import Tuple, Dict, List
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MentalHealthChatbot:
    def __init__(self, intents_file: str = "Mental-Health-Chatbot-using-BERT-/intents.json", 
                 model_path: str = "models/mental_health_chatbot"):
        """
        Initialize the BERT-based mental health chatbot.
        
        Args:
            intents_file: Path to the intents JSON file
            model_path: Path to save/load the trained model
        """
        self.intents_file = intents_file
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 128
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.intent_to_response = {}
        
        # Load or train the model
        self._load_or_train_model()
        
    def _load_intents(self) -> Dict:
        """Load intents from JSON file."""
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Warning: {self.intents_file} not found. Using default intents.")
            return {"intents": []}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for BERT model."""
        # Remove special characters except apostrophes
        text = re.sub('[^a-zA-Z\']', ' ', text)
        text = text.lower()
        text = text.split()
        text = " ".join(text)
        return text
    
    def _prepare_data(self):
        """Prepare data for training."""
        # Load intents
        data = self._load_intents()
        
        # Create DataFrame-like structure
        patterns = []
        tags = []
        responses = []
        
        for intent in data['intents']:
            for pattern in intent['patterns']:
                patterns.append(self._preprocess_text(pattern))
                tags.append(intent['tag'])
                responses.append(intent['responses'])
        
        # Create intent to response mapping
        for intent in data['intents']:
            self.intent_to_response[intent['tag']] = intent['responses']
        
        return patterns, tags
    
    def _encode_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode texts for BERT model."""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)
    
    def _train_model(self, patterns: List[str], tags: List[str]):
        """Train the BERT model for intent classification."""
        print("🔄 Training BERT model for intent classification...")
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(tags)
        num_labels = len(np.unique(y_encoded))
        
        # Encode patterns
        input_ids, attention_masks = self._encode_texts(patterns)
        labels = torch.tensor(y_encoded)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
        validation_dataloader = DataLoader(val_dataset, batch_size=16)
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # Training loop
        epochs = 5  # Reduced for faster training
        print(f"🎯 Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch in train_dataloader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)
                
                self.model.zero_grad()
                outputs = self.model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels
                )
                
                loss = outputs[0]
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"📊 Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.2f}")
        
        print("✅ Model training completed!")
        
        # Save the model
        self._save_model()
    
    def _save_model(self):
        """Save the trained model, tokenizer, and label encoder."""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        
        # Save label encoder
        with open(os.path.join(self.model_path, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        # Save intent to response mapping
        with open(os.path.join(self.model_path, "intent_responses.json"), "w") as f:
            json.dump(self.intent_to_response, f, indent=2)
        
        print(f"💾 Model saved to {self.model_path}")
    
    def _load_model(self) -> bool:
        """Load the trained model, tokenizer, and label encoder."""
        try:
            # Load model and tokenizer
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # Load label encoder
            with open(os.path.join(self.model_path, "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)
            
            # Load intent to response mapping
            with open(os.path.join(self.model_path, "intent_responses.json"), "r") as f:
                self.intent_to_response = json.load(f)
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.model_path) and self._load_model():
            print("🎯 Using pre-trained model")
        else:
            print("🔄 No pre-trained model found. Training new model...")
            patterns, tags = self._prepare_data()
            self._train_model(patterns, tags)
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict intent using BERT model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            return "no-response", 0.0
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Tokenize and encode
            encoded_dict = self.tokenizer.encode_plus(
                processed_text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            input_ids = encoded_dict['input_ids'].to(self.device)
            attention_mask = encoded_dict['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            predicted_label_idx = np.argmax(probabilities, axis=1).flatten()
            
            # Get predicted intent
            predicted_intent = self.label_encoder.inverse_transform(predicted_label_idx)[0]
            confidence = probabilities[0][predicted_label_idx]
            
            return predicted_intent, confidence
            
        except Exception as e:
            print(f"Error in intent prediction: {e}")
            return "no-response", 0.0
    
    def get_response(self, intent: str, confidence: float) -> str:
        """
        Get response based on predicted intent.
        
        Args:
            intent: Predicted intent
            confidence: Confidence score
            
        Returns:
            Generated response text
        """
        # Get responses for the intent
        responses = self.intent_to_response.get(intent, [])
        
        if responses:
            # Randomly select a response
            response = random.choice(responses)
        else:
            # Fallback response
            response = "I'm here to listen. Could you tell me more about how you're feeling?"
        
        return response
    
    def classify_emotion(self, text: str) -> Tuple[str, float]:
        """
        Simple emotion classification based on keywords.
        This is a fallback since the BERT model focuses on intent.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        emotion_keywords = {
            'sad': ['sad', 'depressed', 'lonely', 'empty', 'down', 'blue', 'miserable'],
            'joy': ['happy', 'great', 'wonderful', 'excited', 'joy', 'pleased', 'good'],
            'anger': ['angry', 'frustrated', 'mad', 'irritated', 'annoyed', 'upset'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
            'stress': ['stressed', 'overwhelmed', 'burned out', 'tired', 'exhausted']
        }
        
        max_score = 0
        detected_emotion = 'neutral'
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > max_score:
                max_score = score
                detected_emotion = emotion
        
        confidence = min(max_score / len(text_lower.split()), 1.0)
        return detected_emotion, confidence
    
    def process_message(self, user_input: str) -> Dict:
        """
        Process a user message and return intent, emotion, and response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dictionary containing intent, emotion, confidence, and response
        """
        # Predict intent using BERT
        intent, intent_confidence = self.predict_intent(user_input)
        
        # Classify emotion (simple keyword-based)
        emotion, emotion_confidence = self.classify_emotion(user_input)
        
        # Generate response
        response = self.get_response(intent, intent_confidence)
        
        # Convert numpy values to Python floats for JSON serialization
        intent_confidence = float(intent_confidence) if isinstance(intent_confidence, (np.ndarray, np.integer, np.floating)) else intent_confidence
        emotion_confidence = float(emotion_confidence) if isinstance(emotion_confidence, (np.ndarray, np.integer, np.floating)) else emotion_confidence
        
        return {
            "intent": intent,
            "emotion": emotion,
            "intent_confidence": intent_confidence,
            "emotion_confidence": emotion_confidence,
            "response": response,
            "user_input": user_input
        }
    
    def retrain_model(self):
        """Retrain the model with current data."""
        print("🔄 Retraining model...")
        patterns, tags = self._prepare_data()
        self._train_model(patterns, tags)
        print("✅ Model retraining completed!")
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "max_length": self.max_len,
            "num_intents": len(self.intent_to_response),
            "intents": list(self.intent_to_response.keys())
        } 