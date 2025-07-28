#!/usr/bin/env python3
"""
Mental Health Chatbot - CLI Version
A privacy-respecting command-line interface for emotional support using BERT-based intent classification.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from chatbot import MentalHealthChatbot

class CLIChatbot:
    def __init__(self):
        """Initialize the CLI chatbot."""
        print("🧠 Initializing BERT-based Mental Health Chatbot...")
        self.chatbot = MentalHealthChatbot()
        self.session_start = datetime.now()
        self.message_count = 0
        
    def print_banner(self):
        """Display the chatbot banner."""
        print("=" * 60)
        print("🧠 Mental Health Chatbot - CLI Version (BERT-based)")
        print("=" * 60)
        print("Your AI companion for emotional support")
        print("Powered by BERT for intelligent intent recognition")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'help' for available commands")
        print("Type 'stats' to see session statistics")
        print("Type 'model' to see model information")
        print("=" * 60)
        print()
    
    def print_help(self):
        """Display help information."""
        print("\n📚 Available Commands:")
        print("  - Type your message to chat")
        print("  - 'quit', 'exit', 'bye' - End session")
        print("  - 'help' - Show this help")
        print("  - 'stats' - Show session statistics")
        print("  - 'model' - Show model information")
        print("  - 'clear' - Clear the screen")
        print("  - 'privacy' - Show privacy information")
        print("  - 'retrain' - Retrain the BERT model")
        print()
    
    def print_privacy(self):
        """Display privacy information."""
        print("\n🔒 Privacy Information:")
        print("  - No personal data is stored or logged")
        print("  - Only anonymized statistics are kept")
        print("  - Your messages are processed in memory only")
        print("  - BERT model is trained on anonymized intent data")
        print("  - Session data is cleared when you exit")
        print()
    
    def print_stats(self):
        """Display session statistics."""
        session_duration = datetime.now() - self.session_start
        print(f"\n📊 Session Statistics:")
        print(f"  - Messages exchanged: {self.message_count}")
        print(f"  - Session duration: {session_duration}")
        print(f"  - Started at: {self.session_start.strftime('%H:%M:%S')}")
        print()
    
    def print_model_info(self):
        """Display model information."""
        model_info = self.chatbot.get_model_info()
        print(f"\n🤖 Model Information:")
        print(f"  - Model path: {model_info['model_path']}")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Max sequence length: {model_info['max_length']}")
        print(f"  - Number of intents: {model_info['num_intents']}")
        print(f"  - Available intents: {', '.join(model_info['intents'][:10])}...")
        print()
    
    def format_response(self, result):
        """Format the chatbot response for CLI display."""
        # Convert numpy values to Python floats for display
        intent_confidence = float(result['intent_confidence']) if isinstance(result['intent_confidence'], (np.ndarray, np.integer, np.floating)) else result['intent_confidence']
        emotion_confidence = float(result['emotion_confidence']) if isinstance(result['emotion_confidence'], (np.ndarray, np.integer, np.floating)) else result['emotion_confidence']
        
        print(f"\n🤖 Bot: {result['response']}")
        print(f"   🎯 Intent: {result['intent']} ({intent_confidence:.1%} confidence)")
        print(f"   📊 Emotion: {result['emotion']} ({emotion_confidence:.1%} confidence)")
        print()
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
    
    def retrain_model(self):
        """Retrain the BERT model."""
        print("\n🔄 Retraining BERT model...")
        print("This may take a few minutes...")
        try:
            self.chatbot.retrain_model()
            print("✅ Model retraining completed successfully!")
        except Exception as e:
            print(f"❌ Error retraining model: {e}")
        print()
    
    def run(self):
        """Run the CLI chatbot."""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input("💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for chatting with me. Take care!")
                    self.print_stats()
                    break
                
                elif user_input.lower() == 'help':
                    self.print_help()
                    continue
                
                elif user_input.lower() == 'privacy':
                    self.print_privacy()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.print_stats()
                    continue
                
                elif user_input.lower() == 'model':
                    self.print_model_info()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_screen()
                    continue
                
                elif user_input.lower() == 'retrain':
                    self.retrain_model()
                    continue
                
                elif not user_input:
                    print("Please type a message or use 'help' for commands.")
                    continue
                
                # Process the message
                print("🤔 Processing with BERT model...")
                result = self.chatbot.process_message(user_input)
                
                # Display response
                self.format_response(result)
                
                # Update session stats
                self.message_count += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Take care!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'help' for assistance.")
                continue

def main():
    """Main entry point for the CLI chatbot."""
    try:
        cli = CLIChatbot()
        cli.run()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {str(e)}")
        print("Please check that all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 