#!/usr/bin/env python3
"""
Mental Health Chatbot - Launcher Script
Provides easy access to different interfaces and testing options.
"""

import sys
import os
import subprocess
import argparse

def run_web_server():
    """Start the Flask web server."""
    print("🌐 Starting Mental Health Chatbot Web Server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def run_cli():
    """Start the CLI chatbot."""
    print("💻 Starting Mental Health Chatbot CLI...")
    print("💬 Type your messages to chat")
    print("🛑 Type 'quit' to exit")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "cli_chatbot.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 CLI stopped.")
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")

def run_tests():
    """Run the test suite."""
    print("🧪 Running Mental Health Chatbot Tests...")
    print("-" * 50)
    
    try:
        # Run unit tests
        print("📋 Running unit tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_chatbot.py", "-v"], check=True)
        
        print("\n📋 Running API tests...")
        print("⚠️  Note: API tests require the server to be running")
        print("   Start the server in another terminal: python app.py")
        
        # Ask if user wants to run API tests
        response = input("\n🤔 Do you want to run API tests? (y/n): ").lower()
        if response in ['y', 'yes']:
            subprocess.run([sys.executable, "-m", "pytest", "tests/test_api.py", "-v"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
    except Exception as e:
        print(f"❌ Error running tests: {e}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('flask', 'Flask'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('sklearn', 'scikit-learn'),  # Changed from scikit-learn to sklearn
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('nltk', 'nltk'),
        ('pytest', 'pytest'),
        ('requests', 'requests'),
        ('tokenizers', 'tokenizers'),
        ('sentencepiece', 'sentencepiece')
    ]
    
    missing_packages = []
    version_issues = []
    
    for import_name, package_name in required_packages:
        try:
            # Try to import the package
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)
        except Exception as e:
            print(f"⚠️  {package_name} - VERSION ISSUE: {str(e)}")
            version_issues.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    if version_issues:
        print(f"\n⚠️  Version issues with: {', '.join(version_issues)}")
        print("📦 Try reinstalling: pip install -r requirements.txt --upgrade")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def show_help():
    """Show help information."""
    print("🧠 Mental Health Chatbot - Launcher")
    print("=" * 50)
    print("Usage: python run.py [option]")
    print()
    print("Options:")
    print("  web     - Start web server (default)")
    print("  cli     - Start command-line interface")
    print("  test    - Run test suite")
    print("  check   - Check dependencies")
    print("  help    - Show this help")
    print()
    print("Examples:")
    print("  python run.py web")
    print("  python run.py cli")
    print("  python run.py test")
    print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mental Health Chatbot Launcher")
    parser.add_argument('mode', nargs='?', default='web', 
                       choices=['web', 'cli', 'test', 'check', 'help'],
                       help='Mode to run (default: web)')
    
    args = parser.parse_args()
    
    if args.mode == 'help':
        show_help()
        return
    
    print("🧠 Mental Health Chatbot")
    print("=" * 50)
    
    if args.mode == 'check':
        check_dependencies()
        return
    
    # Check dependencies before running
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        print("💡 Try running: setup_env.bat (Windows) or ./setup_env.sh (Linux/Mac)")
        return
    
    # Run the selected mode
    if args.mode == 'web':
        run_web_server()
    elif args.mode == 'cli':
        run_cli()
    elif args.mode == 'test':
        run_tests()
    else:
        show_help()

if __name__ == "__main__":
    main() 