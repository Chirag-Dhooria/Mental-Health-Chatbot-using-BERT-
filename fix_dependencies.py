#!/usr/bin/env python3
"""
Dependency Fix Script for Mental Health Chatbot
Fixes common dependency issues, especially scikit-learn problems.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_scikit_learn():
    """Check if scikit-learn is properly installed."""
    try:
        import sklearn
        print("✅ scikit-learn is properly installed")
        return True
    except ImportError as e:
        print(f"❌ scikit-learn import error: {e}")
        return False

def fix_scikit_learn():
    """Fix scikit-learn installation issues."""
    print("\n🔧 Fixing scikit-learn issues...")
    
    # Try different installation methods
    methods = [
        ("pip install scikit-learn", "Installing scikit-learn"),
        ("pip install --upgrade scikit-learn", "Upgrading scikit-learn"),
        ("pip install scikit-learn --force-reinstall", "Force reinstalling scikit-learn"),
        ("pip install scikit-learn==1.3.0", "Installing specific scikit-learn version"),
    ]
    
    for command, description in methods:
        if run_command(command, description):
            if check_scikit_learn():
                print("✅ scikit-learn is now working!")
                return True
    
    return False

def fix_all_dependencies():
    """Fix all dependency issues."""
    print("🔧 Fixing all dependencies...")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install/upgrade all requirements
    run_command("pip install -r requirements.txt --upgrade", "Upgrading all dependencies")
    
    # Force reinstall problematic packages
    problematic_packages = [
        "scikit-learn",
        "numpy", 
        "pandas",
        "transformers",
        "torch"
    ]
    
    for package in problematic_packages:
        run_command(f"pip install {package} --force-reinstall", f"Force reinstalling {package}")
    
    return True

def check_all_dependencies():
    """Check all dependencies."""
    print("\n🔍 Checking all dependencies...")
    
    dependencies = [
        ('flask', 'Flask'),
        ('transformers', 'Transformers'),
        ('torch', 'PyTorch'),
        ('sklearn', 'scikit-learn'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('nltk', 'NLTK'),
        ('pytest', 'PyTest'),
        ('requests', 'Requests'),
        ('tokenizers', 'Tokenizers'),
        ('sentencepiece', 'SentencePiece')
    ]
    
    all_good = True
    
    for import_name, package_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name} - MISSING: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {package_name} - ISSUE: {e}")
            all_good = False
    
    return all_good

def main():
    """Main function."""
    print("🧠 Mental Health Chatbot - Dependency Fixer")
    print("=" * 50)
    
    # Check current status
    print("📊 Current dependency status:")
    current_status = check_all_dependencies()
    
    if current_status:
        print("\n✅ All dependencies are working correctly!")
        return
    
    print("\n🔧 Attempting to fix dependencies...")
    
    # Try to fix scikit-learn specifically
    if not check_scikit_learn():
        print("\n🎯 scikit-learn needs fixing...")
        if fix_scikit_learn():
            print("✅ scikit-learn fixed!")
        else:
            print("❌ Could not fix scikit-learn automatically")
    
    # Fix all dependencies
    print("\n🔧 Fixing all dependencies...")
    fix_all_dependencies()
    
    # Check final status
    print("\n📊 Final dependency status:")
    final_status = check_all_dependencies()
    
    if final_status:
        print("\n🎉 All dependencies are now working!")
        print("🚀 You can now run: python run.py web")
    else:
        print("\n❌ Some dependencies still have issues.")
        print("💡 Try manually installing problematic packages:")
        print("   pip install scikit-learn numpy pandas --force-reinstall")

if __name__ == "__main__":
    main() 