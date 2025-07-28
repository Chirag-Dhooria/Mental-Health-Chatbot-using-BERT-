#!/usr/bin/env python3
"""
Test script to verify scikit-learn installation and functionality.
"""

def test_scikit_learn():
    """Test scikit-learn installation and basic functionality."""
    print("🧪 Testing scikit-learn installation...")
    
    try:
        # Test basic import
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        # Test version
        print(f"📦 scikit-learn version: {sklearn.__version__}")
        
        # Test basic functionality
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ TfidfVectorizer imported successfully")
        
        # Test simple vectorization
        texts = ["hello world", "test message", "another test"]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        print(f"✅ Vectorization test passed - Shape: {X.shape}")
        
        print("🎉 All scikit-learn tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ scikit-learn import error: {e}")
        return False
    except Exception as e:
        print(f"❌ scikit-learn functionality error: {e}")
        return False

def test_all_ml_dependencies():
    """Test all ML-related dependencies."""
    print("\n🔍 Testing all ML dependencies...")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers')
    ]
    
    all_good = True
    
    for import_name, package_name in dependencies:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name} - Version: {version}")
        except ImportError as e:
            print(f"❌ {package_name} - MISSING: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {package_name} - ISSUE: {e}")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🧠 Mental Health Chatbot - ML Dependencies Test")
    print("=" * 50)
    
    # Test scikit-learn specifically
    sklearn_ok = test_scikit_learn()
    
    # Test all ML dependencies
    all_ok = test_all_ml_dependencies()
    
    print("\n" + "=" * 50)
    if sklearn_ok and all_ok:
        print("🎉 All ML dependencies are working correctly!")
        print("🚀 You can now run the chatbot successfully.")
    else:
        print("❌ Some ML dependencies have issues.")
        print("💡 Try running: python fix_dependencies.py")
        print("💡 Or manually install: pip install scikit-learn numpy pandas --force-reinstall") 