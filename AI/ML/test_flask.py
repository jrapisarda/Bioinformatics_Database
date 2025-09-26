#!/usr/bin/env python3
"""
Test Flask Application

Simple test to verify the Flask app is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_flask_app():
    """Test Flask application initialization."""
    try:
        # Import the Flask app
        sys.path.insert(0, str(Path(__file__).parent / 'web_interface'))
        from app import app
        
        print("✅ Flask app imported successfully")
        
        # Test app configuration
        print(f"✅ Upload folder: {app.config['UPLOAD_FOLDER']}")
        print(f"✅ Max file size: {app.config['MAX_CONTENT_LENGTH']}")
        
        # Test if we can create a test client
        with app.test_client() as client:
            print("✅ Test client created successfully")
            
            # Test a simple route
            response = client.get('/')
            print(f"✅ Index route response: {response.status_code}")
            
            return True
            
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_templates():
    """Test if templates are accessible."""
    try:
        from jinja2 import Environment, FileSystemLoader
        
        template_dir = Path(__file__).parent / 'web_interface' / 'templates'
        
        if not template_dir.exists():
            print(f"❌ Template directory not found: {template_dir}")
            return False
            
        print(f"✅ Template directory found: {template_dir}")
        
        # List available templates
        templates = list(template_dir.glob('*.html'))
        print(f"✅ Found {len(templates)} templates:")
        for template in templates:
            print(f"   - {template.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Flask Application...")
    print("=" * 50)
    
    tests = [
        ("Flask App", test_flask_app),
        ("Templates", test_templates)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            print(f"{test_name:15} [{status}]")
        except Exception as e:
            print(f"{test_name:15} [ERROR: {e}]")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Flask application is ready to run!")
        print("Start the server with: python web_interface/app.py")
    else:
        print("\n❌ Please fix the issues above before running the server.")

if __name__ == '__main__':
    main()