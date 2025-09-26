#!/usr/bin/env python3
"""
Diagnostic Test Script for Gene Pair ML Analysis Web App
This script will help identify what's missing and what needs to be fixed.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_flask_installation():
    """Check if Flask is installed."""
    print("\n🔍 Checking Flask installation...")
    try:
        import flask
        print(f"✅ Flask version {flask.__version__} is installed")
        return True
    except ImportError:
        print("❌ Flask is not installed")
        print("💡 Install with: pip install flask")
        return False

def check_required_modules():
    """Check for required Python modules."""
    print("\n🔍 Checking required modules...")
    
    required_modules = [
        'pandas',
        'numpy', 
        'plotly',
        'openpyxl'  # For Excel export
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} is installed")
        except ImportError:
            print(f"❌ {module} is missing")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n💡 Install missing modules with:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True

def check_project_structure():
    """Check if project structure is complete."""
    print("\n🔍 Checking project structure...")
    
    # Check for required directories
    required_dirs = [
        'templates',
        'static',
        'uploads', 
        'results'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory is missing")
            missing_dirs.append(dir_name)
    
    # Check for core modules
    core_modules = [
        'gene_pair_agent',
        'visualization'
    ]
    
    for module in core_modules:
        if os.path.exists(module) or os.path.exists(f"{module}.py"):
            print(f"✅ {module} module found")
        else:
            print(f"❌ {module} module is missing")
            missing_dirs.append(module)
    
    return len(missing_dirs) == 0

def test_basic_imports():
    """Test basic imports from app.py."""
    print("\n🔍 Testing basic imports...")
    
    # Read app.py and extract imports
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Extract import statements
        import_lines = []
        for line in content.split('\n'):
            if line.strip().startswith('import') or line.strip().startswith('from'):
                import_lines.append(line.strip())
        
        print("Found imports in app.py:")
        for imp in import_lines[:10]:  # Show first 10 imports
            print(f"  {imp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def check_file_permissions():
    """Check file and directory permissions."""
    print("\n🔍 Checking file permissions...")
    
    # Check if we can create directories
    try:
        test_dir = Path("test_permissions")
        test_dir.mkdir(exist_ok=True)
        test_dir.rmdir()
        print("✅ Can create directories")
    except Exception as e:
        print(f"❌ Cannot create directories: {e}")
        return False
    
    # Check if we can write files
    try:
        test_file = Path("test_write.txt")
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Can write files")
        return True
    except Exception as e:
        print(f"❌ Cannot write files: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    print("🚀 Gene Pair ML Analysis - Diagnostic Test")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Flask Installation", check_flask_installation), 
        ("Required Modules", check_required_modules),
        ("Project Structure", check_project_structure),
        ("Basic Imports", test_basic_imports),
        ("File Permissions", check_file_permissions)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            results.append((check_name, False))
        print()
    
    # Summary
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed! The application should work.")
    else:
        print("⚠️  Some issues found. Please fix the failed checks above.")
        
        print("\n🔧 RECOMMENDED FIXES:")
        for check_name, result in results:
            if not result:
                if check_name == "Flask Installation":
                    print("- Install Flask: pip install flask")
                elif check_name == "Required Modules":
                    print("- Install missing Python modules")
                elif check_name == "Project Structure":
                    print("- Create missing directories and files")
                elif check_name == "File Permissions":
                    print("- Check directory write permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)