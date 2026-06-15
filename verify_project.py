"""
Deepfake Audio Detection - Project Verification Script
Checks if all required files and dependencies are in place.

Usage:
    python verify_project.py
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False


def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).exists():
        files = list(Path(dirpath).glob("*"))
        print(f"✅ {description}: {dirpath} ({len(files)} files)")
        return True
    else:
        print(f"❌ {description}: {dirpath} - NOT FOUND")
        return False


def check_imports():
    """Check if required packages can be imported."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = [
        ("numpy", "NumPy"),
        ("librosa", "Librosa"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("soundfile", "SoundFile"),
        ("joblib", "Joblib"),
        ("pandas", "Pandas"),
        ("xgboost", "XGBoost"),
    ]
    
    optional_packages = [
        ("streamlit", "Streamlit"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed (required)")
            all_good = False
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"⚠️  {name} is NOT installed (optional)")
    
    return all_good


def main():
    """Main verification function."""
    print("="*60)
    print("DEEPFAKE AUDIO DETECTION - PROJECT VERIFICATION")
    print("="*60)
    
    # Get project root
    project_root = Path(__file__).parent
    
    print(f"\nProject Root: {project_root}")
    
    # Check core files
    print("\n" + "="*60)
    print("CHECKING CORE FILES")
    print("="*60)
    
    core_files = [
        ("notebooks/deepfake_detection.ipynb", "Main Notebook"),
        ("app.py", "Streamlit App"),
        ("test_audio.py", "Test Script"),
        ("create_test_samples.py", "Test Sample Generator"),
        ("src/__init__.py", "Source Package Init"),
        ("src/utils.py", "Utilities Module"),
    ]
    
    files_ok = True
    for filepath, description in core_files:
        full_path = project_root / filepath
        if not check_file_exists(full_path, description):
            files_ok = False
    
    # Check documentation
    print("\n" + "="*60)
    print("CHECKING DOCUMENTATION")
    print("="*60)
    
    doc_files = [
        ("README.md", "README"),
        ("PERFORMANCE_REPORT.md", "Performance Report"),
        ("LICENSE", "License"),
        ("PROJECT_SUMMARY.md", "Project Summary"),
    ]
    
    for filepath, description in doc_files:
        full_path = project_root / filepath
        check_file_exists(full_path, description)
    
    # Check configuration
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    config_files = [
        ("requirements.txt", "Requirements"),
        ("requirements_streamlit.txt", "Streamlit Requirements"),
        (".streamlit/config.toml", "Streamlit Config"),
        ("setup.py", "Setup Script"),
        (".gitignore", "Git Ignore"),
    ]
    
    for filepath, description in config_files:
        full_path = project_root / filepath
        check_file_exists(full_path, description)
    
    # Check directories
    print("\n" + "="*60)
    print("CHECKING DIRECTORIES")
    print("="*60)
    
    directories = [
        ("notebooks", "Notebooks Directory"),
        ("models", "Models Directory"),
        ("src", "Source Directory"),
        ("figures", "Figures Directory"),
        ("test_samples", "Test Samples Directory"),
        (".streamlit", "Streamlit Config Directory"),
    ]
    
    for dirpath, description in directories:
        full_path = project_root / dirpath
        check_directory_exists(full_path, description)
    
    # Check source modules
    print("\n" + "="*60)
    print("CHECKING SOURCE MODULES")
    print("="*60)
    
    src_files = [
        ("src/data_preprocessing.py", "Data Preprocessing"),
        ("src/feature_extraction.py", "Feature Extraction"),
        ("src/model.py", "Model Definitions"),
        ("src/train.py", "Training Pipeline"),
        ("src/evaluate.py", "Evaluation Module"),
        ("src/utils.py", "Utilities"),
    ]
    
    for filepath, description in src_files:
        full_path = project_root / filepath
        check_file_exists(full_path, description)
    
    # Check imports
    imports_ok = check_imports()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if files_ok and imports_ok:
        print("\n🎉 PROJECT VERIFICATION: PASSED")
        print("\nAll core files and dependencies are in place!")
        print("\nNext steps:")
        print("  1. Download dataset from Kaggle")
        print("  2. Run notebook to train model")
        print("  3. Test with sample audio")
        print("  4. Deploy Streamlit app")
        print("  5. Record demo video")
    else:
        print("\n⚠️  PROJECT VERIFICATION: ISSUES FOUND")
        print("\nPlease check the missing files/dependencies above.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
