"""
Deepfake Audio Detection - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="deepfake-audio-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning system for detecting deepfake audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepfake-audio-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepfake-detect=test_audio:main",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/deepfake-audio-detection/issues",
        "Documentation": "https://github.com/yourusername/deepfake-audio-detection#readme",
        "Source Code": "https://github.com/yourusername/deepfake-audio-detection",
    },
)
