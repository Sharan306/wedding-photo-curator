#!/usr/bin/env python3
"""
Setup script for Wedding Photo Curator.

This script allows installation using: pip install -e .
Or building distributions: python setup.py sdist bdist_wheel
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="wedding-photo-curator",
    version="1.0.0",
    author="Wedding Photo Curator Contributors",
    description="Quality-first photo curation tool using hard rejection rules and diversity filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sharan306/wedding-photo-curator",
    project_urls={
        "Bug Tracker": "https://github.com/Sharan306/wedding-photo-curator/issues",
        "Documentation": "https://github.com/Sharan306/wedding-photo-curator/blob/main/README.md",
        "Source Code": "https://github.com/Sharan306/wedding-photo-curator",
    },
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wedding-photo-curator=analyze_photos:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: Viewers",
    ],
    keywords=[
        "photo-curation",
        "wedding-photography",
        "image-analysis",
        "computer-vision",
        "opencv",
    ],
    license="MIT",
)
