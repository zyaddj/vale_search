# ValeSearch - Setup Configuration for True Drag-and-Drop Installation

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vale-search",
    version="0.1.0",
    author="Vale Systems",
    author_email="opensource@valesystems.ai",
    description="The hybrid, cached retrieval engine for RAG systems - true drag-and-drop integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zyaddj/vale_search",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0"
        ],
        "redis": ["redis>=5.0.1"],
        "production": [
            "redis>=5.0.1",
            "gunicorn>=21.2.0",
            "uvicorn[standard]>=0.24.0"
        ]
    },
    
    # Entry points for CLI tools (if needed)
    entry_points={
        "console_scripts": [
            "vale-search=vale_search.cli:main",  # Future CLI tool
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    
    # Keywords for discovery
    keywords=[
        "rag", "retrieval", "cache", "ai", "ml", "nlp", 
        "search", "hybrid", "bm25", "vector", "semantic",
        "llm", "chatgpt", "openai", "fastapi", "redis"
    ],
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "vale_search": ["data/*.json", "configs/*.yaml"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/zyaddj/vale_search/issues",
        "Documentation": "https://vale-search.readthedocs.io/",
        "Source": "https://github.com/zyaddj/vale_search",
        "Changelog": "https://github.com/zyaddj/vale_search/blob/main/CHANGELOG.md",
    },
)