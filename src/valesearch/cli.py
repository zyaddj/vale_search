#!/usr/bin/env python3
"""
ValeSearch CLI - Simple command-line interface for ValeSearch setup and management.
"""

import click
import sys
import subprocess
import os
from pathlib import Path

@click.group()
def main():
    """ValeSearch CLI - Hybrid cached retrieval for RAG systems."""
    click.echo("🚀 ValeSearch CLI - Supercharge your RAG with intelligent caching!")

@main.command()
def init():
    """Initialize ValeSearch in the current directory."""
    click.echo("🔧 Initializing ValeSearch project...")
    
    # Create .env configuration
    env_content = """# ValeSearch Configuration
REDIS_URL=redis://localhost:6379
CACHE_TTL=86400
ENABLE_SEMANTIC_CACHE=true
SEMANTIC_THRESHOLD=0.85
BM25_MIN_SCORE=0.1
LOG_LEVEL=INFO
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Create example integration file
    example_content = '''"""
ValeSearch Integration Example

This shows how to integrate ValeSearch with your existing RAG system.
"""

import asyncio
from valesearch import ValeSearch

# Your existing RAG function (replace this with your actual implementation)
def my_rag_function(query: str) -> str:
    """
    Replace this with your actual RAG implementation.
    
    Examples:
    - OpenAI API calls
    - Local LLM inference
    - Vector database queries
    - Existing retrieval pipelines
    """
    # This is just a placeholder - replace with your real RAG logic
    return f"Generated response for: {query}"

async def main():
    """Main function demonstrating ValeSearch usage."""
    
    # Initialize ValeSearch with your RAG function
    print("🚀 Initializing ValeSearch...")
    vale = ValeSearch(fallback_function=my_rag_function)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does neural network training work?", 
        "What is machine learning?",  # Repeat to show caching
    ]
    
    print("\\n🔍 Testing ValeSearch performance:")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\nQuery {i}: {query}")
        
        result = await vale.search(query)
        
        print(f"Source: {result.source}")
        print(f"Latency: {result.latency_ms:.1f}ms")
        print(f"Cached: {result.cached}")
        print(f"Answer: {result.answer[:100]}...")
    
    # Show statistics
    stats = vale.get_stats()
    print(f"\\n📊 Performance Statistics:")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache hit rate: {stats['cache_hits']/max(stats['total_queries'], 1)*100:.1f}%")
    print(f"Average latency: {stats['total_latency']/max(stats['total_queries'], 1):.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('valesearch_example.py', 'w') as f:
        f.write(example_content)
    
    # Create docker-compose for Redis (optional)
    docker_compose = """version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    
    click.echo("✅ ValeSearch project initialized!")
    click.echo("")
    click.echo("📁 Created files:")
    click.echo("   .env                    - Configuration")
    click.echo("   valesearch_example.py   - Integration example")
    click.echo("   docker-compose.yml      - Redis setup (optional)")
    click.echo("")
    click.echo("🚀 Next steps:")
    click.echo("   1. Start Redis: docker-compose up -d")
    click.echo("   2. Run example: python valesearch_example.py")
    click.echo("   3. Integrate with your RAG system!")

@main.command()
def start():
    """Start Redis using Docker (convenience command)."""
    click.echo("🐳 Starting Redis with Docker...")
    
    if not Path("docker-compose.yml").exists():
        click.echo("❌ No docker-compose.yml found. Run 'valesearch init' first.")
        return
    
    try:
        result = subprocess.run(["docker-compose", "up", "-d"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("✅ Redis started successfully!")
            click.echo("📍 Available at: redis://localhost:6379")
        else:
            click.echo(f"❌ Failed to start Redis: {result.stderr}")
    except FileNotFoundError:
        click.echo("❌ Docker Compose not found. Please install Docker Desktop.")
        click.echo("   https://docs.docker.com/get-docker/")

@main.command()
def stop():
    """Stop Redis Docker container."""
    click.echo("🛑 Stopping Redis...")
    
    try:
        result = subprocess.run(["docker-compose", "down"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("✅ Redis stopped successfully!")
        else:
            click.echo(f"❌ Failed to stop Redis: {result.stderr}")
    except FileNotFoundError:
        click.echo("❌ Docker Compose not found.")

@main.command()
def test():
    """Run ValeSearch integration test."""
    click.echo("🧪 Running ValeSearch integration test...")
    
    if Path("valesearch_example.py").exists():
        try:
            result = subprocess.run([sys.executable, "valesearch_example.py"])
            if result.returncode == 0:
                click.echo("✅ Integration test completed!")
            else:
                click.echo("❌ Integration test failed.")
        except Exception as e:
            click.echo(f"❌ Error running test: {e}")
    else:
        click.echo("❌ No valesearch_example.py found. Run 'valesearch init' first.")

@main.command()
def status():
    """Check ValeSearch system status."""
    click.echo("📊 ValeSearch System Status")
    click.echo("-" * 30)
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        click.echo("✅ Redis: Connected")
    except Exception as e:
        click.echo("❌ Redis: Not available")
        click.echo(f"   Error: {e}")
    
    # Check ValeSearch installation
    try:
        import valesearch
        click.echo("✅ ValeSearch: Installed")
        click.echo(f"   Version: {valesearch.__version__}")
    except ImportError:
        click.echo("❌ ValeSearch: Not properly installed")
    
    # Check example files
    if Path("valesearch_example.py").exists():
        click.echo("✅ Example: Available")
    else:
        click.echo("⚠️  Example: Run 'valesearch init' to create")

if __name__ == "__main__":
    main()