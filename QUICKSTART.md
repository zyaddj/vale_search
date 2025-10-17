# ValeSearch Quick Start Guide

## Overview

ValeSearch is a plug-and-play intelligence layer that optimally routes queries through:
1. **Cache** (instant responses for repeated queries)
2. **BM25** (fast keyword search for factual queries) 
3. **Your existing RAG system** (fallback for complex queries)

This means you don't need to replace your current RAG infrastructure - ValeSearch enhances it by handling the easy queries efficiently and only passing complex ones to your RAG system.

## Installation

```bash
pip install vale-search
# OR for development:
pip install -e .
```

## Basic Integration

### 1. Function Callback (Simplest)

```python
import asyncio
from vale_search import ValeSearch, FallbackResult

async def my_rag_callback(query: str, context: dict) -> FallbackResult:
    """Your existing RAG system wrapped as a callback"""
    # Call your existing RAG system here
    response = await your_rag_system.query(query)
    
    return FallbackResult(
        answer=response.answer,
        confidence=response.confidence,
        latency_ms=response.processing_time,
        metadata=response.metadata
    )

# Initialize ValeSearch
vale = ValeSearch(
    fallback_function=my_rag_callback,
    cache_backend="memory",  # or "redis" for production
)

# Use it!
result = await vale.search("What is machine learning?")
print(f"Answer: {result.answer}")
print(f"Source: {result.source}")  # "cache", "bm25", or "user_rag"
print(f"Cached: {result.cached}")
```

### 2. Webhook Integration

```python
vale = ValeSearch(
    fallback_webhook={
        "url": "https://your-rag-api.com/query",
        "headers": {"Authorization": "Bearer your-key"},
        "timeout": 30
    }
)
```

### 3. SDK Integration

```python
vale = ValeSearch(
    fallback_sdk={
        "instance": your_rag_sdk,
        "method": "search",
        "timeout": 30
    }
)
```

## Configuration

```python
vale = ValeSearch(
    # Cache settings
    cache_backend="redis",
    cache_url="redis://localhost:6379/0",
    cache_ttl=3600,  # 1 hour
    quality_threshold=0.7,  # Only cache high-quality responses
    
    # BM25 settings  
    bm25_min_score=0.5,  # Minimum score to trust BM25
    short_query_max_words=3,  # Words or less = try BM25 first
    
    # Fallback settings
    fallback_timeout=30,
    fallback_retries=2,
    
    # Your RAG integration
    fallback_function=my_rag_callback
)
```

## How It Works

```
User Query → ValeSearch Intelligence Layer
    ↓
1. Check Cache (instant if hit)
    ↓ (cache miss)
2. Try BM25 (fast keyword search)
    ↓ (if low confidence or miss)
3. Fallback to Your RAG System
    ↓
Cache the good results for next time
```

## Benefits

- **🚀 Speed**: Cache hits return instantly, BM25 is ~10x faster than vector search
- **💰 Cost**: Reduce LLM API calls by 60-80% with intelligent caching
- **🔌 Plug-and-Play**: No need to replace your existing RAG system
- **📊 Smart**: Learns which queries work best with which method
- **🛡️ Reliable**: Quality gates prevent caching poor responses

## Monitoring

```python
# Get performance statistics
stats = vale.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"BM25 success rate: {stats['bm25_success_rate']:.1%}")
print(f"Fallback rate: {stats['fallback_rate']:.1%}")
print(f"Average latency: {stats['average_latency_ms']:.1f}ms")

# Health check
health = await vale.health_check()
print(f"Status: {health['status']}")
```

## Production Setup

### Redis Cache (Recommended)
```python
vale = ValeSearch(
    cache_backend="redis",
    cache_url="redis://localhost:6379/0",
    cache_ttl=3600,
    max_cache_size=100000,
    quality_threshold=0.8,  # Higher threshold for production
    fallback_function=production_rag_callback
)
```

### Error Handling
```python
async def robust_rag_callback(query: str, context: dict) -> FallbackResult:
    try:
        response = await your_rag_system.query(query)
        return FallbackResult(
            answer=response.answer,
            confidence=response.confidence,
            latency_ms=response.time,
            metadata=response.meta
        )
    except Exception as e:
        # Graceful degradation
        return FallbackResult(
            answer="I'm having trouble finding that information. Please try again later.",
            confidence=0.1,
            latency_ms=0,
            metadata={"error": str(e), "fallback": True}
        )
```

## Next Steps

1. **Try the examples**: Run `python examples/integration_examples.py`
2. **Configure your cache**: Set up Redis for production use
3. **Tune parameters**: Adjust BM25 thresholds and quality gates based on your data
4. **Monitor performance**: Use the stats API to optimize your setup
5. **Scale up**: ValeSearch handles high-throughput production workloads

## API Reference

See the full API documentation for advanced configuration options, custom quality gates, and performance tuning.