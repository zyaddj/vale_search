"""
ValeSearch Quick Start Example

This example shows how to integrate ValeSearch with your RAG system in 3 simple steps.
"""

import asyncio
from valesearch import ValeSearch


def my_rag_function(query: str) -> str:
    """
    Replace this with your actual RAG implementation.
    
    Examples:
    - Call OpenAI API
    - Query your vector database
    - Search your documents
    - Call your existing RAG pipeline
    """
    # This is just a placeholder - replace with your real RAG logic
    return f"RAG response for: {query}"


async def main():
    """Demonstrate ValeSearch integration."""
    
    print("🚀 ValeSearch Integration Example")
    print("=" * 40)
    
    # Step 1: Initialize ValeSearch with your RAG function
    vale = ValeSearch(fallback_function=my_rag_function)
    
    # Step 2: Use ValeSearch for queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is machine learning?",  # This will be cached!
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        result = await vale.search(query)
        
        print(f"  Answer: {result.answer}")
        print(f"  Source: {result.source}")
        print(f"  Cached: {result.cached}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
    
    # Step 3: View performance statistics
    stats = vale.get_stats()
    print(f"\n📊 Performance Stats:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Average latency: {stats['average_latency_ms']:.1f}ms")


if __name__ == "__main__":
    # Make sure Redis is running: brew services start redis
    asyncio.run(main())