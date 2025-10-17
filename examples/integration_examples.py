"""
ValeSearch Integration Examples

This file demonstrates how to integrate ValeSearch with your existing RAG systems
using the plug-and-play fallback integration patterns.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

# For actual usage, install ValeSearch with: pip install vale-search
# For this example, we'll import from the local source
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cache.cache_manager import CacheManager
from retrieval.hybrid_engine import HybridEngine
from retrieval.fallback_integration import FallbackIntegration, FallbackResult


@dataclass
class ExampleRAGResponse:
    """Example structure for your existing RAG system response"""
    answer: str
    confidence: float
    sources: list
    metadata: Dict[str, Any]


class ExampleRAGSystem:
    """
    This represents your existing RAG system that ValeSearch will fallback to.
    Replace this with your actual RAG implementation.
    """
    
    def __init__(self):
        # Your RAG system initialization here
        self.vector_db = "your_vector_database"
        self.embedding_model = "your_embedding_model"
        self.llm = "your_llm_model"
    
    async def query(self, question: str, context: Dict[str, Any] = None) -> ExampleRAGResponse:
        """
        Your existing RAG query method.
        ValeSearch will call this when cache and BM25 don't have good answers.
        """
        # Simulate your RAG processing
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Your actual implementation would:
        # 1. Generate embeddings for the question
        # 2. Search your vector database
        # 3. Retrieve relevant documents
        # 4. Generate answer using your LLM
        
        return ExampleRAGResponse(
            answer=f"This is a response from your RAG system for: {question}",
            confidence=0.85,
            sources=["doc1.pdf", "doc2.pdf"],
            metadata={
                "processing_time_ms": 100,
                "model_used": self.llm,
                "retrieved_docs": 3
            }
        )


# ==========================================
# INTEGRATION PATTERN 1: FUNCTION CALLBACK
# ==========================================

async def rag_function_callback(query: str, context: Dict[str, Any]) -> Optional[FallbackResult]:
    """
    Function callback pattern - simplest integration method.
    
    This function gets called when ValeSearch needs to fallback to your RAG system.
    """
    try:
        # Initialize your RAG system
        rag_system = ExampleRAGSystem()
        
        # Call your RAG system
        rag_response = await rag_system.query(query, context)
        
        # Convert to ValeSearch format
        return FallbackResult(
            answer=rag_response.answer,
            confidence=rag_response.confidence,
            latency_ms=rag_response.metadata.get("processing_time_ms", 0),
            metadata={
                "integration_type": "function_callback",
                "sources": rag_response.sources,
                **rag_response.metadata
            }
        )
        
    except Exception as e:
        print(f"RAG callback error: {e}")
        return None


async def example_function_integration():
    """Example of using ValeSearch with function callback integration."""
    print("=== Function Callback Integration Example ===")
    
    # Configure ValeSearch with your RAG function
    fallback_integration = FallbackIntegration(
        integration_type="function",
        callback_function=rag_function_callback,
        timeout_seconds=30
    )
    
    # Initialize ValeSearch components
    cache_manager = CacheManager()
    
    # Create hybrid engine with fallback to your RAG system
    engine = HybridEngine(
        cache_manager=cache_manager,
        fallback_integration=fallback_integration,
        bm25_min_score=0.5,
        short_query_max_words=3
    )
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does neural network training work?",
        "Explain gradient descent"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await engine.search(query)
        print(f"Answer: {result.answer[:100]}...")
        print(f"Source: {result.source}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Cached: {result.cached}")
    
    # Show statistics
    stats = engine.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")


# ==========================================
# INTEGRATION PATTERN 2: WEBHOOK ENDPOINT
# ==========================================

async def example_webhook_integration():
    """Example of using ValeSearch with webhook integration."""
    print("\n=== Webhook Integration Example ===")
    
    # Your RAG system webhook endpoint
    webhook_config = {
        "url": "https://your-rag-system.com/api/query",
        "method": "POST",
        "headers": {
            "Authorization": "Bearer your-api-key",
            "Content-Type": "application/json"
        },
        "timeout": 30,
        "retry_attempts": 2
    }
    
    # Configure ValeSearch with webhook integration
    fallback_integration = FallbackIntegration(
        integration_type="webhook",
        webhook_config=webhook_config
    )
    
    # Initialize ValeSearch (would use the same pattern as above)
    print("Webhook integration configured. In production, this would call:")
    print(f"POST {webhook_config['url']}")
    print("With payload: {'query': 'user_question', 'context': {...}}")


# ==========================================
# INTEGRATION PATTERN 3: SDK INTEGRATION
# ==========================================

class YourRAGSDK:
    """Example SDK for your RAG system"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Your SDK's search method"""
        # Simulate SDK call
        await asyncio.sleep(0.1)
        
        return {
            "answer": f"SDK response for: {query}",
            "confidence": 0.88,
            "metadata": {
                "model": "your-model-v2",
                "tokens_used": 150
            }
        }


async def example_sdk_integration():
    """Example of using ValeSearch with SDK integration."""
    print("\n=== SDK Integration Example ===")
    
    # Initialize your RAG SDK
    rag_sdk = YourRAGSDK(
        api_key="your-api-key",
        base_url="https://api.your-rag-system.com"
    )
    
    # Configure ValeSearch with SDK integration
    fallback_integration = FallbackIntegration(
        integration_type="sdk",
        sdk_instance=rag_sdk,
        sdk_method="search",  # Method name to call on your SDK
        timeout_seconds=30
    )
    
    # Initialize ValeSearch (same pattern as function example)
    print("SDK integration configured. ValeSearch will call:")
    print("rag_sdk.search(query, context=context)")


# ==========================================
# COMPREHENSIVE PRODUCTION EXAMPLE
# ==========================================

async def production_example():
    """
    Production-ready example showing full ValeSearch integration
    with monitoring, error handling, and optimization.
    """
    print("\n=== Production Integration Example ===")
    
    # Configure cache with Redis for production
    cache_config = {
        "backend": "redis",
        "redis_url": "redis://localhost:6379/0",
        "max_cache_size": 10000,
        "default_ttl": 3600,  # 1 hour
        "quality_threshold": 0.7
    }
    
    # Production-ready fallback integration with retries and monitoring
    async def production_rag_callback(query: str, context: Dict[str, Any]) -> Optional[FallbackResult]:
        """Production RAG callback with error handling and monitoring"""
        import time
        start_time = time.time()
        
        try:
            # Your production RAG system
            rag_system = ExampleRAGSystem()
            
            # Add context about ValeSearch's previous attempts
            enhanced_context = {
                **context,
                "vale_search_attempt": True,
                "cache_miss": context.get("cache_miss", False),
                "bm25_attempted": context.get("bm25_attempted", False)
            }
            
            rag_response = await rag_system.query(query, enhanced_context)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log for monitoring
            print(f"RAG fallback successful: {query[:50]}... ({latency_ms:.1f}ms)")
            
            return FallbackResult(
                answer=rag_response.answer,
                confidence=rag_response.confidence,
                latency_ms=latency_ms,
                metadata={
                    "integration_type": "production_callback",
                    "sources": rag_response.sources,
                    "enhanced_context": enhanced_context,
                    **rag_response.metadata
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"RAG fallback error: {e} ({latency_ms:.1f}ms)")
            
            # Return a graceful degradation response
            return FallbackResult(
                answer="I'm sorry, I'm having trouble accessing that information right now. Please try again later.",
                confidence=0.1,
                latency_ms=latency_ms,
                metadata={
                    "integration_type": "production_callback",
                    "error": str(e),
                    "fallback_response": True
                }
            )
    
    # Configure production ValeSearch
    fallback_integration = FallbackIntegration(
        integration_type="function",
        callback_function=production_rag_callback,
        timeout_seconds=30,
        enable_retries=True,
        max_retries=2
    )
    
    cache_manager = CacheManager(cache_config)
    
    engine = HybridEngine(
        cache_manager=cache_manager,
        fallback_integration=fallback_integration,
        bm25_min_score=0.6,  # Higher threshold for production
        short_query_max_words=4
    )
    
    # Production query examples
    queries = [
        "What are the latest features in our product?",
        "How do I troubleshoot connection issues?",
        "What is our refund policy?",
        "Explain the technical architecture",
        "API rate limits and quotas"
    ]
    
    print("Processing queries with full ValeSearch pipeline...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {query}")
        
        result = await engine.search(query)
        
        print(f"  ✓ Source: {result.source}")
        print(f"  ✓ Confidence: {result.confidence:.2f}")
        print(f"  ✓ Latency: {result.latency_ms:.1f}ms")
        print(f"  ✓ Cached: {result.cached}")
        if result.metadata.get("error"):
            print(f"  ⚠ Error: {result.metadata['error']}")
    
    # Show comprehensive statistics
    stats = engine.get_stats()
    print(f"\n=== Final Statistics ===")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"BM25 success rate: {stats['bm25_success_rate']:.1%}")
    print(f"Fallback rate: {stats['fallback_rate']:.1%}")
    print(f"Average latency: {stats['average_latency_ms']:.1f}ms")


# ==========================================
# MAIN DEMO RUNNER
# ==========================================

async def main():
    """Run all integration examples."""
    print("ValeSearch Integration Examples")
    print("=" * 50)
    print("This demo shows how to integrate ValeSearch with your existing RAG systems.")
    print("ValeSearch provides intelligent routing: Cache → BM25 → Your RAG System")
    print()
    
    # Run all examples
    await example_function_integration()
    await example_webhook_integration()
    await example_sdk_integration()
    await production_example()
    
    print("\n" + "=" * 50)
    print("Integration Examples Complete!")
    print("\nNext Steps:")
    print("1. Choose your integration pattern (function/webhook/SDK)")
    print("2. Configure your RAG system connection")
    print("3. Set up caching backend (Redis recommended for production)")
    print("4. Tune BM25 thresholds and cache quality gates")
    print("5. Monitor performance and adjust as needed")


if __name__ == "__main__":
    asyncio.run(main())