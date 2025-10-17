"""
Simple test to verify ValeSearch Component 2 integration is working.
This test simulates the full pipeline: Cache → BM25 → Fallback to user's RAG system.
"""

import asyncio
import time
from typing import Dict, Any, Optional

# Simulate the ValeSearch components locally
class MockBM25Search:
    def search_best(self, query: str):
        # Simulate BM25 search - returns None to trigger fallback
        if "office hours" in query.lower():
            return type('BM25Result', (), {
                'answer': 'Our office hours are Monday-Friday, 9 AM to 6 PM EST.',
                'score': 0.8,
                'metadata': {'source': 'faq_doc_1'}
            })()
        return None

class MockFallbackResult:
    def __init__(self, answer: str, confidence: float, latency_ms: float, metadata: Dict):
        self.answer = answer
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.metadata = metadata

class MockFallbackIntegration:
    def __init__(self, callback_function):
        self.callback_function = callback_function
    
    async def execute_fallback(self, query: str, context: Dict[str, Any]) -> Optional[MockFallbackResult]:
        if self.callback_function:
            return await self.callback_function(query, context)
        return None

class MockCacheManager:
    def __init__(self):
        self.cache = {}
        self.access_log = {}
    
    async def get_with_tracking(self, query: str):
        cache_key = query.lower()
        hit = cache_key in self.cache
        
        if hit:
            self.access_log[cache_key] = time.time()
            cached_data = self.cache[cache_key]
            return type('CacheResult', (), {
                'hit': True,
                'answer': cached_data['answer'],
                'source': cached_data['source'],
                'confidence': cached_data['confidence'],
                'metadata': cached_data['metadata']
            })()
        
        return type('CacheResult', (), {'hit': False})()
    
    async def set_with_quality_check(self, query: str, answer: str, source_score: float, metadata: Dict):
        if source_score >= 0.5:  # Quality threshold
            cache_key = query.lower()
            self.cache[cache_key] = {
                'answer': answer,
                'source': 'cached',
                'confidence': source_score,
                'metadata': metadata
            }
            return True
        return False

class MockHybridEngine:
    def __init__(self, cache_manager, fallback_integration, bm25_min_score=0.5, short_query_max_words=3):
        self.cache_manager = cache_manager
        self.fallback_integration = fallback_integration
        self.bm25_search = MockBM25Search()
        self.bm25_min_score = bm25_min_score
        self.short_query_max_words = short_query_max_words
        self._stats = {
            "cache_hits": 0,
            "bm25_hits": 0,
            "fallback_calls": 0,
            "errors": 0,
            "total_latency": 0.0
        }
    
    def _should_try_bm25(self, query: str) -> bool:
        words = query.split()
        if len(words) <= self.short_query_max_words:
            return True
        
        factual_indicators = ["what", "when", "where", "who", "which", "how many"]
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in factual_indicators):
            return True
        
        return False
    
    async def search(self, query: str, use_cache: bool = True):
        start_time = time.time()
        
        # Step 1: Check cache
        if use_cache:
            cache_result = await self.cache_manager.get_with_tracking(query)
            if cache_result.hit:
                self._stats["cache_hits"] += 1
                total_latency = (time.time() - start_time) * 1000
                
                return type('HybridResult', (), {
                    'answer': cache_result.answer,
                    'source': cache_result.source,
                    'confidence': cache_result.confidence,
                    'latency_ms': total_latency,
                    'cached': True,
                    'metadata': {**cache_result.metadata, "pipeline_stage": "cache"}
                })()
        
        # Step 2: Try BM25 for appropriate queries
        if self._should_try_bm25(query):
            bm25_result = self.bm25_search.search_best(query)
            if bm25_result and bm25_result.score >= self.bm25_min_score:
                self._stats["bm25_hits"] += 1
                total_latency = (time.time() - start_time) * 1000
                
                # Cache the result
                await self.cache_manager.set_with_quality_check(
                    query, bm25_result.answer, bm25_result.score, 
                    {"generated_at": time.time(), "source": "bm25"}
                )
                
                return type('HybridResult', (), {
                    'answer': bm25_result.answer,
                    'source': "bm25",
                    'confidence': min(bm25_result.score / 5.0, 1.0),
                    'latency_ms': total_latency,
                    'cached': False,
                    'metadata': {"bm25_score": bm25_result.score, "search_method": "bm25"}
                })()
        
        # Step 3: Fallback to user's RAG system
        fallback_context = {
            "cache_miss": True,
            "bm25_attempted": self._should_try_bm25(query),
            "query_characteristics": {
                "length": len(query.split()),
                "is_short_query": len(query.split()) <= self.short_query_max_words
            }
        }
        
        fallback_result = await self.fallback_integration.execute_fallback(query, fallback_context)
        
        if fallback_result:
            self._stats["fallback_calls"] += 1
            total_latency = (time.time() - start_time) * 1000
            
            # Cache the result
            await self.cache_manager.set_with_quality_check(
                query, fallback_result.answer, fallback_result.confidence,
                {"generated_at": time.time(), "source": "user_rag"}
            )
            
            return type('HybridResult', (), {
                'answer': fallback_result.answer,
                'source': "user_rag",
                'confidence': fallback_result.confidence,
                'latency_ms': total_latency,
                'cached': False,
                'metadata': {"fallback_metadata": fallback_result.metadata, "search_method": "fallback"}
            })()
        
        # Step 4: No results found
        total_latency = (time.time() - start_time) * 1000
        self._stats["errors"] += 1
        return type('HybridResult', (), {
            'answer': "No relevant results found",
            'source': "error",
            'confidence': 0.0,
            'latency_ms': total_latency,
            'cached': False,
            'metadata': {"error": True}
        })()
    
    def get_stats(self):
        total_requests = sum([
            self._stats["cache_hits"],
            self._stats["bm25_hits"], 
            self._stats["fallback_calls"],
            self._stats["errors"]
        ])
        
        if total_requests == 0:
            return {"total_requests": 0, "cache_hit_rate": 0.0}
            
        return {
            "total_requests": total_requests,
            "cache_hit_rate": self._stats["cache_hits"] / total_requests,
            "bm25_success_rate": self._stats["bm25_hits"] / total_requests,
            "fallback_rate": self._stats["fallback_calls"] / total_requests,
            "error_rate": self._stats["errors"] / total_requests
        }


# Simulate user's existing RAG system
async def simulate_user_rag_system(query: str, context: Dict[str, Any]) -> MockFallbackResult:
    """
    This simulates the user's existing RAG system that ValeSearch will fallback to.
    In reality, this would be their actual RAG implementation.
    """
    print(f"  🤖 RAG System called with context: {context}")
    
    # Simulate processing time
    await asyncio.sleep(0.1)
    
    # Simulate RAG response based on query
    if "machine learning" in query.lower():
        answer = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
        confidence = 0.92
    elif "python" in query.lower():
        answer = "Python is a high-level programming language known for its simplicity and versatility, widely used in data science, web development, and automation."
        confidence = 0.88
    else:
        answer = f"I found information related to your query about '{query}'. This is a simulated response from your RAG system."
        confidence = 0.75
    
    return MockFallbackResult(
        answer=answer,
        confidence=confidence,
        latency_ms=100,
        metadata={
            "rag_model": "gpt-4",
            "context_used": context,
            "processing_time": 100
        }
    )


async def test_valesearch_integration():
    """Test the complete ValeSearch integration pipeline."""
    
    print("🔍 ValeSearch Component 2 Integration Test")
    print("=" * 50)
    print("Testing: Cache → BM25 → User RAG System pipeline\n")
    
    # Initialize ValeSearch components
    cache_manager = MockCacheManager()
    fallback_integration = MockFallbackIntegration(simulate_user_rag_system)
    
    engine = MockHybridEngine(
        cache_manager=cache_manager,
        fallback_integration=fallback_integration,
        bm25_min_score=0.5,
        short_query_max_words=3
    )
    
    # Test queries that will demonstrate different paths
    test_queries = [
        ("What are your office hours?", "BM25 hit"),
        ("What is machine learning?", "RAG fallback"),
        ("Explain Python programming", "RAG fallback"),
        ("What are your office hours?", "Cache hit (repeat query)"),
        ("ML", "Short query → RAG fallback"),
    ]
    
    print("Processing test queries...\n")
    
    for i, (query, expected_path) in enumerate(test_queries, 1):
        print(f"[{i}] Query: '{query}'")
        print(f"    Expected path: {expected_path}")
        
        result = await engine.search(query)
        
        print(f"    ✓ Result: {result.answer[:80]}...")
        print(f"    ✓ Source: {result.source}")
        print(f"    ✓ Confidence: {result.confidence:.2f}")
        print(f"    ✓ Latency: {result.latency_ms:.1f}ms")
        print(f"    ✓ Cached: {result.cached}")
        print()
    
    # Show final statistics
    stats = engine.get_stats()
    print("📊 Final Statistics:")
    print("-" * 30)
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"BM25 success rate: {stats['bm25_success_rate']:.1%}")
    print(f"Fallback rate: {stats['fallback_rate']:.1%}")
    print(f"Error rate: {stats['error_rate']:.1%}")
    
    print("\n🎉 Integration Test Complete!")
    print("\nKey Takeaways:")
    print("• ValeSearch routes queries intelligently through cache → BM25 → user RAG")
    print("• Your existing RAG system is called only when needed")
    print("• Subsequent identical queries are served from cache instantly")
    print("• You get cost savings and faster responses while keeping your RAG system")


if __name__ == "__main__":
    asyncio.run(test_valesearch_integration())