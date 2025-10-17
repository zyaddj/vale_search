"""
Hybrid Engine for ValeSearch.

The core decision engine that routes queries through the optimal retrieval path:
Cache → BM25 → User's RAG System (via fallback integration)
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from ..cache.cache_manager import CacheManager, CacheResult
from .bm25_search import BM25Search, BM25Result
from .fallback_integration import FallbackIntegration, FallbackConfig, FallbackType, FallbackResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HybridResult:
    """Final result from hybrid search pipeline."""
    answer: str
    source: str  # cache, bm25, user_rag
    confidence: float
    latency_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridEngine:
    """
    The intelligent decision engine for ValeSearch.
    
    Routes queries through the optimal retrieval path:
    1. Cache availability (exact → semantic)
    2. BM25 search for factual/keyword queries
    3. Fallback to user's existing RAG system
    """
    
    def __init__(
        self,
        # Cache configuration
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 86400,
        enable_semantic_cache: bool = True,
        semantic_threshold: float = 0.85,
        
        # BM25 configuration
        data_path: Optional[str] = None,
        short_query_max_words: int = 3,
        bm25_min_score: float = 0.1,
        
        # Fallback integration (user's RAG system)
        fallback_function: Optional[Callable] = None,
        fallback_webhook_url: Optional[str] = None,
        fallback_headers: Optional[Dict[str, str]] = None,
        enable_fallback: bool = True
    ):
        """
        Initialize the hybrid engine with plug-and-play fallback integration.
        
        Args:
            fallback_function: User's RAG function to call on cache/BM25 miss
            fallback_webhook_url: HTTP endpoint for user's RAG system
            fallback_headers: Headers for webhook requests
            enable_fallback: Whether to use fallback integration
        """
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            redis_url=redis_url,
            similarity_threshold=semantic_threshold,
            cache_ttl=cache_ttl,
            enable_semantic=enable_semantic_cache
        )
        
        # Initialize BM25 search
        self.bm25_search = BM25Search(
            data_path=data_path,
            min_score_threshold=bm25_min_score
        )
        
        # Configure fallback integration to user's RAG system
        if enable_fallback:
            if fallback_function:
                # Function-based integration
                fallback_config = FallbackConfig(
                    fallback_type=FallbackType.FUNCTION,
                    function_callback=fallback_function,
                    enable_caching=True
                )
            elif fallback_webhook_url:
                # Webhook-based integration
                fallback_config = FallbackConfig(
                    fallback_type=FallbackType.WEBHOOK,
                    webhook_url=fallback_webhook_url,
                    webhook_headers=fallback_headers or {},
                    enable_caching=True
                )
            else:
                # No fallback configured
                fallback_config = FallbackConfig(fallback_type=FallbackType.DISABLED)
                
            self.fallback_integration = FallbackIntegration(fallback_config)
        else:
            # Fallback disabled
            fallback_config = FallbackConfig(fallback_type=FallbackType.DISABLED)
            self.fallback_integration = FallbackIntegration(fallback_config)
        
        # Configuration
        self.short_query_max_words = short_query_max_words
        self.bm25_min_score = bm25_min_score
        data_path: str = "data/documents.json",
        short_query_max_words: int = 3,
        
        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "bm25_hits": 0,
            "fallback_calls": 0,
            "total_latency": 0.0
        }
        
        logger.info("HybridEngine initialized with plug-and-play fallback integration")
    
    async def search(self, query: str, use_cache: bool = True) -> HybridResult:
        """
        Main search method - routes query through optimal retrieval path.
        
        Pipeline: Cache → BM25 → User's RAG System (fallback)
        
        Flow:
        1. Check cache (exact → semantic) 
        2. Analyze query characteristics
        3. Route to BM25 (factual queries) or fallback (complex queries)
        4. Cache result for future use
        """
        start_time = time.time()
        self._stats["total_queries"] += 1
        
        query = query.strip()
        if not query:
            return self._create_error_result("Empty query provided")
        
        logger.debug(f"Processing query: '{query[:100]}...'")
        
        # Step 1: Check cache first (with access tracking)
        if use_cache:
            cache_result = await self.cache_manager.get_with_tracking(query)
            if cache_result.hit:
                self._stats["cache_hits"] += 1
                
                total_latency = (time.time() - start_time) * 1000
                self._stats["total_latency"] += total_latency
                
                return HybridResult(
                    answer=cache_result.answer,
                    source=cache_result.source,
                    confidence=cache_result.confidence,
                    latency_ms=total_latency,
                    cached=True,
                    metadata={
                        **cache_result.metadata,
                        "pipeline_stage": "cache",
                        "total_latency_ms": total_latency
                    }
                )
        
        # Step 2: Try BM25 for factual/keyword queries  
        if self._should_try_bm25(query):
            bm25_result = self.bm25_search.search_best(query)
            if bm25_result and bm25_result.score >= self.bm25_min_score:
                self._stats["bm25_hits"] += 1
                total_latency = (time.time() - start_time) * 1000
                
                # Cache the BM25 result with quality check
                await self._cache_result_if_quality(query, bm25_result.answer, bm25_result.score, use_cache)
                
                return HybridResult(
                    answer=bm25_result.answer,
                    source="bm25",
                    confidence=min(bm25_result.score / 5.0, 1.0),  # Normalize BM25 score
                    latency_ms=total_latency,
                    cached=False,
                    metadata={
                        "bm25_score": bm25_result.score,
                        "document_id": bm25_result.metadata.get("document_id"),
                        "search_method": "bm25"
                    }
                )
        
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
            
            # Cache the fallback result with quality check
            await self._cache_result_if_quality(query, fallback_result.answer, fallback_result.confidence, use_cache)
            
            return HybridResult(
                answer=fallback_result.answer,
                source="user_rag",
                confidence=fallback_result.confidence,
                latency_ms=total_latency,
                cached=False,
                metadata={
                    "fallback_latency_ms": fallback_result.latency_ms,
                    "fallback_metadata": fallback_result.metadata,
                    "search_method": "fallback"
                }
            )
        
        # Step 4: No results found
        total_latency = (time.time() - start_time) * 1000
        return self._create_error_result(
            "No relevant results found - cache miss, BM25 miss, and fallback unavailable",
            latency_ms=total_latency
        )

    def _should_try_bm25(self, query: str) -> bool:
        """
        Determine if we should try BM25 search for this query.
        
        Args:
            query: The input query string
            
        Returns:
            bool: True if BM25 is likely to perform well for this query
        """
        words = query.split()
        
        # Short queries work well with BM25
        if len(words) <= self.short_query_max_words:
            return True
        
        # Keyword-heavy queries (proper nouns, specific terms)
        keyword_indicators = sum(1 for word in words if word[0].isupper())
        if keyword_indicators / len(words) > 0.3:
            return True
        
        # Factual/specific questions often work well with BM25
        factual_indicators = ["what", "when", "where", "who", "which", "how many"]
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in factual_indicators):
            return True
        
        return False

    async def _cache_result_if_quality(self, query: str, answer: str, score: float, use_cache: bool):
        """
        Cache a result if it meets quality standards and caching is enabled.
        
        Args:
            query: The query that generated this result
            answer: The answer to potentially cache
            score: The confidence/quality score of the result
            use_cache: Whether caching is enabled for this request
        """
        if not use_cache:
            return
            
        cache_success = await self.cache_manager.set_with_quality_check(
            query,
            answer,
            source_score=score,
            metadata={
                "generated_at": time.time(),
                "quality_score": score
            }
        )
        
        if cache_success:
            logger.debug(f"Cached result with quality score {score}")
        else:
            logger.debug(f"Skipped caching result (quality score {score} too low)")

    def _create_error_result(self, message: str, latency_ms: float) -> HybridResult:
        """Create a HybridResult for error cases."""
        self._stats["errors"] += 1
        return HybridResult(
            answer=message,
            source="error",
            confidence=0.0,
            latency_ms=latency_ms,
            cached=False,
            metadata={
                "error": True,
                "error_message": message,
                "latency_ms": latency_ms
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        total_requests = (
            self._stats["cache_hits"] + 
            self._stats["bm25_hits"] + 
            self._stats["fallback_calls"] +
            self._stats["errors"]
        )
        
        if total_requests == 0:
            return {
                "total_requests": 0,
                "cache_hit_rate": 0.0,
                "bm25_success_rate": 0.0,
                "fallback_rate": 0.0,
                "error_rate": 0.0,
                "average_latency_ms": 0.0,
                **self._stats
            }
        
        return {
            "total_requests": total_requests,
            "cache_hit_rate": self._stats["cache_hits"] / total_requests,
            "bm25_success_rate": self._stats["bm25_hits"] / total_requests, 
            "fallback_rate": self._stats["fallback_calls"] / total_requests,
            "error_rate": self._stats["errors"] / total_requests,
            "average_latency_ms": (
                self._stats["total_latency"] / total_requests
                if total_requests > 0 else 0.0
            ),
            **self._stats
        }