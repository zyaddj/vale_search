"""
Hybrid Engine for ValeSearch.

The core decision engine that routes queries through the optimal retrieval path:
Cache → BM25 → Vector → LLM fallback
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from ..cache.cache_manager import CacheManager, CacheResult
from .bm25_search import BM25Search, BM25Result
from .vector_search import VectorSearch, VectorResult
from .reranker import Reranker, RerankedResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HybridResult:
    """Final result from hybrid search pipeline."""
    answer: str
    source: str  # cache, bm25, vector, llm_fallback
    confidence: float
    latency_ms: float
    cached: bool
    metadata: Dict[str, Any]


class HybridEngine:
    """
    The intelligent decision engine for ValeSearch.
    
    Routes queries through the optimal retrieval path based on:
    1. Cache availability (exact → semantic)
    2. Query characteristics (length, complexity)
    3. Retrieval method performance
    """
    
    def __init__(
        self,
        # Cache configuration
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 86400,
        enable_semantic_cache: bool = True,
        semantic_threshold: float = 0.85,
        
        # BM25 configuration
        data_path: str = "data/documents.json",
        short_query_max_words: int = 3,
        bm25_min_score: float = 0.1,
        
        # Vector search configuration
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_similarity_threshold: float = 0.5,
        
        # Reranking configuration
        enable_reranking: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        
        # Fallback configuration
        enable_llm_fallback: bool = False,
        llm_model: str = "gpt-3.5-turbo"
    ):
        # Initialize cache manager
        self.cache_manager = CacheManager(
            redis_url=redis_url,
            similarity_threshold=semantic_threshold,
            cache_ttl=cache_ttl,
            enable_semantic=enable_semantic_cache
        )
        
        # Initialize retrieval engines
        self.bm25_search = BM25Search(
            data_path=data_path,
            min_score_threshold=bm25_min_score
        )
        
        self.vector_search = VectorSearch(
            data_path=data_path,
            embedding_model=embedding_model,
            similarity_threshold=vector_similarity_threshold
        )
        
        # Initialize reranker
        self.reranker = Reranker(
            model_name=rerank_model,
            enable_reranking=enable_reranking
        )
        
        # Configuration
        self.short_query_max_words = short_query_max_words
        self.enable_llm_fallback = enable_llm_fallback
        self.llm_model = llm_model
        
        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "bm25_hits": 0,
            "vector_hits": 0,
            "llm_fallbacks": 0,
            "total_latency": 0.0
        }
        
        logger.info("HybridEngine initialized with all components")
    
    async def search(self, query: str, use_cache: bool = True) -> HybridResult:
        """
        Main search method - routes query through optimal retrieval path.
        
        Flow:
        1. Check cache (exact → semantic)
        2. Analyze query characteristics  
        3. Route to BM25 (short queries) or Vector (complex queries)
        4. Apply reranking if enabled
        5. LLM fallback if no good results
        6. Cache result for future use
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
        
        # Step 2: Determine optimal retrieval method
        retrieval_method = self._choose_retrieval_method(query)
        logger.debug(f"Chosen retrieval method: {retrieval_method}")
        
        # Step 3: Execute retrieval
        if retrieval_method == "bm25":
            result = await self._search_bm25(query)
        elif retrieval_method == "vector":
            result = await self._search_vector(query)
        else:
            # Hybrid: try both and pick best
            result = await self._search_hybrid(query)
        
        # Step 4: LLM fallback if no good results
        if not result and self.enable_llm_fallback:
            result = await self._llm_fallback(query)
        
        # Step 5: Handle no results case
        if not result:
            total_latency = (time.time() - start_time) * 1000
            return self._create_error_result(
                "No relevant results found",
                latency_ms=total_latency
            )
        
        # Step 6: Cache the result with quality check
        if use_cache and result.answer:
            # Use quality-checked caching with source confidence score
            source_score = getattr(result, 'score', result.confidence) if hasattr(result, 'score') else result.confidence
            
            cache_success = await self.cache_manager.set_with_quality_check(
                query,
                result.answer,
                source_score=source_score,
                metadata={
                    "source": result.source,
                    "confidence": result.confidence,
                    "generated_at": time.time(),
                    "search_method": result.source
                }
            )
            
            if cache_success:
                logger.debug(f"Cached response from {result.source} with quality check")
            else:
                logger.debug(f"Skipped caching response from {result.source} (quality too low)")
        
        # Update statistics
        total_latency = (time.time() - start_time) * 1000
        self._stats["total_latency"] += total_latency
        
        # Update result with final timing
        result.latency_ms = total_latency
        result.metadata["total_latency_ms"] = total_latency
        
        return result
    
    def _choose_retrieval_method(self, query: str) -> str:
        """
        Decide optimal retrieval method based on query characteristics.
        
        Logic:
        - Short queries (≤3 words): BM25 for exact keyword matching
        - Long/complex queries: Vector search for semantic understanding
        - Very short queries (1 word): Try both (hybrid)
        """
        # Tokenize query to count meaningful words
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        word_count = len(words)
        
        if word_count <= 1:
            return "hybrid"  # Very short - try both
        elif word_count <= self.short_query_max_words:
            return "bm25"   # Short - keyword search better
        else:
            return "vector" # Long - semantic search better
    
    async def _search_bm25(self, query: str) -> Optional[HybridResult]:
        """Execute BM25 keyword search."""
        try:
            start_time = time.time()
            
            # Get BM25 results
            bm25_results = self.bm25_search.search(query, top_k=5)
            if not bm25_results:
                return None
            
            # Apply reranking if enabled
            reranked_results = self.reranker.rerank_results(query, bm25_results, top_k=1)
            if not reranked_results:
                return None
            
            best_result = reranked_results[0]
            self._stats["bm25_hits"] += 1
            
            search_time = (time.time() - start_time) * 1000
            
            return HybridResult(
                answer=best_result.answer,
                source="bm25",
                confidence=best_result.final_score,
                latency_ms=search_time,
                cached=False,
                metadata={
                    **best_result.metadata,
                    "pipeline_stage": "bm25",
                    "bm25_score": best_result.original_score,
                    "rerank_score": best_result.rerank_score,
                    "search_time_ms": search_time
                }
            )
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return None
    
    async def _search_vector(self, query: str) -> Optional[HybridResult]:
        """Execute vector semantic search."""
        try:
            start_time = time.time()
            
            # Get vector results
            vector_results = self.vector_search.search(query, top_k=5)
            if not vector_results:
                return None
            
            # Apply reranking if enabled
            reranked_results = self.reranker.rerank_results(query, vector_results, top_k=1)
            if not reranked_results:
                return None
            
            best_result = reranked_results[0]
            self._stats["vector_hits"] += 1
            
            search_time = (time.time() - start_time) * 1000
            
            return HybridResult(
                answer=best_result.answer,
                source="vector",
                confidence=best_result.final_score,
                latency_ms=search_time,
                cached=False,
                metadata={
                    **best_result.metadata,
                    "pipeline_stage": "vector",
                    "similarity_score": best_result.original_score,
                    "rerank_score": best_result.rerank_score,
                    "search_time_ms": search_time
                }
            )
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return None
    
    async def _search_hybrid(self, query: str) -> Optional[HybridResult]:
        """Execute both BM25 and vector search, return best result."""
        try:
            start_time = time.time()
            
            # Run both searches concurrently
            bm25_task = asyncio.create_task(self._search_bm25(query))
            vector_task = asyncio.create_task(self._search_vector(query))
            
            bm25_result, vector_result = await asyncio.gather(
                bm25_task, vector_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(bm25_result, Exception):
                logger.error(f"BM25 error in hybrid search: {bm25_result}")
                bm25_result = None
            
            if isinstance(vector_result, Exception):
                logger.error(f"Vector error in hybrid search: {vector_result}")
                vector_result = None
            
            # Choose best result based on confidence
            candidates = []
            if bm25_result:
                candidates.append(bm25_result)
            if vector_result:
                candidates.append(vector_result)
            
            if not candidates:
                return None
            
            # Return result with highest confidence
            best_result = max(candidates, key=lambda x: x.confidence)
            best_result.source = f"hybrid_{best_result.source}"
            
            search_time = (time.time() - start_time) * 1000
            best_result.metadata["hybrid_search_time_ms"] = search_time
            
            return best_result
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return None
    
    async def _llm_fallback(self, query: str) -> Optional[HybridResult]:
        """
        LLM fallback for when retrieval methods find nothing.
        
        Note: This is a placeholder - implement with your preferred LLM API.
        """
        try:
            start_time = time.time()
            
            # Placeholder implementation
            # In a real system, you'd call OpenAI, Anthropic, etc.
            fallback_answer = f"I apologize, but I don't have specific information about '{query}' in my knowledge base. Please contact support for assistance."
            
            self._stats["llm_fallbacks"] += 1
            
            search_time = (time.time() - start_time) * 1000
            
            return HybridResult(
                answer=fallback_answer,
                source="llm_fallback",
                confidence=0.1,  # Low confidence for fallback
                latency_ms=search_time,
                cached=False,
                metadata={
                    "pipeline_stage": "llm_fallback",
                    "model": self.llm_model,
                    "search_time_ms": search_time
                }
            )
            
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return None
    
    def _create_error_result(self, message: str, latency_ms: float = 0.0) -> HybridResult:
        """Create an error result."""
        return HybridResult(
            answer=message,
            source="error",
            confidence=0.0,
            latency_ms=latency_ms,
            cached=False,
            metadata={"error": True, "message": message}
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        cache_stats = await self.cache_manager.get_stats()
        bm25_stats = self.bm25_search.get_stats()
        vector_stats = self.vector_search.get_stats()
        reranker_stats = self.reranker.get_stats()
        
        total_queries = self._stats["total_queries"]
        avg_latency = (
            self._stats["total_latency"] / total_queries 
            if total_queries > 0 else 0.0
        )
        
        return {
            "engine": {
                "total_queries": total_queries,
                "cache_hit_rate": self._stats["cache_hits"] / total_queries if total_queries > 0 else 0.0,
                "bm25_usage": self._stats["bm25_hits"] / total_queries if total_queries > 0 else 0.0,
                "vector_usage": self._stats["vector_hits"] / total_queries if total_queries > 0 else 0.0,
                "llm_fallback_rate": self._stats["llm_fallbacks"] / total_queries if total_queries > 0 else 0.0,
                "average_latency_ms": avg_latency
            },
            "cache": cache_stats.__dict__ if hasattr(cache_stats, '__dict__') else cache_stats,
            "bm25": bm25_stats,
            "vector": vector_stats,
            "reranker": reranker_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components."""
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check cache
        try:
            await self.cache_manager.get("health_check")
            health["components"]["cache"] = "healthy"
        except Exception as e:
            health["components"]["cache"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check BM25
        try:
            self.bm25_search.search("health check", top_k=1)
            health["components"]["bm25"] = "healthy"
        except Exception as e:
            health["components"]["bm25"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check vector search
        try:
            self.vector_search.search("health check", top_k=1)
            health["components"]["vector"] = "healthy"
        except Exception as e:
            health["components"]["vector"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check reranker
        health["components"]["reranker"] = "healthy" if self.reranker.enable_reranking else "disabled"
        
        return health
    
    async def add_document(self, doc_id: str, text: str, answer: str, metadata: Optional[Dict] = None):
        """Add a document to both BM25 and vector indices."""
        self.bm25_search.add_document(doc_id, text, answer, metadata)
        self.vector_search.add_document(doc_id, text, answer, metadata)
        logger.info(f"Added document {doc_id} to hybrid engine")
    
    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from both indices."""
        bm25_success = self.bm25_search.remove_document(doc_id)
        vector_success = self.vector_search.remove_document(doc_id)
        
        success = bm25_success and vector_success
        if success:
            logger.info(f"Removed document {doc_id} from hybrid engine")
        
        return success
    
    async def clear_cache(self) -> bool:
        """Clear all cached results."""
        return await self.cache_manager.clear()
    
    async def close(self):
        """Close all components and connections."""
        await self.cache_manager.close()
        logger.info("HybridEngine closed")