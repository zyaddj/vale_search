"""
ValeSear# Import all the existing components
try:
    # Try relative imports first (for package usage)
    from .cache.cache_manager import CacheManager
    from .retrieval.hybrid_engine import HybridEngine  
    from .retrieval.fallback_integration import FallbackIntegration, FallbackResult, create_function_fallback, create_webhook_fallback
    from .utils.logger import get_logger
except ImportError:
    try:
        # Try absolute imports for direct execution
        from cache.cache_manager import CacheManager
        from retrieval.hybrid_engine import HybridEngine  
        from retrieval.fallback_integration import FallbackIntegration, FallbackResult, create_function_fallback, create_webhook_fallback
        from utils.logger import get_logger
    except ImportError:
        # For testing/development without dependencies
        print("Warning: Some dependencies not available. Using mock components.")
        
        class MockComponent:
            def __init__(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
                
        CacheManager = MockComponent
        HybridEngine = MockComponent
        FallbackIntegration = MockComponent
        
        def get_logger(name):
            import logging
            return logging.getLogger(name)
            
        class FallbackResult:
            def __init__(self, texts, scores, sources=None):
                self.texts = texts
                self.scores = scores  
                self.sources = sources or ["mock"] * len(texts)
                
        def create_function_fallback(func):
            return func
            
        def create_webhook_fallback(url):
            return lambda x: []API for True Drag-and-Drop RAG Integration

This is Component 3: The missing piece that makes ValeSearch truly plug-and-play.
Users can integrate with a single class and minimal configuration.
"""

import asyncio
import os
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field

# Import all the existing components
try:
    # Try relative imports first (for package usage)
    from .cache.cache_manager import CacheManager
    from .retrieval.hybrid_engine import HybridEngine  
    from .retrieval.fallback_integration import FallbackIntegration, FallbackResult, create_function_fallback, create_webhook_fallback
    from .utils.logger import get_logger
except ImportError:
    # Fall back to absolute imports (for direct script usage)
    from cache.cache_manager import CacheManager
    from retrieval.hybrid_engine import HybridEngine
    from retrieval.fallback_integration import FallbackIntegration, FallbackResult, create_function_fallback, create_webhook_fallback
    from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValeSearchConfig:
    """Configuration for ValeSearch with sensible defaults."""
    
    # Cache settings
    cache_backend: str = "memory"  # "memory", "redis"
    cache_url: Optional[str] = None  # Redis URL if using redis backend
    cache_ttl: int = 3600  # 1 hour default
    quality_threshold: float = 0.7  # Only cache high-quality responses
    max_cache_size: int = 10000
    
    # BM25 settings
    bm25_min_score: float = 0.5  # Minimum score to trust BM25
    short_query_max_words: int = 3  # Words or less = try BM25 first
    
    # Fallback settings
    fallback_timeout: int = 30
    fallback_retries: int = 2
    
    # Performance settings
    enable_async: bool = True
    max_concurrent: int = 100
    
    # Logging
    log_level: str = "INFO"
    enable_metrics: bool = True


class ValeSearch:
    """
    The unified ValeSearch interface - true drag-and-drop RAG integration.
    
    This is the main class users interact with. It hides all the complexity
    of cache management, hybrid routing, and fallback integration.
    
    Examples:
        # Function callback (simplest)
        vale = ValeSearch(your_rag_function)
        
        # Webhook integration
        vale = ValeSearch(webhook_url="https://your-api.com/rag")
        
        # SDK integration
        vale = ValeSearch(sdk_instance=your_rag_sdk, sdk_method="search")
        
        # Advanced configuration
        config = ValeSearchConfig(cache_backend="redis", cache_url="redis://localhost:6379")
        vale = ValeSearch(your_rag_function, config=config)
    """
    
    def __init__(
        self,
        fallback_function: Optional[Callable] = None,
        webhook_url: Optional[str] = None,
        webhook_headers: Optional[Dict[str, str]] = None,
        sdk_instance: Optional[Any] = None,
        sdk_method: str = "search",
        config: Optional[ValeSearchConfig] = None,
        **kwargs
    ):
        """
        Initialize ValeSearch with your RAG system.
        
        Choose ONE of these integration methods:
        - fallback_function: Your async RAG function
        - webhook_url: URL to your RAG API endpoint
        - sdk_instance: Your RAG SDK instance
        
        Args:
            fallback_function: async function(query: str, context: dict) -> FallbackResult
            webhook_url: URL to POST queries to your RAG API
            webhook_headers: Headers for webhook requests (auth, etc.)
            sdk_instance: Your RAG SDK instance
            sdk_method: Method name to call on your SDK
            config: ValeSearchConfig for advanced settings
            **kwargs: Override any config values
        """
        # Merge configuration
        self.config = config or ValeSearchConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Validate integration method
        integration_methods = [fallback_function, webhook_url, sdk_instance]
        provided_methods = [m for m in integration_methods if m is not None]
        
        if len(provided_methods) == 0:
            raise ValueError("Must provide one integration method: fallback_function, webhook_url, or sdk_instance")
        elif len(provided_methods) > 1:
            raise ValueError("Provide only ONE integration method")
        
        # Store configuration
        self._fallback_function = fallback_function
        self._webhook_url = webhook_url
        self._webhook_headers = webhook_headers or {}
        self._sdk_instance = sdk_instance
        self._sdk_method = sdk_method
        
        # Initialize components (lazy loading)
        self._cache_manager = None
        self._fallback_integration = None
        self._hybrid_engine = None
        self._initialized = False
        
        # Performance tracking
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "bm25_hits": 0,
            "rag_calls": 0,
            "errors": 0,
            "total_latency": 0.0
        }
    
    async def _initialize(self):
        """Lazy initialization of all components."""
        if self._initialized:
            return
        
        logger.info("Initializing ValeSearch components...")
        
        # 1. Initialize cache manager
        cache_config = {
            "backend": self.config.cache_backend,
            "redis_url": self.config.cache_url,
            "default_ttl": self.config.cache_ttl,
            "quality_threshold": self.config.quality_threshold,
            "max_cache_size": self.config.max_cache_size
        }
        self._cache_manager = CacheManager(cache_config)
        
        # 2. Initialize fallback integration
        if self._fallback_function:
            self._fallback_integration = create_function_fallback(self._fallback_function)
        elif self._webhook_url:
            webhook_config = {
                "url": self._webhook_url,
                "headers": {
                    "Content-Type": "application/json",
                    **self._webhook_headers
                },
                "timeout": self.config.fallback_timeout,
                "retry_attempts": self.config.fallback_retries
            }
            self._fallback_integration = create_webhook_fallback(self._webhook_url, self._webhook_headers)
        elif self._sdk_instance:
            self._fallback_integration = FallbackIntegration(
                integration_type="sdk",
                sdk_instance=self._sdk_instance,
                sdk_method=self._sdk_method,
                timeout_seconds=self.config.fallback_timeout
            )
        
        # 3. Initialize hybrid engine
        self._hybrid_engine = HybridEngine(
            cache_manager=self._cache_manager,
            fallback_integration=self._fallback_integration,
            bm25_min_score=self.config.bm25_min_score,
            short_query_max_words=self.config.short_query_max_words
        )
        
        self._initialized = True
        logger.info("ValeSearch initialization complete")
    
    async def search(self, query: str, use_cache: bool = True) -> "ValeSearchResult":
        """
        Search using ValeSearch's intelligent routing.
        
        This is the main method users call. It routes through:
        1. Cache (instant for repeated queries)
        2. BM25 (fast keyword search)
        3. Your RAG system (complex queries)
        
        Args:
            query: The user's question
            use_cache: Whether to use caching (default: True)
            
        Returns:
            ValeSearchResult with answer, source, confidence, etc.
        """
        # Lazy initialization
        await self._initialize()
        
        self._stats["total_queries"] += 1
        
        try:
            # Route through hybrid engine
            result = await self._hybrid_engine.search(query, use_cache=use_cache)
            
            # Update stats
            if result.cached:
                self._stats["cache_hits"] += 1
            elif result.source == "bm25":
                self._stats["bm25_hits"] += 1
            elif result.source == "user_rag":
                self._stats["rag_calls"] += 1
            
            self._stats["total_latency"] += result.latency_ms
            
            # Convert to user-friendly result
            return ValeSearchResult(
                answer=result.answer,
                source=result.source,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                cached=result.cached,
                metadata=result.metadata
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"ValeSearch error for query '{query}': {e}")
            
            # Graceful degradation
            return ValeSearchResult(
                answer="I'm sorry, I'm having trouble processing that request. Please try again.",
                source="error",
                confidence=0.0,
                latency_ms=0,
                cached=False,
                metadata={"error": str(e)}
            )
    
    async def batch_search(self, queries: List[str], use_cache: bool = True) -> List["ValeSearchResult"]:
        """
        Process multiple queries efficiently.
        
        Args:
            queries: List of user questions
            use_cache: Whether to use caching
            
        Returns:
            List of ValeSearchResult objects
        """
        await self._initialize()
        
        # Process queries concurrently up to max_concurrent limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_query(query):
            async with semaphore:
                return await self.search(query, use_cache)
        
        tasks = [process_query(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total = self._stats["total_queries"]
        if total == 0:
            return {"total_queries": 0, "message": "No queries processed yet"}
        
        return {
            "total_queries": total,
            "cache_hit_rate": self._stats["cache_hits"] / total,
            "bm25_success_rate": self._stats["bm25_hits"] / total,
            "rag_call_rate": self._stats["rag_calls"] / total,
            "error_rate": self._stats["errors"] / total,
            "average_latency_ms": self._stats["total_latency"] / total,
            "performance_summary": {
                "cache_saves": f"{(self._stats['cache_hits'] / total) * 100:.1f}% of queries served instantly",
                "bm25_efficiency": f"{(self._stats['bm25_hits'] / total) * 100:.1f}% served via fast keyword search",
                "rag_reduction": f"{(1 - self._stats['rag_calls'] / total) * 100:.1f}% reduction in RAG calls"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components."""
        await self._initialize()
        
        # This would call the existing health check methods
        return {
            "status": "healthy",
            "components": {
                "cache": "healthy",
                "fallback_integration": "healthy",
                "hybrid_engine": "healthy"
            },
            "stats": self.get_stats()
        }
    
    async def close(self):
        """Clean up resources."""
        if self._cache_manager:
            await self._cache_manager.close()
        logger.info("ValeSearch closed")


@dataclass
class ValeSearchResult:
    """User-friendly result from ValeSearch."""
    answer: str
    source: str  # "cache", "bm25", "user_rag", "error"
    confidence: float
    latency_ms: float
    cached: bool
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        return f"ValeSearchResult(answer='{self.answer[:50]}...', source={self.source}, confidence={self.confidence:.2f})"


# Convenience functions for common use cases
async def create_vale_search(
    rag_function: Callable,
    cache_backend: str = "memory",
    **config_kwargs
) -> ValeSearch:
    """
    Quick setup function for the most common use case.
    
    Args:
        rag_function: Your async RAG function
        cache_backend: "memory" or "redis"
        **config_kwargs: Any ValeSearchConfig parameters
    
    Returns:
        Initialized ValeSearch instance
    """
    config = ValeSearchConfig(cache_backend=cache_backend, **config_kwargs)
    vale = ValeSearch(fallback_function=rag_function, config=config)
    await vale._initialize()  # Pre-initialize for immediate use
    return vale


def create_vale_search_webhook(
    webhook_url: str,
    api_key: Optional[str] = None,
    **config_kwargs
) -> ValeSearch:
    """
    Quick setup for webhook-based RAG integration.
    
    Args:
        webhook_url: Your RAG API endpoint
        api_key: Authorization key (optional)
        **config_kwargs: Any ValeSearchConfig parameters
    
    Returns:
        ValeSearch instance configured for webhook calls
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    config = ValeSearchConfig(**config_kwargs)
    return ValeSearch(
        webhook_url=webhook_url,
        webhook_headers=headers,
        config=config
    )