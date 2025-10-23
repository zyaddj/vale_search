"""
ValeSearch - The hybrid, cached retrieval engine for RAG systems.

ValeSearch provides a plug-and-play retrieval layer that combines semantic caching,
keyword search (BM25), and vector retrieval for maximum accuracy and efficiency.

True drag-and-drop integration:
    from vale_search import ValeSearch
    
    vale = ValeSearch(your_rag_function)
    result = await vale.search("What is machine learning?")
"""

__version__ = "0.1.0"
__author__ = "Zyad Djouad"
__email__ = "zyaddj@valesolutions.net"

# Main unified API - this is what users import
try:
    from .vale_search import ValeSearch, ValeSearchResult, ValeSearchConfig, create_vale_search, create_vale_search_webhook
except ImportError:
    # Fallback for development/testing
    print("Warning: Core ValeSearch components not available in development mode")
    
    class ValeSearch:
        def __init__(self, *args, **kwargs): pass
        async def search(self, query): return []
    
    class ValeSearchResult:
        def __init__(self, texts, scores, sources=None):
            self.texts = texts
            self.scores = scores
            self.sources = sources or []
    
    class ValeSearchConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def create_vale_search(*args, **kwargs):
        return ValeSearch(*args, **kwargs)
    
    def create_vale_search_webhook(*args, **kwargs):
        return ValeSearch(*args, **kwargs)

# Advanced components for power users  
try:
    from .cache.cache_manager import CacheManager
    from .retrieval.hybrid_engine import HybridEngine
    from .retrieval.fallback_integration import FallbackIntegration, FallbackResult
    from .api.schemas import QueryRequest, QueryResponse
except ImportError:
    # Mock components for development
    class MockComponent:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    
    CacheManager = MockComponent
    HybridEngine = MockComponent
    FallbackIntegration = MockComponent
    FallbackResult = MockComponent
    QueryRequest = MockComponent
    QueryResponse = MockComponent

__all__ = [
    # Main unified API (recommended for most users)
    "ValeSearch",
    "ValeSearchResult", 
    "ValeSearchConfig",
    "create_vale_search",
    "create_vale_search_webhook",
    
    # Advanced components
    "CacheManager",
    "HybridEngine",
    "FallbackIntegration",
    "FallbackResult",
    "QueryRequest",
    "QueryResponse"
]