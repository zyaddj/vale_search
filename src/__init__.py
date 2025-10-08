"""
ValeSearch - The hybrid, cached retrieval engine for RAG systems.

ValeSearch provides a plug-and-play retrieval layer that combines semantic caching,
keyword search (BM25), and vector retrieval for maximum accuracy and efficiency.
"""

__version__ = "0.1.0"
__author__ = "Vale Systems"
__email__ = "opensource@valesystems.ai"

from .cache.cache_manager import CacheManager
from .retrieval.hybrid_engine import HybridEngine
from .api.schemas import QueryRequest, QueryResponse

__all__ = [
    "CacheManager",
    "HybridEngine", 
    "QueryRequest",
    "QueryResponse"
]