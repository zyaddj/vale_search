"""
API schemas for ValeSearch FastAPI endpoints.

Defines request/response models using Pydantic for type safety and validation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class QueryRequest(BaseModel):
    """Request model for search queries."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="The search query text"
    )
    use_cache: bool = Field(
        default=True, 
        description="Whether to use cache for this query"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include detailed metadata in response"
    )
    
    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for search queries."""
    answer: str = Field(description="The answer to the query")
    source: str = Field(description="Source of the answer (cache, bm25, vector, llm_fallback)")
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the answer"
    )
    latency_ms: float = Field(description="Response latency in milliseconds")
    cached: bool = Field(description="Whether this result was served from cache")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the search"
    )


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    total_queries: int = Field(description="Total number of queries processed")
    exact_hits: int = Field(description="Number of exact cache hits")
    semantic_hits: int = Field(description="Number of semantic cache hits")
    misses: int = Field(description="Number of cache misses")
    hit_rate: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Overall cache hit rate"
    )
    avg_latency_ms: float = Field(description="Average cache lookup latency")
    cache_size: int = Field(description="Number of items in cache")


class EngineStatsResponse(BaseModel):
    """Response model for engine statistics."""
    total_queries: int = Field(description="Total queries processed")
    cache_hit_rate: float = Field(description="Cache hit rate")
    bm25_usage: float = Field(description="Percentage of queries using BM25")
    vector_usage: float = Field(description="Percentage of queries using vector search")
    llm_fallback_rate: float = Field(description="Percentage of queries falling back to LLM")
    average_latency_ms: float = Field(description="Average response latency")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Overall system status")
    components: Dict[str, str] = Field(description="Status of individual components")
    timestamp: Optional[str] = Field(description="Health check timestamp")


class AddDocumentRequest(BaseModel):
    """Request model for adding documents."""
    document_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the document"
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Document text/question"
    )
    answer: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Answer/response for the document"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the document"
    )
    
    @validator('document_id')
    def document_id_validation(cls, v):
        if not v or not v.strip():
            raise ValueError('Document ID cannot be empty')
        # Allow alphanumeric, underscore, hyphen, dot
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Document ID can only contain letters, numbers, dots, underscores, and hyphens')
        return v.strip()


class AddDocumentResponse(BaseModel):
    """Response model for adding documents."""
    success: bool = Field(description="Whether the document was added successfully")
    document_id: str = Field(description="The document ID that was added")
    message: str = Field(description="Success or error message")


class RemoveDocumentRequest(BaseModel):
    """Request model for removing documents."""
    document_id: str = Field(
        ...,
        min_length=1,
        description="ID of the document to remove"
    )


class RemoveDocumentResponse(BaseModel):
    """Response model for removing documents."""
    success: bool = Field(description="Whether the document was removed successfully")
    document_id: str = Field(description="The document ID that was processed")
    message: str = Field(description="Success or error message")


class ClearCacheRequest(BaseModel):
    """Request model for clearing cache."""
    confirm: bool = Field(
        default=False,
        description="Confirmation flag to prevent accidental cache clearing"
    )
    
    @validator('confirm')
    def confirm_must_be_true(cls, v):
        if not v:
            raise ValueError('Must confirm cache clearing by setting confirm=true')
        return v


class ClearCacheResponse(BaseModel):
    """Response model for clearing cache."""
    success: bool = Field(description="Whether the cache was cleared successfully")
    message: str = Field(description="Success or error message")
    items_cleared: int = Field(description="Number of cache items cleared")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    queries: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of queries to process"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cache for these queries"
    )
    max_results_per_query: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum results per query"
    )
    
    @validator('queries')
    def validate_queries(cls, v):
        # Remove empty queries
        valid_queries = [q.strip() for q in v if q and q.strip()]
        if not valid_queries:
            raise ValueError('At least one non-empty query required')
        return valid_queries


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    results: List[QueryResponse] = Field(description="Results for each query")
    total_queries: int = Field(description="Total number of queries processed")
    total_latency_ms: float = Field(description="Total processing time")
    average_latency_ms: float = Field(description="Average latency per query")


class SearchSource(str, Enum):
    """Enum for search sources."""
    CACHE = "cache"
    EXACT_CACHE = "exact_cache"
    SEMANTIC_CACHE = "semantic_cache"
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID_BM25 = "hybrid_bm25"
    HYBRID_VECTOR = "hybrid_vector"
    LLM_FALLBACK = "llm_fallback"
    ERROR = "error"


class ConfigResponse(BaseModel):
    """Response model for configuration info."""
    version: str = Field(description="ValeSearch version")
    cache_enabled: bool = Field(description="Whether caching is enabled")
    semantic_cache_enabled: bool = Field(description="Whether semantic caching is enabled")
    reranking_enabled: bool = Field(description="Whether reranking is enabled")
    llm_fallback_enabled: bool = Field(description="Whether LLM fallback is enabled")
    embedding_model: str = Field(description="Name of the embedding model")
    short_query_threshold: int = Field(description="Word count threshold for short queries")


# Response models for specific endpoints
class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    engine_stats: EngineStatsResponse
    cache_stats: CacheStatsResponse
    uptime_seconds: float = Field(description="System uptime in seconds")
    memory_usage_mb: float = Field(description="Memory usage in MB")
    requests_per_minute: float = Field(description="Recent requests per minute")


class WarmCacheRequest(BaseModel):
    """Request model for cache warming."""
    queries_and_answers: List[Dict[str, str]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of query-answer pairs to cache"
    )
    
    @validator('queries_and_answers')
    def validate_pairs(cls, v):
        for i, pair in enumerate(v):
            if 'query' not in pair or 'answer' not in pair:
                raise ValueError(f'Item {i} must have "query" and "answer" keys')
            if not pair['query'] or not pair['answer']:
                raise ValueError(f'Item {i} has empty query or answer')
        return v


class WarmCacheResponse(BaseModel):
    """Response model for cache warming."""
    success: bool = Field(description="Whether cache warming completed successfully")
    items_cached: int = Field(description="Number of items successfully cached")
    total_items: int = Field(description="Total number of items attempted")
    message: str = Field(description="Success or error message")