"""
FastAPI routes for ValeSearch API.

Defines all endpoints for the ValeSearch intelligence layer.
"""

import time
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from ..retrieval.hybrid_engine import HybridEngine
from .schemas import (
    QueryRequest, QueryResponse, CacheStatsResponse, EngineStatsResponse,
    HealthCheckResponse, AddDocumentRequest, AddDocumentResponse,
    RemoveDocumentRequest, RemoveDocumentResponse, ClearCacheRequest,
    ClearCacheResponse, ErrorResponse, BatchQueryRequest, BatchQueryResponse,
    ConfigResponse, MetricsResponse, WarmCacheRequest, WarmCacheResponse
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Global engine instance (will be initialized in main.py)
engine: HybridEngine = None

# Router for all ValeSearch endpoints
router = APIRouter()


def get_engine() -> HybridEngine:
    """Dependency to get the hybrid engine instance."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ValeSearch engine not initialized"
        )
    return engine


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "ValeSearch",
        "description": "The hybrid, cached retrieval engine for RAG systems",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    engine: HybridEngine = Depends(get_engine)
) -> QueryResponse:
    """
    Main search endpoint - processes queries through the hybrid pipeline.
    
    Routes query through:
    1. Cache check (exact → semantic)
    2. Query analysis and method selection
    3. BM25/Vector search with reranking
    4. Result caching for future use
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: '{request.query[:100]}...'")
        
        # Execute search through hybrid engine
        result = await engine.search(
            query=request.query,
            use_cache=request.use_cache
        )
        
        # Prepare response
        response = QueryResponse(
            answer=result.answer,
            source=result.source,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            cached=result.cached,
            metadata=result.metadata if request.include_metadata else None
        )
        
        logger.info(f"Query processed in {result.latency_ms:.1f}ms, source: {result.source}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/ask/batch", response_model=BatchQueryResponse)
async def ask_batch_questions(
    request: BatchQueryRequest,
    engine: HybridEngine = Depends(get_engine)
) -> BatchQueryResponse:
    """
    Batch query endpoint for processing multiple queries efficiently.
    
    Processes queries concurrently for better performance.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing batch of {len(request.queries)} queries")
        
        # Create tasks for concurrent processing
        tasks = []
        for query in request.queries:
            task = engine.search(query=query, use_cache=request.use_cache)
            tasks.append(task)
        
        # Execute all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        query_responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch query {i}: {result}")
                # Create error response
                error_response = QueryResponse(
                    answer=f"Error processing query: {str(result)}",
                    source="error",
                    confidence=0.0,
                    latency_ms=0.0,
                    cached=False,
                    metadata={"error": True, "message": str(result)}
                )
                query_responses.append(error_response)
            else:
                response = QueryResponse(
                    answer=result.answer,
                    source=result.source,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                    cached=result.cached,
                    metadata=result.metadata
                )
                query_responses.append(response)
        
        total_latency = (time.time() - start_time) * 1000
        avg_latency = total_latency / len(request.queries)
        
        logger.info(f"Processed {len(request.queries)} queries in {total_latency:.1f}ms")
        
        return BatchQueryResponse(
            results=query_responses,
            total_queries=len(request.queries),
            total_latency_ms=total_latency,
            average_latency_ms=avg_latency
        )
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch queries: {str(e)}"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(engine: HybridEngine = Depends(get_engine)) -> CacheStatsResponse:
    """Get cache performance statistics."""
    try:
        cache_stats = await engine.cache_manager.get_stats()
        
        return CacheStatsResponse(
            total_queries=cache_stats.total_queries,
            exact_hits=cache_stats.exact_hits,
            semantic_hits=cache_stats.semantic_hits,
            misses=cache_stats.misses,
            hit_rate=cache_stats.hit_rate,
            avg_latency_ms=cache_stats.avg_latency_ms,
            cache_size=cache_stats.cache_size
        )
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cache statistics: {str(e)}"
        )


@router.get("/stats", response_model=MetricsResponse)
async def get_engine_stats(engine: HybridEngine = Depends(get_engine)) -> MetricsResponse:
    """Get comprehensive engine statistics and metrics."""
    try:
        stats = await engine.get_stats()
        
        # Get cache stats separately
        cache_stats = await engine.cache_manager.get_stats()
        
        # Calculate system metrics
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # System uptime (approximate)
        uptime = time.time() - getattr(get_engine_stats, '_start_time', time.time())
        
        # Requests per minute (approximate)
        total_queries = stats["engine"]["total_queries"]
        uptime_minutes = max(uptime / 60, 1)  # Avoid division by zero
        rpm = total_queries / uptime_minutes
        
        return MetricsResponse(
            engine_stats=EngineStatsResponse(**stats["engine"]),
            cache_stats=CacheStatsResponse(
                total_queries=cache_stats.total_queries,
                exact_hits=cache_stats.exact_hits,
                semantic_hits=cache_stats.semantic_hits,
                misses=cache_stats.misses,
                hit_rate=cache_stats.hit_rate,
                avg_latency_ms=cache_stats.avg_latency_ms,
                cache_size=cache_stats.cache_size
            ),
            uptime_seconds=uptime,
            memory_usage_mb=memory_mb,
            requests_per_minute=rpm
        )
        
    except Exception as e:
        logger.error(f"Error getting engine stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving engine statistics: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(engine: HybridEngine = Depends(get_engine)) -> HealthCheckResponse:
    """Check the health of all system components."""
    try:
        health = await engine.health_check()
        
        return HealthCheckResponse(
            status=health["status"],
            components=health["components"],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Return unhealthy status instead of raising exception
        return HealthCheckResponse(
            status="unhealthy",
            components={"system": f"error: {str(e)}"},
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/config", response_model=ConfigResponse)
async def get_config(engine: HybridEngine = Depends(get_engine)) -> ConfigResponse:
    """Get current system configuration."""
    try:
        return ConfigResponse(
            version="0.1.0",
            cache_enabled=True,
            semantic_cache_enabled=engine.cache_manager.enable_semantic,
            reranking_enabled=engine.reranker.enable_reranking,
            llm_fallback_enabled=engine.enable_llm_fallback,
            embedding_model=engine.vector_search.embedding_model_name,
            short_query_threshold=engine.short_query_max_words
        )
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving configuration: {str(e)}"
        )


@router.post("/documents/add", response_model=AddDocumentResponse)
async def add_document(
    request: AddDocumentRequest,
    engine: HybridEngine = Depends(get_engine)
) -> AddDocumentResponse:
    """Add a new document to the search indices."""
    try:
        await engine.add_document(
            doc_id=request.document_id,
            text=request.text,
            answer=request.answer,
            metadata=request.metadata
        )
        
        logger.info(f"Added document: {request.document_id}")
        
        return AddDocumentResponse(
            success=True,
            document_id=request.document_id,
            message=f"Document '{request.document_id}' added successfully"
        )
        
    except Exception as e:
        logger.error(f"Error adding document {request.document_id}: {e}")
        return AddDocumentResponse(
            success=False,
            document_id=request.document_id,
            message=f"Error adding document: {str(e)}"
        )


@router.post("/documents/remove", response_model=RemoveDocumentResponse)
async def remove_document(
    request: RemoveDocumentRequest,
    engine: HybridEngine = Depends(get_engine)
) -> RemoveDocumentResponse:
    """Remove a document from the search indices."""
    try:
        success = await engine.remove_document(request.document_id)
        
        if success:
            logger.info(f"Removed document: {request.document_id}")
            return RemoveDocumentResponse(
                success=True,
                document_id=request.document_id,
                message=f"Document '{request.document_id}' removed successfully"
            )
        else:
            return RemoveDocumentResponse(
                success=False,
                document_id=request.document_id,
                message=f"Document '{request.document_id}' not found"
            )
            
    except Exception as e:
        logger.error(f"Error removing document {request.document_id}: {e}")
        return RemoveDocumentResponse(
            success=False,
            document_id=request.document_id,
            message=f"Error removing document: {str(e)}"
        )


@router.post("/cache/clear", response_model=ClearCacheResponse)
async def clear_cache(
    request: ClearCacheRequest,
    engine: HybridEngine = Depends(get_engine)
) -> ClearCacheResponse:
    """Clear all cached results."""
    try:
        # Get cache size before clearing
        cache_stats = await engine.cache_manager.get_stats()
        items_before = cache_stats.cache_size
        
        # Clear cache
        success = await engine.clear_cache()
        
        if success:
            logger.info(f"Cleared cache: {items_before} items removed")
            return ClearCacheResponse(
                success=True,
                message="Cache cleared successfully",
                items_cleared=items_before
            )
        else:
            return ClearCacheResponse(
                success=False,
                message="Failed to clear cache",
                items_cleared=0
            )
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return ClearCacheResponse(
            success=False,
            message=f"Error clearing cache: {str(e)}",
            items_cleared=0
        )


@router.post("/cache/warm", response_model=WarmCacheResponse)
async def warm_cache(
    request: WarmCacheRequest,
    engine: HybridEngine = Depends(get_engine)
) -> WarmCacheResponse:
    """Warm the cache with pre-computed query-answer pairs."""
    try:
        # Convert request format to what cache manager expects
        query_answer_pairs = [
            (item["query"], item["answer"]) 
            for item in request.queries_and_answers
        ]
        
        # Warm the cache
        items_cached = await engine.cache_manager.warm_cache(query_answer_pairs)
        
        logger.info(f"Cache warming completed: {items_cached}/{len(query_answer_pairs)} items cached")
        
        return WarmCacheResponse(
            success=True,
            items_cached=items_cached,
            total_items=len(query_answer_pairs),
            message=f"Successfully cached {items_cached} out of {len(query_answer_pairs)} items"
        )
        
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        return WarmCacheResponse(
            success=False,
            items_cached=0,
            total_items=len(request.queries_and_answers),
            message=f"Error warming cache: {str(e)}"
        )


# Error handlers
@router.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "path": str(request.url.path)
        }
    )


@router.exception_handler(422)
async def validation_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": exc.errors() if hasattr(exc, 'errors') else str(exc)
        }
    )


@router.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "details": str(exc)
        }
    )


# Initialize start time for uptime calculation
get_engine_stats._start_time = time.time()