"""
Fallback Integration System for ValeSearch.

Handles integration with user's existing RAG systems as fallback when 
cache and BM25 don't provide sufficient results.
"""

import asyncio
import httpx
import time
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FallbackType(Enum):
    """Types of fallback integration."""
    FUNCTION = "function"          # Direct function callback
    WEBHOOK = "webhook"            # HTTP webhook to user's endpoint
    SDK = "sdk"                    # User's SDK/client integration
    DISABLED = "disabled"          # No fallback (BM25 only)


@dataclass
class FallbackResult:
    """Result from user's RAG system fallback."""
    answer: str
    confidence: float = 0.8
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FallbackConfig:
    """Configuration for fallback integration."""
    fallback_type: FallbackType
    webhook_url: Optional[str] = None
    webhook_timeout: int = 30
    webhook_headers: Optional[Dict[str, str]] = None
    function_callback: Optional[Callable] = None
    max_retries: int = 2
    enable_caching: bool = True


class FallbackIntegration:
    """
    Handles integration with user's existing RAG systems.
    
    Provides multiple integration patterns:
    1. Function callback - Direct Python function
    2. Webhook - HTTP POST to user's endpoint  
    3. SDK integration - User's existing client
    """
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self._http_client = None
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_latency": 0.0
        }
        
        logger.info(f"Fallback integration initialized: {config.fallback_type.value}")
    
    async def execute_fallback(self, query: str, context: Optional[Dict] = None) -> Optional[FallbackResult]:
        """
        Execute fallback to user's RAG system.
        
        Args:
            query: User's original query
            context: Additional context from ValeSearch (cache misses, BM25 results, etc.)
            
        Returns:
            FallbackResult or None if fallback disabled/failed
        """
        if self.config.fallback_type == FallbackType.DISABLED:
            return None
        
        start_time = time.time()
        self._stats["total_calls"] += 1
        
        try:
            result = None
            
            if self.config.fallback_type == FallbackType.FUNCTION:
                result = await self._function_fallback(query, context)
            elif self.config.fallback_type == FallbackType.WEBHOOK:
                result = await self._webhook_fallback(query, context)
            elif self.config.fallback_type == FallbackType.SDK:
                result = await self._sdk_fallback(query, context)
            
            if result:
                self._stats["successful_calls"] += 1
                latency_ms = (time.time() - start_time) * 1000
                result.latency_ms = latency_ms
                self._stats["total_latency"] += latency_ms
                
                logger.debug(f"Fallback successful: {query[:50]}... ({latency_ms:.1f}ms)")
                return result
            else:
                self._stats["failed_calls"] += 1
                logger.warning(f"Fallback returned no result: {query[:50]}...")
                return None
                
        except Exception as e:
            self._stats["failed_calls"] += 1
            logger.error(f"Fallback error for query '{query[:50]}...': {e}")
            return None
    
    async def _function_fallback(self, query: str, context: Optional[Dict]) -> Optional[FallbackResult]:
        """Execute function callback fallback."""
        if not self.config.function_callback:
            logger.error("Function callback not configured")
            return None
        
        try:
            # Call user's function (could be sync or async)
            if asyncio.iscoroutinefunction(self.config.function_callback):
                result = await self.config.function_callback(query, context)
            else:
                result = self.config.function_callback(query, context)
            
            # Handle different return formats
            if isinstance(result, str):
                return FallbackResult(answer=result)
            elif isinstance(result, dict):
                return FallbackResult(
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0.8),
                    metadata=result.get("metadata", {})
                )
            elif isinstance(result, FallbackResult):
                return result
            else:
                logger.error(f"Invalid function callback return type: {type(result)}")
                return None
                
        except Exception as e:
            logger.error(f"Function callback error: {e}")
            return None
    
    async def _webhook_fallback(self, query: str, context: Optional[Dict]) -> Optional[FallbackResult]:
        """Execute webhook fallback."""
        if not self.config.webhook_url:
            logger.error("Webhook URL not configured")
            return None
        
        try:
            # Prepare webhook payload
            payload = {
                "query": query,
                "context": context or {},
                "valesearch_version": "1.0",
                "timestamp": time.time()
            }
            
            # Get HTTP client
            if not self._http_client:
                self._http_client = httpx.AsyncClient(
                    timeout=self.config.webhook_timeout,
                    headers=self.config.webhook_headers or {}
                )
            
            # Make webhook request with retries
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await self._http_client.post(
                        self.config.webhook_url,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return FallbackResult(
                            answer=data.get("answer", ""),
                            confidence=data.get("confidence", 0.8),
                            metadata=data.get("metadata", {})
                        )
                    else:
                        logger.warning(f"Webhook returned {response.status_code}: {response.text}")
                        
                except httpx.RequestError as e:
                    logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(1)  # Brief delay before retry
            
            return None
            
        except Exception as e:
            logger.error(f"Webhook fallback error: {e}")
            return None
    
    async def _sdk_fallback(self, query: str, context: Optional[Dict]) -> Optional[FallbackResult]:
        """Execute SDK integration fallback."""
        # This would be implemented based on specific SDK patterns
        # For now, treat as function callback
        return await self._function_fallback(query, context)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback integration statistics."""
        total_calls = self._stats["total_calls"]
        
        return {
            "total_calls": total_calls,
            "successful_calls": self._stats["successful_calls"],
            "failed_calls": self._stats["failed_calls"],
            "success_rate": self._stats["successful_calls"] / max(total_calls, 1),
            "avg_latency_ms": self._stats["total_latency"] / max(self._stats["successful_calls"], 1),
            "fallback_type": self.config.fallback_type.value
        }
    
    async def close(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Helper functions for easy integration
def create_function_fallback(callback_function: Callable) -> FallbackIntegration:
    """Create function-based fallback integration."""
    config = FallbackConfig(
        fallback_type=FallbackType.FUNCTION,
        function_callback=callback_function
    )
    return FallbackIntegration(config)


def create_webhook_fallback(webhook_url: str, headers: Optional[Dict[str, str]] = None) -> FallbackIntegration:
    """Create webhook-based fallback integration."""
    config = FallbackConfig(
        fallback_type=FallbackType.WEBHOOK,
        webhook_url=webhook_url,
        webhook_headers=headers or {}
    )
    return FallbackIntegration(config)


def create_disabled_fallback() -> FallbackIntegration:
    """Create disabled fallback (BM25 only mode)."""
    config = FallbackConfig(fallback_type=FallbackType.DISABLED)
    return FallbackIntegration(config)