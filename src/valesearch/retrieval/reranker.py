"""
Reranking module for ValeSearch.

Provides lightweight reranking using cross-encoders to improve result quality.
"""

import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RerankedResult:
    """Reranked search result."""
    answer: str
    original_score: float
    rerank_score: float
    final_score: float
    document_id: str
    metadata: Dict[str, Any]
    source: str


class Reranker:
    """
    Cross-encoder based reranking for improving search result quality.
    
    Takes initial search results and reranks them based on query-document
    relevance using a more sophisticated but slower model.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        enable_reranking: bool = True,
        rerank_threshold: float = 0.1,
        max_rerank_candidates: int = 20
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_reranking = enable_reranking
        self.rerank_threshold = rerank_threshold
        self.max_rerank_candidates = max_rerank_candidates
        
        # Initialize cross-encoder model
        self.model: Optional[CrossEncoder] = None
        if enable_reranking:
            self._load_model()
        
        logger.info(f"Reranker initialized with model: {model_name}, enabled: {enable_reranking}")
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranking model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranking model: {e}")
            self.enable_reranking = False
    
    def rerank_results(
        self,
        query: str,
        results: List[Union[Dict, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Original search query
            results: List of search results (from BM25 or vector search)
            top_k: Number of top results to return
            
        Returns:
            List of reranked results sorted by final score
        """
        if not self.enable_reranking or not self.model or not results:
            # If reranking disabled, convert to RerankedResult format
            return self._convert_without_reranking(results, top_k)
        
        start_time = time.time()
        
        # Limit candidates for efficiency
        candidates = results[:self.max_rerank_candidates]
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        for result in candidates:
            # Extract text content for reranking
            if hasattr(result, 'answer'):
                text = result.answer
            elif isinstance(result, dict):
                text = result.get('answer', result.get('text', ''))
            else:
                text = str(result)
            
            pairs.append([query, text])
        
        # Get reranking scores
        try:
            rerank_scores = self.model.predict(pairs)
            
            # Convert to float if needed
            if hasattr(rerank_scores, 'tolist'):
                rerank_scores = rerank_scores.tolist()
            elif not isinstance(rerank_scores, list):
                rerank_scores = [float(rerank_scores)]
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._convert_without_reranking(results, top_k)
        
        # Build reranked results
        reranked_results = []
        for result, rerank_score in zip(candidates, rerank_scores):
            # Extract original data
            if hasattr(result, '__dict__'):
                # Handle dataclass objects
                original_score = getattr(result, 'score', 0.0)
                answer = getattr(result, 'answer', '')
                document_id = getattr(result, 'document_id', '')
                metadata = getattr(result, 'metadata', {})
                source = getattr(result, 'source', 'unknown')
            elif isinstance(result, dict):
                # Handle dict objects
                original_score = result.get('score', 0.0)
                answer = result.get('answer', result.get('text', ''))
                document_id = result.get('document_id', result.get('id', ''))
                metadata = result.get('metadata', {})
                source = result.get('source', 'unknown')
            else:
                # Fallback
                original_score = 0.0
                answer = str(result)
                document_id = ''
                metadata = {}
                source = 'unknown'
            
            # Calculate final score (weighted combination)
            final_score = self._calculate_final_score(original_score, rerank_score)
            
            # Add reranking metadata
            rerank_metadata = {
                **metadata,
                'original_score': original_score,
                'rerank_score': float(rerank_score),
                'rerank_model': self.model_name,
                'rerank_time_ms': (time.time() - start_time) * 1000
            }
            
            reranked_result = RerankedResult(
                answer=answer,
                original_score=original_score,
                rerank_score=float(rerank_score),
                final_score=final_score,
                document_id=document_id,
                metadata=rerank_metadata,
                source=f"{source}_reranked"
            )
            
            reranked_results.append(reranked_result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Filter by threshold and limit results
        filtered_results = [
            result for result in reranked_results 
            if result.rerank_score >= self.rerank_threshold
        ]
        
        if top_k:
            filtered_results = filtered_results[:top_k]
        
        rerank_time = (time.time() - start_time) * 1000
        logger.debug(f"Reranked {len(candidates)} results in {rerank_time:.1f}ms, returned {len(filtered_results)}")
        
        return filtered_results
    
    def _calculate_final_score(self, original_score: float, rerank_score: float) -> float:
        """
        Calculate final score from original and rerank scores.
        
        Uses weighted combination with bias toward reranking score.
        """
        # Normalize scores to [0, 1] range
        norm_original = max(0, min(1, original_score))
        norm_rerank = max(0, min(1, rerank_score))
        
        # Weighted combination (70% rerank, 30% original)
        final_score = 0.7 * norm_rerank + 0.3 * norm_original
        
        return final_score
    
    def _convert_without_reranking(
        self,
        results: List[Union[Dict, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """Convert results to RerankedResult format without reranking."""
        converted_results = []
        
        for result in results:
            # Extract data based on result type
            if hasattr(result, '__dict__'):
                original_score = getattr(result, 'score', 0.0)
                answer = getattr(result, 'answer', '')
                document_id = getattr(result, 'document_id', '')
                metadata = getattr(result, 'metadata', {})
                source = getattr(result, 'source', 'unknown')
            elif isinstance(result, dict):
                original_score = result.get('score', 0.0)
                answer = result.get('answer', result.get('text', ''))
                document_id = result.get('document_id', result.get('id', ''))
                metadata = result.get('metadata', {})
                source = result.get('source', 'unknown')
            else:
                original_score = 0.0
                answer = str(result)
                document_id = ''
                metadata = {}
                source = 'unknown'
            
            reranked_result = RerankedResult(
                answer=answer,
                original_score=original_score,
                rerank_score=original_score,  # Use original score as rerank score
                final_score=original_score,
                document_id=document_id,
                metadata={**metadata, 'reranked': False},
                source=source
            )
            
            converted_results.append(reranked_result)
        
        if top_k:
            converted_results = converted_results[:top_k]
        
        return converted_results
    
    def rerank_single(self, query: str, candidate_text: str) -> float:
        """
        Get reranking score for a single query-document pair.
        
        Returns the relevance score (higher = more relevant).
        """
        if not self.enable_reranking or not self.model:
            return 0.5  # Default neutral score
        
        try:
            score = self.model.predict([query, candidate_text])
            return float(score)
        except Exception as e:
            logger.error(f"Single reranking failed: {e}")
            return 0.5
    
    def batch_rerank(self, query: str, candidate_texts: List[str]) -> List[float]:
        """
        Get reranking scores for multiple candidates efficiently.
        
        Returns list of relevance scores in same order as input.
        """
        if not self.enable_reranking or not self.model:
            return [0.5] * len(candidate_texts)  # Default neutral scores
        
        pairs = [[query, text] for text in candidate_texts]
        
        try:
            scores = self.model.predict(pairs)
            
            # Convert to list of floats
            if hasattr(scores, 'tolist'):
                return scores.tolist()
            elif isinstance(scores, list):
                return [float(score) for score in scores]
            else:
                return [float(scores)]
                
        except Exception as e:
            logger.error(f"Batch reranking failed: {e}")
            return [0.5] * len(candidate_texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "model_name": self.model_name,
            "enabled": self.enable_reranking,
            "model_loaded": self.model is not None,
            "batch_size": self.batch_size,
            "rerank_threshold": self.rerank_threshold,
            "max_candidates": self.max_rerank_candidates
        }
    
    def enable(self):
        """Enable reranking."""
        if not self.model:
            self._load_model()
        self.enable_reranking = True
        logger.info("Reranking enabled")
    
    def disable(self):
        """Disable reranking."""
        self.enable_reranking = False
        logger.info("Reranking disabled")
    
    def update_threshold(self, threshold: float):
        """Update the reranking threshold."""
        self.rerank_threshold = threshold
        logger.info(f"Updated rerank threshold to {threshold}")
    
    def update_max_candidates(self, max_candidates: int):
        """Update the maximum number of candidates to rerank."""
        self.max_rerank_candidates = max_candidates
        logger.info(f"Updated max rerank candidates to {max_candidates}")