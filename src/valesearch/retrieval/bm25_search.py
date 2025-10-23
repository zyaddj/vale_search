"""
BM25 keyword search for ValeSearch.

Provides fast keyword-based retrieval for short queries.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import nltk
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


@dataclass
class BM25Result:
    """Result from BM25 search."""
    answer: str
    score: float
    document_id: str
    metadata: Dict[str, Any]
    source: str = "bm25"


class BM25Search:
    """
    Fast keyword search using BM25 algorithm.
    
    Optimized for short queries (≤3 words) where keyword matching
    is more effective than semantic search.
    """
    
    def __init__(
        self,
        data_path: str = "data/documents.json",
        k1: float = 1.5,
        b: float = 0.75,
        use_stopwords: bool = True,
        min_score_threshold: float = 0.1
    ):
        self.data_path = data_path
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords
        self.min_score_threshold = min_score_threshold
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english')) if use_stopwords else set()
        
        # Storage for documents and BM25 index
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        # Load documents and build index
        self._load_documents()
        self._build_index()
        
        logger.info(f"BM25Search initialized with {len(self.documents)} documents")
    
    def _load_documents(self):
        """Load documents from JSON file."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Document file not found: {self.data_path}")
            self._create_sample_documents()
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            logger.info(f"Loaded {len(self.documents)} documents from {self.data_path}")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            self._create_sample_documents()
    
    def _create_sample_documents(self):
        """Create sample documents for demonstration."""
        self.documents = [
            {
                "id": "doc_001",
                "text": "How to reset your password? Go to settings and click forgot password.",
                "answer": "To reset your password, go to Settings > Account > Forgot Password and follow the instructions.",
                "metadata": {"category": "account", "priority": "high"}
            },
            {
                "id": "doc_002", 
                "text": "What are office hours? Our office hours are Monday to Friday 9 AM to 6 PM EST.",
                "answer": "Our office hours are Monday-Friday, 9 AM to 6 PM EST.",
                "metadata": {"category": "general", "priority": "medium"}
            },
            {
                "id": "doc_003",
                "text": "How to contact support? Email us at support@company.com or call 1-800-SUPPORT.",
                "answer": "You can contact support by emailing support@company.com or calling 1-800-SUPPORT.",
                "metadata": {"category": "support", "priority": "high"}
            },
            {
                "id": "doc_004",
                "text": "API rate limits documentation. Our API has a rate limit of 1000 requests per hour.",
                "answer": "The API rate limit is 1000 requests per hour per API key.",
                "metadata": {"category": "technical", "priority": "medium"}
            },
            {
                "id": "doc_005",
                "text": "Billing questions and payment methods. We accept credit cards and PayPal for payments.",
                "answer": "We accept credit cards (Visa, MasterCard, AmEx) and PayPal for billing.",
                "metadata": {"category": "billing", "priority": "high"}
            }
        ]
        
        # Save sample documents
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created sample documents at {self.data_path}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text for BM25."""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_tokens = [
            token for token in tokens 
            if token.isalpha() and token not in self.stop_words
        ]
        
        return filtered_tokens
    
    def _build_index(self):
        """Build BM25 index from documents."""
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        # Tokenize all documents
        self.tokenized_docs = []
        for doc in self.documents:
            # Combine text and answer for indexing (gives more context)
            full_text = f"{doc['text']} {doc.get('answer', '')}"
            tokens = self._tokenize_text(full_text)
            self.tokenized_docs.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"Built BM25 index with {len(self.tokenized_docs)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        """
        Search for documents using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of BM25Result objects sorted by relevance score
        """
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index not available")
            return []
        
        start_time = time.time()
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        if not query_tokens:
            logger.debug(f"No valid tokens in query: {query}")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top documents with scores above threshold
        doc_scores = [
            (i, score) for i, score in enumerate(scores) 
            if score >= self.min_score_threshold
        ]
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_idx, score in doc_scores[:top_k]:
            doc = self.documents[doc_idx]
            
            result = BM25Result(
                answer=doc.get('answer', doc['text']),
                score=float(score),
                document_id=doc['id'],
                metadata={
                    **doc.get('metadata', {}),
                    'bm25_score': float(score),
                    'query_tokens': query_tokens,
                    'search_time_ms': (time.time() - start_time) * 1000
                }
            )
            results.append(result)
        
        logger.debug(f"BM25 search for '{query}' returned {len(results)} results in {(time.time() - start_time)*1000:.1f}ms")
        return results
    
    def search_best(self, query: str) -> Optional[BM25Result]:
        """
        Search for the single best matching document.
        
        Returns None if no document meets the minimum score threshold.
        """
        results = self.search(query, top_k=1)
        return results[0] if results else None
    
    def is_short_query(self, query: str, max_words: int = 3) -> bool:
        """
        Check if query is short enough for BM25 optimization.
        
        Short queries (≤3 words) typically work better with keyword search.
        """
        tokens = self._tokenize_text(query)
        return len(tokens) <= max_words
    
    def add_document(self, doc_id: str, text: str, answer: str, metadata: Optional[Dict] = None):
        """
        Add a new document to the index.
        
        Note: This rebuilds the entire index, so use sparingly in production.
        """
        new_doc = {
            "id": doc_id,
            "text": text,
            "answer": answer,
            "metadata": metadata or {}
        }
        
        self.documents.append(new_doc)
        self._build_index()  # Rebuild index
        
        logger.info(f"Added document {doc_id} to BM25 index")
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.
        
        Returns True if document was found and removed.
        """
        original_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc['id'] != doc_id]
        
        if len(self.documents) < original_count:
            self._build_index()  # Rebuild index
            logger.info(f"Removed document {doc_id} from BM25 index")
            return True
        
        logger.warning(f"Document {doc_id} not found in index")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 search statistics."""
        return {
            "document_count": len(self.documents),
            "index_available": self.bm25 is not None,
            "k1_parameter": self.k1,
            "b_parameter": self.b,
            "use_stopwords": self.use_stopwords,
            "min_score_threshold": self.min_score_threshold,
            "data_path": self.data_path
        }
    
    def rebuild_index(self):
        """Manually rebuild the BM25 index."""
        self._build_index()
        logger.info("BM25 index rebuilt")
    
    def save_documents(self, file_path: Optional[str] = None):
        """Save current documents to file."""
        save_path = file_path or self.data_path
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.documents)} documents to {save_path}")