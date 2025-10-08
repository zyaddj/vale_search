"""
Vector search implementation for ValeSearch.

Handles embedding-based semantic retrieval using FAISS.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VectorResult:
    """Result from vector search."""
    answer: str
    score: float
    document_id: str
    metadata: Dict[str, Any]
    source: str = "vector"


class VectorSearch:
    """
    FAISS-based vector search for semantic retrieval.
    
    Optimized for complex queries where semantic understanding
    is more important than exact keyword matching.
    """
    
    def __init__(
        self,
        data_path: str = "data/documents.json",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        index_type: str = "flat",  # flat, ivf, hnsw
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ):
        self.data_path = data_path
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Storage for documents and FAISS index
        self.documents: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None
        self.document_embeddings: Optional[np.ndarray] = None
        
        # Load documents and build index
        self._load_documents()
        self._build_index()
        
        logger.info(f"VectorSearch initialized with {len(self.documents)} documents")
    
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
            },
            {
                "id": "doc_006",
                "text": "How to integrate our API with your application using REST endpoints and authentication.",
                "answer": "To integrate our API, use the REST endpoints with your API key in the Authorization header.",
                "metadata": {"category": "technical", "priority": "high"}
            },
            {
                "id": "doc_007",
                "text": "Data privacy and security policies. We encrypt all data and follow GDPR compliance.",
                "answer": "We take data privacy seriously with encryption, GDPR compliance, and regular security audits.",
                "metadata": {"category": "privacy", "priority": "high"}
            }
        ]
        
        # Save sample documents
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created sample documents at {self.data_path}")
    
    def _create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index based on configuration."""
        n_docs, dim = embeddings.shape
        
        if self.index_type == "flat":
            # Simple flat index (exact search)
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, max(1, n_docs // 10))  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # Train the index
            index.train(embeddings)
            
        elif self.index_type == "hnsw":
            # HNSW index for very fast approximate search
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 connections per node
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            
        else:
            logger.warning(f"Unknown index type '{self.index_type}', using flat")
            index = faiss.IndexFlatIP(dim)
        
        return index
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings.astype(np.float32)
    
    def _build_index(self):
        """Build FAISS index from documents."""
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        start_time = time.time()
        
        # Extract texts for embedding (combine text and answer for better context)
        texts = []
        for doc in self.documents:
            # Combine question and answer for richer embeddings
            combined_text = f"{doc['text']} {doc.get('answer', '')}"
            texts.append(combined_text)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        self.document_embeddings = self._generate_embeddings(texts)
        
        # Create and populate index
        self.index = self._create_index(self.document_embeddings)
        self.index.add(self.document_embeddings)
        
        build_time = time.time() - start_time
        logger.info(f"Built vector index in {build_time:.2f}s with {self.index.ntotal} documents")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[VectorResult]:
        """
        Search for semantically similar documents.
        
        Args:
            query: Search query
            top_k: Maximum number of results (defaults to max_results)
            
        Returns:
            List of VectorResult objects sorted by similarity score
        """
        if not self.index or not self.documents:
            logger.warning("Vector index not available")
            return []
        
        if top_k is None:
            top_k = self.max_results
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])
        
        # Search the index
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results
        results = []
        for score, doc_idx in zip(scores[0], indices[0]):
            # Skip results below threshold
            if score < self.similarity_threshold:
                continue
                
            # Get document
            doc = self.documents[doc_idx]
            
            result = VectorResult(
                answer=doc.get('answer', doc['text']),
                score=float(score),
                document_id=doc['id'],
                metadata={
                    **doc.get('metadata', {}),
                    'similarity_score': float(score),
                    'search_time_ms': (time.time() - start_time) * 1000,
                    'embedding_model': self.embedding_model_name
                }
            )
            results.append(result)
        
        search_time = (time.time() - start_time) * 1000
        logger.debug(f"Vector search for '{query}' returned {len(results)} results in {search_time:.1f}ms")
        
        return results
    
    def search_best(self, query: str) -> Optional[VectorResult]:
        """
        Search for the single best matching document.
        
        Returns None if no document meets the similarity threshold.
        """
        results = self.search(query, top_k=1)
        return results[0] if results else None
    
    def add_document(self, doc_id: str, text: str, answer: str, metadata: Optional[Dict] = None):
        """
        Add a new document to the index.
        
        Note: This rebuilds the entire index, so use sparingly in production.
        For frequent updates, consider using a more sophisticated indexing strategy.
        """
        new_doc = {
            "id": doc_id,
            "text": text,
            "answer": answer,
            "metadata": metadata or {}
        }
        
        self.documents.append(new_doc)
        self._build_index()  # Rebuild index
        
        logger.info(f"Added document {doc_id} to vector index")
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.
        
        Returns True if document was found and removed.
        """
        original_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc['id'] != doc_id]
        
        if len(self.documents) < original_count:
            self._build_index()  # Rebuild index
            logger.info(f"Removed document {doc_id} from vector index")
            return True
        
        logger.warning(f"Document {doc_id} not found in index")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector search statistics."""
        return {
            "document_count": len(self.documents),
            "index_available": self.index is not None,
            "index_type": self.index_type,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "data_path": self.data_path,
            "index_total": self.index.ntotal if self.index else 0
        }
    
    def save_index(self, index_path: str):
        """Save the FAISS index to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved vector index to {index_path}")
    
    def load_index(self, index_path: str) -> bool:
        """Load a FAISS index from disk."""
        if not os.path.exists(index_path):
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded vector index from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def rebuild_index(self):
        """Manually rebuild the vector index."""
        self._build_index()
        logger.info("Vector index rebuilt")
    
    def get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get the embedding for a specific document."""
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                if self.document_embeddings is not None:
                    return self.document_embeddings[i]
        return None
    
    def similarity_search_by_embedding(self, embedding: np.ndarray, top_k: int = 5) -> List[VectorResult]:
        """Search using a pre-computed embedding."""
        if not self.index:
            return []
        
        # Ensure embedding is the right shape and type
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        embedding = embedding.astype(np.float32)
        
        scores, indices = self.index.search(embedding, min(top_k, self.index.ntotal))
        
        results = []
        for score, doc_idx in zip(scores[0], indices[0]):
            if score < self.similarity_threshold:
                continue
                
            doc = self.documents[doc_idx]
            result = VectorResult(
                answer=doc.get('answer', doc['text']),
                score=float(score),
                document_id=doc['id'],
                metadata={
                    **doc.get('metadata', {}),
                    'similarity_score': float(score)
                }
            )
            results.append(result)
        
        return results