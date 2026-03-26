"""
Hybrid Search for TemporalGuard-RAG

Combines semantic and keyword-based search for improved retrieval quality.
Implements re-ranking and result fusion strategies.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
import json
from pathlib import Path
from datetime import datetime
import logging
import re

from .vector_store import TemporalVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining semantic and lexical retrieval.
    
    Features:
    - BM25 keyword search
    - Semantic vector search
    - Reciprocal rank fusion
    - Cross-encoder re-ranking (optional)
    """
    
    def __init__(self,
                 vector_store: TemporalVectorStore = None,
                 chunks_path: str = "data/processed/chunks/temporal_chunks.jsonl"):
        """
        Initialize Hybrid Search.
        
        Args:
            vector_store: TemporalVectorStore instance
            chunks_path: Path to chunks file for BM25 indexing
        """
        self.vector_store = vector_store or TemporalVectorStore()
        self.chunks_path = Path(chunks_path)
        
        # BM25 index components
        self.bm25 = None
        self.chunk_texts = []
        self.chunk_metadata = []
        
        # Load chunks and build BM25 index
        if self.chunks_path.exists():
            self._build_bm25_index()
        else:
            logger.warning(f"Chunks file not found: {chunks_path}")
            
    def _build_bm25_index(self):
        """Build BM25 index from chunks."""
        logger.info(f"Building BM25 index from {self.chunks_path}...")
        
        self.chunk_texts = []
        self.chunk_metadata = []
        tokenized_corpus = []
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                text = chunk['text']
                
                self.chunk_texts.append(text)
                self.chunk_metadata.append({
                    'chunk_id': chunk['chunk_id'],
                    'ticker': chunk.get('ticker'),
                    'filing_date': chunk.get('filing_date'),
                    'filing_type': chunk.get('filing_type')
                })
                
                # Tokenize for BM25
                tokens = self._tokenize(text)
                tokenized_corpus.append(tokens)
                
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"BM25 index built with {len(self.chunk_texts)} documents")
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
        
    def bm25_search(self,
                   query: str,
                   n_results: int = 10,
                   cutoff_date: str = None,
                   ticker: str = None) -> Dict:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            n_results: Number of results
            cutoff_date: Temporal filter (YYYYMMDD)
            ticker: Company filter
            
        Returns:
            Search results dictionary
        """
        if self.bm25 is None:
            logger.error("BM25 index not built")
            return {'documents': [], 'metadatas': [], 'scores': []}
            
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply filters and sort
        results = []
        for i, score in enumerate(scores):
            meta = self.chunk_metadata[i]
            
            # Apply temporal filter
            if cutoff_date and meta.get('filing_date'):
                if meta['filing_date'] > cutoff_date:
                    continue
                    
            # Apply ticker filter
            if ticker and meta.get('ticker') != ticker:
                continue
                
            results.append({
                'index': i,
                'score': float(score),
                'text': self.chunk_texts[i],
                'metadata': meta
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:n_results]
        
        return {
            'documents': [r['text'] for r in results],
            'metadatas': [r['metadata'] for r in results],
            'scores': [r['score'] for r in results],
            'indices': [r['index'] for r in results]
        }
        
    def semantic_search(self,
                       query: str,
                       n_results: int = 10,
                       cutoff_date: str = None,
                       ticker: str = None) -> Dict:
        """
        Perform semantic vector search.
        
        Args:
            query: Search query
            n_results: Number of results
            cutoff_date: Temporal filter
            ticker: Company filter
            
        Returns:
            Search results dictionary
        """
        results = self.vector_store.temporal_search(
            query=query,
            cutoff_date=cutoff_date,
            ticker=ticker,
            n_results=n_results
        )
        
        return results
        
    def hybrid_search(self,
                     query: str,
                     n_results: int = 10,
                     cutoff_date: str = None,
                     ticker: str = None,
                     semantic_weight: float = 0.7,
                     bm25_weight: float = 0.3) -> Dict:
        """
        Perform hybrid search combining semantic and BM25.
        
        Args:
            query: Search query
            n_results: Number of results
            cutoff_date: Temporal filter
            ticker: Company filter
            semantic_weight: Weight for semantic results
            bm25_weight: Weight for BM25 results
            
        Returns:
            Combined search results
        """
        # Get more results from each method for fusion
        k = n_results * 3
        
        # Semantic search
        semantic_results = self.semantic_search(
            query=query,
            n_results=k,
            cutoff_date=cutoff_date,
            ticker=ticker
        )
        
        # BM25 search
        bm25_results = self.bm25_search(
            query=query,
            n_results=k,
            cutoff_date=cutoff_date,
            ticker=ticker
        )
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
        
        # Take top n_results
        fused_results = fused_results[:n_results]
        
        return {
            'documents': [r['text'] for r in fused_results],
            'metadatas': [r['metadata'] for r in fused_results],
            'fusion_scores': [r['fusion_score'] for r in fused_results],
            'search_metadata': {
                'query': query,
                'n_results': len(fused_results),
                'cutoff_date': cutoff_date,
                'ticker': ticker,
                'semantic_weight': semantic_weight,
                'bm25_weight': bm25_weight,
                'method': 'reciprocal_rank_fusion'
            }
        }
        
    def _reciprocal_rank_fusion(self,
                               semantic_results: Dict,
                               bm25_results: Dict,
                               semantic_weight: float = 0.7,
                               bm25_weight: float = 0.3,
                               k: int = 60) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = Σ (weight / (k + rank))
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            semantic_weight: Weight for semantic results
            bm25_weight: Weight for BM25 results
            k: RRF constant (typically 60)
            
        Returns:
            Fused and sorted results
        """
        fusion_scores = {}  # chunk_id -> score
        result_map = {}  # chunk_id -> result data
        
        # Process semantic results
        for rank, (doc, meta) in enumerate(zip(
            semantic_results.get('documents', [[]])[0],
            semantic_results.get('metadatas', [[]])[0]
        )):
            chunk_id = meta.get('chunk_hash', str(rank))
            rrf_score = semantic_weight / (k + rank + 1)
            
            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = 0
                result_map[chunk_id] = {
                    'text': doc,
                    'metadata': meta
                }
            fusion_scores[chunk_id] += rrf_score
            
        # Process BM25 results
        for rank, (doc, meta) in enumerate(zip(
            bm25_results.get('documents', []),
            bm25_results.get('metadatas', [])
        )):
            chunk_id = meta.get('chunk_id', str(hash(doc)))
            rrf_score = bm25_weight / (k + rank + 1)
            
            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = 0
                result_map[chunk_id] = {
                    'text': doc,
                    'metadata': meta
                }
            fusion_scores[chunk_id] += rrf_score
            
        # Sort by fusion score
        sorted_ids = sorted(fusion_scores.keys(), 
                           key=lambda x: fusion_scores[x], 
                           reverse=True)
        
        results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id].copy()
            result['fusion_score'] = fusion_scores[chunk_id]
            result['chunk_id'] = chunk_id
            results.append(result)
            
        return results
        
    def search_with_reranking(self,
                             query: str,
                             n_results: int = 10,
                             cutoff_date: str = None,
                             ticker: str = None,
                             rerank_top_k: int = 30) -> Dict:
        """
        Hybrid search with cross-encoder re-ranking.
        
        Args:
            query: Search query
            n_results: Final number of results
            cutoff_date: Temporal filter
            ticker: Company filter
            rerank_top_k: Number of candidates for re-ranking
            
        Returns:
            Re-ranked search results
        """
        # Get candidates through hybrid search
        candidates = self.hybrid_search(
            query=query,
            n_results=rerank_top_k,
            cutoff_date=cutoff_date,
            ticker=ticker
        )
        
        # Try to use cross-encoder for re-ranking
        try:
            from sentence_transformers import CrossEncoder
            
            # Load cross-encoder (caching would be better in production)
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare pairs for re-ranking
            pairs = [[query, doc] for doc in candidates['documents']]
            
            # Get cross-encoder scores
            ce_scores = cross_encoder.predict(pairs)
            
            # Sort by cross-encoder score
            sorted_indices = np.argsort(ce_scores)[::-1][:n_results]
            
            return {
                'documents': [candidates['documents'][i] for i in sorted_indices],
                'metadatas': [candidates['metadatas'][i] for i in sorted_indices],
                'rerank_scores': [float(ce_scores[i]) for i in sorted_indices],
                'search_metadata': {
                    **candidates['search_metadata'],
                    'reranking': 'cross-encoder',
                    'rerank_model': 'ms-marco-MiniLM-L-6-v2'
                }
            }
            
        except ImportError:
            logger.warning("Cross-encoder not available, returning hybrid results")
            return {
                'documents': candidates['documents'][:n_results],
                'metadatas': candidates['metadatas'][:n_results],
                'fusion_scores': candidates['fusion_scores'][:n_results],
                'search_metadata': {
                    **candidates['search_metadata'],
                    'reranking': 'none'
                }
            }
            
    def get_search_stats(self) -> Dict:
        """
        Get statistics about the search indices.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'bm25_documents': len(self.chunk_texts),
            'vector_store_documents': self.vector_store.collection.count() if self.vector_store else 0,
            'bm25_index_ready': self.bm25 is not None,
            'chunks_path': str(self.chunks_path)
        }


# Usage
if __name__ == "__main__":
    # Initialize hybrid search
    hybrid_search = HybridSearch()
    
    # Print stats
    stats = hybrid_search.get_search_stats()
    print("Search Index Statistics:")
    print(json.dumps(stats, indent=2))
    
    if stats['bm25_documents'] > 0:
        # Test hybrid search
        print("\nTesting hybrid search...")
        
        results = hybrid_search.hybrid_search(
            query="What are the major risk factors affecting the business?",
            n_results=5,
            cutoff_date="20230701"
        )
        
        print(f"\nFound {len(results['documents'])} results")
        for i, (doc, meta, score) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['fusion_scores']
        )):
            print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
            print(f"Ticker: {meta.get('ticker')}, Date: {meta.get('filing_date')}")
            print(f"Preview: {doc[:150]}...")
    else:
        print("\nNo documents indexed. Run preprocessing pipeline first.")
