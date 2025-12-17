"""
E5-Multilingual Embedding Service

Provides embedding functionality using intfloat/multilingual-e5-base model
for semantic similarity, deduplication, and chunk relation detection.
"""

import numpy as np
from typing import List, Tuple, Optional
import torch


class E5EmbeddingService:
    """E5-Multilingual embedding service for semantic operations"""
    
    MODEL_NAME = "intfloat/multilingual-e5-base"
    
    def __init__(self, device: str = None, use_fp16: bool = True):
        """
        Initialize E5 embedding service.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_fp16: Use half precision for faster inference on GPU
        """
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        print(f"Loading E5-multilingual model on {device}...")
        
        # Modern approach for sentence-transformers 3.x (Dec 2025)
        # Use model_kwargs instead of deprecated model.half()
        model_kwargs = {}
        if self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        self.model = SentenceTransformer(
            self.MODEL_NAME, 
            device=device,
            model_kwargs=model_kwargs  # Pass dict (empty or with dtype)
        )
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Embedding cache for repeated texts (Issue #20 optimization)
        self._cache = {}
        self._cache_max_size = 1000
        print(f"  Model loaded. Embedding dim: {self.embedding_dim}")
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               prefix: str = "passage: ", show_progress: bool = True,
               use_cache: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings with optional caching.
        
        E5 models require a prefix:
        - "query: " for search queries
        - "passage: " for documents/chunks to be searched
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            prefix: Prefix to add to each text ("passage: " or "query: ")
            show_progress: Show progress bar
            use_cache: Whether to use embedding cache (default True)
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache for previously computed embeddings
        results = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = f"{prefix}{text}"
            if use_cache and cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            prefixed_texts = [f"{prefix}{t}" for t in texts_to_encode]
            
            new_embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress and len(prefixed_texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
            
            # Store in cache and results
            for j, (idx, text) in enumerate(zip(text_indices, texts_to_encode)):
                cache_key = f"{prefix}{text}"
                emb = new_embeddings[j]
                
                # Only cache if under size limit
                if use_cache and len(self._cache) < self._cache_max_size:
                    self._cache[cache_key] = emb
                
                results.append((idx, emb))
        
        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        embeddings = np.stack([emb for _, emb in results])
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query text"""
        return self.encode([query], prefix="query: ", show_progress=False)[0]
    
    def encode_passages(self, passages: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple passages/documents"""
        return self.encode(passages, batch_size=batch_size, prefix="passage: ")
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Since embeddings are L2-normalized, this is just dot product.
        """
        return float(np.dot(emb1, emb2))
    
    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all embeddings.
        
        Args:
            embeddings: Array of shape (n, dim)
            
        Returns:
            Similarity matrix of shape (n, n)
        """
        # Since embeddings are normalized, similarity = dot product
        return np.dot(embeddings, embeddings.T)
    
    def find_similar(self, query_emb: np.ndarray, corpus_embs: np.ndarray,
                     top_k: int = 5, threshold: float = 0.0) -> List[Tuple[int, float]]:
        """
        Find most similar items in corpus to query.
        
        Args:
            query_emb: Query embedding (1D array)
            corpus_embs: Corpus embeddings (2D array)
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity) tuples, sorted by similarity descending
        """
        if len(corpus_embs) == 0:
            return []
        
        # Compute similarities
        similarities = np.dot(corpus_embs, query_emb)
        
        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        valid_sims = similarities[valid_indices]
        
        # Sort and get top-k
        sorted_order = np.argsort(valid_sims)[::-1][:top_k]
        
        results = [
            (int(valid_indices[i]), float(valid_sims[i]))
            for i in sorted_order
        ]
        
        return results
    
    def find_duplicates(self, embeddings: np.ndarray, 
                        threshold: float = 0.92) -> List[Tuple[int, int, float]]:
        """
        Find pairs of embeddings with similarity above threshold.
        
        Args:
            embeddings: Array of embeddings
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            List of (idx1, idx2, similarity) for pairs above threshold
        """
        n = len(embeddings)
        if n < 2:
            return []
        
        duplicates = []
        sim_matrix = self.similarity_matrix(embeddings)
        
        # Only check upper triangle (avoid duplicates and self-similarity)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    duplicates.append((i, j, float(sim_matrix[i, j])))
        
        return duplicates
    
    def find_related_pairs(self, embeddings: np.ndarray,
                           threshold: float = 0.75,
                           max_per_item: int = 5) -> List[Tuple[int, int, float]]:
        """
        Find related (but not duplicate) pairs based on similarity.
        
        Args:
            embeddings: Array of embeddings
            threshold: Minimum similarity for relation
            max_per_item: Maximum relations per item
            
        Returns:
            List of (idx1, idx2, similarity) for related pairs
        """
        n = len(embeddings)
        if n < 2:
            return []
        
        sim_matrix = self.similarity_matrix(embeddings)
        relations = []
        
        # For each item, find top related items
        for i in range(n):
            # Get similarities for this item (excluding self)
            sims = sim_matrix[i].copy()
            sims[i] = -1  # Exclude self
            
            # Find items above threshold
            valid = np.where(sims >= threshold)[0]
            
            # Sort by similarity and take top-k
            sorted_valid = sorted(valid, key=lambda j: sims[j], reverse=True)[:max_per_item]
            
            for j in sorted_valid:
                if i < j:  # Only add once (i < j)
                    relations.append((i, j, float(sims[j])))
        
        # Remove duplicates and sort
        unique_relations = list(set(relations))
        unique_relations.sort(key=lambda x: x[2], reverse=True)
        
        return unique_relations


# Convenience function
def create_embedding_service(device: str = None) -> E5EmbeddingService:
    """Create and return E5 embedding service instance"""
    return E5EmbeddingService(device=device)


if __name__ == "__main__":
    # Quick test
    print("Testing E5EmbeddingService...")
    
    svc = E5EmbeddingService(device="cpu")
    
    test_texts = [
        "Biaya kuliah Teknik Informatika di UNSIQ",
        "SPP jurusan TI semester 1 adalah Rp 6.045.000",
        "Beasiswa Tahfidz 30 Juz memberikan gratis SPP",
        "Pendaftaran mahasiswa baru gelombang 1",
    ]
    
    embeddings = svc.encode_passages(test_texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim_01 = svc.similarity(embeddings[0], embeddings[1])
    sim_02 = svc.similarity(embeddings[0], embeddings[2])
    sim_03 = svc.similarity(embeddings[0], embeddings[3])
    
    print(f"\nSimilarities to '{test_texts[0]}':")
    print(f"  '{test_texts[1][:40]}...': {sim_01:.3f}")
    print(f"  '{test_texts[2][:40]}...': {sim_02:.3f}")
    print(f"  '{test_texts[3][:40]}...': {sim_03:.3f}")
    
    # Test find similar
    query_emb = svc.encode_query("Berapa biaya kuliah TI?")
    results = svc.find_similar(query_emb, embeddings, top_k=2)
    
    print(f"\nTop matches for query 'Berapa biaya kuliah TI?':")
    for idx, sim in results:
        print(f"  [{sim:.3f}] {test_texts[idx]}")
    
    print("\nTest PASSED!")
