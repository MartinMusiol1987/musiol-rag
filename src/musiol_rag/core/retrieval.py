"""
FAISS-based retrieval system.
"""
from typing import List, Tuple
import numpy as np
import faiss
import os
from ..database.base import BaseDatabase
from .embeddings import EmbeddingModel
from ..config import settings

class FAISSRetriever:
    def __init__(self, embedding_model: EmbeddingModel, index_path: str):
        """
        Initialize the FAISS retriever.
        
        Args:
            embedding_model: The embedding model to use
            index_path: Path where the FAISS index will be saved/loaded
        """
        if not index_path:
            raise ValueError("index_path must be provided for FAISS index storage")
            
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.text_lookup: List[str] = []
        
        # Initialize or load existing index
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load FAISS index from {index_path}: {str(e)}")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_model.dimension)

    async def update_index(self, database: BaseDatabase):
        """Update the FAISS index with all chunks from the database."""
        try:
            # Get all chunks
            chunks = await database.get_chunks()
            if not chunks:
                return

            # Clear existing index
            self.index = faiss.IndexFlatL2(self.embedding_model.dimension)
            self.text_lookup = chunks

            # Create embeddings and add to index
            try:
                embeddings = self.embedding_model.encode(chunks)
                self.index.add(embeddings)
            except Exception as e:
                raise RuntimeError(f"Failed to generate or add embeddings: {str(e)}")

            # Save index
            try:
                faiss.write_index(self.index, self.index_path)
            except Exception as e:
                raise RuntimeError(f"Failed to save FAISS index to {self.index_path}: {str(e)}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to update index: {str(e)}")

    async def get_relevant_texts(
        self, 
        query: str, 
        database: BaseDatabase,
        k: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        Get the most relevant chunks for a query.
        
        Args:
            query: The query text
            database: Database instance
            k: Number of results to return (defaults to settings.top_k)
            
        Returns:
            Tuple of (relevant_chunks, distances)
            
        Raises:
            RuntimeError: If embedding generation or search fails
        """
        k = k or settings.top_k
        
        try:
            # Encode query
            query_vector = self.embedding_model.encode_single(query)
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}")
        
        try:
            # Search index
            distances, indices = self.index.search(query_vector, k)
        except Exception as e:
            raise RuntimeError(f"Failed to search FAISS index: {str(e)}")
        
        # Get corresponding chunks and their distances
        relevant_chunks = []
        relevant_distances = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.text_lookup):
                relevant_chunks.append(self.text_lookup[idx])
                relevant_distances.append(dist)
        
        return relevant_chunks, relevant_distances 