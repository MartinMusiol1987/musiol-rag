"""
FAISS-based retrieval system.
"""
from typing import List, Tuple
import numpy as np
import faiss
from ..database.base import BaseDatabase
from .embeddings import EmbeddingModel
from ..config import settings

class FAISSRetriever:
    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize the FAISS retriever."""
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatL2(self.embedding_model.dimension)
        self.text_lookup: List[str] = []

    async def update_index(self, database: BaseDatabase):
        """Update the FAISS index with all texts from the database."""
        # Get all texts
        texts = await database.get_texts()
        if not texts:
            return

        # Clear existing index
        self.index = faiss.IndexFlatL2(self.embedding_model.dimension)
        self.text_lookup = texts

        # Create embeddings and add to index
        embeddings = self.embedding_model.encode(texts)
        self.index.add(embeddings)

        # Save index if path is configured
        if settings.faiss_index_path:
            faiss.write_index(self.index, settings.faiss_index_path)

    def load_index(self, path: str = None):
        """Load a saved FAISS index."""
        path = path or settings.faiss_index_path
        if path:
            self.index = faiss.read_index(path)

    async def get_relevant_texts(
        self, 
        query: str, 
        database: BaseDatabase,
        k: int = None
    ) -> List[str]:
        """
        Get the most relevant texts for a query.
        
        Args:
            query: The query text
            database: Database instance
            k: Number of results to return (defaults to settings.top_k)
            
        Returns:
            List of relevant texts
        """
        k = k or settings.top_k
        
        # Encode query
        query_vector = self.embedding_model.encode_single(query)
        
        # Search index
        distances, indices = self.index.search(query_vector, k)
        
        # Get corresponding texts
        relevant_texts = []
        for idx in indices[0]:
            if 0 <= idx < len(self.text_lookup):
                relevant_texts.append(self.text_lookup[idx])
        
        return relevant_texts 