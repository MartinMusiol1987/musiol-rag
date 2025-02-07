"""
Embeddings handler for text encoding.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import settings

class EmbeddingModel:
    def __init__(self, model_name: str = None):
        """Initialize the embedding model."""
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self._dimension = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array of the embedding
        """
        return self.model.encode(text, convert_to_numpy=True).reshape(1, -1) 