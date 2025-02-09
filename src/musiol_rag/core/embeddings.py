"""
Embeddings handler for text encoding.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import settings

class EmbeddingModel:
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            self.model_name = model_name or settings.embedding_model
            self.model = SentenceTransformer(self.model_name)
            self._dimension = None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model {self.model_name}: {str(e)}")

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Raises:
            RuntimeError: If dimension cannot be determined
        """
        if self._dimension is None:
            try:
                self._dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(f"Failed to get embedding dimension: {str(e)}")
        return self._dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of embeddings
            
        Raises:
            ValueError: If texts is empty or contains invalid entries
            RuntimeError: If encoding fails
        """
        if not texts:
            raise ValueError("Cannot encode empty text list")
            
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("All texts must be non-empty strings")
            
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {str(e)}")

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array of the embedding
            
        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If encoding fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
            
        try:
            return self.model.encode(text, convert_to_numpy=True).reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Failed to encode text: {str(e)}") 