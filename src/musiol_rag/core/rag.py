"""
RAG (Retrieval-Augmented Generation) wrapper class.
Provides a clean interface for integrating RAG functionality into larger projects.
"""
from typing import List, Optional, Protocol
import numpy as np

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        ...
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        ...

class DatabaseProvider(Protocol):
    """Protocol for database providers."""
    async def add_text(self, text: str) -> None:
        """Add a text to the database."""
        ...
    
    async def get_texts(self) -> List[str]:
        """Get all texts from the database."""
        ...
    
    async def clear(self) -> None:
        """Clear all texts from the database."""
        ...

class RetrieverProvider(Protocol):
    """Protocol for retriever providers."""
    async def update_index(self, database: DatabaseProvider) -> None:
        """Update the retrieval index."""
        ...
    
    async def get_relevant_texts(
        self, 
        query: str, 
        database: DatabaseProvider,
        k: Optional[int] = None
    ) -> List[str]:
        """Get relevant texts for a query."""
        ...

class RAGWrapper:
    """
    Wrapper class for RAG functionality.
    Uses dependency injection for flexibility and testability.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        database_provider: DatabaseProvider,
        retriever_provider: RetrieverProvider
    ):
        """
        Initialize the RAG wrapper.
        
        Args:
            embedding_provider: Provider for text embeddings
            database_provider: Provider for text storage
            retriever_provider: Provider for text retrieval
        """
        self.embedding_provider = embedding_provider
        self.database_provider = database_provider
        self.retriever_provider = retriever_provider
    
    async def add_document(self, text: str) -> None:
        """
        Add a document to the RAG system.
        
        Args:
            text: The text to add
        """
        await self.database_provider.add_text(text)
        await self.retriever_provider.update_index(self.database_provider)
    
    async def add_documents(self, texts: List[str]) -> None:
        """
        Add multiple documents to the RAG system.
        
        Args:
            texts: List of texts to add
        """
        for text in texts:
            await self.database_provider.add_text(text)
        await self.retriever_provider.update_index(self.database_provider)
    
    async def query(self, query: str, k: Optional[int] = None) -> List[str]:
        """
        Query the RAG system for relevant texts.
        
        Args:
            query: The query text
            k: Number of results to return (optional)
            
        Returns:
            List of relevant texts
        """
        return await self.retriever_provider.get_relevant_texts(
            query, 
            self.database_provider,
            k=k
        )
    
    async def clear(self) -> None:
        """Clear all documents from the RAG system."""
        await self.database_provider.clear()
        await self.retriever_provider.update_index(self.database_provider) 