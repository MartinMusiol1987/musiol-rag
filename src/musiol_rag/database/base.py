"""
Base database interface for RAG system.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDatabase(ABC):
    """
    Abstract base class for database implementations.
    All database providers must implement these methods.
    """
    
    @abstractmethod
    async def add_text(self, text: str) -> None:
        """
        Add a text to the database.
        
        Args:
            text: The text to add
        """
        pass
        
    @abstractmethod
    async def get_texts(self) -> List[str]:
        """
        Get all texts from the database.
        
        Returns:
            List of all stored texts
        """
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """Clear all texts from the database."""
        pass

    @classmethod
    @abstractmethod
    async def from_connection_string(cls, connection_string: str) -> 'BaseDatabase':
        """
        Create a database instance from a connection string.
        
        Args:
            connection_string: Database connection string
            
        Returns:
            Database instance
        """
        pass

    @abstractmethod
    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the database.
        
        Returns:
            Dictionary containing metadata
        """
        pass

    @abstractmethod
    async def get_text_by_id(self, text_id: int) -> Optional[str]:
        """
        Get a text by its ID.
        
        Args:
            text_id: ID of the text to retrieve
            
        Returns:
            The text if found, None otherwise
        """
        pass 