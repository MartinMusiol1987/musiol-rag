"""
In-memory database implementation for testing and simple use cases.
"""
from typing import List, Dict, Any, Optional
from .base import BaseDatabase

class InMemoryDatabase(BaseDatabase):
    """
    Simple in-memory database implementation.
    Useful for testing and small-scale applications.
    """
    
    def __init__(self):
        """Initialize an empty database."""
        self.texts: List[str] = []
        
    async def add_text(self, text: str) -> None:
        """
        Add a text to the database.
        
        Args:
            text: The text to add
        """
        self.texts.append(text)
        
    async def get_texts(self) -> List[str]:
        """
        Get all texts from the database.
        
        Returns:
            List of all stored texts
        """
        return self.texts
        
    async def clear(self) -> None:
        """Clear all texts from the database."""
        self.texts = []

    @classmethod
    async def from_connection_string(cls, connection_string: str) -> 'InMemoryDatabase':
        """
        Create a database instance from a connection string.
        For in-memory database, this just creates a new instance.
        
        Args:
            connection_string: Ignored for in-memory database
            
        Returns:
            New InMemoryDatabase instance
        """
        return cls()

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the database.
        
        Returns:
            Dictionary containing metadata
        """
        return {
            "type": "in-memory",
            "text_count": len(self.texts)
        }

    async def get_text_by_id(self, text_id: int) -> Optional[str]:
        """
        Get a text by its ID (index in this case).
        
        Args:
            text_id: Index of the text to retrieve
            
        Returns:
            The text if found, None otherwise
            
        Raises:
            ValueError: If text_id is out of range
        """
        if 0 <= text_id < len(self.texts):
            return self.texts[text_id]
        raise ValueError(f"Text ID {text_id} not found") 