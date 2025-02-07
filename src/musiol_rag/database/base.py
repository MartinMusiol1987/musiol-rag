"""
Abstract base class for database connections.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDatabase(ABC):
    """Base class for database connections."""
    
    @abstractmethod
    async def get_texts(self, query: Optional[str] = None) -> List[str]:
        """
        Retrieve texts based on optional query.
        If query is None, retrieve all texts.
        """
        pass

    @abstractmethod
    async def get_text_by_id(self, text_id: str) -> Optional[str]:
        """Retrieve specific text by ID."""
        pass
    
    @abstractmethod
    async def get_metadata(self, text_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific text."""
        pass

    @classmethod
    @abstractmethod
    def from_connection_string(cls, connection_string: str) -> 'BaseDatabase':
        """Create database instance from connection string."""
        pass 