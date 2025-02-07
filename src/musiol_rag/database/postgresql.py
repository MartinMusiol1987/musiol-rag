"""
PostgreSQL database implementation.
"""
from typing import List, Dict, Any, Optional
import asyncpg
from .base import BaseDatabase

class PostgreSQLDatabase(BaseDatabase):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def from_connection_string(cls, connection_string: str) -> 'PostgreSQLDatabase':
        """Create a PostgreSQL database instance from connection string."""
        pool = await asyncpg.create_pool(connection_string)
        return cls(pool)

    async def get_texts(self, query: Optional[str] = None) -> List[str]:
        """Retrieve texts based on optional query."""
        async with self.pool.acquire() as conn:
            if query:
                # Adjust the query based on your table structure
                rows = await conn.fetch(
                    "SELECT content FROM documents WHERE content ILIKE $1",
                    f"%{query}%"
                )
            else:
                rows = await conn.fetch("SELECT content FROM documents")
            return [row['content'] for row in rows]

    async def get_text_by_id(self, text_id: str) -> Optional[str]:
        """Retrieve specific text by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT content FROM documents WHERE id = $1",
                text_id
            )
            return row['content'] if row else None

    async def get_metadata(self, text_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific text."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT metadata FROM documents WHERE id = $1",
                text_id
            )
            return row['metadata'] if row else {} 