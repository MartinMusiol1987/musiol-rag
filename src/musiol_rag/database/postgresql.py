"""
PostgreSQL database implementation for RAG system.
"""
from typing import List, Dict, Any, Optional
import asyncpg
from .base import BaseDatabase

class PostgreSQLDatabase(BaseDatabase):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    @classmethod
    async def from_connection_string(cls, connection_string: str) -> 'PostgreSQLDatabase':
        """Create database instance and initialize tables."""
        pool = await asyncpg.create_pool(connection_string)
        
        # Create tables if they don't exist
        async with pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        return cls(pool)
    
    async def add_text(self, text: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO documents (text) VALUES ($1)',
                text
            )
    
    async def get_texts(self) -> List[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT text FROM documents ORDER BY id')
            return [row['text'] for row in rows]
    
    async def clear(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute('TRUNCATE documents')
    
    async def get_metadata(self) -> Dict[str, Any]:
        async with self.pool.acquire() as conn:
            count = await conn.fetchval('SELECT COUNT(*) FROM documents')
            return {
                "type": "postgresql",
                "text_count": count
            }
    
    async def get_text_by_id(self, text_id: int) -> Optional[str]:
        async with self.pool.acquire() as conn:
            text = await conn.fetchval(
                'SELECT text FROM documents WHERE id = $1',
                text_id
            )
            if text is None:
                raise ValueError(f"Text ID {text_id} not found")
            return text 