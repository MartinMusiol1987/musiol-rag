"""
PostgreSQL database implementation for RAG system.
"""
from typing import List, Dict, Any, Optional, Tuple
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
            # Create documents table for full texts
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create chunks table with reference to source document
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, chunk_index)
                )
            ''')
        
        return cls(pool)
    
    async def add_text(self, text: str, chunks: List[str] = None) -> int:
        """
        Add a document and optionally its chunks to the database.
        
        Args:
            text: The full document text
            chunks: Optional list of text chunks
            
        Returns:
            document_id: The ID of the inserted document
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert the full document
                document_id = await conn.fetchval(
                    'INSERT INTO documents (text) VALUES ($1) RETURNING id',
                    text
                )
                
                # If chunks are provided, insert them
                if chunks:
                    for i, chunk in enumerate(chunks):
                        await conn.execute(
                            'INSERT INTO chunks (document_id, chunk_text, chunk_index) VALUES ($1, $2, $3)',
                            document_id, chunk, i
                        )
                
                return document_id
    
    async def get_texts(self) -> List[str]:
        """Get all full document texts."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT text FROM documents ORDER BY id')
            return [row['text'] for row in rows]
    
    async def get_chunks(self) -> List[str]:
        """Get all chunks across all documents."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT chunk_text FROM chunks ORDER BY document_id, chunk_index')
            return [row['chunk_text'] for row in rows]
    
    async def get_document_with_chunks(self, document_id: int) -> Tuple[str, List[str]]:
        """
        Get a document and its chunks.
        
        Returns:
            Tuple of (document_text, list_of_chunks)
        """
        async with self.pool.acquire() as conn:
            # Get the document
            document = await conn.fetchrow(
                'SELECT text FROM documents WHERE id = $1',
                document_id
            )
            if not document:
                raise ValueError(f"Document ID {document_id} not found")
            
            # Get its chunks
            chunks = await conn.fetch(
                'SELECT chunk_text FROM chunks WHERE document_id = $1 ORDER BY chunk_index',
                document_id
            )
            
            return document['text'], [chunk['chunk_text'] for chunk in chunks]
    
    async def clear(self) -> None:
        """Clear all documents and chunks."""
        async with self.pool.acquire() as conn:
            # Chunks will be automatically deleted due to CASCADE
            await conn.execute('TRUNCATE documents CASCADE')
    
    async def get_metadata(self) -> Dict[str, Any]:
        async with self.pool.acquire() as conn:
            doc_count = await conn.fetchval('SELECT COUNT(*) FROM documents')
            chunk_count = await conn.fetchval('SELECT COUNT(*) FROM chunks')
            return {
                "type": "postgresql",
                "document_count": doc_count,
                "chunk_count": chunk_count
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