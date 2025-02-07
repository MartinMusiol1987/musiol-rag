"""
Script to populate the database with sample data.
"""
import asyncio
import asyncpg
from dotenv import load_dotenv
import os

# Sample documents about various topics
SAMPLE_DOCUMENTS = [
    {
        "content": "Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data.",
        "metadata": {"topic": "AI", "source": "sample"}
    },
    {
        "content": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It supports multiple programming paradigms.",
        "metadata": {"topic": "Programming", "source": "sample"}
    },
    {
        "content": "A database is an organized collection of structured information, or data, typically stored electronically in a computer system.",
        "metadata": {"topic": "Databases", "source": "sample"}
    },
]

async def populate_database():
    """Populate the database with sample documents."""
    load_dotenv()
    
    # Connect to database
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    
    # Create table if it doesn't exist
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'
        )
    ''')
    
    # Insert sample documents
    for doc in SAMPLE_DOCUMENTS:
        await conn.execute(
            'INSERT INTO documents (content, metadata) VALUES ($1, $2)',
            doc["content"],
            doc["metadata"]
        )
    
    # Verify insertion
    count = await conn.fetchval('SELECT COUNT(*) FROM documents')
    print(f"Successfully inserted {count} documents")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(populate_database()) 