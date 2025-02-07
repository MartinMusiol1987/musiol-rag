"""
Basic usage example of Musiol-RAG.
"""
import asyncio
import os
from dotenv import load_dotenv
from src.database.postgresql import PostgreSQLDatabase
from src.core.rag_engine import RAGEngine

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize database connection
    database = await PostgreSQLDatabase.from_connection_string(
        os.getenv("DATABASE_URL")
    )
    
    # Initialize RAG engine
    rag = RAGEngine(
        database=database,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize the system (build/load index)
    await rag.initialize()
    
    # Example query
    result = await rag.query("What is machine learning?")
    
    print("\nQuestion:", result["question"])
    print("\nRelevant Context:")
    for i, ctx in enumerate(result["context"], 1):
        print(f"\n{i}. {ctx[:200]}...")
    print("\nAnswer:", result["answer"])

if __name__ == "__main__":
    asyncio.run(main()) 