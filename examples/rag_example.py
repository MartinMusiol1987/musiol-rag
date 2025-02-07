"""
Example usage of the RAG wrapper.
"""
import asyncio
from pathlib import Path
from src.core.embeddings import EmbeddingModel
from src.core.retrieval import FAISSRetriever
from src.core.rag import RAGWrapper
from tests.samples.test_database import InMemoryDatabase

async def main():
    # Initialize components with dependency injection
    embedding_model = EmbeddingModel()  # Uses default model from settings
    database = InMemoryDatabase()
    retriever = FAISSRetriever(embedding_model)
    
    # Create RAG wrapper
    rag = RAGWrapper(
        embedding_provider=embedding_model,
        database_provider=database,
        retriever_provider=retriever
    )
    
    # Add some sample documents
    sample_texts = [
        """Quantum computing is a type of computation that harnesses quantum mechanics.
        It uses qubits which can exist in multiple states simultaneously.""",
        
        """Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without explicit programming.""",
        
        """Climate change refers to long-term shifts in global weather patterns
        and temperatures, primarily caused by human activities."""
    ]
    
    await rag.add_documents(sample_texts)
    print("Added sample documents")
    
    # Test some queries
    queries = [
        "What is quantum computing?",
        "How does machine learning work?",
        "What causes climate change?"
    ]
    
    print("\nTesting queries:")
    for query in queries:
        print(f"\nQuery: {query}")
        relevant_texts = await rag.query(query, k=1)
        print(f"Most relevant text: {relevant_texts[0][:200]}...")

if __name__ == "__main__":
    asyncio.run(main()) 