"""
Example usage of the RAG wrapper with detailed logging.
"""
import asyncio
import logging
from pathlib import Path
from musiol_rag.core.embeddings import EmbeddingModel
from musiol_rag.core.retrieval import FAISSRetriever
from musiol_rag.core.rag import RAGWrapper
from musiol_rag.database.memory import InMemoryDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_test")

async def test_embedding_model():
    """Test the embedding model functionality."""
    logger.info("Testing Embedding Model...")
    
    model = EmbeddingModel()
    test_text = "This is a test sentence."
    
    logger.info(f"Model name: {model.model_name}")
    logger.info(f"Embedding dimension: {model.dimension}")
    
    # Test single text embedding
    embedding = model.encode_single(test_text)
    logger.info(f"Single text embedding shape: {embedding.shape}")
    
    # Test batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = model.encode(texts)
    logger.info(f"Batch embeddings shape: {embeddings.shape}")
    
    return model

async def test_database():
    """Test the database functionality."""
    logger.info("\nTesting Database...")
    
    db = InMemoryDatabase()
    
    # Test adding texts
    texts = [
        "Document 1: This is the first test document.",
        "Document 2: Here's another document for testing.",
        "Document 3: And a third one to make it interesting."
    ]
    
    for text in texts:
        logger.info(f"Adding text: {text[:50]}...")
        await db.add_text(text)
    
    # Verify stored texts
    stored_texts = await db.get_texts()
    logger.info(f"Number of stored texts: {len(stored_texts)}")
    
    return db

async def test_retriever(embedding_model, database):
    """Test the retriever functionality."""
    logger.info("\nTesting Retriever...")
    
    retriever = FAISSRetriever(embedding_model)
    
    # Update index
    logger.info("Updating FAISS index...")
    await retriever.update_index(database)
    
    # Test queries
    test_queries = [
        "What is document one about?",
        "Tell me about document two",
        "Find the third document"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        relevant_texts = await retriever.get_relevant_texts(query, database, k=2)
        logger.info("Top 2 most relevant documents:")
        for i, text in enumerate(relevant_texts, 1):
            logger.info(f"{i}. {text[:100]}...")
    
    return retriever

async def test_rag_wrapper(embedding_model, database, retriever):
    """Test the complete RAG wrapper."""
    logger.info("\nTesting RAG Wrapper...")
    
    # Initialize RAG wrapper
    rag = RAGWrapper(
        embedding_provider=embedding_model,
        database_provider=database,
        retriever_provider=retriever
    )
    
    # Test document addition
    new_docs = [
        """Quantum computing is a type of computation that harnesses quantum mechanics.
        It uses qubits which can exist in multiple states simultaneously.""",
        
        """Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without explicit programming.""",
        
        """Climate change refers to long-term shifts in global weather patterns
        and temperatures, primarily caused by human activities."""
    ]
    
    logger.info("Adding new documents to RAG system...")
    await rag.add_documents(new_docs)
    
    # Test queries
    test_queries = [
        "What is quantum computing?",
        "How does machine learning work?",
        "What causes climate change?"
    ]
    
    logger.info("\nTesting queries on RAG system:")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        relevant_texts = await rag.query(query, k=1)
        logger.info(f"Most relevant text: {relevant_texts[0][:200]}...")
    
    return rag

async def main():
    """Run all tests in sequence."""
    logger.info("Starting RAG system tests...\n")
    
    try:
        # Test individual components
        embedding_model = await test_embedding_model()
        database = await test_database()
        retriever = await test_retriever(embedding_model, database)
        
        # Test complete system
        rag = await test_rag_wrapper(embedding_model, database, retriever)
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 