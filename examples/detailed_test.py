"""
Detailed test of the RAG system with PostgreSQL, showing each step of the process.
"""
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
import os
import getpass

from musiol_rag.core.embeddings import EmbeddingModel
from musiol_rag.core.retrieval import FAISSRetriever
from musiol_rag.database.postgresql import PostgreSQLDatabase
from musiol_rag.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_detailed_test")

def print_separator(title: str):
    """Print a section separator."""
    logger.info("\n" + "="*50)
    logger.info(title)
    logger.info("="*50 + "\n")

async def inspect_original_texts(db: PostgreSQLDatabase):
    """Step 1: Show original texts in the database."""
    print_separator("STEP 1: Original Texts in Database")
    texts = await db.get_texts()
    for i, text in enumerate(texts, 1):
        logger.info(f"Document {i}:")
        logger.info(f"Length: {len(text)} characters")
        logger.info(f"Preview: {text[:200]}...")
        logger.info("")

def create_text_chunks(texts: List[str], chunk_size: int = None, overlap: int = None) -> List[str]:
    """Create overlapping chunks from texts."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    
    logger.info(f"Chunking with size={chunk_size}, overlap={overlap}")
    chunks = []
    for text in texts:
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            for i in range(0, len(text) - overlap, chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
    return chunks

def inspect_chunks(chunks: List[str]):
    """Step 2: Show text chunks."""
    print_separator("STEP 2: Text Chunks")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Chunk {i}:")
        logger.info(f"Length: {len(chunk)} characters")
        logger.info(f"Content: {chunk[:100]}...")
        logger.info("")

def inspect_chunk_vectors(embeddings: np.ndarray, chunks: List[str]):
    """Step 3: Show vector representations of chunks."""
    print_separator("STEP 3: Chunk Vectors")
    logger.info(f"Number of chunks: {len(chunks)}")
    logger.info(f"Vector dimension: {embeddings.shape[1]}")
    logger.info(f"Vector shape: {embeddings.shape}")
    logger.info("\nSample vector (first chunk, first 10 dimensions):")
    logger.info(embeddings[0][:10])

def inspect_query_vector(query: str, embedding_model: EmbeddingModel):
    """Step 4: Show query vector."""
    print_separator("STEP 4: Query Vector")
    logger.info(f"Query: {query}")
    query_vector = embedding_model.encode_single(query)
    logger.info(f"Vector shape: {query_vector.shape}")
    logger.info("\nQuery vector (first 10 dimensions):")
    logger.info(query_vector[0][:10])
    return query_vector

async def find_closest_vectors(query: str, chunks: List[str], embedding_model: EmbeddingModel, k: int = 4) -> Tuple[List[str], List[float]]:
    """Step 5: Find k closest vectors using FAISS."""
    print_separator("STEP 5: Closest Vectors")
    
    # Initialize FAISS retriever
    retriever = FAISSRetriever(embedding_model)
    
    # Create a temporary in-memory database for the chunks
    from musiol_rag.database.memory import InMemoryDatabase
    temp_db = InMemoryDatabase()
    for chunk in chunks:
        await temp_db.add_text(chunk)
    
    # Update the index with our chunks
    await retriever.update_index(temp_db)
    
    # Get relevant texts
    relevant_texts = await retriever.get_relevant_texts(query, temp_db, k=k)
    
    # Get distances for logging
    query_vector = embedding_model.encode_single(query)
    chunk_vectors = embedding_model.encode(chunks)
    distances = np.linalg.norm(chunk_vectors - query_vector, axis=1)
    
    # Log results
    logger.info(f"Found {len(relevant_texts)} closest vectors")
    for i, text in enumerate(relevant_texts, 1):
        chunk_idx = chunks.index(text)
        logger.info(f"\nChunk {i}:")
        logger.info(f"Distance: {distances[chunk_idx]:.4f}")
        logger.info(f"First 10 dimensions: {chunk_vectors[chunk_idx][:10]}")
    
    return relevant_texts, [distances[chunks.index(text)] for text in relevant_texts]

def show_closest_chunks(closest_chunks: List[str]):
    """Step 6: Show the text of closest chunks."""
    print_separator("STEP 6: Closest Chunks")
    for i, chunk in enumerate(closest_chunks, 1):
        logger.info(f"\nChunk {i}:")
        logger.info(f"Content: {chunk}")

async def main():
    # Initialize components
    username = getpass.getuser()
    connection_string = os.environ.get(
        "DATABASE_URL",
        f"postgresql://{username}@localhost/rag_test"  # Use system username
    )
    
    try:
        db = await PostgreSQLDatabase.from_connection_string(connection_string)
        embedding_model = EmbeddingModel()
        
        # Clear any existing data
        await db.clear()
        
        # Add sample documents
        sample_texts = [
            """Quantum computing is a type of computation that harnesses quantum mechanics.
            It uses qubits which can exist in multiple states simultaneously. This makes
            quantum computers particularly good at solving certain types of problems that
            classical computers struggle with.""",
            
            """Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience. It uses various algorithms
            and statistical models to analyze and draw inferences from patterns in data.
            Deep learning, a subset of machine learning, uses neural networks with multiple layers.""",
            
            """Climate change refers to long-term shifts in global weather patterns
            and temperatures. The primary driver is human activity, particularly the
            emission of greenhouse gases. Effects include rising sea levels, extreme
            weather events, and threats to biodiversity."""
        ]
        
        for text in sample_texts:
            await db.add_text(text)
        
        # Step 1: Show original texts
        await inspect_original_texts(db)
        
        # Step 2: Create and show chunks
        texts = await db.get_texts()
        chunks = create_text_chunks(texts)
        inspect_chunks(chunks)
        
        # Step 3: Create and show chunk vectors
        chunk_vectors = embedding_model.encode(chunks)
        inspect_chunk_vectors(chunk_vectors, chunks)
        
        # Step 4: Create and show query vector
        query = "How does quantum computing work?"
        query_vector = inspect_query_vector(query, embedding_model)
        
        # Step 5: Find closest vectors using FAISS
        closest_chunks, distances = await find_closest_vectors(query, chunks, embedding_model)
        
        # Step 6: Show closest chunks
        show_closest_chunks(closest_chunks)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 