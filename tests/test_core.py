"""
Basic tests for the RAG system.
"""
import pytest
from musiol_rag.core.embeddings import EmbeddingModel
from musiol_rag.core.retrieval import FAISSRetriever

@pytest.fixture
def embedding_model():
    return EmbeddingModel()

@pytest.fixture
def retriever(embedding_model):
    return FAISSRetriever(embedding_model)

def test_embedding_dimension(embedding_model):
    """Test that embedding dimension is correct."""
    assert embedding_model.dimension > 0

def test_encode_single(embedding_model):
    """Test single text encoding."""
    text = "This is a test text"
    embedding = embedding_model.encode_single(text)
    assert embedding.shape[1] == embedding_model.dimension

def test_encode_batch(embedding_model):
    """Test batch text encoding."""
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedding_model.encode(texts)
    assert embeddings.shape == (3, embedding_model.dimension) 