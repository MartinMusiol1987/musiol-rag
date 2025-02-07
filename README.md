# Musiol RAG

A modular Retrieval-Augmented Generation (RAG) system designed for easy integration into larger projects.

## Features

- Modular design with dependency injection
- Simple interface for document management and querying
- Flexible embedding and retrieval providers
- Intelligent text chunking with sentence boundary detection
- Async support
- Easy to extend and customize

## Installation

To integrate this module into your project, ensure the following dependencies are installed:

```bash
pip install sentence-transformers faiss-cpu pydantic pydantic-settings numpy spacy

# Install required spaCy model
python -m spacy download en_core_web_sm
```

## Testing the Integration

Run the detailed test example to verify the integration and understand how the system works:

```bash
PYTHONPATH=. python examples/detailed_test.py
```

This script provides a comprehensive demonstration of the RAG system, showing each step of the process:

1. **Text Management**: Adding and retrieving documents from the database
2. **Intelligent Text Chunking**: Breaking documents into semantically meaningful pieces using sentence boundary detection
3. **Embedding Generation**: Converting text into vector representations
4. **Vector Search**: Finding the most relevant text chunks for a query
5. **Results Inspection**: Detailed logging of the entire process

The example uses PostgreSQL for storage and FAISS for efficient similarity search.

## Configuration

Key settings can be configured through environment variables or the `config.py` file:

```python
# Embedding settings
embedding_model: str = "all-MiniLM-L6-v2"

# Retrieval settings
top_k: int = 3
chunk_size: int = 200  # Maximum size of text chunks in characters

# Database settings
database_url: Optional[str] = None
database_type: str = "postgresql"

# FAISS settings
faiss_index_path: Optional[str] = "faiss_index.bin"
```

## Architecture

The system uses a modular architecture with four main components:

1. **Text Chunker**: Splits documents into semantically meaningful chunks using spaCy's sentence boundary detection
2. **Embedding Provider**: Converts text into vector embeddings
3. **Database Provider**: Stores and manages text documents
4. **Retriever Provider**: Indexes and retrieves relevant documents

Each component follows a Protocol interface, making it easy to swap implementations or create custom ones.

## License

MIT License - see LICENSE file for details. 