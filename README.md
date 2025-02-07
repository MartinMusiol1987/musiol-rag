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

There are several ways to integrate this package into your project:

### 1. Install from GitHub (Recommended during development)

```bash
pip install git+https://github.com/MartinMusiol1987/musiol-rag.git

# Install required spaCy model
python -m spacy download en_core_web_sm
```

### 2. Local Development Install

If you want to modify the package while using it:

```bash
# Clone the repository
git clone https://github.com/MartinMusiol1987/musiol-rag.git
cd musiol-rag

# Install in editable mode
pip install -e .

# Install required spaCy model
python -m spacy download en_core_web_sm
```

### 3. Copy Required Components

If you need to heavily customize the implementation or can't use pip install:

1. Copy the `src/musiol_rag` directory into your project
2. Install the required dependencies:
```bash
pip install sentence-transformers>=2.2.2 faiss-cpu>=1.7.4 pydantic>=2.7.0 pydantic-settings>=2.7.0 numpy>=1.24.3 spacy>=3.7.0

# Install required spaCy model
python -m spacy download en_core_web_sm
```

## Testing the Installation

Run the detailed test example to verify the installation and understand how the system works:

```bash
python examples/detailed_test.py
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

## Development

To set up for development:

```bash
# Clone the repository
git clone https://github.com/MartinMusiol1987/musiol-rag.git
cd musiol-rag

# Install development dependencies
pip install -e ".[dev]"

# Install required spaCy model
python -m spacy download en_core_web_sm
```

## License

MIT License - see LICENSE file for details. 