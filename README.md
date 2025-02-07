# Musiol RAG

A modular Retrieval-Augmented Generation (RAG) system designed for easy integration into larger projects.

## Features

- Modular design with dependency injection
- Simple interface for document management and querying
- Flexible embedding and retrieval providers
- Async support
- Easy to extend and customize

## Installation

There are several ways to integrate this package into your project:

### 1. Install from GitHub (Recommended during development)

```bash
pip install git+https://github.com/MartinMusiol1987/musiol-rag.git
```

### 2. Local Development Install

If you want to modify the package while using it:

```bash
# Clone the repository
git clone https://github.com/MartinMusiol1987/musiol-rag.git
cd musiol-rag

# Install in editable mode
pip install -e .
```

### 3. Copy Required Components

If you need to heavily customize the implementation or can't use pip install:

1. Copy the `src/musiol_rag` directory into your project
2. Install the required dependencies:
```bash
pip install sentence-transformers>=2.2.2 faiss-cpu>=1.7.4 pydantic>=2.7.0 pydantic-settings>=2.7.0 numpy>=1.24.3
```

## Quick Start

```python
import asyncio
from musiol_rag.core.embeddings import EmbeddingModel
from musiol_rag.core.retrieval import FAISSRetriever
from musiol_rag.core.rag import RAGWrapper
from musiol_rag.database.memory import InMemoryDatabase

async def main():
    # Initialize components
    embedding_model = EmbeddingModel()
    database = InMemoryDatabase()
    retriever = FAISSRetriever(embedding_model)
    
    # Create RAG wrapper
    rag = RAGWrapper(
        embedding_provider=embedding_model,
        database_provider=database,
        retriever_provider=retriever
    )
    
    # Add documents
    await rag.add_document("Your text document here...")
    
    # Query
    results = await rag.query("Your query here", k=3)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing the Installation

After installation, you can run the example script to verify everything works:

```bash
# Clone the repository if you haven't already
git clone https://github.com/MartinMusiol1987/musiol-rag.git
cd musiol-rag

# Run the example
python examples/rag_example.py
```

## Integration Patterns

### 1. Basic Integration

Use the provided components as is:

```python
from musiol_rag.core.rag import RAGWrapper
from musiol_rag.core.embeddings import EmbeddingModel
from musiol_rag.core.retrieval import FAISSRetriever
from musiol_rag.database.memory import InMemoryDatabase

# Create and use the RAG system
rag = RAGWrapper(
    embedding_provider=EmbeddingModel(),
    database_provider=InMemoryDatabase(),
    retriever_provider=FAISSRetriever(embedding_model)
)
```

### 2. Custom Database Integration

Implement your own database provider:

```python
from musiol_rag.core.rag import DatabaseProvider
from typing import List

class YourDatabaseProvider(DatabaseProvider):
    async def add_text(self, text: str) -> None:
        # Your implementation
        ...
    
    async def get_texts(self) -> List[str]:
        # Your implementation
        ...
    
    async def clear(self) -> None:
        # Your implementation
        ...
```

### 3. Custom Embedding Provider

Use a different embedding model:

```python
from musiol_rag.core.rag import EmbeddingProvider
import numpy as np
from typing import List

class YourEmbeddingProvider(EmbeddingProvider):
    def encode(self, texts: List[str]) -> np.ndarray:
        # Your implementation
        ...
    
    def encode_single(self, text: str) -> np.ndarray:
        # Your implementation
        ...
```

## Architecture

The system uses a modular architecture with three main components:

1. **Embedding Provider**: Converts text into vector embeddings
2. **Database Provider**: Stores and manages text documents
3. **Retriever Provider**: Indexes and retrieves relevant documents

Each component follows a Protocol interface, making it easy to swap implementations or create custom ones.

## Development

To set up for development:

```bash
# Clone the repository
git clone https://github.com/MartinMusiol1987/musiol-rag.git
cd musiol-rag

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT License - see LICENSE file for details. 