# Musiol RAG

A modular Retrieval-Augmented Generation (RAG) system designed for easy integration into larger projects.

## Features

- Modular design with dependency injection
- Simple interface for document management and querying
- Flexible embedding and retrieval providers
- Async support
- Easy to extend and customize

## Installation

```bash
pip install musiol-rag
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

## Architecture

The system uses a modular architecture with three main components:

1. **Embedding Provider**: Converts text into vector embeddings
2. **Database Provider**: Stores and manages text documents
3. **Retriever Provider**: Indexes and retrieves relevant documents

Each component follows a Protocol interface, making it easy to swap implementations or create custom ones.

## Customization

You can create custom providers by implementing the following protocols:

```python
class EmbeddingProvider(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...
    def encode_single(self, text: str) -> np.ndarray: ...

class DatabaseProvider(Protocol):
    async def add_text(self, text: str) -> None: ...
    async def get_texts(self) -> List[str]: ...
    async def clear(self) -> None: ...

class RetrieverProvider(Protocol):
    async def update_index(self, database: DatabaseProvider) -> None: ...
    async def get_relevant_texts(
        self, 
        query: str, 
        database: DatabaseProvider,
        k: Optional[int] = None
    ) -> List[str]: ...
```

## License

MIT License - see LICENSE file for details. 