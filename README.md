# Musiol-RAG

A powerful, modular Retrieval-Augmented Generation (RAG) system that combines database retrieval with GPT-4 for accurate, context-aware responses. This system is designed to be database-agnostic and easily integrable into existing projects.

## Features

### Core Features
- 🔍 **Vector Similarity Search**: FAISS-powered efficient similarity search
- 🧠 **Flexible Embedding Models**: Default to 'all-MiniLM-L6-v2' with support for other models
- 💾 **Database Agnostic**: Support for multiple database types (PostgreSQL implemented)
- 🤖 **GPT-4 Integration**: Advanced language model for high-quality responses
- 🌐 **REST API**: FastAPI-based interface for service integration
- ⚡ **Async Support**: Built with asyncio for high performance
- 📊 **Comprehensive Logging**: Detailed system monitoring

### Technical Capabilities
- Semantic search using FAISS vector similarity
- Configurable number of context chunks (top-k) for retrieval
- Customizable embedding models
- Persistent vector index storage
- Automatic index updates
- Error handling and recovery

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd musiol_rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DATABASE_TYPE=postgresql  # or sqlite

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# API Configuration (optional)
API_HOST=0.0.0.0
API_PORT=8000

# Embedding Configuration (optional)
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Configuration Options (`config.py`)
- `embedding_model`: Choice of sentence transformer model
- `top_k`: Number of relevant chunks to retrieve (default: 3)
- `database_type`: Database backend to use
- `api_host` and `api_port`: API server configuration
- `faiss_index_path`: Location for storing the FAISS index

## Database Setup

### PostgreSQL Setup
1. Create a database and user:
```sql
CREATE DATABASE your_database;
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE your_database TO your_user;
```

2. Create the required table:
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'
);
```

## Usage

### As a Library

1. Basic Usage:
```python
import asyncio
from musiol_rag.database.postgresql import PostgreSQLDatabase
from musiol_rag.core.rag_engine import RAGEngine

async def main():
    # Initialize database
    database = await PostgreSQLDatabase.from_connection_string(
        "postgresql://user:password@localhost:5432/dbname"
    )
    
    # Initialize RAG
    rag = RAGEngine(
        database=database,
        openai_api_key="your-api-key"
    )
    
    # Initialize system
    await rag.initialize()
    
    # Query
    result = await rag.query("Your question here?")
    print(result["answer"])

asyncio.run(main())
```

### As a Service

1. Start the API server:
```bash
uvicorn musiol_rag.api.rest:app --host 0.0.0.0 --port 8000
```

2. Use the API endpoints:
- Query endpoint:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here?", "max_tokens": 500}'
```

- Update index endpoint:
```bash
curl -X POST "http://localhost:8000/update-index"
```

## Core Components

### 1. RAG Engine (`core/rag_engine.py`)
The main orchestrator that:
- Manages the retrieval process
- Handles LLM interactions
- Combines context with queries
- Formats responses

### 2. Embedding System (`core/embeddings.py`)
Handles text embeddings with features:
- Multiple model support
- Caching capabilities
- Batch processing
- Dimension management

### 3. Retrieval System (`core/retrieval.py`)
FAISS-based retrieval with:
- Efficient similarity search
- Index persistence
- Dynamic updates
- Configurable retrieval size

### 4. Database Layer (`database/`)
Abstract database interface with:
- Connection pooling
- Async operations
- Error handling
- Metadata support

### 5. API Layer (`api/`)
FastAPI implementation providing:
- RESTful endpoints
- Request validation
- Error handling
- Swagger documentation

## Advanced Features

### Custom Embedding Models
Change the embedding model:
```python
rag = RAGEngine(
    database=database,
    embedding_model="all-mpnet-base-v2"  # Higher quality, slower
)
```

### Adjusting Retrieved Context
Modify the number of context chunks:
```python
# In config.py or environment
settings.top_k = 5  # Retrieve more context
```

### Custom Database Implementation
Implement new databases by extending `BaseDatabase`:
```python
class YourDatabase(BaseDatabase):
    async def get_texts(self, query: Optional[str] = None) -> List[str]:
        # Your implementation
        pass
    # ... implement other methods
```

## Error Handling

The system includes comprehensive error handling:
- Database connection errors
- Embedding model errors
- LLM API errors
- Index corruption detection
- API request validation

## Performance Considerations

- FAISS index is memory-resident for fast retrieval
- Connection pooling for database operations
- Async operations for better concurrency
- Batch processing for embeddings
- Configurable caching

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here] 