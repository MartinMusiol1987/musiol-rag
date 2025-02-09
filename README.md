# Musiol RAG

A modular Retrieval-Augmented Generation (RAG) system designed for easy integration into larger projects.

## Features

- Modular design with dependency injection
- Simple interface for document management and querying
- Efficient vector similarity search with FAISS
- Intelligent text chunking with sentence boundary detection
- Async support
- PostgreSQL storage for both documents and chunks
- Distance-based relevance scoring using cosine similarity

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database server (you need to set up and run your own PostgreSQL instance)

## Installation

To integrate this module into your project, ensure the following dependencies are installed:

```bash
pip install sentence-transformers faiss-cpu pydantic pydantic-settings numpy spacy asyncpg

# Install required spaCy model
python -m spacy download en_core_web_sm
```

## Database Setup

1. Ensure you have PostgreSQL installed and running
2. Create a database for the RAG system:
```bash
createdb rag_test  # Or your preferred database name
```
3. Set your database URL in the environment:
```bash
export DATABASE_URL="postgresql://username@localhost/rag_test"  # Replace with your credentials
```

**Note:** The `DATABASE_URL` environment variable is crucial for the script to connect to your PostgreSQL database. Ensure it is correctly set before running any tests or scripts that require database access. Below is an example of how to set it:

```bash
export DATABASE_URL="postgresql://your_username:your_password@localhost/your_database_name"
```

Replace `your_username`, `your_password`, and `your_database_name` with your actual PostgreSQL credentials and database name.

## Testing the Integration

Run the detailed test example to verify the integration and understand how the system works:

```bash
DATABASE_URL="postgresql://username@localhost/rag_test" PYTHONPATH=. python examples/detailed_test.py
```

This script provides a comprehensive demonstration of the RAG system, showing each step of the process:

1. **Document Storage**: Storing full documents in PostgreSQL
2. **Intelligent Text Chunking**: Breaking documents into semantically meaningful pieces using sentence boundary detection
3. **Chunk Storage**: Storing document chunks with references to their source documents
4. **Embedding Generation**: Converting text into vector representations
5. **Vector Search**: Finding the most relevant text chunks for a query with distance-based scoring using cosine similarity
6. **Results Inspection**: Detailed logging of the entire process with similarity scores

The system uses PostgreSQL for document and chunk storage, and FAISS for efficient similarity search.

## Configuration

Key settings can be configured through environment variables or the `config.py` file:

```python
# Embedding settings
embedding_model: str = "all-MiniLM-L6-v2"

# Retrieval settings
top_k: int = 3
chunk_size: int = 200  # Maximum size of text chunks in characters
chunk_overlap: int = 50  # Overlap between consecutive chunks

# Database settings
database_url: str  # Required PostgreSQL connection string

# FAISS settings
faiss_index_path: str = "faiss_index.bin"  # Path for storing FAISS index
```

## Architecture

The system uses a modular architecture with four main components:

1. **Text Chunker**: Splits documents into semantically meaningful chunks using spaCy's sentence boundary detection
2. **Embedding Provider**: Converts text into vector embeddings using Sentence Transformers
3. **PostgreSQL Database**: Stores both full documents and their chunks with proper relationships
4. **FAISS Retriever**: Indexes chunks and retrieves relevant ones with distance-based scoring

The retrieval process includes:
- Storing both full documents and their chunks in PostgreSQL
- Converting chunks to vector embeddings
- Building and maintaining a FAISS index for efficient similarity search
- Computing cosine similarity between query and chunk vectors for relevance scoring

## Example Output

When querying the system, you get results like this:

```
Query: "How does quantum computing work?"

Chunk 1:
Distance: 0.4261
Content: "Quantum computing is a type of computation that harnesses quantum mechanics. 
         It uses qubits which can exist in multiple states simultaneously."

Chunk 2:
Distance: 0.6489
Content: "This makes quantum computers particularly good at solving certain types 
         of problems that classical computers struggle with."
```

Lower distance scores indicate higher relevance to the query.

## License

MIT License - see LICENSE file for details.

## Performance Metrics

Below is a table showing the retrieval time for different versions of the system:

| Version (Note) | Retrieval Time |
|----------------|----------------|
| Standard       | 24.50 sec       | 
| Cached (exact query)         | 21.05 sec       |
| Cached (fuzzy query)         | ??      |

Note: The "Standard" version is the initial implementation without caching. The "Cached" version includes caching of query results, and the "Optimized" version includes both caching and a more efficient retrieval method.

**Performance Metrics**: The retrieval times were measured by running 11 queries 20 times each.

## Codebase Structure

The `musiol_rag` project is organized as follows:

```
musiol_rag/
│
├── core/
│   ├── chunking.py          # Handles text chunking with sentence boundary detection
│   ├── embeddings.py        # Manages text encoding into vector embeddings
│   ├── retrieval.py         # Implements FAISS-based retrieval system
│   └── rag.py               # RAG (Retrieval-Augmented Generation) wrapper class
│
├── database/
│   ├── base.py              # Base class for database interactions
│   └── postgresql.py        # PostgreSQL database implementation
│
├── examples/
│   ├── detailed_test.py     # Detailed test script for the RAG system
│   └── texts/               # Sample text files for testing
│
├── config.py                # Configuration settings for the project
├── LICENSE                  # License information
└── README.md                # Project documentation
```

