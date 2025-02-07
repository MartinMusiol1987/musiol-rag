"""
FastAPI-based REST API implementation.
"""
from fastapi import FastAPI, HTTPException
from ..core.rag_engine import RAGEngine
from ..database.postgresql import PostgreSQLDatabase
from .schemas import QueryRequest, QueryResponse, ErrorResponse
from ..config import settings

app = FastAPI(title="Musiol-RAG API")
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG engine on startup."""
    global rag_engine
    try:
        # Initialize database connection
        database = await PostgreSQLDatabase.from_connection_string(
            settings.database_url
        )
        
        # Initialize RAG engine
        rag_engine = RAGEngine(database=database)
        await rag_engine.initialize()
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a query through the RAG system.
    
    Args:
        request: QueryRequest containing the question
        
    Returns:
        QueryResponse containing the answer and context
    """
    if not rag_engine:
        raise HTTPException(
            status_code=500,
            detail="RAG engine not initialized"
        )
    
    try:
        result = await rag_engine.query(
            request.question,
            max_tokens=request.max_tokens
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/update-index")
async def update_index():
    """Update the FAISS index with current database content."""
    if not rag_engine:
        raise HTTPException(
            status_code=500,
            detail="RAG engine not initialized"
        )
    
    try:
        await rag_engine.initialize()
        return {"status": "success", "message": "Index updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating index: {str(e)}"
        ) 