"""
API request and response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Schema for query requests."""
    question: str
    max_tokens: Optional[int] = 500

class QueryResponse(BaseModel):
    """Schema for query responses."""
    question: str
    context: List[str]
    answer: str

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str 