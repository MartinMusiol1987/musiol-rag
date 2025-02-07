"""
Main RAG engine implementation.
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..database.base import BaseDatabase
from .embeddings import EmbeddingModel
from .retrieval import FAISSRetriever
from ..config import settings

class RAGEngine:
    def __init__(
        self,
        database: BaseDatabase,
        embedding_model: str = None,
        openai_api_key: str = None
    ):
        """Initialize the RAG engine."""
        self.database = database
        self.embeddings = EmbeddingModel(model_name=embedding_model)
        self.retriever = FAISSRetriever(self.embeddings)
        self.openai_client = OpenAI(api_key=openai_api_key)

    async def initialize(self):
        """Initialize the system by updating the index."""
        await self.retriever.update_index(self.database)

    async def query(
        self,
        question: str,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Process a question through the RAG system.
        
        Args:
            question: User's question
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary containing question, context, and answer
        """
        # Get relevant texts
        relevant_texts = await self.retriever.get_relevant_texts(
            question,
            self.database
        )
        
        # Combine context
        context = "\n\n".join(relevant_texts)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Get answer from GPT
        response = await self._get_completion(prompt, max_tokens)
        
        return {
            "question": question,
            "context": relevant_texts,
            "answer": response
        }

    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM."""
        return f"""Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

    async def _get_completion(
        self,
        prompt: str,
        max_tokens: int = 500
    ) -> str:
        """Get completion from OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}" 