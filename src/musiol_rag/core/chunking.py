"""
Text chunking module with intelligent sentence boundary detection.
"""
from typing import List, Optional
import spacy
from ..config import settings

class TextChunker:
    """
    Intelligent text chunker that uses spaCy for sentence boundary detection.
    This ensures that chunks preserve semantic context by respecting sentence boundaries.
    """
    
    def __init__(self, model: str = "en_core_web_sm", max_chunk_size: Optional[int] = None):
        """
        Initialize the text chunker.
        
        Args:
            model: spaCy model to use for sentence detection
            max_chunk_size: Maximum size of a chunk in characters (defaults to settings.chunk_size)
        """
        self.nlp = spacy.load(model, disable=["ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        # Only enable sentence segmentation for better performance
        self.nlp.enable_pipe("senter")
        self.max_chunk_size = max_chunk_size or settings.chunk_size
        
    def create_chunks(self, text: str) -> List[str]:
        """
        Create chunks from text using sentence boundary detection.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks that respect sentence boundaries
        """
        # Process the text with spaCy
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)
            
            # If a single sentence is longer than max_chunk_size,
            # we need to split it using a sliding window
            if sent_length > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence using sliding window
                for i in range(0, sent_length, self.max_chunk_size // 2):
                    chunk = sent_text[i:i + self.max_chunk_size]
                    if chunk:
                        chunks.append(chunk)
                continue
            
            # If adding this sentence would exceed max_chunk_size,
            # save current chunk and start a new one
            if current_length + sent_length + 1 > self.max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Add sentence to current chunk
            current_chunk.append(sent_text)
            current_length += sent_length + 1  # +1 for space
        
        # Add any remaining text
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @classmethod
    def from_settings(cls) -> 'TextChunker':
        """
        Create a TextChunker instance using settings from config.
        
        Returns:
            TextChunker instance
        """
        return cls(max_chunk_size=settings.chunk_size) 