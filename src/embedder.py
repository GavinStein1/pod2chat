"""
Embedding generation service using OpenAI embeddings API.

Generates embeddings for text chunks to enable semantic search.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import BadRequestError


class Embedder:
    """Embedding generation service with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize embedder with OpenAI API key.
        If not provided, will try to get from OPENAI_API_KEY environment variable.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"  # Cost-effective embedding model
        
        # Track token usage for embeddings (if needed for cost calculation)
        self._total_tokens = 0

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding
            # Track usage if available
            if hasattr(response, 'usage'):
                self._total_tokens += response.usage.total_tokens
            return embedding
        except Exception as e:
            print(f"    Warning: Embedding generation failed: {e}")
            raise

    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks using batch processing.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            batch_size: Number of texts to embed in each batch (OpenAI supports up to 2048)
            
        Returns:
            List of chunks with 'embedding' key added
        """
        if not chunks:
            return []
        
        print(f"    Embedding {len(chunks)} chunks...")
        
        # Prepare texts for embedding
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Track usage
                if hasattr(response, 'usage'):
                    self._total_tokens += response.usage.total_tokens
                
                print(f"      Embedded batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)")
                
                # Small delay between batches to avoid rate limits (embeddings have higher limits)
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            except BadRequestError as e:
                if "rate_limit_exceeded" in str(e).lower():
                    print(f"      Rate limit exceeded, waiting 1 second...")
                    time.sleep(1.0)
                    # Retry this batch
                    i -= batch_size
                    continue
                else:
                    print(f"    Warning: Batch {batch_num} failed: {e}")
                    # Fill with empty embeddings for failed batch
                    all_embeddings.extend([[]] * len(batch_texts))
            except Exception as e:
                print(f"    Warning: Batch {batch_num} failed: {e}")
                # Fill with empty embeddings for failed batch
                all_embeddings.extend([[]] * len(batch_texts))
        
        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding
            chunks_with_embeddings.append(chunk_copy)
        
        print(f"    Embedding complete: {len([c for c in chunks_with_embeddings if c.get('embedding')])} chunks embedded")
        
        return chunks_with_embeddings
