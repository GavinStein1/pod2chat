"""
RAG (Retrieval-Augmented Generation) chat service.

Retrieves relevant chunks from vector store and generates responses using only
the retrieved context, with no external knowledge.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import BadRequestError
from vector_store import VectorStore
from embedder import Embedder
from chunk import format_ts
from prompts import RAG_CHAT_SYSTEM_MESSAGE, RAG_CHAT_USER_MESSAGE_TEMPLATE


class RAGChat:
    """RAG-based chat service with retrieval and response generation."""

    def __init__(self, db_path: str, video_url: str, api_key: Optional[str] = None):
        """
        Initialize RAG chat service.
        
        Args:
            db_path: Path to SQLite database with chunks
            video_url: Video URL to filter chunks
            api_key: OpenAI API key (optional, uses env var if not provided)
        """
        self.db_path = db_path
        self.video_url = video_url
        self.store = VectorStore(db_path)
        self.embedder = Embedder(api_key)
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5-nano"

    def chat(self, query: str, max_chunks: int = 10) -> Dict[str, Any]:
        """
        Main chat method - retrieve chunks and generate response.
        
        Args:
            query: User's question
            max_chunks: Maximum number of chunks to use for context
            
        Returns:
            Dictionary with response, chunks used, sources, and token usage
        """
        # Retrieve relevant chunks
        chunks = self._retrieve_chunks(query, max_chunks)
        
        if not chunks:
            return {
                "response": "I couldn't find relevant information in the transcript to answer your question.",
                "chunks_used": [],
                "sources": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            }
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate response
        response_text, input_tokens, output_tokens = self._generate_response(query, context)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * 0.05
        output_cost = (output_tokens / 1_000_000) * 0.40
        total_cost = input_cost + output_cost
        
        # Extract sources (timestamps)
        sources = [format_ts(chunk["start"]) for chunk in chunks]
        
        return {
            "response": response_text,
            "chunks_used": chunks,
            "sources": sources,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def _retrieve_chunks(self, query: str, max_chunks: int) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using cosine similarity + keyword re-ranking.
        
        Args:
            query: User's query
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of relevant chunks
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Step 1: Retrieve top chunks using cosine similarity (get more than needed for re-ranking)
        similar_chunks = self.store.search_similar(
            query_embedding, 
            video_url=self.video_url, 
            limit=30  # Get top 30 for re-ranking
        )
        
        if not similar_chunks:
            return []
        
        # Step 2: Re-rank using FTS5 keyword matching
        reranked_chunks = self.store.rerank_with_keywords(
            similar_chunks,
            query,
            limit=max_chunks
        )
        
        return reranked_chunks

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks as context for LLM.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for chunk in chunks:
            start_ts = format_ts(chunk["start"])
            end_ts = format_ts(chunk["end"])
            text = chunk["text"]
            context_parts.append(f"[{start_ts}-{end_ts}] {text}")
        
        return "\n\n".join(context_parts)

    def _generate_response(self, query: str, context: str) -> tuple[str, int, int]:
        """
        Generate LLM response with strict grounding.
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        system_message = RAG_CHAT_SYSTEM_MESSAGE
        user_message = RAG_CHAT_USER_MESSAGE_TEMPLATE.format(context=context, query=query)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract token usage
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            else:
                # Fallback: estimate tokens
                input_tokens = len(system_message.split()) + len(user_message.split())
                output_tokens = len(response_text.split())
            
            return response_text, input_tokens, output_tokens
            
        except BadRequestError as e:
            raise Exception(f"API error: {e}")
        except Exception as e:
            raise Exception(f"Failed to generate response: {e}")

    def close(self) -> None:
        """Close database connection."""
        if self.store:
            self.store.close()
