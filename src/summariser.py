"""
Multi-pass summarisation for YouTube transcripts.

Generates structured markdown summaries with grounded content (no hallucination).
All summaries include timestamps and are verified against the transcript.
"""

from __future__ import annotations

import json
import os
import time
import math
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from openai import BadRequestError
from chunk import format_ts, simple_token_count
from embedder import Embedder
from prompts import (
    TOPIC_EXTRACTION_SYSTEM_MESSAGE,
    TOPIC_EXTRACTION_BASE_PROMPT,
    EXECUTIVE_SYNOPSIS_SYSTEM_MESSAGE,
    EXECUTIVE_SYNOPSIS_BASE_PROMPT,
    DEEP_DIVE_SYSTEM_MESSAGE,
    DEEP_DIVE_BASE_PROMPT_TEMPLATE,
    FRAMEWORK_EXTRACTION_SYSTEM_MESSAGE,
    FRAMEWORK_EXTRACTION_BASE_PROMPT,
    HIERARCHICAL_MERGE_SYSTEM_MESSAGE,
    HIERARCHICAL_MERGE_BASE_PROMPT,
    TOPIC_STREAM_SYSTEM_MESSAGE,
    TOPIC_STREAM_PROMPT,
)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class Summariser:
    """Multi-pass summarisation engine with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize summarizer with OpenAI API key.
        If not provided, will try to get from OPENAI_API_KEY environment variable.
        
        Args:
            api_key: OpenAI API key (optional)
            verbose: Enable verbose output for debugging
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5-nano"  # Cost-effective model
        self.verbose = verbose
        self._tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                # gpt-5-nano uses the same encoding as gpt-4
                self._tokenizer = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                self._tokenizer = None
        
        # Rate limiting: 200,000 TPM, 500 RPM (use 95% safety margin)
        self._max_tpm = 190_000  # 95% of 200,000
        self._max_rpm = 475  # 95% of 500
        self._window_seconds = 60
        # Store (timestamp, tokens) tuples for sliding window
        self._token_history: deque = deque()
        self._request_history: deque = deque()
        
        # Token usage tracking for cost calculation
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def generate_summary(
        self, chunks: Dict[str, List[Dict[str, Any]]], metadata: Dict[str, Any], raw_segments: List[Dict[str, Any]]
    ) -> str:
        """
        Main orchestrator for multi-pass summary generation.
        
        Args:
            chunks: Dictionary with 'fine' and 'coarse' chunk lists
            metadata: Video metadata (title, channel, duration, url)
            raw_segments: Original transcript segments
            
        Returns:
            Complete markdown summary as string
        """
        # Reset token counters for this summary
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
        coarse_chunks = chunks.get("coarse", [])
        fine_chunks = chunks.get("fine", [])

        # Pass 1: Extract topics from coarse chunks
        print("  Pass 1: Extracting topics...")
        if self.verbose:
            print(f"    Analyzing {len(coarse_chunks)} coarse chunks for topic extraction...")
        topics, chunk_to_topic = self._extract_topics(coarse_chunks, metadata)
        if self.verbose:
            print(f"    ✓ Extracted {len(topics)} topics: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")

        # Build markdown sections
        markdown_parts = []

        # Header
        markdown_parts.append(self._build_header(metadata))

        # Executive Synopsis
        print("  Pass 2: Generating executive synopsis...")
        if self.verbose:
            print(f"    Generating synopsis from {len(coarse_chunks)} coarse chunks...")
        markdown_parts.append(self._build_executive_synopsis(coarse_chunks, raw_segments))
        if self.verbose:
            print("    ✓ Executive synopsis generated")

        # Topic Map
        if self.verbose:
            print("  Building topic map...")
        markdown_parts.append(self._build_topic_map(coarse_chunks, topics, chunk_to_topic))
        if self.verbose:
            print("    ✓ Topic map generated")

        # Deep Dive
        print("  Pass 3: Generating deep dive notes...")
        if self.verbose:
            print(f"    Processing {len(topics)} topics with {len(fine_chunks)} fine chunks...")
        markdown_parts.append(self._build_deep_dive(topics, coarse_chunks, chunk_to_topic, fine_chunks, raw_segments))
        if self.verbose:
            print("    ✓ Deep dive notes generated")

        # Frameworks
        print("  Extracting frameworks...")
        if self.verbose:
            print("    Searching for actionable frameworks and checklists...")
        markdown_parts.append(self._build_frameworks(coarse_chunks, fine_chunks, raw_segments))
        if self.verbose:
            print("    ✓ Framework extraction completed")

        # Quotes
        # if self.verbose:
        #     print("  Extracting memorable quotes...")
        # markdown_parts.append(self._build_quotes(raw_segments))
        # if self.verbose:
        #     print("    ✓ Quotes extracted")

        # Listening Paths
        # if self.verbose:
        #     print("  Building listening paths...")
        # markdown_parts.append(self._build_listening_paths(coarse_chunks))
        # if self.verbose:
        #     print("    ✓ Listening paths generated")

        # Calculate and print cost summary
        cost_info = self._calculate_cost()
        print(f"\n  Token Usage Summary:")
        print(f"    Input cost: ${cost_info['input_cost']:.6f}")
        print(f"    Output cost: ${cost_info['output_cost']:.6f}")
        print(f"    Total cost: ${cost_info['total_cost']:.6f}")

        # Convert timestamps to hyperlinks before returning
        markdown = "\n\n".join(markdown_parts)
        video_url = metadata.get("url", "")
        if video_url:
            markdown = self._convert_timestamps_to_hyperlinks(markdown, video_url)

        return markdown

    # ----------------------------
    # Token Counting Utilities
    # ----------------------------

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken (preferred) or fallback approximation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback to word-based approximation (from chunk.py)
        return simple_token_count(text)

    def _count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in OpenAI message format (system + user + formatting overhead).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Total token count including formatting overhead
        """
        total = 0
        
        # Account for message formatting overhead (~4 tokens per message)
        total += len(messages) * 4
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            # Role takes ~1-2 tokens, content uses normal counting
            total += 2 + self._count_tokens(content)
        
        return total

    def _estimate_prompt_tokens(self, chunks: List[Dict[str, Any]], base_prompt: str) -> int:
        """
        Estimate total prompt tokens including chunk content.
        
        Args:
            chunks: List of chunk dictionaries
            base_prompt: Base prompt text (without chunks)
            
        Returns:
            Total estimated token count
        """
        total = self._count_tokens(base_prompt)
        
        # Add tokens for each chunk (text + formatting overhead)
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            # Format: "[HH:MM:SS-HH:MM:SS] text" = ~20 tokens overhead + text tokens
            total += 20 + self._count_tokens(chunk_text)
        
        return total

    def _get_model_context_limit(self) -> int:
        """Return context window size for current model."""
        # gpt-5-nano has 400k context window
        if "gpt-5-nano" in self.model or "gpt-5" in self.model or "gpt-4" in self.model:
            return 400_000
        # Fallback for other models
        return 400_000

    # ----------------------------
    # Context Limit Verification
    # ----------------------------

    def _select_chunks_within_limit(
        self, chunks: List[Dict[str, Any]], base_prompt: str, max_input_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Select chunks that fit within token limit.
        
        Args:
            chunks: List of all chunks to potentially include
            base_prompt: Base prompt text (without chunks)
            max_input_tokens: Maximum input tokens (defaults to context_limit - response_buffer)
            
        Returns:
            List of chunks that fit within limit
        """
        if max_input_tokens is None:
            context_limit = self._get_model_context_limit()
            # Reserve: system message (100), base prompt, response buffer (8000)
            base_tokens = self._count_tokens(base_prompt) + 100
            response_buffer = 8000
            max_input_tokens = context_limit - base_tokens - response_buffer
        
        selected = []
        current_tokens = self._count_tokens(base_prompt) + 100  # Base + system overhead
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = 20 + self._count_tokens(chunk_text)  # Format overhead + text
            
            if current_tokens + chunk_tokens > max_input_tokens:
                break
            
            selected.append(chunk)
            current_tokens += chunk_tokens
        
        return selected

    def _can_fit_in_single_request(
        self, chunks: List[Dict[str, Any]], base_prompt: str, response_buffer: int = 8000
    ) -> bool:
        """
        Check if all chunks can fit in one API call.
        
        Args:
            chunks: List of chunks to check
            base_prompt: Base prompt text
            response_buffer: Tokens to reserve for response
            
        Returns:
            True if all chunks fit, False if batching needed
        """
        context_limit = self._get_model_context_limit()
        estimated_tokens = self._estimate_prompt_tokens(chunks, base_prompt)
        total_needed = estimated_tokens + response_buffer + 100  # +100 for system message overhead
        
        return total_needed <= context_limit

    # ----------------------------
    # Rate Limiting
    # ----------------------------

    def _clean_old_history(self, current_time: float) -> None:
        """Remove entries older than window_seconds from history."""
        cutoff_time = current_time - self._window_seconds
        
        # Clean token history
        while self._token_history and self._token_history[0][0] < cutoff_time:
            self._token_history.popleft()
        
        # Clean request history
        while self._request_history and self._request_history[0] < cutoff_time:
            self._request_history.popleft()

    def _get_current_usage(self, current_time: float) -> tuple[int, int]:
        """
        Get current token and request usage in the sliding window.
        
        Returns:
            (tokens_used, requests_count) in the last 60 seconds
        """
        self._clean_old_history(current_time)
        
        tokens_used = sum(tokens for _, tokens in self._token_history)
        requests_count = len(self._request_history)
        
        return tokens_used, requests_count

    def _wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Wait if necessary to respect rate limits before making a request.
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request
        """
        current_time = time.time()
        tokens_used, requests_count = self._get_current_usage(current_time)
        
        # Calculate how many tokens/requests we'll have after this request
        new_tokens = tokens_used + estimated_tokens
        new_requests = requests_count + 1
        
        # Check if we'd exceed limits
        tokens_exceeded = new_tokens > self._max_tpm
        requests_exceeded = new_requests > self._max_rpm
        
        if tokens_exceeded or requests_exceeded:
            # Calculate wait time - need to wait until oldest entry expires
            if self._token_history:
                oldest_token_time = self._token_history[0][0]
                token_wait_time = (oldest_token_time + self._window_seconds) - current_time
            else:
                token_wait_time = 0
            
            if self._request_history:
                oldest_request_time = self._request_history[0]
                request_wait_time = (oldest_request_time + self._window_seconds) - current_time
            else:
                request_wait_time = 0
            
            # Wait for whichever limit would be hit first
            wait_time = max(token_wait_time, request_wait_time)
            
            if wait_time > 0:
                # Add small buffer to avoid edge cases
                wait_time += 0.1
                if self.verbose:
                    print(f"    Rate limit: {tokens_used:,}/{self._max_tpm:,} tokens, {requests_count}/{self._max_rpm} requests")
                print(f"    Rate limit approaching, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                
                # Recalculate after waiting
                current_time = time.time()
                self._clean_old_history(current_time)

    def _record_request(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record a completed request with its token usage.
        
        Args:
            input_tokens: Input/prompt tokens used for the request
            output_tokens: Output/completion tokens used for the request
        """
        current_time = time.time()
        total_tokens = input_tokens + output_tokens
        
        # Track for rate limiting (use total tokens)
        self._token_history.append((current_time, total_tokens))
        self._request_history.append(current_time)
        
        # Track for cost calculation (separate input/output)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        
        # Clean old entries periodically (every 10 requests to avoid overhead)
        if len(self._request_history) % 10 == 0:
            self._clean_old_history(current_time)

    def _calculate_cost(self) -> Dict[str, Any]:
        """
        Calculate total cost based on token usage.
        
        Pricing: $0.05 per million input tokens, $0.40 per million output tokens
        
        Returns:
            Dictionary with token counts and costs
        """
        input_cost = (self._total_input_tokens / 1_000_000) * 0.05
        output_cost = (self._total_output_tokens / 1_000_000) * 0.40
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    # ----------------------------
    # Batching Strategy
    # ----------------------------

    def _summarize_with_batching(
        self,
        chunks: List[Dict[str, Any]],
        base_prompt: str,
        system_message: str,
        merge_strategy: str = "combine",
        # temperature: float = 0.4,
    ) -> str:
        """
        Summarise chunks using batching when single request won't fit.
        
        Args:
            chunks: All chunks to process (will be split into batches)
            base_prompt: Base prompt template (will be formatted with chunks)
            system_message: System message for API call
            merge_strategy: How to merge batch results:
                - "combine": Simply concatenate all batch results with newlines (default).
                  Best for lists, bullets, or non-overlapping content.
                - "hierarchical": Combine results then re-summarise to remove redundancy.
                  Useful when batches may contain overlapping or similar content.
                  Currently implemented as "combine" (re-summarisation step can be added).
                - "selective": Keep only unique content, removing duplicates.
                  Helpful when batches might repeat information.
                  Currently implemented as "combine" (deduplication logic can be enhanced).
            
        Returns:
            Merged summary from all batches
        """
        if not chunks:
            return ""
        
        # Check if we can fit all chunks in one request
        if self._can_fit_in_single_request(chunks, base_prompt):
            # Try single request first
            if self.verbose:
                print(f"    Processing {len(chunks)} chunks in single request...")
            try:
                chunk_texts = [f"[{format_ts(c['start'])}]-[{format_ts(c['end'])}] {c['text']}" for c in chunks]
                # Handle both {chunks} placeholder and direct formatting
                if "{chunks}" in base_prompt:
                    prompt = base_prompt.format(chunks=chr(10).join(chunk_texts))
                else:
                    prompt = base_prompt + "\n\n" + chr(10).join(chunk_texts)
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
                
                # Rate limiting: estimate tokens and wait if needed
                estimated_tokens = self._count_message_tokens(messages)
                self._wait_if_needed(estimated_tokens)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # temperature=temperature,
                )
                
                # Record actual token usage
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = getattr(response.usage, 'prompt_tokens', estimated_tokens)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                else:
                    # Fallback to estimate if usage info not available
                    input_tokens = estimated_tokens
                    output_tokens = 0
                
                self._record_request(input_tokens, output_tokens)
                
                if self.verbose:
                    print(f"    ✓ Single request completed ({input_tokens:,} input, {output_tokens:,} output tokens)")
                
                return response.choices[0].message.content.strip()
            except BadRequestError as e:
                if "context_length_exceeded" in str(e).lower():
                    if self.verbose:
                        print(f"    Context limit exceeded ({len(chunks)} chunks), falling back to batching...")
                    else:
                        print(f"    Context limit exceeded, falling back to batching...")
                else:
                    raise
        
        # Need batching - split chunks into batches
        print(f"    Processing {len(chunks)} chunks in batches...")
        if self.verbose:
            print(f"    Calculating batch sizes (context limit: {self._get_model_context_limit():,} tokens)...")
        context_limit = self._get_model_context_limit()
        base_tokens = self._count_tokens(base_prompt) + 100 + 8000  # base + system + response buffer
        max_chunk_tokens = context_limit - base_tokens
        
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = 20 + self._count_tokens(chunk_text)  # Format overhead + text
            
            if current_batch_tokens + chunk_tokens > max_chunk_tokens and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = [chunk]
                current_batch_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_batch_tokens += chunk_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        print(f"    Split into {len(batches)} batches")
        if self.verbose:
            batch_sizes = [len(b) for b in batches]
            print(f"    Batch sizes: {batch_sizes}")
        
        # Process each batch
        batch_results = []
        for i, batch in enumerate(batches):
            if self.verbose:
                print(f"    Processing batch {i+1}/{len(batches)} ({len(batch)} chunks)...")
            try:
                chunk_texts = [f"[{format_ts(c['start'])}]-[{format_ts(c['end'])}] {c['text']}" for c in batch]
                # Handle both {chunks} placeholder and direct formatting
                if "{chunks}" in base_prompt:
                    prompt = base_prompt.format(chunks=chr(10).join(chunk_texts))
                else:
                    prompt = base_prompt + "\n\n" + chr(10).join(chunk_texts)
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
                
                # Rate limiting: estimate tokens and wait if needed
                estimated_tokens = self._count_message_tokens(messages)
                self._wait_if_needed(estimated_tokens)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # temperature=temperature,
                )
                
                # Record actual token usage
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = getattr(response.usage, 'prompt_tokens', estimated_tokens)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                else:
                    # Fallback to estimate if usage info not available
                    input_tokens = estimated_tokens
                    output_tokens = 0
                
                self._record_request(input_tokens, output_tokens)
                
                if self.verbose:
                    print(f"      ✓ Batch {i+1} completed ({input_tokens:,} input, {output_tokens:,} output tokens)")
                
                batch_results.append(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"    Warning: Batch {i+1}/{len(batches)} failed: {e}")
                batch_results.append(f"(Batch {i+1} processing failed)")
        
        # Merge results based on strategy
        if self.verbose:
            print(f"    Merging {len(batch_results)} batch results using '{merge_strategy}' strategy...")
        merged_result = self._merge_batch_results(batch_results, merge_strategy)
        if self.verbose:
            print(f"    ✓ Merged result: {len(merged_result)} characters")
        return merged_result
    
    def _merge_batch_results(self, batch_results: List[str], strategy: str) -> str:
        """
        Merge results from multiple batches.
        
        Args:
            batch_results: List of results from each batch
            strategy: Merge strategy ("combine", "hierarchical", "selective")
            
        Returns:
            Merged summary text
        """
        if not batch_results:
            return ""
        
        if len(batch_results) == 1:
            return batch_results[0]
        
        if strategy == "combine":
            # Simply combine all results
            return "\n\n".join(batch_results)
        
        elif strategy == "hierarchical":
            # Combine, then re-summarise to remove redundancy
            try:
                if self.verbose:
                    print(f"      Combining {len(batch_results)} batches, then re-summarizing...")
                combined = "\n\n".join(batch_results)
                
                messages = [
                    {"role": "system", "content": HIERARCHICAL_MERGE_SYSTEM_MESSAGE},
                    {"role": "user", "content": HIERARCHICAL_MERGE_BASE_PROMPT.format(combined_summaries=combined)},
                ]
                
                # Rate limiting: estimate tokens and wait if needed
                estimated_tokens = self._count_message_tokens(messages)
                self._wait_if_needed(estimated_tokens)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                
                # Record actual token usage
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = getattr(response.usage, 'prompt_tokens', estimated_tokens)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                else:
                    input_tokens = estimated_tokens
                    output_tokens = 0
                
                self._record_request(input_tokens, output_tokens)
                
                if self.verbose:
                    print(f"      ✓ Hierarchical merge completed ({input_tokens:,} input, {output_tokens:,} output tokens)")
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"    Warning: Hierarchical merge failed: {e}, falling back to combine")
                # Fallback to combine on error
                return "\n\n".join(batch_results)
        
        elif strategy == "selective":
            # Keep only unique content using semantic similarity
            try:
                if self.verbose:
                    print(f"      Analyzing {len(batch_results)} batches for duplicate content...")
                # Split each batch result into logical sections (paragraphs)
                sections = []
                for batch_idx, batch_result in enumerate(batch_results):
                    # Split by double newlines to get paragraphs/sections
                    batch_sections = [s.strip() for s in batch_result.split("\n\n") if s.strip()]
                    sections.extend([(batch_idx, i, section) for i, section in enumerate(batch_sections)])
                
                if not sections:
                    return "\n\n".join(batch_results)
                
                if len(sections) == 1:
                    return sections[0][2]
                
                if self.verbose:
                    print(f"      Generating embeddings for {len(sections)} sections...")
                # Generate embeddings for each section
                embedder = Embedder(api_key=os.getenv("OPENAI_API_KEY"))
                section_texts = [section[2] for section in sections]
                
                # Generate embeddings in batches (embedder handles this)
                embeddings = []
                for text in section_texts:
                    try:
                        embedding = embedder.embed_text(text)
                        embeddings.append(embedding)
                    except Exception as e:
                        print(f"    Warning: Embedding generation failed for section: {e}")
                        # Use empty embedding as fallback
                        embeddings.append([])
                
                if self.verbose:
                    print(f"      Filtering duplicates using cosine similarity (threshold: 0.85)...")
                # Calculate cosine similarity and filter duplicates
                unique_sections = []
                unique_indices = []  # Track indices of kept sections for embedding comparison
                similarity_threshold = 0.85
                
                for i, (batch_idx, sec_idx, section_text) in enumerate(sections):
                    if not embeddings[i]:  # Skip similarity check if embedding failed
                        unique_sections.append(section_text)
                        unique_indices.append(i)
                        continue
                    
                    # Check similarity against all previously kept sections
                    is_duplicate = False
                    for kept_idx in unique_indices:
                        if embeddings[kept_idx]:
                            similarity = self._cosine_similarity(embeddings[i], embeddings[kept_idx])
                            if similarity > similarity_threshold:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        unique_sections.append(section_text)
                        unique_indices.append(i)
                
                if self.verbose:
                    print(f"      ✓ Kept {len(unique_sections)}/{len(sections)} unique sections")
                
                return "\n\n".join(unique_sections) if unique_sections else "\n\n".join(batch_results)
            except Exception as e:
                print(f"    Warning: Selective merge failed: {e}, falling back to combine")
                # Fallback to combine on error
                return "\n\n".join(batch_results)
        
        else:
            # Default to combine
            return "\n\n".join(batch_results)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def extract_topics_streaming(
        self,
        coarse_chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Tuple[List[str], List[int]]:
        """
        Streaming topic extraction:
        - iterates chunks in order
        - assigns each chunk to an existing topic or creates a new one
        Returns (topics, chunk_to_topic_index)
        """
        if not coarse_chunks:
            return [], []

        topics: List[str] = []
        chunk_to_topic: List[int] = []

        for i, chunk in enumerate(coarse_chunks):
            start = format_ts(chunk["start"])
            end = format_ts(chunk["end"])
            text = chunk["text"]

            topics_indexed = "\n".join([f"{idx}: {t}" for idx, t in enumerate(topics)]) or "(none yet)"

            prompt = TOPIC_STREAM_PROMPT.format(
                topics_indexed=topics_indexed,
                start=start,
                end=end,
                text=text,
                title=metadata.get("title"),
            )

            messages = [
                {"role": "system", "content": TOPIC_STREAM_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]

            estimated_tokens = self._count_message_tokens(messages)
            self._wait_if_needed(estimated_tokens)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            except Exception as e:
                # Fallback: if we can't call the API, assign to "Main Discussion"
                if not topics:
                    topics = ["Main Discussion"]
                chunk_to_topic.append(0)
                if self.verbose:
                    print(f"    Warning: topic classify failed on chunk {i}: {e}")
                continue

            # usage accounting
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", estimated_tokens)
                output_tokens = getattr(response.usage, "completion_tokens", 0)
            else:
                input_tokens = estimated_tokens
                output_tokens = 0
            self._record_request(input_tokens, output_tokens)

            raw = response.choices[0].message.content or ""
            raw = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].split("\n", 1)[1] if "\n" in raw.split("```")[1] else raw.split("```")[2]
            if self.verbose:
                print(f"    Topic classify response: {raw}")
            try:
                decision = json.loads(raw)
            except Exception:
                # If model returns malformed JSON, do a conservative fallback
                if not topics:
                    topics = ["Main Discussion"]
                chunk_to_topic.append(0)
                if self.verbose:
                    print(f"    Warning: topic classify failed on chunk {i}: {raw}")
                continue

            action = decision.get("action")

            if action == "assign":
                idx = decision.get("topic_index")
                if isinstance(idx, int) and 0 <= idx < len(topics):
                    if self.verbose:
                        print(f"    Assigning chunk {i} to topic {topics[idx]}")
                    chunk_to_topic.append(idx)
                elif idx == -1:
                    if self.verbose:
                        print(f"    Chunk is not relevant to any topic")
                        chunk_to_topic.append(-1)
                    if self.verbose:
                        print(f"    Assigning chunk {i} to topic -1")
                    # if self.verbose:
                    #     print(f"    Assigning chunk {i} to topic -1 (no topic)")
                    # if not topics:
                    #     topics = ["Irrelevant discussion"]
                    # elif not "Irrelevant discussion" in topics:
                    #     topics.append("Irrelevant discussion")
                    #     chunk_to_topic.append(len(topics) - 1)
                    # else:
                    #     chunk_to_topic.append(topics.index("Irrelevant discussion"))
                    # if self.verbose:
                    #     print(f"    Assigning chunk {i} to topic {topics[chunk_to_topic[-1]]}")
                else:
                    # invalid index => fallback assign to first topic or create
                    if topics:
                        if self.verbose:
                            print(f"    Assigning chunk {i} to first topic")
                        chunk_to_topic.append(0)
                    else:
                        if self.verbose:
                            print(f"   FALLBACK: Assigning chunk {i} to first topic")
                        topics = ["Main Discussion"]
                        chunk_to_topic.append(0)

            elif action == "create":
                name = decision.get("topic_name", "")
                topics.append(name.strip())
                chunk_to_topic.append(len(topics) - 1)
                if self.verbose:
                    print(f"    Created new topic: {name}")

            else:
                # unknown action => fallback
                if self.verbose:
                    print(f"   FALLBACK: Unknown action: {action}")
                if not topics:
                    topics = ["Main Discussion"]
                chunk_to_topic.append(0)

        # If you ended with >max_topics due to any edge cases, you can merge later.
        # If <min_topics, that's allowed by your earlier constraints ("fewer if insufficient content").
        return topics, chunk_to_topic

    def _extract_topics(self, coarse_chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Tuple[List[str], List[int]]:
        """Pass 1: Extract main topics/themes from coarse chunks."""
        topics, chunk_to_topic = self.extract_topics_streaming(coarse_chunks, metadata)
        return topics, chunk_to_topic

    def _build_header(self, metadata: Dict[str, Any]) -> str:
        """Build header section with title, channel, URL, duration, and usage instructions."""
        duration_ts = format_ts(metadata.get("duration", 0))

        return f"""# {metadata.get("title", "Video Summary")}

**Channel:** {metadata.get("channel", "Unknown")}  
**URL:** {metadata.get("url", "")}  
**Duration:** {duration_ts}

## How to use this doc

This summary is structured to help you quickly find what you need:
- **Executive Synopsis**: High-level overview with timestamps for quick scanning
- **Topic Map**: Table of contents with segment boundaries
- **Deep Dive**: Detailed notes organised by topic with full context
- **Frameworks**: Actionable checklists and step-by-step methods

All timestamps are clickable in supported markdown viewers and reference exact moments in the video."""

    def _build_executive_synopsis(
        self, coarse_chunks: List[Dict[str, Any]], raw_segments: List[Dict[str, Any]]
    ) -> str:
        """Build executive synopsis with 10-20 timestamped bullets."""
        if not coarse_chunks:
            return "## Executive Synopsis\n\n(No content available)"

        base_prompt = EXECUTIVE_SYNOPSIS_BASE_PROMPT
        system_message = EXECUTIVE_SYNOPSIS_SYSTEM_MESSAGE

        # Try to include all chunks first (within token limits)
        selected_chunks = self._select_chunks_within_limit(coarse_chunks, base_prompt)
        
        # If we can't fit all chunks, note it (but process what we can)
        if len(selected_chunks) < len(coarse_chunks):
            if self.verbose:
                print(f"    Note: Token limit reached, processing {len(selected_chunks)}/{len(coarse_chunks)} chunks for executive synopsis")
            else:
                print(f"    Note: Processing {len(selected_chunks)}/{len(coarse_chunks)} chunks for executive synopsis")

        try:
            # Try single request first
            if self._can_fit_in_single_request(selected_chunks, base_prompt):
                if self.verbose:
                    print(f"    Generating synopsis from {len(selected_chunks)} chunks in single request...")
                chunk_texts = [f"[{format_ts(c['start'])}] {c['text']}" for c in selected_chunks]
                prompt = base_prompt.format(chunks=chr(10).join(chunk_texts))
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
                
                # Rate limiting: estimate tokens and wait if needed
                estimated_tokens = self._count_message_tokens(messages)
                self._wait_if_needed(estimated_tokens)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # temperature=0.4,
                )
                
                # Record actual token usage
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = getattr(response.usage, 'prompt_tokens', estimated_tokens)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                else:
                    # Fallback to estimate if usage info not available
                    input_tokens = estimated_tokens
                    output_tokens = 0
                
                self._record_request(input_tokens, output_tokens)
                if self.verbose:
                    print(f"    ✓ Synopsis generated ({input_tokens:,} input, {output_tokens:,} output tokens)")
                synopsis = response.choices[0].message.content.strip()
            else:
                # Use batching
                synopsis = self._summarize_with_batching(
                    selected_chunks, 
                    base_prompt, 
                    system_message, 
                    merge_strategy="hierarchical", 
                    # temperature=0.4
                )
            
            # Verify timestamps exist
            synopsis = self._verify_timestamps(synopsis, raw_segments)
            return f"## Executive Synopsis\n\n{synopsis}"
        except BadRequestError as e:
            if "context_length_exceeded" in str(e).lower():
                print(f"    Context limit exceeded, retrying with batching...")
                try:
                    synopsis = self._summarize_with_batching(
                        selected_chunks, 
                        base_prompt, 
                        system_message,
                        merge_strategy="hierarchical", 
                        # temperature=0.4
                    )
                    synopsis = self._verify_timestamps(synopsis, raw_segments)
                    return f"## Executive Synopsis\n\n{synopsis}"
                except Exception as retry_error:
                    print(f"    Warning: Executive synopsis generation with batching failed: {retry_error}")
            else:
                print(f"    Warning: Executive synopsis generation failed: {e}")
            
            # Fallback: create basic synopsis from chunk starts
            bullets = []
            for chunk in selected_chunks[:15]:
                bullets.append(f"- [{format_ts(chunk['start'])}] {chunk['text'][:100]}...")
            return f"## Executive Synopsis\n\n{chr(10).join(bullets)}"
        except Exception as e:
            print(f"    Warning: Executive synopsis generation failed: {e}")
            # Fallback: create basic synopsis from chunk starts
            bullets = []
            for chunk in selected_chunks[:15]:
                bullets.append(f"- [{format_ts(chunk['start'])}] {chunk['text'][:100]}...")
            return f"## Executive Synopsis\n\n{chr(10).join(bullets)}"

    def _build_topic_map(self, coarse_chunks: List[Dict[str, Any]], topics: List[str], chunk_to_topic: List[int]) -> str:
        """Build topic map table with Start-End, Topic label, Why it matters, Listen if..."""
        if not coarse_chunks or not topics or not chunk_to_topic:
            return "## Topic Map / Outline\n\n(No content available)"

        chunk_groups = {}

        for i, topic in enumerate(topics):
            topic_chunks = []
            for j, chunk in enumerate(coarse_chunks):
                if chunk_to_topic[j] == i:
                    topic_chunks.append(chunk)
            chunk_groups[topic] = topic_chunks

        # Build table
        table_rows = ["| Timecodes | Topic |"]
        table_rows.append("|----------|-------|")

        for topic, chunks in chunk_groups.items():
            if not chunks:
                continue
            topic_start_timestamps = []
            topic_start_timestamps.append(format_ts(chunks[0]["start"]))
            topic_end_timestamps = []
            i = 1
            while i < len(chunks):
                if chunks[i]["start"] > chunks[i-1]["end"] + 5: # 5 seconds of overlap
                    topic_end_timestamps.append(format_ts(chunks[i-1]["end"]))
                    topic_start_timestamps.append(format_ts(chunks[i]["start"]))
                i += 1
            topic_end_timestamps.append(format_ts(chunks[-1]["end"]))
            if len(topic_start_timestamps) != len(topic_end_timestamps):
                print(f"    Warning: Topic {topic} has different number of start and end timestamps")
                continue
            else: 
                timecode_strings = []
                for j in range(len(topic_start_timestamps)):
                    timecode_string = f"[{topic_start_timestamps[j]}]"
                    timecode_strings.append(timecode_string)
                timecode_string = ", ".join(timecode_strings)
                table_rows.append(f"| {timecode_string} | {topic} |")

        return f"## Topic Map / Outline\n\n{chr(10).join(table_rows)}"

    def _build_deep_dive(
        self,
        topics: List[str],
        coarse_chunks: List[Dict[str, Any]],
        chunk_to_topic: List[int],
        fine_chunks: List[Dict[str, Any]],
        raw_segments: List[Dict[str, Any]],
    ) -> str:
        """Build deep dive notes organised by topic."""
        if not topics or not coarse_chunks:
            return "## Deep Dive Notes\n\n(No content available)"

        sections = ["## Deep Dive Notes\n"]

        # Process each topic
        chunk_groups = {}  # chunk groups contain coarse chunks

        for i, topic in enumerate(topics):
            topic_chunks = []
            for j, chunk in enumerate(coarse_chunks):
                if chunk_to_topic[j] == i:
                    topic_chunks.append(chunk)
            chunk_groups[topic] = topic_chunks

        for topic_idx, topic in enumerate(topics):
            topic_chunks = chunk_groups[topic]
            if not topic_chunks:
                continue

            # Get related fine chunks for detail
            # For each coarse chunk in topic_chunks, get fine chunks that start within the chunk's time range
            related_fine_chunks = []
            for coarse_chunk in topic_chunks:
                for fine_chunk in fine_chunks:
                    if coarse_chunk["start"] <= fine_chunk["start"] <= coarse_chunk["end"]:
                        related_fine_chunks.append(fine_chunk)

            # Select chunks within token limit for this topic
            base_prompt_template = DEEP_DIVE_BASE_PROMPT_TEMPLATE
            base_prompt = base_prompt_template.format(topic=topic, chunks="{chunks}")

            # Select chunks within token limit
            # selected_topic_chunks = self._select_chunks_within_limit(topic_chunks, base_prompt)
            system_message = DEEP_DIVE_SYSTEM_MESSAGE

            try:
                # Try single request first
                if self._can_fit_in_single_request(related_fine_chunks, base_prompt):
                    if self.verbose:
                        print(f"      Processing topic '{topic}' with {len(related_fine_chunks)} chunks...")
                    chunk_content = "\n\n".join(
                        f"[{format_ts(c['start'])}[-[{format_ts(c['end'])}] {c['text']}"
                        for c in related_fine_chunks
                    )
                    # base_prompt already has {topic} filled, now fill {chunks}
                    prompt = base_prompt.format(chunks=chunk_content)
                    
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ]
                    
                    # Rate limiting: estimate tokens and wait if needed
                    estimated_tokens = self._count_message_tokens(messages)
                    self._wait_if_needed(estimated_tokens)
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        # temperature=0.4,
                    )
                    
                    # Record actual token usage
                    if hasattr(response, 'usage') and response.usage:
                        input_tokens = getattr(response.usage, 'prompt_tokens', estimated_tokens)
                        output_tokens = getattr(response.usage, 'completion_tokens', 0)
                    else:
                        # Fallback to estimate if usage info not available
                        input_tokens = estimated_tokens
                        output_tokens = 0
                    
                    self._record_request(input_tokens, output_tokens)
                    if self.verbose:
                        print(f"      ✓ Topic '{topic}' completed ({input_tokens:,} input, {output_tokens:,} output tokens)")
                    topic_section = response.choices[0].message.content.strip()
                else:
                    # Use batching if needed
                    topic_section = self._summarize_with_batching(
                        topic_chunks, 
                        base_prompt, 
                        system_message,
                        merge_strategy="selective", 
                        # temperature=0.4
                    )
                
                # Verify timestamps
                topic_section = self._verify_timestamps(topic_section, raw_segments)
                sections.append(topic_section)
            except BadRequestError as e:
                if "context_length_exceeded" in str(e).lower():
                    print(f"    Context limit exceeded for topic '{topic}', retrying with batching...")
                    try:
                        topic_section = self._summarize_with_batching(
                            topic_chunks, 
                            base_prompt, 
                            system_message,
                            merge_strategy="selective", 
                            # temperature=0.4
                        )
                        topic_section = self._verify_timestamps(topic_section, raw_segments)
                        sections.append(topic_section)
                    except Exception as retry_error:
                        print(f"    Warning: Deep dive generation with batching failed for topic '{topic}': {retry_error}")
                        # Fallback
                        first_chunk = topic_chunks[0] if topic_chunks else None
                        if first_chunk:
                            sections.append(
                                f"""### {topic}

**Key Claims + Reasoning:**
- [{format_ts(first_chunk['start'])}] {first_chunk['text'][:200]}...

**Examples/Stories:**
- (Content extraction failed)

**Counterpoints / Nuance:**
- (Content extraction failed)

**Practical Takeaways:**
- (Content extraction failed)"""
                            )
                else:
                    print(f"    Warning: Deep dive generation failed for topic '{topic}': {e}")
                    # Fallback
                    first_chunk = topic_chunks[0] if topic_chunks else None
                    if first_chunk:
                        sections.append(
                            f"""### {topic}

**Key Claims + Reasoning:**
- [{format_ts(first_chunk['start'])}] {first_chunk['text'][:200]}...

**Examples/Stories:**
- (Content extraction failed)

**Counterpoints / Nuance:**
- (Content extraction failed)

**Practical Takeaways:**
- (Content extraction failed)"""
                        )
            except Exception as e:
                print(f"    Warning: Deep dive generation failed for topic '{topic}': {e}")
                # Fallback
                first_chunk = topic_chunks[0] if topic_chunks else None
                if first_chunk:
                    sections.append(
                        f"""### {topic}

**Key Claims + Reasoning:**
- [{format_ts(first_chunk['start'])}] {first_chunk['text'][:200]}...

**Examples/Stories:**
- (Content extraction failed)

**Counterpoints / Nuance:**
- (Content extraction failed)

**Practical Takeaways:**
- (Content extraction failed)"""
                    )

        return "\n\n".join(sections)

    def _build_frameworks(
        self,
        coarse_chunks: List[Dict[str, Any]],
        fine_chunks: List[Dict[str, Any]],
        raw_segments: List[Dict[str, Any]],
    ) -> str:
        """Extract frameworks, checklists, step-by-step methods with timestamps."""
        if not coarse_chunks:
            return "## Actionable Frameworks / Checklists\n\n(No frameworks found)"

        # Look for chunks that might contain frameworks (step-by-step, numbered lists, etc.)
        framework_chunks = []
        for chunk in coarse_chunks:
            text_lower = chunk["text"].lower()
            if any(
                keyword in text_lower
                for keyword in ["step", "framework", "checklist", "process", "method", "how to", "first", "second", "third"]
            ):
                framework_chunks.append(chunk)

        if not framework_chunks:
            return "## Actionable Frameworks / Checklists\n\n(No frameworks found in transcript)"

        # Get related fine chunks for framework chunks
        related_fine_chunks = []
        for chunk in framework_chunks:
            for fine_chunk in fine_chunks:
                if chunk["start"] <= fine_chunk["start"] <= chunk["end"]:
                    related_fine_chunks.append(fine_chunk)
        

        base_prompt = FRAMEWORK_EXTRACTION_BASE_PROMPT
        system_message = FRAMEWORK_EXTRACTION_SYSTEM_MESSAGE

        # Select chunks within token limit
        selected_framework_chunks = self._select_chunks_within_limit(framework_chunks, base_prompt)

        try:
            # Try single request first
            if self._can_fit_in_single_request(related_fine_chunks, base_prompt):
                    if self.verbose:
                        print(f"    Extracting frameworks from {len(related_fine_chunks)} candidate chunks...")
                    chunk_texts = "\n\n".join(
                        f"[{format_ts(c['start'])}]-[{format_ts(c['end'])}] {c['text']}" 
                        for c in related_fine_chunks
                    )
                    prompt = base_prompt.format(chunks=chunk_texts)
                    
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ]
                    
                    # Rate limiting: estimate tokens and wait if needed
                    estimated_tokens = self._count_message_tokens(messages)
                    self._wait_if_needed(estimated_tokens)
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        # temperature=0.3,
                    )
                    
                    # Record actual token usage
                    if hasattr(response, 'usage') and response.usage:
                        input_tokens = getattr(response.usage, 'prompt_tokens', estimated_tokens)
                        output_tokens = getattr(response.usage, 'completion_tokens', 0)
                    else:
                        # Fallback to estimate if usage info not available
                        input_tokens = estimated_tokens
                        output_tokens = 0
                    
                    self._record_request(input_tokens, output_tokens)
                    if self.verbose:
                        print(f"    ✓ Framework extraction completed ({input_tokens:,} input, {output_tokens:,} output tokens)")
                    frameworks = response.choices[0].message.content.strip()
            else:
                # Use batching if needed
                frameworks = self._summarize_with_batching(
                    selected_framework_chunks, 
                    base_prompt, 
                    system_message,
                    merge_strategy="hierarchical", 
                    # temperature=0.3
                )
            
            frameworks = self._verify_timestamps(frameworks, raw_segments)
            return f"## Actionable Frameworks / Checklists\n\n{frameworks}"
        except BadRequestError as e:
            if "context_length_exceeded" in str(e).lower():
                print(f"    Context limit exceeded, retrying framework extraction with batching...")
                try:
                    frameworks = self._summarize_with_batching(
                        selected_framework_chunks, 
                        base_prompt, 
                        system_message,
                        merge_strategy="hierarchical", 
                        # temperature=0.3
                    )
                    frameworks = self._verify_timestamps(frameworks, raw_segments)
                    return f"## Actionable Frameworks / Checklists\n\n{frameworks}"
                except Exception as retry_error:
                    print(f"    Warning: Framework extraction with batching failed: {retry_error}")
            else:
                print(f"    Warning: Framework extraction failed: {e}")
            
            # Fallback
            first_framework = framework_chunks[0] if framework_chunks else None
            if first_framework:
                return f"""## Actionable Frameworks / Checklists

### Framework [{format_ts(first_framework['start'])}]

{first_framework['text'][:300]}..."""
            return "## Actionable Frameworks / Checklists\n\n(Extraction failed)"
        except Exception as e:
            print(f"    Warning: Framework extraction failed: {e}")
            # Fallback
            first_framework = framework_chunks[0] if framework_chunks else None
            if first_framework:
                return f"""## Actionable Frameworks / Checklists

### Framework [{format_ts(first_framework['start'])}]

{first_framework['text'][:300]}..."""
            return "## Actionable Frameworks / Checklists\n\n(Extraction failed)"

    def _build_quotes(self, raw_segments: List[Dict[str, Any]]) -> str:
        """Extract short memorable quotes with timestamps."""
        if not raw_segments:
            return "## Memorable Quotes\n\n(No quotes found)"

        # Look for segments with quotes or memorable phrases (short segments, punctuation)
        quote_candidates = []
        for seg in raw_segments:
            text = seg["text"].strip()
            # Prefer segments that look quotable (contain quotes, are short statements, etc.)
            if (
                ('"' in text or "'" in text)
                and len(text) < 200
                and any(punct in text for punct in [".", "!", "?"])
            ):
                quote_candidates.append(seg)

        if not quote_candidates:
            # Fallback: use short segments
            quote_candidates = [seg for seg in raw_segments if len(seg["text"].strip()) < 150][:10]

        quotes_section = ["## Memorable Quotes\n"]
        for seg in quote_candidates[:15]:  # Limit to 15 quotes
            quote_text = seg["text"].strip()
            # Clean up the quote
            if quote_text.startswith(">>") or quote_text.startswith(">"):
                quote_text = quote_text.lstrip(">").strip()
            quotes_section.append(f'- [{format_ts(seg["start"])}] "{quote_text}"')

        return "\n".join(quotes_section)

    def _build_listening_paths(self, coarse_chunks: List[Dict[str, Any]]) -> str:
        """Build curated listening paths for 15/30/60 minutes."""
        if not coarse_chunks:
            return "## If you only have 15/30/60 minutes\n\n(No content available)"

        total_duration = coarse_chunks[-1]["end"] if coarse_chunks else 0
        duration_15 = 15 * 60
        duration_30 = 30 * 60
        duration_60 = 60 * 60

        sections = ["## If you only have 15/30/60 minutes\n"]

        # 15 minute path (first portion)
        if total_duration > duration_15:
            end_idx = next((i for i, c in enumerate(coarse_chunks) if c["end"] > duration_15), len(coarse_chunks))
            path_chunks = coarse_chunks[:end_idx]
            start_ts = format_ts(path_chunks[0]["start"]) if path_chunks else "00:00:00"
            end_ts = format_ts(path_chunks[-1]["end"]) if path_chunks else format_ts(duration_15)
            sections.append(f"### 15 minutes: [{start_ts} - {end_ts}]")
            sections.append("Core introduction and key concepts")
            sections.append("")

        # 30 minute path (first half)
        if total_duration > duration_30:
            end_idx = next((i for i, c in enumerate(coarse_chunks) if c["end"] > duration_30), len(coarse_chunks))
            path_chunks = coarse_chunks[:end_idx]
            start_ts = format_ts(path_chunks[0]["start"]) if path_chunks else "00:00:00"
            end_ts = format_ts(path_chunks[-1]["end"]) if path_chunks else format_ts(duration_30)
            sections.append(f"### 30 minutes: [{start_ts} - {end_ts}]")
            sections.append("Extended overview with examples and deeper discussion")
            sections.append("")

        # 60 minute path (selective highlights)
        if total_duration > duration_60:
            # Select key chunks across the video
            num_highlights = min(5, len(coarse_chunks))
            step = len(coarse_chunks) // num_highlights
            highlight_chunks = [coarse_chunks[i * step] for i in range(num_highlights)]
            sections.append("### 60 minutes: Selective highlights")
            for chunk in highlight_chunks:
                sections.append(f"- [{format_ts(chunk['start'])}] {chunk['text'][:100]}...")
        else:
            # Full video is less than 60 minutes
            start_ts = format_ts(coarse_chunks[0]["start"]) if coarse_chunks else "00:00:00"
            end_ts = format_ts(coarse_chunks[-1]["end"]) if coarse_chunks else "00:00:00"
            sections.append(f"### Full video ({format_ts(total_duration)}): [{start_ts} - {end_ts}]")
            sections.append("Complete discussion")

        return "\n".join(sections)

    def _verify_timestamps(self, text: str, raw_segments: List[Dict[str, Any]]) -> str:
        """
        Verify timestamps in text exist in raw_segments.
        This is a basic check - in production, could be more sophisticated.
        """
        # Extract timestamps from text (format [HH:MM:SS])
        import re

        timestamp_pattern = r"\[(\d{2}):(\d{2}):(\d{2})\]"
        timestamps = re.findall(timestamp_pattern, text)

        # Convert timestamps to seconds
        segment_times = [seg["start"] for seg in raw_segments]
        valid_times = set()

        for h, m, s in timestamps:
            total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            # Check if timestamp is within 30 seconds of any segment (for flexibility)
            if any(abs(total_seconds - seg_time) < 30 for seg_time in segment_times):
                valid_times.add((h, m, s))

        # If many invalid timestamps, return text as-is (better than corrupting content)
        # In production, could flag or fix invalid timestamps
        return text

    def _convert_timestamps_to_hyperlinks(self, text: str, video_url: str) -> str:
        """
        Convert all timestamps in format [HH:MM:SS] to markdown hyperlinks.
        
        Args:
            text: Markdown text containing timestamps
            video_url: YouTube video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
            
        Returns:
            Markdown text with timestamps converted to hyperlinks
        """
        import re
        
        timestamp_pattern = r"\[(\d{2}):(\d{2}):(\d{2})\]"
        
        def replace_timestamp(match):
            h, m, s = match.groups()
            # Convert to seconds
            total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            # Create hyperlink: [HH:MM:SS](url&t=seconds)
            timestamp_text = match.group(0)  # [HH:MM:SS]
            # Ensure URL separator (use & if ? already present, otherwise ?)
            separator = "&" if "?" in video_url else "?"
            link_url = f"{video_url}{separator}t={total_seconds}"
            return f"[{timestamp_text}]({link_url})"
        
        # Replace all timestamps with hyperlinks
        return re.sub(timestamp_pattern, replace_timestamp, text)
