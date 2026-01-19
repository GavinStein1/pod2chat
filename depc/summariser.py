def _extract_topics(self, coarse_chunks: List[Dict[str, Any]]) -> List[str]:
        """Pass 1: Extract main topics/themes from coarse chunks."""
        topics, _ = self.extract_topics_streaming(coarse_chunks)
        return topics
        if not coarse_chunks:
            return []

        base_prompt = TOPIC_EXTRACTION_BASE_PROMPT

        # Try to include all chunks, but select within limit if needed
        selected_chunks = self._select_chunks_within_limit(coarse_chunks, base_prompt)
        
        # If we had to limit chunks, note it (but still try to extract good topics)
        if len(selected_chunks) < len(coarse_chunks):
            if self.verbose:
                print(f"    Note: Token limit reached, using {len(selected_chunks)}/{len(coarse_chunks)} chunks for topic extraction")
            else:
                print(f"    Note: Using {len(selected_chunks)}/{len(coarse_chunks)} chunks for topic extraction")

        try:
            # Build chunk summaries for topic extraction
            chunk_summaries = []
            for chunk in selected_chunks:
                # Use full text for better topic extraction (within token limits)
                chunk_summaries.append(
                    f"[{format_ts(chunk['start'])}-{format_ts(chunk['end'])}] {chunk['text']}"
                )
            
            prompt = base_prompt.format(chunks=chr(10).join(chunk_summaries))
            
            messages = [
                {"role": "system", "content": TOPIC_EXTRACTION_SYSTEM_MESSAGE},
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
                print(f"    API call completed ({input_tokens:,} input, {output_tokens:,} output tokens)")
            topics_json = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if topics_json.startswith("```"):
                topics_json = topics_json.split("```")[1].split("\n", 1)[1] if "\n" in topics_json.split("```")[1] else topics_json.split("```")[2]
            topics = json.loads(topics_json)
            if self.verbose:
                print(f"    Parsed {len(topics)} topics from response")
            return topics if isinstance(topics, list) else []
        except BadRequestError as e:
            if "context_length_exceeded" in str(e).lower():
                print(f"    Context limit exceeded, using batching for topic extraction...")
                # Use batching if single request fails
                result = self._summarize_with_batching(
                    selected_chunks, base_prompt, 
                    TOPIC_EXTRACTION_SYSTEM_MESSAGE,
                    merge_strategy="selective",
                    # temperature=0.3,
                )
                # Extract topics from batched result
                try:
                    topics_json = result.strip()
                    if topics_json.startswith("```"):
                        topics_json = topics_json.split("```")[1].split("\n", 1)[1] if "\n" in topics_json.split("```")[1] else topics_json.split("```")[2]
                    topics = json.loads(topics_json)
                    return topics if isinstance(topics, list) else []
                except Exception:
                    pass
            print(f"    Warning: Topic extraction failed: {e}. Using fallback topics.")
            return ["Main Discussion", "Key Points", "Examples", "Conclusion"]
        except Exception as e:
            print(f"    Warning: Topic extraction failed: {e}. Using fallback topics.")
            return ["Main Discussion", "Key Points", "Examples", "Conclusion"]