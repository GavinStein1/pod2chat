"""
Vector store using SQLite with FTS5 for full-text search.

Stores chunks with embeddings and enables keyword search.
"""

from __future__ import annotations

import sqlite3
import json
from typing import List, Dict, Any, Optional


class VectorStore:
    """SQLite-based vector store with FTS5 for full-text search."""

    def __init__(self, db_path: str):
        """
        Initialize vector store with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        self.create_tables()

    def create_tables(self) -> None:
        """Create chunks table and FTS5 virtual table with triggers."""
        cursor = self.conn.cursor()
        
        # Main chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                video_url TEXT NOT NULL,
                chunk_id TEXT NOT NULL UNIQUE,
                tier TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text TEXT NOT NULL,
                segment_ids TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add video_url column if it doesn't exist (migration for existing databases)
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN video_url TEXT")
            # Set default for existing rows (empty string, will be filtered out)
            cursor.execute("UPDATE chunks SET video_url = '' WHERE video_url IS NULL")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Create index on chunk_id for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)
        """)
        
        # Create index on video_id for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_id ON chunks(video_id)
        """)
        
        # Create index on video_url for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_url ON chunks(video_url)
        """)
        
        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                tier,
                content='chunks',
                content_rowid='id'
            )
        """)
        
        # Trigger to keep FTS5 in sync when inserting
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, chunk_id, text, tier)
                VALUES (new.id, new.chunk_id, new.text, new.tier);
            END
        """)
        
        # Trigger to keep FTS5 in sync when updating
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');
            END
        """)
        
        # Trigger to keep FTS5 in sync when deleting
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');
            END
        """)
        
        self.conn.commit()

    def insert_chunks(self, chunks_with_embeddings: List[Dict[str, Any]], video_id: str, video_url: str) -> None:
        """
        Insert chunks with embeddings into the database.
        
        Args:
            chunks_with_embeddings: List of chunk dictionaries with 'embedding' key
            video_id: Video ID to associate chunks with
            video_url: Video URL to associate chunks with
        """
        if not chunks_with_embeddings:
            return
        
        cursor = self.conn.cursor()
        
        inserted = 0
        for chunk in chunks_with_embeddings:
            chunk_id = chunk.get("chunk_id")
            tier = chunk.get("tier", "fine")
            start = chunk.get("start", 0.0)
            end = chunk.get("end", 0.0)
            text = chunk.get("text", "")
            segment_ids = json.dumps(chunk.get("segment_ids", []))
            embedding = chunk.get("embedding")
            
            # Store embedding as JSON string
            embedding_json = json.dumps(embedding) if embedding else None
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (video_id, video_url, chunk_id, tier, start, end, text, segment_ids, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (video_id, video_url, chunk_id, tier, start, end, text, segment_ids, embedding_json))
                inserted += 1
            except sqlite3.Error as e:
                print(f"    Warning: Failed to insert chunk {chunk_id}: {e}")
        
        self.conn.commit()
        print(f"    Inserted {inserted}/{len(chunks_with_embeddings)} chunks into vector store")

    def search_text(self, query: str, video_url: Optional[str] = None, limit: int = 10, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search chunks using FTS5 full-text search.
        
        Args:
            query: Search query text
            video_url: Optional video URL to filter results
            limit: Maximum number of results to return
            tier: Optional tier filter ('fine' or 'coarse')
            
        Returns:
            List of matching chunk dictionaries
        """
        cursor = self.conn.cursor()
        
        # Build query with optional filters
        where_clauses = ["chunks_fts MATCH ?"]
        params = [query]
        
        if video_url:
            where_clauses.append("c.video_url = ?")
            params.append(video_url)
        
        if tier:
            where_clauses.append("chunks_fts.tier = ?")
            params.append(tier)
        
        where_clause = " AND ".join(where_clauses)
        
        query_sql = f"""
            SELECT c.*, bm25(chunks_fts) as bm25_score
            FROM chunks c
            JOIN chunks_fts ON c.id = chunks_fts.rowid
            WHERE {where_clause}
            ORDER BY bm25_score
            LIMIT ?
        """
        
        params.append(limit)
        
        try:
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                try:
                    video_url = row["video_url"]
                except (KeyError, IndexError):
                    video_url = ""
                
                try:
                    bm25_score = row["bm25_score"]
                except (KeyError, IndexError):
                    bm25_score = 0
                
                chunk_dict = {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "video_url": video_url,
                    "chunk_id": row["chunk_id"],
                    "tier": row["tier"],
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "segment_ids": json.loads(row["segment_ids"]) if row["segment_ids"] else [],
                    "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                    "created_at": row["created_at"],
                    "bm25_score": bm25_score,
                }
                results.append(chunk_dict)
            
            return results
        except sqlite3.Error as e:
            print(f"    Warning: Search failed: {e}")
            return []

    def search_similar(self, query_embedding: List[float], video_url: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Search chunks using cosine similarity on embeddings.
        
        Args:
            query_embedding: Query embedding vector
            video_url: Optional video URL to filter results
            limit: Maximum number of results to return
            
        Returns:
            List of matching chunk dictionaries with similarity scores
        """
        cursor = self.conn.cursor()
        
        # Build query to get all chunks (with optional video_url filter)
        if video_url:
            cursor.execute("""
                SELECT id, video_id, video_url, chunk_id, tier, start, end, text, segment_ids, embedding
                FROM chunks
                WHERE video_url = ?
            """, (video_url,))
        else:
            cursor.execute("""
                SELECT id, video_id, video_url, chunk_id, tier, start, end, text, segment_ids, embedding
                FROM chunks
            """)
        
        rows = cursor.fetchall()
        
        # Calculate cosine similarity for each chunk
        results = []
        for row in rows:
            embedding_json = row["embedding"]
            if not embedding_json:
                continue
            
            try:
                chunk_embedding = json.loads(embedding_json)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                try:
                    video_url = row["video_url"]
                except (KeyError, IndexError):
                    video_url = ""
                
                chunk_dict = {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "video_url": video_url,
                    "chunk_id": row["chunk_id"],
                    "tier": row["tier"],
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "segment_ids": json.loads(row["segment_ids"]) if row["segment_ids"] else [],
                    "embedding": chunk_embedding,
                    "similarity": similarity,
                }
                results.append(chunk_dict)
            except (json.JSONDecodeError, TypeError) as e:
                # Skip chunks with invalid embeddings
                continue
        
        # Sort by similarity (descending) and return top N
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate norms
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def rerank_with_keywords(self, chunks: List[Dict[str, Any]], query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using FTS5 keyword matching.
        
        Args:
            chunks: List of chunks from similarity search (must have 'similarity' score)
            query: Search query text for keyword matching
            limit: Maximum number of results to return after re-ranking
            
        Returns:
            Re-ranked list of chunks with combined scores
        """
        if not chunks:
            return []
        
        cursor = self.conn.cursor()
        
        # Get keyword match scores for each chunk using FTS5 bm25 (lower/more negative = better)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        placeholders = ",".join("?" * len(chunk_ids))
        
        try:
            cursor.execute(f"""
                SELECT c.chunk_id, bm25(chunks_fts) as bm25_score
                FROM chunks c
                JOIN chunks_fts ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND c.chunk_id IN ({placeholders})
            """, [query] + chunk_ids)
            
            keyword_scores = {row["chunk_id"]: row["bm25_score"] for row in cursor.fetchall()}
        except sqlite3.Error:
            # If FTS5 search fails, use similarity scores only
            keyword_scores = {}
        
        # Normalize keyword scores (bm25 is negative, more negative = better)
        if keyword_scores:
            min_bm25 = min(keyword_scores.values())  # Most negative (best)
            max_bm25 = max(keyword_scores.values())  # Least negative (worst)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
        else:
            min_bm25 = 0.0
            max_bm25 = 0.0
            bm25_range = 1.0
        
        # Combine similarity and keyword scores
        for chunk in chunks:
            similarity_score = chunk.get("similarity", 0.0)
            chunk_id = chunk["chunk_id"]
            
            if chunk_id in keyword_scores:
                # Normalize bm25 score (invert so higher = better, range 0-1)
                bm25_score = keyword_scores[chunk_id]
                if bm25_range > 0:
                    # More negative = better, so invert: (max - score) / range
                    normalized_keyword = (max_bm25 - bm25_score) / bm25_range
                else:
                    normalized_keyword = 1.0
            else:
                normalized_keyword = 0.0
            
            # Weighted combination: 0.7 * similarity + 0.3 * keyword
            combined_score = 0.7 * similarity_score + 0.3 * normalized_keyword
            chunk["combined_score"] = combined_score
            chunk["keyword_score"] = normalized_keyword
        
        # Sort by combined score (descending) and return top N
        chunks.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        return chunks[:limit]

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
