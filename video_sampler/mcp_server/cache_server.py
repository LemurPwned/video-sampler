import json
from datetime import datetime
from typing import Any

import numpy as np
import psycopg2
import psycopg2.pool
from PIL import Image
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, ConfigDict

from .embedders import OpenAIMultimodalEmbedder


class SegmentClip(BaseModel):
    start: float
    end: float
    description: str
    embedding: list[float] | None = None

    def to_embedding_text(self) -> str:
        """Convert segment to text for embedding generation"""
        return f"Video segment from {self.start}s to {self.end}s: {self.description}"


class VideoCache(BaseModel):
    video_path: str
    video_name: str
    video_summary: str
    segments: list[SegmentClip]
    srt_subtitles: str | None = None

    def to_embedding_text(self) -> str:
        """Convert video cache to text for embedding generation"""
        segments_text = " ".join([seg.description for seg in self.segments])
        subtitles_text = self.srt_subtitles or ""
        return f"{self.video_summary} {segments_text} {subtitles_text}"


# New Pydantic models for search results
class SegmentSearchResult(BaseModel):
    """Pydantic model for segment search results"""

    model_config = ConfigDict(from_attributes=True)

    type: str = "segment"
    segment_id: str
    video_name: str
    video_path: str
    start_time: float
    end_time: float
    description: str
    score: float


class VideoSearchResult(BaseModel):
    """Pydantic model for video search results"""

    model_config = ConfigDict(from_attributes=True)

    type: str = "video"
    video_name: str
    score: float
    video_cache: VideoCache


class VideoMetadata(BaseModel):
    """Pydantic model for video metadata listing"""

    model_config = ConfigDict(from_attributes=True)

    video_name: str
    video_path: str
    video_summary: str
    created_at: datetime
    updated_at: datetime
    segment_count: int


class TimeRangeSegment(BaseModel):
    """Pydantic model for time range search results"""

    model_config = ConfigDict(from_attributes=True)

    start_time: float
    end_time: float
    description: str


class VideoCacheServer:
    """
    Video cache server with PostgreSQL + pgvector for semantic search.

    Features:
    - Native vector similarity search with pgvector
    - ACID transactions
    - Scalable to millions of vectors
    - Advanced indexing (IVFFlat, HNSW)
    - Full PostgreSQL ecosystem support
    - Pydantic ORM integration for type-safe results
    """

    def __init__(
        self,
        connection_params: dict[str, Any] | None = None,
        model_name: str = "ViT-B-32",
        embedding_dim: int = 512,
        max_connections: int = 20,
    ):
        """
        Initialize the video cache server with PostgreSQL + pgvector.

        Args:
            connection_params: PostgreSQL connection parameters
            model_name: CLIP model name for embeddings
            embedding_dim: Embedding dimension (512 for ViT-B-32)
            max_connections: Maximum database connections in pool
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.embedder = OpenAIMultimodalEmbedder()
        # Default connection parameters
        if connection_params is None:
            connection_params = {
                "host": "localhost",
                "port": 5432,
                "database": "video_cache",
                "user": "postgres",
                "password": "postgres",
            }

        self.connection_params = connection_params

        # Initialize database connection pool
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1, max_connections, **connection_params
            )
            self._init_database()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def _init_database(self):
        """Initialize database schema with pgvector extension"""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create main video cache table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS video_cache (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        video_name VARCHAR(255) UNIQUE NOT NULL,
                        video_path TEXT NOT NULL,
                        video_summary TEXT,
                        segments JSONB,
                        srt_subtitles TEXT,
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                )

                # Create segment-level table for fine-grained search
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS video_segments (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        video_id UUID REFERENCES video_cache(id) ON DELETE CASCADE,
                        start_time FLOAT NOT NULL,
                        end_time FLOAT NOT NULL,
                        description TEXT NOT NULL,
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                )

                # Create indexes for better performance
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_video_cache_video_name
                    ON video_cache(video_name);
                """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_video_segments_video_id
                    ON video_segments(video_id);
                """
                )

                # Create vector indexes (choose based on your needs)
                # HNSW is generally better for most use cases
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_video_cache_embedding_hnsw
                        ON video_cache USING hnsw (embedding vector_cosine_ops);
                    """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_video_segments_embedding_hnsw
                        ON video_segments USING hnsw (embedding vector_cosine_ops);
                    """
                    )
                except Exception:
                    # Fallback to IVFFlat if HNSW is not available
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_video_cache_embedding_ivfflat
                        ON video_cache USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_video_segments_embedding_ivfflat
                        ON video_segments USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """
                    )

                # Create updated_at trigger
                cur.execute(
                    """
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """
                )

                cur.execute(
                    """
                    DROP TRIGGER IF EXISTS update_video_cache_updated_at ON video_cache;
                    CREATE TRIGGER update_video_cache_updated_at
                        BEFORE UPDATE ON video_cache
                        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                """
                )

                conn.commit()
        finally:
            self.connection_pool.putconn(conn)

    def _generate_embedding(self, text: str) -> np.ndarray | None:
        """Generate embedding for text"""
        return self.embedder.embed_text(text)

    def _frame_embedding(self, frame: Image.Image) -> np.ndarray | None:
        """Generate embedding for frame"""
        return self.embedder.embed_image(frame)

    def get_by_id(self, id: str) -> VideoCache | None:
        """Get video cache by id"""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM video_cache WHERE id = %s", (id,))
                row = cur.fetchone()
                return VideoCache(**row) if row else None
        finally:
            self.connection_pool.putconn(conn)

    def store(self, video: VideoCache) -> bool:
        """
        Store video cache with semantic embedding.

        Args:
            video: VideoCache object to store

        Returns:
            bool: Success status
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Generate embedding for the overall video
                embedding_text = video.to_embedding_text()
                video_embedding = self._generate_embedding(embedding_text)

                # Convert embedding to list for PostgreSQL
                embedding_list = (
                    video_embedding.tolist() if video_embedding is not None else None
                )

                # Store main video record
                cur.execute(
                    """
                    INSERT INTO video_cache
                    (video_name, video_path, video_summary, segments, srt_subtitles, embedding)
                    VALUES (%(video_name)s, %(video_path)s, %(video_summary)s, %(segments)s, %(srt_subtitles)s, %(embedding)s)
                    ON CONFLICT (video_name)
                    DO UPDATE SET
                        video_path = EXCLUDED.video_path,
                        video_summary = EXCLUDED.video_summary,
                        segments = EXCLUDED.segments,
                        srt_subtitles = EXCLUDED.srt_subtitles,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """,
                    {
                        "video_name": video.video_name,
                        "video_path": video.video_path,
                        "video_summary": video.video_summary,
                        "segments": json.dumps([seg.dict() for seg in video.segments]),
                        "srt_subtitles": video.srt_subtitles,
                        "embedding": embedding_list,
                    },
                )

                video_id = cur.fetchone()["id"]

                # Delete existing segments for this video
                cur.execute(
                    "DELETE FROM video_segments WHERE video_id = %s", (video_id,)
                )

                # Store individual segments with their embeddings
                if video.segments:
                    segment_data = []
                    for segment in video.segments:
                        segment_embedding = self._generate_embedding(
                            segment.to_embedding_text()
                        )
                        segment_embedding_list = (
                            segment_embedding.tolist()
                            if segment_embedding is not None
                            else None
                        )

                        segment_data.append(
                            {
                                "video_id": video_id,
                                "start_time": segment.start,
                                "end_time": segment.end,
                                "description": segment.description,
                                "embedding": segment_embedding_list,
                            }
                        )

                    # Batch insert segments
                    cur.executemany(
                        """
                        INSERT INTO video_segments
                        (video_id, start_time, end_time, description, embedding)
                        VALUES (%(video_id)s, %(start_time)s, %(end_time)s, %(description)s, %(embedding)s)
                    """,
                        segment_data,
                    )

                conn.commit()
                return True

        except Exception as e:
            conn.rollback()
            print(f"Error storing video cache: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def update_cache_element(self, video: VideoCache) -> bool:
        """Update existing cache element (same as store with upsert)"""
        return self.store(video)

    def retrieve(self, video_name: str) -> VideoCache | None:
        """
        Retrieve video cache by exact video name.

        Args:
            video_name: Name of the video to retrieve

        Returns:
            VideoCache object or None if not found
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT video_path, video_summary, segments, srt_subtitles
                    FROM video_cache
                    WHERE video_name = %s
                """,
                    (video_name,),
                )

                row = cur.fetchone()
                if not row:
                    return None

                segments = [SegmentClip(**seg) for seg in json.loads(row["segments"])]

                return VideoCache(
                    video_path=row["video_path"],
                    video_name=video_name,
                    video_summary=row["video_summary"],
                    segments=segments,
                    srt_subtitles=row["srt_subtitles"],
                )

        except Exception as e:
            print(f"Error retrieving video: {e}")
            return None
        finally:
            self.connection_pool.putconn(conn)

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_segments: bool = False,
        similarity_threshold: float = 0.7,
    ) -> list[SegmentSearchResult | VideoSearchResult]:
        """
        Semantic search using query embedding with pgvector.

        Args:
            query: Search query text
            top_k: Number of top results to return
            search_segments: If True, search individual segments instead of videos
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of Pydantic models with search results
        """
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            print("Warning: Cannot perform semantic search without embeddings")
            return self._fallback_text_search(query, top_k)

        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if search_segments:
                    # Search individual segments
                    cur.execute(
                        """
                        SELECT
                            vs.id as segment_id,
                            vs.start_time,
                            vs.end_time,
                            vs.description,
                            vc.video_name,
                            vc.video_path,
                            vc.video_summary,
                            (1 - (vs.embedding <=> %s::vector)) as similarity_score
                        FROM video_segments vs
                        JOIN video_cache vc ON vs.video_id = vc.id
                        WHERE vs.embedding IS NOT NULL
                            AND (1 - (vs.embedding <=> %s::vector)) >= %s
                        ORDER BY vs.embedding <=> %s::vector
                        LIMIT %s;
                    """,
                        (
                            query_embedding.tolist(),
                            query_embedding.tolist(),
                            similarity_threshold,
                            query_embedding.tolist(),
                            top_k,
                        ),
                    )

                    results = []
                    for row in cur.fetchall():
                        # Create Pydantic model from database row
                        result = SegmentSearchResult(
                            segment_id=str(row["segment_id"]),
                            video_name=row["video_name"],
                            video_path=row["video_path"],
                            start_time=row["start_time"],
                            end_time=row["end_time"],
                            description=row["description"],
                            score=float(row["similarity_score"]),
                        )
                        results.append(result)

                else:
                    # Search entire videos
                    cur.execute(
                        """
                        SELECT
                            video_name,
                            video_path,
                            video_summary,
                            segments,
                            srt_subtitles,
                            (1 - (embedding <=> %s::vector)) as similarity_score
                        FROM video_cache
                        WHERE embedding IS NOT NULL
                            AND (1 - (embedding <=> %s::vector)) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """,
                        (
                            query_embedding.tolist(),
                            query_embedding.tolist(),
                            similarity_threshold,
                            query_embedding.tolist(),
                            top_k,
                        ),
                    )

                    results = []
                    for row in cur.fetchall():
                        segments = [
                            SegmentClip(**seg) for seg in json.loads(row["segments"])
                        ]
                        video_cache = VideoCache(
                            video_path=row["video_path"],
                            video_name=row["video_name"],
                            video_summary=row["video_summary"],
                            segments=segments,
                            srt_subtitles=row["srt_subtitles"],
                        )

                        # Create Pydantic model for video search result
                        result = VideoSearchResult(
                            video_name=row["video_name"],
                            score=float(row["similarity_score"]),
                            video_cache=video_cache,
                        )
                        results.append(result)

                return results

        except Exception as e:
            print(f"Error performing semantic search: {e}")
            return []
        finally:
            self.connection_pool.putconn(conn)

    def search_by_time_range(
        self, video_name: str, start_time: float, end_time: float
    ) -> list[TimeRangeSegment]:
        """
        Search segments within a specific time range for a video.

        Args:
            video_name: Name of the video
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of TimeRangeSegment Pydantic models
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        vs.start_time,
                        vs.end_time,
                        vs.description
                    FROM video_segments vs
                    JOIN video_cache vc ON vs.video_id = vc.id
                    WHERE vc.video_name = %s
                        AND vs.start_time >= %s
                        AND vs.end_time <= %s
                    ORDER BY vs.start_time;
                """,
                    (video_name, start_time, end_time),
                )

                return [
                    TimeRangeSegment.model_validate(dict(row)) for row in cur.fetchall()
                ]

        except Exception as e:
            print(f"Error searching by time range: {e}")
            return []
        finally:
            self.connection_pool.putconn(conn)

    def _fallback_text_search(self, query: str, top_k: int) -> list[VideoSearchResult]:
        """Fallback text search using PostgreSQL full-text search"""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        video_name,
                        video_path,
                        video_summary,
                        segments,
                        srt_subtitles,
                        ts_rank(
                            to_tsvector('english', video_summary || ' ' || COALESCE(srt_subtitles, '')),
                            plainto_tsquery('english', %s)
                        ) as rank
                    FROM video_cache
                    WHERE to_tsvector('english', video_summary || ' ' || COALESCE(srt_subtitles, ''))
                          @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                """,
                    (query, query, top_k),
                )

                results = []
                for row in cur.fetchall():
                    segments = [
                        SegmentClip(**seg) for seg in json.loads(row["segments"])
                    ]
                    video_cache = VideoCache(
                        video_path=row["video_path"],
                        video_name=row["video_name"],
                        video_summary=row["video_summary"],
                        segments=segments,
                        srt_subtitles=row["srt_subtitles"],
                    )

                    # Create Pydantic model for fallback search result
                    result = VideoSearchResult(
                        video_name=row["video_name"],
                        score=float(row["rank"]),
                        video_cache=video_cache,
                    )
                    results.append(result)

                return results

        except Exception as e:
            print(f"Error in fallback text search: {e}")
            return []
        finally:
            self.connection_pool.putconn(conn)

    def list_all_videos(self) -> list[VideoMetadata]:
        """List all cached videos with metadata as Pydantic models"""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        video_name,
                        video_path,
                        video_summary,
                        created_at,
                        updated_at,
                        jsonb_array_length(segments) as segment_count
                    FROM video_cache
                    ORDER BY created_at DESC;
                """
                )

                return [
                    VideoMetadata.model_validate(dict(row)) for row in cur.fetchall()
                ]

        except Exception as e:
            print(f"Error listing videos: {e}")
            return []
        finally:
            self.connection_pool.putconn(conn)

    def delete_video(self, video_name: str) -> bool:
        """
        Delete a video from cache (cascades to segments).

        Args:
            video_name: Name of the video to delete

        Returns:
            bool: True if deleted, False if not found
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM video_cache WHERE video_name = %s", (video_name,)
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            conn.rollback()
            print(f"Error deleting video: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def close(self):
        """Close database connection pool"""
        if hasattr(self, "connection_pool"):
            self.connection_pool.closeall()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
