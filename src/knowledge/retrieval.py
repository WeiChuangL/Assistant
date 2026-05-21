from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from src.config import settings
from src.llm.client import llm_client
from src.vector_store.db import get_conn, vec_to_str


@dataclass
class RetrievalResult:
    content: str
    score: float
    document_id: int
    chunk_index: int
    filename: str | None = None


async def search_chunks(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    document_id: int | None = None,
) -> list[RetrievalResult]:
    """Search for chunks similar to the query using cosine similarity."""
    if top_k is None:
        top_k = settings.agent_top_k_chunks
    if threshold is None:
        threshold = settings.agent_similarity_threshold

    embeddings = await llm_client.embed([query])
    query_vec = vec_to_str(embeddings[0])

    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            if document_id is not None:
                await cur.execute(
                    "SELECT content, score, document_id, chunk_index, filename "
                    "FROM ("
                    "  SELECT c.content, 1 - (c.embedding <=> %s::vector) AS score, "
                    "  c.document_id, c.chunk_index, d.filename "
                    "  FROM chunks c JOIN documents d ON c.document_id = d.id "
                    "  WHERE c.document_id = %s"
                    ") sub "
                    "WHERE score > %s "
                    "ORDER BY score DESC "
                    "LIMIT %s",
                    (query_vec, document_id, threshold, top_k),
                )
            else:
                await cur.execute(
                    "SELECT content, score, document_id, chunk_index, filename "
                    "FROM ("
                    "  SELECT c.content, 1 - (c.embedding <=> %s::vector) AS score, "
                    "  c.document_id, c.chunk_index, d.filename "
                    "  FROM chunks c JOIN documents d ON c.document_id = d.id"
                    ") sub "
                    "WHERE score > %s "
                    "ORDER BY score DESC "
                    "LIMIT %s",
                    (query_vec, threshold, top_k),
                )
            rows = await cur.fetchall()
        return [
            RetrievalResult(
                content=r["content"],
                score=r["score"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                filename=r["filename"],
            )
            for r in rows
        ]
    finally:
        await conn.close()
