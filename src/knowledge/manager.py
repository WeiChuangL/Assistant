from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from src.config import settings
from src.vector_store.db import get_conn


@dataclass
class DocInfo:
    id: int
    filename: str
    file_path: str | None
    file_type: str | None
    chunk_count: int
    created_at: str


async def list_documents() -> list[DocInfo]:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT d.id, d.filename, d.file_path, d.file_type, d.created_at, "
                "COUNT(c.id) AS chunk_count "
                "FROM documents d LEFT JOIN chunks c ON d.id = c.document_id "
                "GROUP BY d.id ORDER BY d.created_at DESC"
            )
            rows = await cur.fetchall()
        return [
            DocInfo(
                id=r["id"],
                filename=r["filename"],
                file_path=r["file_path"],
                file_type=r["file_type"],
                chunk_count=r["chunk_count"],
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
    finally:
        await conn.close()


async def delete_document(doc_id: int) -> bool:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            return cur.rowcount > 0
    finally:
        await conn.close()


async def get_chunks_for_doc(doc_id: int, limit: int = 5) -> list[dict]:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, content, chunk_index, metadata FROM chunks "
                "WHERE document_id = %s ORDER BY chunk_index LIMIT %s",
                (doc_id, limit),
            )
            return await cur.fetchall()
    finally:
        await conn.close()
