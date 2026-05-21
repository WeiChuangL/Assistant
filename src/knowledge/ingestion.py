import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from src.config import settings
from src.llm.client import llm_client
from src.vector_store.chunker import chunk_text
from src.vector_store.db import get_conn, vec_to_str


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in (".md", ".txt"):
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


async def ingest_file(file_path: Path) -> int:
    """Ingest a single file: read -> chunk -> embed -> store. Returns document_id."""
    content = read_file(file_path)
    chunks = chunk_text(content)
    if not chunks:
        return 0

    embeddings = await llm_client.embed_documents([c.content for c in chunks])

    conn = await get_conn()
    try:
        doc_id = await _insert_document(conn, file_path)
        await _insert_chunks(conn, doc_id, chunks, embeddings)
        return doc_id
    finally:
        await conn.close()


async def ingest_directory(dir_path: Path) -> list[int]:
    """Recursively ingest all supported files in a directory. Returns list of document_ids."""
    doc_ids: list[int] = []
    for root, _dirs, files in os.walk(dir_path):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    doc_id = await ingest_file(fpath)
                    doc_ids.append(doc_id)
                    print(f"  [OK] {fpath}")
                except Exception as e:
                    print(f"  [FAIL] {fpath}: {e}")
    return doc_ids


async def _insert_document(conn, file_path: Path) -> int:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "INSERT INTO documents (filename, file_path, file_type) "
            "VALUES (%(fn)s, %(fp)s, %(ft)s) RETURNING id",
            {"fn": file_path.name, "fp": str(file_path.absolute()), "ft": file_path.suffix.lower()},
        )
        row = await cur.fetchone()
        return row["id"]


async def _insert_chunks(conn, doc_id: int, chunks, embeddings):
    async with conn.cursor() as cur:
        for c, emb in zip(chunks, embeddings):
            vec_str = vec_to_str(emb)
            await cur.execute(
                "INSERT INTO chunks (document_id, content, chunk_index, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (doc_id, c.content, c.index, vec_str),
            )
