import psycopg
from psycopg import AsyncConnection

from src.config import settings


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT,
    file_type TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER,
    embedding vector({embedding_dim}),
    metadata JSONB DEFAULT '{{}}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    title TEXT DEFAULT '新会话',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    embedding vector({embedding_dim}),
    memory_type TEXT DEFAULT 'fact',
    importance FLOAT DEFAULT 0.5,
    metadata JSONB DEFAULT '{{}}',
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_profile (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mcp_servers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    transport TEXT NOT NULL DEFAULT 'stdio',
    command TEXT,
    args TEXT,
    url TEXT,
    headers TEXT,
    env TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
"""


def vec_to_str(embedding: list[float]) -> str:
    """Convert a Python float list to a pgvector-compatible string."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def get_conn() -> AsyncConnection:
    return await AsyncConnection.connect(settings.pg_dsn, autocommit=True)


async def init_db():
    conn = await get_conn()
    try:
        # Extension must already be installed by superuser
        # CREATE EXTENSION IF NOT EXISTS vector;
        from src.llm.client import llm_client
        dim = llm_client.embedding_dim
        sql = CREATE_TABLES_SQL.format(embedding_dim=dim)
        await conn.execute(sql)
        # Migration: add session_id to existing memories table
        await conn.execute(
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS memories_embedding_idx "
            "ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
    finally:
        await conn.close()
