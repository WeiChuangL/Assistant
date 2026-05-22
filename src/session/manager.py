from dataclasses import dataclass

from psycopg.rows import dict_row

from src.vector_store.db import get_conn


@dataclass
class Session:
    id: int
    title: str
    created_at: str
    updated_at: str


async def create_session(title: str = "新会话") -> Session:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO sessions (title) VALUES (%s) RETURNING id, title, created_at, updated_at",
                (title,),
            )
            row = await cur.fetchone()
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
    finally:
        await conn.close()


async def list_sessions() -> list[Session]:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            rows = await cur.fetchall()
        return [
            Session(
                id=r["id"],
                title=r["title"],
                created_at=str(r["created_at"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]
    finally:
        await conn.close()


async def rename_session(session_id: int, title: str):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE sessions SET title = %s, updated_at = NOW() WHERE id = %s",
                (title, session_id),
            )
    finally:
        await conn.close()


async def delete_session(session_id: int):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    finally:
        await conn.close()


async def touch_session(session_id: int):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE id = %s",
                (session_id,),
            )
    finally:
        await conn.close()


async def get_session(session_id: int) -> Session | None:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
    finally:
        await conn.close()


async def ensure_default_session() -> Session:
    """Get first existing session or create a default one."""
    sessions = await list_sessions()
    if sessions:
        return sessions[0]
    return await create_session("默认会话")
