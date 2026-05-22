import json

from psycopg.rows import dict_row

from src.vector_store.db import get_conn


async def add_server(
    name: str,
    transport: str = "stdio",
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict | None = None,
    env: dict | None = None,
    auto_connect: bool = True,
) -> int:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO mcp_servers (name, transport, command, args, url, headers, env, auto_connect) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    name,
                    transport,
                    command,
                    json.dumps(args or []),
                    url,
                    json.dumps(headers or {}),
                    json.dumps(env or {}),
                    auto_connect,
                ),
            )
            row = await cur.fetchone()
        return row["id"]
    finally:
        await conn.close()


async def remove_server(server_id: int):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM mcp_servers WHERE id = %s", (server_id,))
    finally:
        await conn.close()


async def update_server(server_id: int, **kwargs):
    """Update server fields. Supports: name, transport, command, args, url, headers, env, enabled, auto_connect."""
    allowed = {"name", "transport", "command", "args", "url", "headers", "env", "enabled", "auto_connect"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return

    set_parts = []
    params = []
    for k, v in updates.items():
        if k in ("args", "headers", "env"):
            v = json.dumps(v or []) if k == "args" else json.dumps(v or {})
        set_parts.append(f"{k} = %s")
        params.append(v)

    params.append(server_id)
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE mcp_servers SET {', '.join(set_parts)} WHERE id = %s",
                params,
            )
    finally:
        await conn.close()


async def list_servers() -> list[dict]:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, name, transport, command, args, url, headers, env, enabled, auto_connect, created_at "
                "FROM mcp_servers ORDER BY created_at DESC"
            )
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "transport": r["transport"],
                "command": r["command"],
                "args": r["args"],
                "url": r["url"],
                "headers": r["headers"],
                "env": r["env"],
                "enabled": r["enabled"],
                "auto_connect": r.get("auto_connect", True),
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ]
    finally:
        await conn.close()


async def get_server(server_id: int) -> dict | None:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, name, transport, command, args, url, headers, env, enabled, auto_connect, created_at "
                "FROM mcp_servers WHERE id = %s",
                (server_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "transport": row["transport"],
            "command": row["command"],
            "args": row["args"],
            "url": row["url"],
            "headers": row["headers"],
            "env": row["env"],
            "enabled": row["enabled"],
            "auto_connect": row.get("auto_connect", True),
            "created_at": str(row["created_at"]),
        }
    finally:
        await conn.close()


async def toggle_server(server_id: int, enabled: bool):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE mcp_servers SET enabled = %s WHERE id = %s",
                (enabled, server_id),
            )
    finally:
        await conn.close()


async def set_auto_connect(server_id: int, auto_connect: bool):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE mcp_servers SET auto_connect = %s WHERE id = %s",
                (auto_connect, server_id),
            )
    finally:
        await conn.close()
