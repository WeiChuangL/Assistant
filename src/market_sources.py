import httpx
from psycopg.rows import dict_row

from src.vector_store.db import get_conn


async def add_source(market_type: str, name: str, url: str) -> int:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO market_sources (market_type, name, url) VALUES (%s, %s, %s) RETURNING id",
                (market_type, name, url),
            )
            row = await cur.fetchone()
        return row["id"]
    finally:
        await conn.close()


async def remove_source(source_id: int):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM market_sources WHERE id = %s", (source_id,))
    finally:
        await conn.close()


async def list_sources(market_type: str) -> list[dict]:
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, market_type, name, url, enabled, created_at "
                "FROM market_sources WHERE market_type = %s ORDER BY created_at DESC",
                (market_type,),
            )
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "market_type": r["market_type"],
                "name": r["name"],
                "url": r["url"],
                "enabled": r["enabled"],
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ]
    finally:
        await conn.close()


async def toggle_source(source_id: int, enabled: bool):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE market_sources SET enabled = %s WHERE id = %s",
                (enabled, source_id),
            )
    finally:
        await conn.close()


async def fetch_third_party_market(market_type: str) -> list[dict]:
    """Fetch market entries from all enabled third-party sources."""
    sources = await list_sources(market_type)
    results = []
    for src in sources:
        if not src["enabled"]:
            continue
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(src["url"])
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        item["_source"] = src["name"]
                        item["_source_url"] = src["url"]
                        results.append(item)
        except Exception:
            continue
    return results
