import json
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from src.config import settings
from src.llm.client import llm_client
from src.vector_store.db import get_conn, vec_to_str


@dataclass
class MemoryEntry:
    id: int
    content: str
    summary: str | None
    score: float
    memory_type: str
    importance: float
    created_at: str


async def store_memory(
    content: str,
    summary: str | None = None,
    memory_type: str = "fact",
    importance: float = 0.5,
    metadata: dict | None = None,
):
    """Store a memory with its embedding."""
    text_to_embed = summary or content
    embeddings = await llm_client.embed([text_to_embed])
    vec_str = vec_to_str(embeddings[0])

    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO memories (content, summary, embedding, memory_type, importance, metadata) "
                "VALUES (%s, %s, %s::vector, %s, %s, %s)",
                (content, summary, vec_str, memory_type, importance, json.dumps(metadata or {})),
            )
    finally:
        await conn.close()


async def search_memories(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    memory_type: str | None = None,
) -> list[MemoryEntry]:
    """Search memories by semantic similarity."""
    if top_k is None:
        top_k = settings.agent_top_k_memories
    if threshold is None:
        threshold = settings.agent_similarity_threshold

    embeddings = await llm_client.embed([query])
    query_vec = vec_to_str(embeddings[0])

    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            if memory_type:
                await cur.execute(
                    "SELECT id, content, summary, score, memory_type, importance, created_at "
                    "FROM ("
                    "  SELECT id, content, summary, "
                    "  1 - (embedding <=> %s::vector) AS score, "
                    "  memory_type, importance, created_at "
                    "  FROM memories "
                    "  WHERE memory_type = %s"
                    ") sub "
                    "WHERE score > %s "
                    "ORDER BY score DESC "
                    "LIMIT %s",
                    (query_vec, memory_type, threshold, top_k),
                )
            else:
                await cur.execute(
                    "SELECT id, content, summary, score, memory_type, importance, created_at "
                    "FROM ("
                    "  SELECT id, content, summary, "
                    "  1 - (embedding <=> %s::vector) AS score, "
                    "  memory_type, importance, created_at "
                    "  FROM memories"
                    ") sub "
                    "WHERE score > %s "
                    "ORDER BY score DESC "
                    "LIMIT %s",
                    (query_vec, threshold, top_k),
                )
            rows = await cur.fetchall()

        if rows:
            ids = [r["id"] for r in rows]
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE memories SET last_accessed = NOW() WHERE id = ANY(%s)",
                    (ids,),
                )

        return [
            MemoryEntry(
                id=r["id"],
                content=r["content"],
                summary=r["summary"],
                score=r["score"],
                memory_type=r["memory_type"],
                importance=r["importance"],
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
    finally:
        await conn.close()


async def extract_and_store_memories(conversation_text: str):
    """Use LLM to extract important facts/preferences from conversation and store them."""
    from src.llm.client import ChatMessage, llm_client as client

    prompt = f"""从以下对话中提取值得长期记忆的信息。返回 JSON 数组，每个元素包含:
- "content": 记忆内容（一句话）
- "summary": 简短摘要
- "type": "fact" | "preference" | "conversation"
- "importance": 0-1 的重要程度

只提取确实有价值、以后可能需要回忆的信息。如果没什么重要的，返回空数组 []。

对话内容:
{conversation_text}

只返回 JSON 数组，不要其他文字:"""

    response = await client.chat(
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.3,
        max_tokens=2048,
    )

    try:
        import re
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            items = json.loads(json_match.group())
            for item in items:
                await store_memory(
                    content=item["content"],
                    summary=item.get("summary"),
                    memory_type=item.get("type", "fact"),
                    importance=item.get("importance", 0.5),
                )
    except (json.JSONDecodeError, KeyError):
        pass
