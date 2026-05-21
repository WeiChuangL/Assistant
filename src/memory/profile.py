import json
import re

import psycopg
from psycopg.rows import dict_row

from src.config import settings
from src.vector_store.db import get_conn


async def get_profile() -> dict[str, str]:
    """Get all user profile key-value pairs."""
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT key, value FROM user_profile ORDER BY key")
            rows = await cur.fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        await conn.close()


async def get_profile_value(key: str) -> str | None:
    """Get a single profile value by key."""
    conn = await get_conn()
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT value FROM user_profile WHERE key = %s", (key,))
            row = await cur.fetchone()
        return row["value"] if row else None
    finally:
        await conn.close()


async def set_profile_value(key: str, value: str):
    """Set or update a profile key-value pair."""
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO user_profile (key, value, updated_at) VALUES (%s, %s, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                (key, value),
            )
    finally:
        await conn.close()


async def delete_profile_value(key: str) -> bool:
    """Delete a profile key."""
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM user_profile WHERE key = %s", (key,))
            return cur.rowcount > 0
    finally:
        await conn.close()


async def extract_and_update_profile(conversation_text: str):
    """Use LLM to detect user preferences and update profile."""
    from src.llm.client import ChatMessage, llm_client as client

    prompt = f"""分析以下对话，识别用户的偏好、习惯或个人信息。返回 JSON 对象，每个 key 是偏好的类别，value 是具体内容。

例如：
{{"编程语言": "偏好 Python", "工作习惯": "喜欢在晚上coding", "身份": "全栈工程师"}}

只提取明确的、用户自己表达的偏好或信息。不要推测。如果没有明显的偏好，返回空对象 {{}}。

对话内容:
{conversation_text}

只返回 JSON 对象，不要其他文字:"""

    response = await client.chat(
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.3,
        max_tokens=1024,
    )

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            prefs = json.loads(json_match.group())
            for key, value in prefs.items():
                if value and isinstance(value, str):
                    await set_profile_value(key, value)
    except (json.JSONDecodeError, KeyError):
        pass
