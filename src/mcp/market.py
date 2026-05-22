import json
from pathlib import Path

from src.market_sources import fetch_third_party_market
from src.mcp.manager import list_servers

_MARKET_FILE = Path(__file__).parent.parent.parent / "data" / "mcp_market.json"


def _load_local_market() -> list[dict]:
    if not _MARKET_FILE.exists():
        return []
    try:
        return json.loads(_MARKET_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


async def get_market_servers() -> list[dict]:
    """Return market servers merged from local + third-party sources."""
    data = _load_local_market()
    local_names = {item["name"] for item in data}

    for item in data:
        item["source"] = "builtin"

    try:
        third_party = await fetch_third_party_market("mcp")
    except Exception:
        third_party = []

    for entry in third_party:
        if entry.get("name", "") in local_names:
            continue
        data.append(entry)

    return data


async def get_market_servers_with_status() -> list[dict]:
    """Return market servers with 'added' flag based on existing configs."""
    servers = await list_servers()
    existing_names = {s["name"] for s in servers}
    market = await get_market_servers()
    for item in market:
        item["added"] = item["name"] in existing_names
    return market


def find_market_server(name: str) -> dict | None:
    market = _load_local_market()
    for item in market:
        if item["name"] == name:
            return item
    return None
