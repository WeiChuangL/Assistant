import json
from pathlib import Path

from src.market_sources import fetch_third_party_market
from src.skill.registry import skill_registry

_MARKET_FILE = Path(__file__).parent.parent.parent / "data" / "skill_market.json"


def _load_local_market() -> list[dict]:
    if not _MARKET_FILE.exists():
        return []
    try:
        return json.loads(_MARKET_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


async def get_market_skills() -> list[dict]:
    # Local entries
    data = _load_local_market()
    local_names = {item["name"] for item in data}

    # Mark installed
    for item in data:
        item["installed"] = item["name"] in skill_registry
        item["source"] = "builtin"

    # Third-party entries
    try:
        third_party = await fetch_third_party_market("skill")
    except Exception:
        third_party = []

    for entry in third_party:
        entry["installed"] = entry.get("name", "") in skill_registry
        # Local takes priority — skip duplicates
        if entry.get("name", "") in local_names:
            continue
        data.append(entry)

    return data


def install_from_market(name: str) -> bool:
    """Install a skill from the marketplace. Works with built-in entries.
    For third-party installs, the market data is already loaded at call time."""
    # First try local market
    local = _load_local_market()
    for item in local:
        if item["name"] == name:
            skill_registry.register({
                "name": item["name"],
                "display_name": item.get("display_name", item["name"]),
                "description": item.get("description", ""),
                "icon": item.get("icon", "⚡"),
                "prompt_append": item.get("prompt_append", ""),
                "tools": item.get("tools", []),
                "enabled": True,
                "auto_trigger": item.get("auto_trigger", True),
                "trigger_keywords": item.get("trigger_keywords", []),
            })
            return True
    return False
