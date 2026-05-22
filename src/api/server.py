import asyncio
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.core import Agent
from src.knowledge.ingestion import ingest_file
from src.knowledge.manager import delete_document, list_documents
from src.knowledge.retrieval import search_chunks
from src.mcp.client import mcp_client
from src.mcp.manager import (
    add_server as mcp_add_server,
    get_server as mcp_get_server,
    list_servers as mcp_list_servers,
    remove_server as mcp_remove_server,
    set_auto_connect as mcp_set_auto_connect,
    toggle_server as mcp_toggle_server,
    update_server as mcp_update_server,
)
from src.mcp.market import find_market_server, get_market_servers, get_market_servers_with_status
from src.memory.long_term import search_memories
from src.memory.profile import delete_profile_value, get_profile, set_profile_value
from src.session.manager import (
    create_session,
    delete_session,
    list_sessions,
    rename_session,
)
from src.market_sources import (
    add_source as market_add_source,
    fetch_third_party_market,
    list_sources as market_list_sources,
    remove_source as market_remove_source,
    toggle_source as market_toggle_source,
)
from src.skill.market import get_market_skills, install_from_market
from src.skill.registry import skill_registry
from src.vector_store.db import init_db

app = FastAPI(title="智能助手")

HTML_PATH = Path(__file__).parent / "index.html"


class ChatRequest(BaseModel):
    message: str
    session_id: int | None = None
    triggered_skills: list[str] | None = None


class ProfileRequest(BaseModel):
    key: str
    value: str


@app.on_event("startup")
async def startup():
    await init_db()
    # Connect auto-connect MCP servers
    try:
        servers = await mcp_list_servers()
        await mcp_client.connect_auto(servers)
    except Exception:
        pass  # MCP connection failures are non-fatal


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent = Agent(session_id=req.session_id)

    async def sse_generator():
        try:
            async for token in agent.chat_stream(
                req.message, triggered_skills=req.triggered_skills
            ):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/kb/upload")
async def kb_upload(file: UploadFile = File(...)):
    import tempfile
    tmp = Path(tempfile.mktemp(suffix="." + (file.filename or "txt")))
    tmp.write_bytes(await file.read())
    try:
        doc_id = await ingest_file(tmp)
        return {"ok": True, "document_id": doc_id, "filename": file.filename}
    finally:
        tmp.unlink()


@app.get("/api/kb/list")
async def kb_list():
    docs = await list_documents()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at,
        }
        for d in docs
    ]


@app.delete("/api/kb/{doc_id}")
async def kb_delete(doc_id: int):
    ok = await delete_document(doc_id)
    return {"ok": ok}


@app.get("/api/profile")
async def profile_get():
    return await get_profile()


@app.post("/api/profile")
async def profile_set(req: ProfileRequest):
    await set_profile_value(req.key, req.value)
    return {"ok": True}


@app.delete("/api/profile/{key}")
async def profile_del(key: str):
    await delete_profile_value(key)
    return {"ok": True}


# ── Session endpoints ──

@app.post("/api/session/new")
async def session_new(title: str = "新会话"):
    session = await create_session(title)
    return {"id": session.id, "title": session.title, "created_at": session.created_at, "updated_at": session.updated_at}


@app.get("/api/session/list")
async def session_list():
    sessions = await list_sessions()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at} for s in sessions]


@app.delete("/api/session/{session_id}")
async def session_del(session_id: int):
    await delete_session(session_id)
    return {"ok": True}


@app.put("/api/session/{session_id}/rename")
async def session_rename(session_id: int, title: str = ""):
    if not title:
        return {"ok": False, "error": "title is required"}
    await rename_session(session_id, title)
    return {"ok": True}


# ── Skill endpoints ──

@app.get("/api/skill/list")
async def skill_list():
    return skill_registry.list_all()


@app.post("/api/skill/{name}/enable")
async def skill_enable(name: str):
    skill_registry.enable(name)
    return {"ok": True}


@app.post("/api/skill/{name}/disable")
async def skill_disable(name: str):
    skill_registry.disable(name)
    return {"ok": True}


@app.delete("/api/skill/{name}")
async def skill_remove(name: str):
    skill_registry.remove(name)
    return {"ok": True}


@app.get("/api/skill/market")
async def skill_market():
    return await get_market_skills()


@app.post("/api/skill/market/{name}/install")
async def skill_market_install(name: str):
    ok = install_from_market(name)
    return {"ok": ok}


@app.put("/api/skill/{name}/auto-trigger")
async def skill_auto_trigger(name: str, body: dict):
    auto_trigger = body.get("auto_trigger", True)
    skill_registry.toggle_auto_trigger(name, auto_trigger)
    return {"ok": True, "auto_trigger": auto_trigger}


@app.post("/api/skill/{name}/trigger")
async def skill_trigger(name: str):
    """Return skill info so frontend can append it to next message's triggered_skills."""
    skill = skill_registry.get(name)
    if not skill:
        return {"ok": False, "error": "skill not found"}
    return {"ok": True, "name": name, "keywords": skill.trigger_keywords}


# ── MCP endpoints ──

@app.get("/api/mcp/servers")
async def mcp_server_list():
    servers = await mcp_list_servers()
    for s in servers:
        s["status"] = mcp_client.get_connection_status(s["id"])
        s["tool_count"] = mcp_client.get_tool_count(s["id"])
    return servers


@app.post("/api/mcp/servers")
async def mcp_server_add(req: dict):
    sid = await mcp_add_server(
        name=req.get("name", ""),
        transport=req.get("transport", "stdio"),
        command=req.get("command"),
        args=req.get("args", []),
        url=req.get("url"),
        headers=req.get("headers", {}),
        env=req.get("env", {}),
    )
    srv = await mcp_get_server(sid)
    if srv:
        tools = await mcp_client.connect(sid, srv)
        return {"ok": True, "id": sid, "tools": len(tools)}
    return {"ok": False, "error": "failed to create server"}


@app.delete("/api/mcp/servers/{server_id}")
async def mcp_server_del(server_id: int):
    await mcp_client.disconnect(server_id)
    await mcp_remove_server(server_id)
    return {"ok": True}


@app.put("/api/mcp/servers/{server_id}")
async def mcp_server_update(server_id: int, req: dict):
    await mcp_update_server(server_id, **req)
    return {"ok": True}


@app.post("/api/mcp/servers/{server_id}/toggle")
async def mcp_server_toggle(server_id: int):
    srv = await mcp_get_server(server_id)
    if not srv:
        return {"ok": False, "error": "not found"}
    new_state = not srv["enabled"]
    await mcp_toggle_server(server_id, new_state)
    if new_state:
        await mcp_client.connect(server_id, srv)
    else:
        await mcp_client.disconnect(server_id)
    return {"ok": True, "enabled": new_state}


@app.post("/api/mcp/servers/{server_id}/reconnect")
async def mcp_server_reconnect(server_id: int):
    await mcp_client.disconnect(server_id)
    srv = await mcp_get_server(server_id)
    if not srv:
        return {"ok": False, "error": "server not found"}
    tools = await mcp_client.connect(server_id, srv)
    return {"ok": True, "tools": tools, "count": len(tools)}


@app.put("/api/mcp/servers/{server_id}/auto-connect")
async def mcp_server_auto_connect(server_id: int, body: dict):
    auto_connect = body.get("auto_connect", True)
    await mcp_set_auto_connect(server_id, auto_connect)
    return {"ok": True, "auto_connect": auto_connect}


@app.get("/api/mcp/servers/{server_id}/tools")
async def mcp_server_tools(server_id: int):
    return mcp_client.get_server_tools(server_id)


@app.get("/api/mcp/market")
async def mcp_market():
    return await get_market_servers_with_status()


@app.post("/api/mcp/market/{name}/add")
async def mcp_market_add(name: str, config: dict | None = None):
    item = find_market_server(name)
    if not item:
        return {"ok": False, "error": "market item not found"}

    # Merge user-provided config overrides (API keys etc.)
    env = item.get("env", {})
    if config:
        if "env" in config:
            env.update(config["env"])
        if "args" in config:
            item["args"] = config["args"]
        if "headers" in config:
            item["headers"] = {**item.get("headers", {}), **config["headers"]}

    sid = await mcp_add_server(
        name=item["name"],
        transport=item.get("transport", "stdio"),
        command=item.get("command"),
        args=item.get("args", []),
        url=item.get("url"),
        headers=item.get("headers", {}),
        env=env,
    )
    srv = await mcp_get_server(sid)
    if srv:
        tools = await mcp_client.connect(sid, srv)
        return {"ok": True, "id": sid, "tools": len(tools)}
    return {"ok": False, "error": "failed to add server"}


# ── Market Source endpoints ──

@app.get("/api/skill/market/sources")
async def skill_market_sources():
    return await market_list_sources("skill")


@app.post("/api/skill/market/sources")
async def skill_market_source_add(body: dict):
    name = body.get("name", "").strip()
    url = body.get("url", "").strip()
    if not name or not url:
        return {"ok": False, "error": "name and url are required"}
    sid = await market_add_source("skill", name, url)
    return {"ok": True, "id": sid}


@app.delete("/api/skill/market/sources/{source_id}")
async def skill_market_source_del(source_id: int):
    await market_remove_source(source_id)
    return {"ok": True}


@app.put("/api/skill/market/sources/{source_id}")
async def skill_market_source_toggle(source_id: int, body: dict):
    enabled = body.get("enabled", True)
    await market_toggle_source(source_id, enabled)
    return {"ok": True, "enabled": enabled}


@app.get("/api/mcp/market/sources")
async def mcp_market_sources():
    return await market_list_sources("mcp")


@app.post("/api/mcp/market/sources")
async def mcp_market_source_add(body: dict):
    name = body.get("name", "").strip()
    url = body.get("url", "").strip()
    if not name or not url:
        return {"ok": False, "error": "name and url are required"}
    sid = await market_add_source("mcp", name, url)
    return {"ok": True, "id": sid}


@app.delete("/api/mcp/market/sources/{source_id}")
async def mcp_market_source_del(source_id: int):
    await market_remove_source(source_id)
    return {"ok": True}


@app.put("/api/mcp/market/sources/{source_id}")
async def mcp_market_source_toggle(source_id: int, body: dict):
    enabled = body.get("enabled", True)
    await market_toggle_source(source_id, enabled)
    return {"ok": True, "enabled": enabled}
