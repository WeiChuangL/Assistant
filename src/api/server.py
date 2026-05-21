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
from src.memory.long_term import search_memories
from src.memory.profile import delete_profile_value, get_profile, set_profile_value
from src.vector_store.db import init_db

app = FastAPI(title="智能助手")

HTML_PATH = Path(__file__).parent / "index.html"


class ChatRequest(BaseModel):
    message: str


class ProfileRequest(BaseModel):
    key: str
    value: str


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent = Agent()

    async def sse_generator():
        try:
            async for token in agent.chat_stream(req.message):
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
