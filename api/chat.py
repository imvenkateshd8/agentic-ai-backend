from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from fastapi.responses import StreamingResponse
import asyncio

from tools.rag import ingest_pdf

router = APIRouter()


# -----------------------
# Request / Response Models
# -----------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class SourceInfo(BaseModel):
    type: str  # model | rag | tool | mcp
    name: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    sources: list[SourceInfo]


class UploadPDFResponse(BaseModel):
    thread_id: str
    filename: str
    documents: int
    chunks: int


# -----------------------
# Chat Endpoint
# -----------------------
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    thread_id = req.thread_id or str(uuid.uuid4())

    chat_service = request.app.state.chat_service
    result = await chat_service.invoke(req.message, thread_id)

    return ChatResponse(
        thread_id=thread_id,
        answer=result["answer"],
        sources=result["sources"],
    )


# -----------------------
# Chat Stream Endpoint
# -----------------------

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    thread_id = req.thread_id or str(uuid.uuid4())
    chat_service = request.app.state.chat_service

    async def event_generator():
        async for chunk in chat_service.stream(req.message, thread_id):
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

# -----------------------
# Upload PDF Endpoint
# -----------------------
@router.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    thread_id: Optional[str] = None,
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    thread_id = thread_id or str(uuid.uuid4())
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        metadata = ingest_pdf(
            file_bytes=file_bytes,
            thread_id=thread_id,
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadPDFResponse(
        thread_id=thread_id,
        filename=metadata.get("filename"),
        documents=metadata.get("documents", 0),
        chunks=metadata.get("chunks", 0),
    )


# -----------------------
# Router registration helper
# -----------------------
def register_routes(app):
    app.include_router(router)
