from fastapi import FastAPI
from contextlib import asynccontextmanager

from llm.base import get_llm
from graph.factory import build_async_graph
from core.db import get_async_checkpointer
from services.chat_service import ChatService
from api.chat import register_routes

from tools.builtin import search_tool, calculator, get_stock_price
from tools.rag import rag_tool
from tools.mcp import load_mcp_tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for async startup/shutdown.
    """
    # -----------------------
    # Startup
    # -----------------------
    llm = get_llm()

    tools = [
        search_tool,
        calculator,
        get_stock_price,
        rag_tool,
        *load_mcp_tools(),  # MCP tools (async-only)
    ]

    checkpointer = await get_async_checkpointer()
    chatbot = build_async_graph(llm, tools, checkpointer)

    app.state.chatbot = chatbot
    app.state.chat_service = ChatService(chatbot)

    yield

    # -----------------------
    # Shutdown (optional cleanup)
    # -----------------------


app = FastAPI(
    title="Agentic AI App",
    version="1.0.0",
    lifespan=lifespan,
)

register_routes(app)
