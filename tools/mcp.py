from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
import asyncio
import threading

_ASYNC_LOOP = asyncio.new_event_loop()
threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True).start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()

_mcp_client = MultiServerMCPClient(
    {
        "microsoft-learn": {
            "transport": "http",
            "url": "https://learn.microsoft.com/api/mcp",
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(_mcp_client.get_tools())
    except Exception as e:
        print("⚠️ MCP tools unavailable:", e)
        return []
