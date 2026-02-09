import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def get_async_checkpointer(db_path="chatbot.db"):
    conn = await aiosqlite.connect(db_path)
    return AsyncSqliteSaver(conn)
