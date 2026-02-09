from langchain_core.messages import HumanMessage, AIMessage
from typing import AsyncGenerator
import asyncio


class ChatService:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    # -------------------------
    # Normal (non-streaming)
    # -------------------------
    async def invoke(self, message: str, thread_id: str):
        result = await self.chatbot.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        final_msg: AIMessage = result["messages"][-1]

        sources = []

        if final_msg.tool_calls:
            for call in final_msg.tool_calls:
                tool_name = call.get("name")

                if tool_name == "rag_tool":
                    sources.append({"type": "rag"})
                elif tool_name in ("calculator", "get_stock_price"):
                    sources.append({"type": "tool", "name": tool_name})
                else:
                    sources.append({"type": "mcp", "name": tool_name})

        if not sources:
            sources.append({"type": "model"})

        return {
            "answer": final_msg.content,
            "sources": sources,
        }

    # -------------------------
    # Streaming (SSE)
    # -------------------------
    async def stream(
        self, message: str, thread_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens using LangGraph astream().
        """
        async for event in self.chatbot.astream(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        ):
            if "messages" in event:
                for msg in event["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        yield msg.content
                        await asyncio.sleep(0)  # yield control

    # -------------------------
    # Conversation history
    # -------------------------
    def history(self, thread_id: str):
        state = self.chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", [])