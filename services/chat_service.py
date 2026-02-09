from langchain_core.messages import HumanMessage, AIMessage

class ChatService:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    async def invoke(self, message: str, thread_id: str):
        result = await self.chatbot.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        messages = result["messages"]
        final_msg: AIMessage = messages[-1]

        sources = []
        seen = set()

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.get("name")
                    if name == "rag_tool":
                        seen.add(("rag", None))
                    elif name in ("calculator", "get_stock_price"):
                        seen.add(("tool", name))
                    else:
                        seen.add(("mcp", name))

        if not seen:
            sources.append({"type": "model"})
        else:
            for t, n in seen:
                sources.append({"type": t, "name": n})

        return {
            "answer": final_msg.content,
            "sources": sources,
        }

    def history(self, thread_id: str):
        state = self.chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", [])



