from langchain_core.messages import SystemMessage
from tools.rag import thread_has_document

def make_async_chat_node(llm_with_tools):
    async def chat_node(state, config=None):
        thread_id = config.get("configurable", {}).get("thread_id")
        has_doc = thread_id and thread_has_document(thread_id)

        system = SystemMessage(
            content=(
                "You are a helpful assistant with access to tools.\n\n"
                "DECISION RULES:\n"
                "1. Use `rag_tool` ONLY IF:\n"
                "   - A document exists for this thread AND\n"
                "   - The user's question is about the document's content.\n\n"
                "2. Do NOT use `rag_tool` for:\n"
                "   - General world knowledge questions\n"
                "   - Company overviews, definitions, explanations\n"
                "   - Questions unrelated to the uploaded document\n\n"
                "3. Use MCP tools when external systems or enterprise data are required.\n"
                "4. Use local tools (calculator, etc.) when appropriate.\n"
                "5. Answer directly from the model if no tool is applicable.\n\n"
                f"Document available: {bool(has_doc)}\n"
                f"Current thread_id: {thread_id}"
            )
        )

        messages = [system, *state["messages"]]
        response = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}

    return chat_node
