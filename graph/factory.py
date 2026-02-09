from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from core.state import ChatState
from graph.nodes import make_async_chat_node

def build_async_graph(llm, tools, checkpointer):
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    graph = StateGraph(ChatState)

    graph.add_node("chat", make_async_chat_node(llm_with_tools))

    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "chat")
        graph.add_conditional_edges("chat", tools_condition)
        graph.add_edge("tools", "chat")
    else:
        graph.add_edge(START, "chat")
        graph.add_edge("chat", END)

    return graph.compile(checkpointer=checkpointer)
