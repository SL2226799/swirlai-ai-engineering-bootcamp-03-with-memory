
from src.api.core.config import config
from src.api.rag.tools import get_formatted_context
from src.api.rag.agent import agent_node
from src.api.rag.utils.utils import get_tool_descriptions_from_node

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from src.api.rag.common import State



from qdrant_client import QdrantClient

import logging

logger = logging.getLogger(__name__)

def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def create_agent_graph():

    workflow = StateGraph(State)

    tools = [get_formatted_context]
    tool_node = ToolNode(tools)

    tool_descriptions = get_tool_descriptions_from_node(tool_node)

    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tool_node", tool_node)

    workflow.add_edge(START, "agent_node")

    workflow.add_conditional_edges(
        "agent_node",
        tool_router,
        {
            "tools": "tool_node",
            "end": END
        }
    )

    workflow.add_edge("tool_node", "agent_node")

    graph = workflow.compile()

    return graph, tool_descriptions, workflow

def run_agent(query: str, thread_id: str):

    graph, tool_descriptions, workflow = create_agent_graph()

    initial_state = {
    "messages": [{"role": "user", "content": query}],
    "iteration": 0,
    "available_tools": tool_descriptions
    }

    configurable = {"configurable" : {"thread_id": thread_id}}

    print("config: ", configurable)

    with PostgresSaver.from_conn_string(config.POSTGRES_CONN_STRING) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config=configurable)

    print(result)

    return result
    

def run_agent_wrapper(query: str, thread_id: str):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    
    result = run_agent(query, thread_id)

    image_list = []
    for context in result.get("retrieved_context_ids"):
        payload = qdrant_client.retrieve(
            collection_name=config.QDRANT_COLLECTION_NAME,
            ids=[context.id]
        )[0].payload

        logger.info("payload: ", payload)

        image_url = payload.get("first_large_image")
        price = payload.get("price")
        if image_url and price:
            image_list.append({"image_url": image_url, "price": price, "description": context.short_description})

    logger.info("image_list: ", image_list)

    return {
        "answer": result.get("answer"),
        "retrieved_images": image_list
    }
    