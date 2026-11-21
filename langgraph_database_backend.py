import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

model = os.getenv("GROQ_MODEL")

llm = ChatGroq(model=model)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# For Streaming

# for message_chunk, metadata in chatbot.stream(
#     {"messages": [HumanMessage(content="What is the recipe to make pasta")]},
#     config={"configurable": {"thread_id": "thread-1"}},
#     stream_mode="messages",
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(
            checkpoint.config['configurable']['thread_id']
        )

    return list(all_threads)