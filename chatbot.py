"""
A simple chatbot application built using LangGraph.

This script defines a stateful graph that manages a conversation
with a language model. It initializes a chat model, sets up a
state graph with a single chatbot node, and provides a command-line
interface for users to interact with the chatbot.
"""

from typing import Annotated, Any, Literal
from urllib import response
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


# Load Environment Variables
load_dotenv()

# Initialize Chat Model
llm: BaseChatModel = init_chat_model(model="gemma3:4b", model_provider="ollama")


# State messages schema
class State(TypedDict):
    """
    Represents the state of the conversation graph.

    Attributes:
        messages: A list of messages, managed by `add_messages` to append new messages.
    """

    messages: Annotated[list, add_messages]


# Initialize State Graph
state_graph = StateGraph(state_schema=State)


# Define chatbot
def chatbot(state: State) -> dict[str, list[BaseMessage]]:
    """
    Invokes the language model with the current state's messages and returns the LLM's response.

    Args:
        state: The current state of the graph, containing the message history.

    Returns:
        A dictionary with the LLM's response appended to the messages list.
    """
    llm_response: BaseMessage = llm.invoke(input=state["messages"])
    return {"messages": [llm_response]}


# Add chatbot to state graph
state_graph.add_node(node="chatbot", action=chatbot)
state_graph.add_edge(start_key=START, end_key="chatbot")
state_graph.add_edge(start_key="chatbot", end_key=END)

# Compile state graph
graph: CompiledStateGraph = state_graph.compile()


def stream_graph_updates(user_input: str) -> None:
    """
    Streams updates from the graph as they are generated.

    Args:
        user_input: The user's input to send to the chatbot.
    """
    for event in graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]}
    ):
        for value in event.values():
            print(f'\nChatbot: {value["messages"][-1].content}')


# Main entrypoint
def main() -> None:
    """
    Runs the main interaction loop for the chatbot.

    Prompts the user for input, sends it to the chatbot graph,
    and prints the chatbot's response. The loop continues until
    the user types 'q' to quit.
    """
    while True:
        user_input: str = input("Ask any question to the chatbot (type 'x' to exit): ")
        if user_input != "x":
            # Run graph
            stream_graph_updates(user_input=user_input)
        else:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
