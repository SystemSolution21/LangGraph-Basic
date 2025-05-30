"""
This script demonstrates a basic LangGraph setup for a chatbot
that can use a Tavily search tool to answer questions.

It defines a state graph with two main nodes:
1.  `chatbot`: An LLM (OpenAI's gpt-4.1) that can respond to user queries
    and decide to use tools.
2.  `tools`: A node that executes the Tavily search tool if requested by the chatbot.

The script initializes the necessary components, including:
-   Environment variable loading for API keys (Tavily and OpenAI).
-   TavilySearch tool.
-   OpenAI chat model.
-   LangGraph StateGraph and nodes.

It then compiles the graph and provides a simple command-line interface
to interact with the chatbot, streaming the assistant's responses.
The conversation flow allows the chatbot to call the search tool,
receive its output, and then generate a final response to the user.
"""

import os

from typing import Annotated, List
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
from getpass import getpass

# --- Constants ---
TAVILY_API_KEY = "TAVILY_API_KEY"
OPENAI_API_KEY = "OPENAI_API_KEY"
MODEL_NAME = "gpt-4.1"
PROVIDER_NAME = "openai"
TAVILY_MAX_RESULTS = 3


# Load environment variables
load_dotenv()

# Set Tavily API key
tavily_api_key: str | None = os.getenv(key=TAVILY_API_KEY)
if not tavily_api_key:
    tavily_api_key = getpass(prompt=f"Enter your {TAVILY_API_KEY}:\n")
    if not tavily_api_key:
        print(f"Please set your {TAVILY_API_KEY}. Exiting...")
        exit(code=1)
    os.environ[TAVILY_API_KEY] = tavily_api_key


# Initialize Tavily Search
tools: List[TavilySearch] = [TavilySearch(max_results=TAVILY_MAX_RESULTS)]


# Set OpenAI API key
openai_api_key: str | None = os.getenv(key=OPENAI_API_KEY)
if not openai_api_key:
    openai_api_key = getpass(prompt=f"Enter your {OPENAI_API_KEY}:\n")
    if not openai_api_key:
        print(f"Please set your {OPENAI_API_KEY}. Exiting...")
        exit(code=1)
    os.environ[OPENAI_API_KEY] = openai_api_key


# Initialize chat model
llm: BaseChatModel = init_chat_model(model=MODEL_NAME, model_provider=PROVIDER_NAME)


# State schema
class State(TypedDict):
    """
    Represents the state of the conversation graph.

    Attributes:
        messages: A list of messages, managed by `add_messages` to append new messages.
    """

    messages: Annotated[List[BaseMessage], add_messages]


# Initialize state graph
graph_builder = StateGraph(state_schema=State)


# Bind tools with llm
llm_with_tools: Runnable = llm.bind_tools(tools=tools)


# Define chatbot
def chatbot(state: State) -> dict[str, list[BaseMessage]]:
    """
    Invokes the LLM with the current state's messages and returns the LLM's response.
    If the LLM decides to use a tool, the response will include tool calls.

    Args:
        state: The current state of the graph, containing the message history.

    Returns:
        A dictionary with the LLM's response appended to the messages list.
    """
    return {"messages": [llm_with_tools.invoke(input=state["messages"])]}


# Add chatbot to state graph
graph_builder.add_node(node="chatbot", action=chatbot)


# Add tool node to state graph
tool_node = ToolNode(tools=tools)
graph_builder.add_node(node="tools", action=tool_node)


# Add conditional edges
graph_builder.add_conditional_edges(
    source="chatbot",
    path=tools_condition,
)


# Add edges
graph_builder.add_edge(start_key="tools", end_key="chatbot")
graph_builder.add_edge(start_key=START, end_key="chatbot")
graph: CompiledStateGraph = graph_builder.compile()


# Stream graph updates
def stream_graph_updates(user_input: str) -> None:
    """
    Streams updates from the graph as they are generated in response to user input.

    Args:
        user_input: The user's input string to send to the chatbot.
    """
    for event in graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]}
    ):
        for value in event.values():
            if "messages" in value and value["messages"]:
                print("\nAssistant:", value["messages"][-1].content)


# Main entrypoint
def main() -> None:
    """
    Runs the main interaction loop for the chatbot.

    Prompts the user for input, sends it to the chatbot graph,
    and prints the chatbot's streamed responses. The loop continues until
    the user types 'q' to quit.
    """
    while True:
        try:
            user_input: str = input("Ask a question to Chatbot (type 'q' to 'quit'): ")
            if user_input.lower() == "q":
                print("Goodbye!")
                break
            if user_input.strip():
                stream_graph_updates(user_input=user_input)
            else:
                print("Please enter a question.")
        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
