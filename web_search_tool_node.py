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

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
from getpass import getpass


# Load environment variables
load_dotenv()

# Set Tavily API key
if not os.getenv(key="TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass(prompt="Enter your Tavily API Key:\n")

# Initialize Tavily Search
tool = TavilySearch(max_results=1)
tools: list[TavilySearch] = [tool]

# Set OpenAI API key
if not os.getenv(key="OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass(prompt="Enter your OpenAI API Key:\n")

# Initialize chat model
llm: BaseChatModel = init_chat_model(model="gpt-4.1", model_provider="openai")


# State schema
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize state graph
graph_builder = StateGraph(state_schema=State)

# Bind tools with llm
llm_with_tools: Runnable[LanguageModelInput, BaseMessage] = llm.bind_tools(tools=tools)


# Define chatbot
def chatbot(state: State) -> dict[str, list[BaseMessage]]:
    return {"messages": [llm_with_tools.invoke(input=state["messages"])]}


# Add chatbot to state graph
graph_builder.add_node(node="chatbot", action=chatbot)

# Add tool node to state graph
tool_node = ToolNode(tools=[tool])
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
    for event in graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]}
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# Main entrypoint
if __name__ == "__main__":

    while True:
        try:
            user_input: str = input("Ask a question to Chatbot (type 'q' to 'quit'): ")
            if user_input.lower() == "q":
                print("Goodbye!")
                break

            stream_graph_updates(user_input=user_input)
        except KeyboardInterrupt:
            print("Goodbye!")
            break
