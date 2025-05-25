# Tavily Search
from typing import Any
from unittest import result
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
from getpass import getpass

load_dotenv()

if not os.getenv(key="TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API Key:\n")

tool: TavilySearch = TavilySearch(max_results=5)
tools: list[TavilySearch] = [tool]

response: Any = tool.run(tool_input="What is Python?")
print(f"\n{response}")
