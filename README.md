# LangGraph-Basic

Rename .env.example to .env and add API keys.

## chatbot.py

```
A simple chatbot application built using LangGraph.
This script defines a stateful graph that manages a conversation
with a language model. It initializes a chat model, sets up a
state graph with a single chatbot node, and provides a command-line
interface for users to interact with the chatbot.
```

## web_search_tool.py

```
This script demonstrates a basic LangGraph setup for a chatbot
that can use a Tavily search tool to answer questions.

It defines a state graph with two main nodes:
1.  `chatbot`: An LLM (OpenAI's gpt-4.1) that can respond to user queries
    and decide to use tools.
2.  `tools`: A node that executes the Tavily search tool if requested by the chatbot.

```
