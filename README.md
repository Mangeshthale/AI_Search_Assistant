# AI Search Assistant — Multi-Source Intelligent Research Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-green)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)](https://searchenginellmandagents.streamlit.app/)
[![Model](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange)](https://groq.com/)

An autonomous AI research agent that routes your query to the **right source automatically** — DuckDuckGo for current news, Wikipedia for factual knowledge, and Arxiv for academic research — with real-time reasoning visualization.

🔗 **[Live Demo](https://searchenginellmandagents.streamlit.app/)**

---

## What Makes It Different From a Chatbot?

A regular chatbot guesses. This agent **reasons, acts, and observes** — it decides which tool to use based on your query, executes the search, reads the result, and decides if it needs to search again. You can watch every step happen in real time.

---

## How It Works (ReAct Framework)

```
User Query
    │
    ▼
ReAct Agent (Llama 3.3 70B via Groq)
    │
    ├── Thought: "This looks like a research paper topic"
    │
    ├── Action: ArxivQueryRun("transformer attention mechanism")
    │
    ├── Observation: [results returned]
    │
    └── Final Answer ──► Streamed to UI in real time
```

The agent autonomously picks from 3 tools:
- **DuckDuckGo** — current events, news, general web
- **Wikipedia** — factual, encyclopedic knowledge
- **Arxiv** — academic papers and research

---

## Key Features

- **Zero-Shot ReAct Agent** — no examples needed, reasons from scratch every query
- **Automatic source selection** — agent picks the best tool based on query intent
- **Real-time reasoning visualization** via StreamlitCallbackHandler — see the agent think
- **Stateful multi-turn chat** with full session management
- **Groq Llama 3.3 70B** for fast, high-quality responses
- **Custom dark-themed UI** with configurable API key input

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangChain Zero-Shot ReAct |
| LLM | Groq (Llama 3.3 70B) |
| Tools | DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun |
| Streaming | StreamlitCallbackHandler |
| UI | Streamlit |

---

## Run Locally

```bash
git clone https://github.com/Mangeshthale/AI_Search_Assistant
cd AI_Search_Assistant
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key
```

```bash
streamlit run app.py
```

Or enter your API key directly in the sidebar when the app loads.

---

## Example Queries to Try

| Query | Tool Agent Uses |
|---|---|
| "Latest news about GPT-5" | DuckDuckGo |
| "What is the Turing Test?" | Wikipedia |
| "Recent papers on RAG systems" | Arxiv |
| "Who won the 2024 US election?" | DuckDuckGo |

---

## Author

**Mangesh Thale** — [LinkedIn](https://www.linkedin.com/in/mangesh-thale/) | [GitHub](https://github.com/Mangeshthale)
