import time
import streamlit as st
from groq import APIStatusError
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_community.callbacks import StreamlitCallbackHandler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Search Chat", page_icon="🤖", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}
.chat-user {
    background: #1e293b;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
}
.chat-assistant {
    background: #020617;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    border-left: 3px solid #38bdf8;
}
.title {
    font-size: 2.2rem;
    font-weight: bold;
    color: #38bdf8;
}
section[data-testid="stSidebar"] {
    background-color: #020617;
}
.stChatInput input {
    border-radius: 10px !important;
}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Search Assistant</div>', unsafe_allow_html=True)
st.caption("Chat with AI + Web + Wikipedia + Arxiv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password").strip()
if api_key:
    st.sidebar.success("API Key loaded ✅")
else:
    st.sidebar.warning("Enter API Key to start")
st.sidebar.markdown("---")
st.sidebar.info("💡 Supports:\n- Web Search\n- Wikipedia\n- Arxiv Papers")

# ---------------- TOOLS ----------------
_arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=150)
)
_wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=150)
)
_search_tool = DuckDuckGoSearchRun(name="Search")


def _safe_search_run(query: str) -> str:
    try:
        return _search_tool.run(query)
    except Exception as e:
        return f"Search failed ({e}). Try Wikipedia or Arxiv instead, or answer from general knowledge."


search = Tool(name=_search_tool.name, description=_search_tool.description, func=_safe_search_run)


def _safe_arxiv_run(query: str) -> str:
    try:
        return _arxiv_tool.run(query)
    except Exception as e:
        return f"Arxiv lookup failed ({e}). Try the Search tool instead."


def _safe_wiki_run(query: str) -> str:
    try:
        return _wiki_tool.run(query)
    except Exception as e:
        return f"Wikipedia lookup failed ({e}). Try the Search tool instead."


arxiv = Tool(name=_arxiv_tool.name, description=_arxiv_tool.description, func=_safe_arxiv_run)
wiki = Tool(name=_wiki_tool.name, description=_wiki_tool.description, func=_safe_wiki_run)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "✨ Discover answers, insights, and research in seconds."}
    ]

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
prompt = st.chat_input("Ask anything...")

if prompt:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-user">{prompt}</div>', unsafe_allow_html=True)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3.6-27b",
        streaming=True,
        reasoning_format="parsed"
    )

    tools = [search, arxiv, wiki]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=(
            "Invalid format. You must respond with EITHER:\n"
            "Thought: <reasoning>\nAction: <tool name>\nAction Input: <input>\n"
            "OR, if you already have enough information:\n"
            "Thought: <reasoning>\nFinal Answer: <answer>"
        ),
        max_iterations=6,
        early_stopping_method="generate",
    )

    with st.spinner("Thinking... 🤔"):
        with st.container():
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent.run(prompt, callbacks=[st_cb])
            except APIStatusError as e:
                if "rate_limit_exceeded" in str(e):
                    time.sleep(5)
                    try:
                        response = agent.run(prompt, callbacks=[st_cb])
                    except APIStatusError:
                        response = (
                            "⚠️ The model is currently rate-limited (free tier). "
                            "Please wait a minute and try again, or ask a shorter question."
                        )
                else:
                    response = f"⚠️ An API error occurred: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f'<div class="chat-assistant">{response}</div>', unsafe_allow_html=True)
