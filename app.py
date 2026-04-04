import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Search Chat", page_icon="🔎", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Chat bubbles */
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

/* Title */
.title {
    font-size: 2.2rem;
    font-weight: bold;
    color: #38bdf8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Input box */
.stChatInput input {
    border-radius: 10px !important;
}

/* Buttons */
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🔎 AI Search Assistant</div>', unsafe_allow_html=True)
st.caption("Chat with AI + Web + Wikipedia + Arxiv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")

if api_key:
    st.sidebar.success("API Key loaded ✅")
else:
    st.sidebar.warning("Enter API Key to start")

st.sidebar.markdown("---")
st.sidebar.info("💡 Supports:\n- Web Search\n- Wikipedia\n- Arxiv Papers")

# ---------------- TOOLS ----------------
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

search = DuckDuckGoSearchRun(name="Search")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey 👋 I'm your AI search assistant. Ask me anything!"}
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

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-user">{prompt}</div>', unsafe_allow_html=True)

    # LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        streaming=True
    )

    tools = [search, arxiv, wiki]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )

    # Assistant response
    with st.spinner("Thinking... 🤔"):
        with st.container():
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(prompt, callbacks=[st_cb])

    # Store + display
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f'<div class="chat-assistant">{response}</div>', unsafe_allow_html=True)
