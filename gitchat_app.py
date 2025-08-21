"""
GitHub Repo Chatbot with PydanticAI and Ollama
---------------------------------------------
This application allows you to chat with the contents of any GitHub repository.
It uses PydanticAI as the agent framework with a local Ollama model for LLM calls.
"""

# =========== SECTION 1: IMPORTS ===========
import os
import subprocess
import tempfile

import streamlit as st

# PydanticAI for agent-based LLM interaction
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# FAISS vector store & Ollama embeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# =========== SECTION 2: REPOSITORY TEXT EXTRACTION ===========
def get_repo_text(repo_url: str, allowed_extensions: set | None = None) -> str:
    if not allowed_extensions:
        allowed_extensions = {
            ".py", ".md", ".txt", ".js", ".html", ".css", ".json",
            ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".rb", ".go", ".rs",
        }
    repo_texts = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            subprocess.run(
                ["git", "clone", repo_url, tmpdirname],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as e:
            st.error(f"Error cloning the repository: {e}")
            return ""
        for root, _, files in os.walk(tmpdirname):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in allowed_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if content.strip():
                            repo_texts.append(f"Filename: {file}\n{content}")
                    except Exception as e:
                        st.warning(f"Could not read {file_path}: {e}")
        if not repo_texts:
            st.error("No allowed text files found in the repository.")
            return ""
        return "\n".join(repo_texts)

# =========== SECTION 3: TEXT PROCESSING ===========
def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split `text` into chunks of up to `chunk_size` characters,
    with `overlap` characters between chunks.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# =========== SECTION 4: VECTOR STORE CREATION ===========
def create_vectorstore(chunks: list[str]) -> FAISS:
    """
    Build a FAISS vector store using Ollama embeddings.
    """
    embeddings = OllamaEmbeddings(model="all-minilm:33m")
    return FAISS.from_texts(chunks, embeddings)

# =========== SECTION 5: AGENT INITIALIZATION ===========
def initialize_agent(vectorstore: FAISS) -> Agent[None, str]:
    """
    Create a PydanticAI Agent wrapping our vectorstore-based retrieval.
    """
    # Configure the Ollama model as an OpenAI-compatible provider :contentReference[oaicite:0]{index=0}
    ollama_model = OpenAIModel(
        model_name="llama3.2",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1")
        
    )
    sys_prompt = """You are a concise assistant for answering questions about the currently loaded GitHub repository.
    Always ground answers in repository content.
    Use the 'retrieve' tool first with a targeted query to fetch relevant context before answering.
    If retrieved context is insufficient or unrelated, ask a brief clarifying question.
    Prefer facts found in the repo; do not guess. If information is missing, say so and suggest next steps.
    Cite specific files and short snippets when helpful; keep snippets minimal.
    For run/build/test/setup questions, check README/docs/manifests/config files and provide exact commands and file locations.
    When explaining code, mention function/class names, responsibilities, parameters, and data flow.
    Keep responses short and actionable.
    """
    agent = Agent(
        ollama_model,
        system_prompt=(sys_prompt),
    )  # Minimal agent usage :contentReference[oaicite:1]{index=1}

    # Register a plain tool for retrieving the top-4 similar chunks
    @agent.tool_plain
    def retrieve(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=4)
        return "\n\n".join(doc.page_content for doc in docs)

    return agent

# =========== SECTION 6: STREAMLIT UI ===========
def main():
    st.title("📂 GitHub Repo Chatbot with PydanticAI & Ollama")

    repo_url = st.sidebar.text_input(
        "GitHub repository URL:", 
        value="https://github.com/JustCodeIt7/GitHub_Repo_Chat"
    )

    all_exts = [
        ".py", ".md", ".txt", ".js", ".html", ".css", ".json",
        ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".rb", ".go", ".rs"
    ]
    selected_exts = st.sidebar.multiselect(
        "File types to include:", all_exts, default=all_exts[:7]
    )

    if st.sidebar.button("Load Repository"):
        with st.spinner("Cloning and processing..."):
            text = get_repo_text(repo_url, set(selected_exts))
            if text:
                chunks = split_text(text)
                vs = create_vectorstore(chunks)
                st.session_state.vectorstore = vs
                st.session_state.agent = initialize_agent(vs)
                st.success("✅ Repository loaded!")
            else:
                st.error("Failed to load repository.")

    st.header("Chat with the Repository")

    # Initialize history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    result = st.session_state.agent.run_sync(prompt)
                    answer = result.output
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("Enter a repo URL and click **Load Repository** to begin.")

if __name__ == "__main__":
    main()
