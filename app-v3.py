import os
import re
import time
import json
import hashlib
from typing import List, Tuple, Optional, Dict

import requests
import streamlit as st

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage

# ---------------------------
# Config & Utilities
# ---------------------------

st.set_page_config(page_title="GitHub Repo Chat (RAG)", page_icon="💬", layout="wide")

DEFAULT_EXTENSIONS = [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml", ".java", ".go", ".rs"]
MAX_README_TOKENS = 2000  # approx
TOKEN_TO_CHAR_RATIO = 4  # ~4 chars ≈ 1 token (rough heuristic)
MAX_README_CHARS = MAX_README_TOKENS * TOKEN_TO_CHAR_RATIO
MAX_FILE_CHARS = 300_000  # safety limit for large files


def approximate_truncate(text: str, max_tokens: int = MAX_README_TOKENS) -> str:
    max_chars = max_tokens * TOKEN_TO_CHAR_RATIO
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated to fit ~2000 tokens for context...]"


def parse_github_url(url: str) -> Tuple[str, str]:
    """
    Parse 'https://github.com/{owner}/{repo}[...optional...]' and return (owner, repo).
    """
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        raise ValueError("Please provide a valid GitHub repo URL like https://github.com/owner/repo")
    owner, repo = m.group(1), m.group(2)
    # Strip .git if present
    repo = repo[:-4] if repo.endswith(".git") else repo
    return owner, repo


def gh_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "streamlit-rag-app"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def get_default_branch(owner: str, repo: str, token: Optional[str]) -> str:
    resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=gh_headers(token), timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get repo info: HTTP {resp.status_code} - {resp.text}")
    data = resp.json()
    return data.get("default_branch", "main")


def fetch_readme(owner: str, repo: str, token: Optional[str]) -> str:
    """
    Use GitHub API to get the README's download URL, then fetch raw content.
    Fallback to common README filenames if needed.
    """
    # Primary API approach
    resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=gh_headers(token), timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        download_url = data.get("download_url")
        if download_url:
            raw = requests.get(download_url, headers=gh_headers(token), timeout=30)
            if raw.status_code == 200:
                return raw.text

    # Fallback attempts on default branch
    branch = get_default_branch(owner, repo, token)
    candidates = ["README.md", "README.MD", "Readme.md", "readme.md", "README", "README.txt", "README.rst"]
    for name in candidates:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}"
        raw = requests.get(raw_url, headers=gh_headers(token), timeout=30)
        if raw.status_code == 200 and raw.text.strip():
            return raw.text

    return "README not found or empty."


def list_repo_files(owner: str, repo: str, branch: str, token: Optional[str]) -> List[Dict]:
    """
    Return the Git tree (recursive) as list of dicts with 'path' and 'type' from the GitHub API.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=gh_headers(token), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to list files: HTTP {resp.status_code} - {resp.text}")
    data = resp.json()
    tree = data.get("tree", [])
    # Each entry: {'path': '...', 'mode': '100644', 'type': 'blob' or 'tree', 'sha': '...'}
    return tree


def fetch_raw_file(owner: str, repo: str, branch: str, path: str, token: Optional[str]) -> Optional[str]:
    """
    Fetch raw file text content from raw.githubusercontent.com.
    Returns None for binary/too-large/unreadable content.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url, headers=gh_headers(token), timeout=60)
    if resp.status_code != 200:
        return None
    text = resp.text
    if not text or not text.strip():
        return None
    if len(text) > MAX_FILE_CHARS:
        return None
    # Heuristic to skip likely binary: if many NULLs or very high proportion of non-text
    if "\x00" in text:
        return None
    return text


def repo_slug(owner: str, repo: str) -> str:
    return f"{owner}_{repo}".lower()


def format_docs(docs: List[Document]) -> str:
    chunks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        chunks.append(f"Source: {src}\n{d.page_content}")
    return "\n\n-----\n\n".join(chunks) if chunks else "(no additional relevant code snippets found)"


def ensure_ollama_instructions():
    st.info(
        "This app uses Ollama locally for embeddings (nomic-embed-text) and a chat model.\n\n"
        "Before running queries, ensure you have Ollama installed and running, then pull the models:\n"
        "- ollama pull nomic-embed-text\n"
        "- ollama pull llama3\n\n"
        "You can change the chat model name in the sidebar."
    )


# ---------------------------
# Streamlit UI - Sidebar
# ---------------------------

with st.sidebar:
    st.header("Settings")
    chat_model_name = st.text_input(
        "Ollama Chat Model", value="llama3.2", help="Any local Ollama chat model, e.g., llama3, phi3, qwen2.5, etc."
    )
    k_retrieval = st.slider("Top-K retrieved chunks", 2, 10, value=4, step=1)
    chunk_size = st.slider("Chunk size (chars)", 500, 2000, value=1200, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, value=200, step=50)
    st.caption("Tip: Reduce chunk size for more granular retrieval; increase overlap for better context continuity.")

# ---------------------------
# Streamlit UI - Main
# ---------------------------

st.title("💬 Chat with a GitHub Repository (LangChain + Chroma + Ollama)")
st.write(
    "Ask questions about a repo’s README and code. README is always included in context; selected file types are embedded for retrieval."
)

ensure_ollama_instructions()

repo_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/owner/repo",
    help="Public repos work best. For higher rate limits, provide a GitHub token below.",
)
gh_token = st.text_input(
    "GitHub Token (optional)", type="password", help="Optional. Increases GitHub API rate limits during indexing."
)

exts = st.multiselect(
    "File extensions to index (README is always included regardless of selection):",
    DEFAULT_EXTENSIONS,
    default=[".py", ".md", ".txt"],
)

index_button = st.button("Load & Index Repository", type="primary")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # for UI display only
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []  # List[BaseMessage] for LangChain prompt
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "readme_text" not in st.session_state:
    st.session_state.readme_text = ""
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "repo_id" not in st.session_state:
    st.session_state.repo_id = None

# ---------------------------
# Indexing Flow
# ---------------------------


def build_embeddings():
    # Ollama embeddings with the required model
    return OllamaEmbeddings(model="nomic-embed-text")


def build_llm(model_name: str):
    # Local Ollama LLM for chat
    return ChatOllama(model=model_name, temperature=0.2)


def index_repository(url: str, token: Optional[str], selected_exts: List[str], chunk_size: int, chunk_overlap: int):
    owner, repo = parse_github_url(url)
    branch = get_default_branch(owner, repo, token)

    with st.status("Fetching README...", expanded=False) as status:
        readme = fetch_readme(owner, repo, token)
        readme_trunc = approximate_truncate(readme, MAX_README_TOKENS)
        st.session_state.readme_text = readme_trunc
        status.update(label="README fetched.", state="complete")

    with st.status("Listing repository files...", expanded=False) as status:
        tree = list_repo_files(owner, repo, branch, token)
        status.update(label=f"Found {len(tree)} items in repo tree.", state="complete")

    # Filter files by extension
    blob_paths = [t["path"] for t in tree if t.get("type") == "blob"]
    selected_paths = [p for p in blob_paths if any(p.lower().endswith(ext.lower()) for ext in selected_exts)]

    # Fetch and build documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []

    progress = st.progress(0)
    total = max(1, len(selected_paths))
    fetched = 0

    for i, path in enumerate(selected_paths, start=1):
        content = fetch_raw_file(owner, repo, branch, path, token)
        if content:
            splits = text_splitter.split_text(content)
            for j, s in enumerate(splits):
                docs.append(
                    Document(
                        page_content=s,
                        metadata={
                            "source": path,
                            "repo": f"{owner}/{repo}",
                            "branch": branch,
                            "chunk": j,
                            "url": f"https://github.com/{owner}/{repo}/blob/{branch}/{path}",
                        },
                    )
                )
        fetched += 1
        progress.progress(min(1.0, fetched / total))

    st.success(f"Prepared {len(docs)} text chunks from {len(selected_paths)} files matching {selected_exts}.")

    # Build Chroma vectorstore
    with st.status("Building Chroma vector database with Ollama embeddings...", expanded=False) as status:
        embeddings = build_embeddings()
        # Persist per-repo so re-runs don’t conflict across different repos
        repo_id = repo_slug(owner, repo)
        persist_dir = os.path.join(".chroma_db", repo_id)
        os.makedirs(persist_dir, exist_ok=True)
        # Recreate collection for fresh indexing
        vectorstore = Chroma(collection_name=repo_id, embedding_function=embeddings, persist_directory=persist_dir)
        # Remove existing docs (if any) then add
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = Chroma(collection_name=repo_id, embedding_function=embeddings, persist_directory=persist_dir)
        if docs:
            vectorstore.add_documents(docs)
            vectorstore.persist()
        status.update(label="Chroma vector DB ready.", state="complete")

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = vectorstore.as_retriever(
        search_kwargs={"k": int(st.session_state.get("k_retrieval", 4))}
    )
    st.session_state.repo_id = repo_id


# Handle Index Button
if index_button:
    if not repo_url.strip():
        st.error("Please enter a valid GitHub repository URL.")
    elif not exts:
        st.error("Please select at least one file extension.")
    else:
        # Store k to session so retrieval stays consistent
        st.session_state.k_retrieval = k_retrieval
        with st.spinner("Indexing repository..."):
            try:
                index_repository(
                    repo_url.strip(), gh_token.strip() if gh_token else None, exts, chunk_size, chunk_overlap
                )
            except Exception as e:
                st.exception(e)

# ---------------------------
# Chat Chain Setup
# ---------------------------


def build_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful code assistant. Use the repository README (below) and retrieved code/document snippets "
            "to answer questions. If you do not know, say so. Prefer citing file paths and lines when relevant.",
        ),
        ("system", "Repository README (truncated to ~2000 tokens):\n\n{readme}\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\n\nRelevant code/document snippets:\n{context}"),
    ])


def answer_question(question: str):
    if not st.session_state.retriever:
        st.warning("Please load and index a repository first.")
        return "Please load and index a repository first.", []

    # Retrieve relevant chunks
    docs = st.session_state.retriever.get_relevant_documents(question)
    st.session_state.last_docs = docs

    context = format_docs(docs)
    readme = st.session_state.readme_text or "(no README found)"

    prompt = build_prompt()
    messages = prompt.format_messages(
        readme=readme, chat_history=st.session_state.lc_history, question=question, context=context
    )

    llm = build_llm(chat_model_name)
    response = llm.invoke(messages)

    # Update LC history with the formatted last turn? We append below in the UI flow.
    return response.content, docs


# ---------------------------
# Chat UI
# ---------------------------

st.markdown("### Chat")
chat_disabled = st.session_state.retriever is None or not st.session_state.readme_text

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about the repository's README or code...", disabled=chat_disabled)

if user_input:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, used_docs = answer_question(user_input)
            except Exception as e:
                answer = f"Error generating answer: {e}"
                used_docs = []

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.lc_history.append(AIMessage(content=answer))

        # Show sources
        if used_docs:
            with st.expander("Sources (retrieved snippets)"):
                unique_sources = []
                for d in used_docs:
                    src = d.metadata.get("source")
                    url = d.metadata.get("url")
                    if src and src not in unique_sources:
                        unique_sources.append(src)
                        st.write(f"- {src}")
                        if url:
                            st.write(f"  {url}")

# ---------------------------
# Minimal Tips
# ---------------------------

st.markdown("#### Quick Tips")
st.markdown(
    "- Always includes README in context (truncated to ~2000 tokens).\n"
    "- Uses Ollama for embeddings (nomic-embed-text) and an Ollama chat model.\n"
    "- You can adjust chunk size/overlap and top-K retrieval from the sidebar.\n"
    "- Optional GitHub token helps avoid API rate limits when indexing."
)
