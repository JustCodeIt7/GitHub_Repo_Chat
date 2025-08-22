import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from git import Repo

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama

################################ Setup & Configuration ################################
st.set_page_config(page_title="GitHub Repo Chat (RAG)", page_icon="💬", layout="wide")

################################ Git Utilities ################################

def approx(text, max_tokens=2000, ratio=4):
    """Truncate text to an approximate number of characters based on token count."""
    mx = max_tokens * ratio
    return text if len(text) <= mx else text[:mx] + "\n\n[...truncated...]"

def parse(url):
    """Extract the owner and repo name from a GitHub URL using regex."""
    match = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url or "")
    if not match:
        raise ValueError("Enter a valid GitHub URL like https://github.com/owner/repo")
    owner, repo = match.group(1), match.group(2)
    # Remove .git suffix if present
    return owner, repo[:-4] if repo.endswith(".git") else repo


@st.cache_resource(show_spinner=False)
def clone_repo(url: str):
    """Clone the repository to a temporary directory and cache the result by URL."""
    temp_dir = tempfile.mkdtemp(prefix="git_rag_")
    # Clone the repository using the default branch
    repo = Repo.clone_from(url, temp_dir)
    return repo, temp_dir


def get_default_branch(repo: Repo) -> str:
    """Detect the default branch without using the GitHub API."""
    # Attempt to get the currently active branch
    try:
        return repo.active_branch.name
    except Exception:
        pass

    # If all else fails, assume 'main'
    return "main"

def find_readme_blob(repo: Repo):
    """Find a README file in the repository, checking the root first."""
    common_names = [
        "README.md", "README.MD", "Readme.md", "readme.md",
        "README", "README.txt", "README.rst"
    ]

    head_tree = repo.head.commit.tree

    # Check the repository root for common README file names
    names_in_root = {e.name: e for e in head_tree}
    for name in common_names:
        entry = names_in_root.get(name)
        if entry and entry.type == "blob":
            return entry

    # Search the entire tree if no README is found in the root
    for entry in head_tree.traverse():
        if entry.type == "blob":
            lower_name = entry.name.lower()
            if lower_name == "readme" or lower_name.startswith("readme."):
                return entry

    return None

def read_blob_text(blob, max_bytes=300_000):
    """Read blob content as text, skipping empty, oversized, or binary files."""
    # Read slightly more than the max to detect oversized files
    data = blob.data_stream.read(max_bytes + 1)
    # Decode to text, replacing any errors
    text = data.decode("utf-8", errors="replace")

    return text

def fetch_readme_git(repo: Repo) -> str:
    """Fetch the README content from the cloned repository."""
    blob = find_readme_blob(repo)
    if not blob:
        return "README not found or empty."
    text = read_blob_text(blob)
    return text if text else "README not found or empty."


def iter_matching_blobs(repo: Repo, exts):
    """Yield blobs that match the specified file extensions."""
    ext_set = {e.lower() for e in exts}
    # Traverse all files in the repository's HEAD commit
    for entry in repo.head.commit.tree.traverse():
        if entry.type == "blob":
            path_str = entry.path
            # Yield the path and blob if the file extension matches
            if any(path_str.lower().endswith(e) for e in ext_set):
                yield path_str, entry


def format_docs(docs):
    """Prepare retrieved documents for inclusion in the LLM prompt."""
    if not docs:
        return "(no relevant snippets)"
    # Format each document with its source metadata
    return "\n\n-----\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs
    )

################################ RAG & Indexing ################################
def index_repo(url, exts, chunk_size, overlap, k):
    """Clone a repo, split files into chunks, and embed them into a vector store."""
    owner, repo_name = parse(url)

    # Clone the repository (uses cached result if available)
    repo, local_dir = clone_repo(url)

    # Determine the default branch to create correct source URLs
    branch = get_default_branch(repo)

    # Fetch and store the repository's README content
    st.session_state.readme = approx(fetch_readme_git(repo))

    # Configure the text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    docs = []
    file_count = 0
    # Process each file that matches the selected extensions
    for rel_path, blob in iter_matching_blobs(repo, exts):
        text = read_blob_text(blob, max_bytes=300_000)
        # Skip empty or invalid files
        if not text:
            continue
        file_count += 1
        # Split the file's text into chunks and create Document objects
        for j, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": rel_path,
                        "repo": f"{owner}/{repo_name}",
                        "branch": branch,
                        "chunk": j,
                        "url": f"https://github.com/{owner}/{repo_name}/blob/{branch}/{rel_path}",
                    },
                )
            )

    # Initialize the embedding model and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(embedding_function=embeddings)
    # Add the document chunks to the vector store if any were created
    if docs:
        vector_store.add_documents(docs)

    # Create and store the retriever in the session state
    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": int(k)})
    st.success(f"Indexed {len(docs)} chunks from {file_count} files.")


def answer(question, model):
    """Retrieve relevant documents, build a prompt, and query the LLM."""
    if not st.session_state.get("retriever"):
        return "Please index a repository first.", []

    # Retrieve documents relevant to the user's question
    docs = st.session_state.retriever.invoke(question)
    context = format_docs(docs)
    readme = st.session_state.get("readme") or "(no README found)"

    # Construct the prompt with context from the README and retrieved snippets
    prompt = (
        "You are a helpful code assistant. Use the repository README (below) and retrieved code/document snippets "
        "to answer questions. If you do not know, say so. Prefer citing file paths and lines when relevant.\n\n"
        f"Repository README (truncated):\n{readme}\n\n"
        f"Question: {question}\n\nRelevant snippets:\n{context}\n\n"
        "Answer:"
    )

    # Query the language model with the constructed prompt
    llm = ChatOllama(model=model, temperature=0.2)
    response = llm.invoke(prompt)
    return response.content, docs

################################ Streamlit UI ################################
def main():
    """Define the main Streamlit user interface."""

    st.title("💬 Chat with a GitHub Repository (LangChain + Chroma + Ollama)")
    st.write(
        "Ask questions about a repo’s README and code. README is always included; selected file types are embedded for retrieval."
    )

    # Configure the settings panel in the sidebar
    with st.sidebar:
        st.header("Settings")
        model_name = "phi4-mini" # Default model
        model = st.text_input("Ollama Chat Model", value=model_name)
        k = st.slider("Top-K retrieved chunks", 2, 10, value=4)
        chunk = st.slider("Chunk size (chars)", 500, 2000, value=1200, step=100)
        overlap = st.slider("Chunk overlap (chars)", 0, 400, value=200, step=50)

    # Define default file extensions to index
    default_exts = [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml", ".java", ".go", ".rs"]

    url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        value="https://github.com/JustCodeIt7/GitHub_Repo_Chat",
    )

    exts = st.multiselect(
        "File extensions to index (README always included):", default_exts, default=[".py", ".md", ".txt"]
    )

    # Handle the repository indexing process on button click
    if st.button("Load & Index Repository", type="primary"):
        try:
            # Validate user inputs before proceeding
            if not url.strip():
                st.error("Please enter a valid GitHub repository URL.")
            elif not exts:
                st.error("Please select at least one file extension.")
            else:
                index_repo(url.strip(), exts, chunk, overlap, k)
        except Exception as e:
            st.exception(e)

    # Initialize session state variables if they don't exist
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("readme", "")

    st.markdown("### Chat")
    # Display the chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Disable the chat input until the repository is indexed
    is_disabled = st.session_state.retriever is None or not st.session_state.readme
    user_input = st.chat_input("Ask about the repository's README or code...", disabled=is_disabled)
    st.sidebar.code("What is this repository about?") # Display example question

    # Process user input and generate a response
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ans, used_docs = answer(user_input, model)
                except Exception as e:
                    ans, used_docs = f"Error: {e}", []
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

            # Display the sources used for the answer
            if used_docs:
                with st.expander("Sources (retrieved snippets)"):
                    seen_sources = set()
                    for doc in used_docs:
                        src, url_ = doc.metadata.get("source"), doc.metadata.get("url")
                        if src and src not in seen_sources:
                            seen_sources.add(src)
                            st.write(f"- {src}")
                            if url_:
                                st.write(f"  {url_}")

################################ Application Entry Point ################################
if __name__ == "__main__":
    main()