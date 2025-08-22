import re, requests, streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama

################################ Setup & Configuration ################################
st.set_page_config(page_title="GitHub Repo Chat (RAG)", page_icon="💬", layout="wide")

################################ GitHub API Utilities ################################
# Set common headers for GitHub API requests to identify the client.
H = {"Accept": "application/vnd.github+json", "User-Agent": "streamlit-rag-app"}


def approx(text, max_tokens=2000, ratio=4):
    """Truncate text to an approximate number of characters based on token count."""
    # Estimate character count from token count for truncation.
    mx = max_tokens * ratio
    return text if len(text) <= mx else text[:mx] + "\n\n[...truncated...]"


def parse(url):
    """Extract owner and repo name from a GitHub URL."""
    # Use regex to find owner and repo in a standard GitHub URL.
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url or "")
    # Raise an error if the URL format is invalid.
    if not m:
        raise ValueError("Enter a valid GitHub URL like https://github.com/owner/repo")
    owner, repo = m.group(1), m.group(2)
    return owner, repo[:-4] if repo.endswith(".git") else repo  # Clean .git suffix if present


def gh_json(url, timeout=30):
    """Fetch JSON data from a GitHub API endpoint with error handling."""
    r = requests.get(url, headers=H, timeout=timeout)
    # Check for non-successful status codes and raise an error.
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:200]}")
    return r.json()


def default_branch(owner, repo):
    """Get the default branch name for a repository."""
    # Query the repository details endpoint to find the default branch.
    return gh_json(f"https://api.github.com/repos/{owner}/{repo}").get("default_branch", "main")


def fetch_readme(owner, repo):
    """Fetch the repository's README, trying the API first, then common filenames."""
    # Attempt to fetch README metadata from the dedicated API endpoint.
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=H, timeout=30)
    if r.status_code == 200 and (u := r.json().get("download_url")):
        raw = requests.get(u, headers=H, timeout=30)
        # If the download is successful, return the text content.
        if raw.status_code == 200:
            return raw.text
    # As a fallback, try fetching common README filenames directly.
    branch = default_branch(owner, repo)
    for name in ["README.md", "README.MD", "Readme.md", "readme.md", "README", "README.txt", "README.rst"]:
        raw = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}", headers=H, timeout=30)
        # Return the content of the first found and non-empty README.
        if raw.status_code == 200 and raw.text.strip():
            return raw.text
    return "README not found or empty."


def list_paths(owner, repo, branch):
    """Get a flat list of all file paths in the repository."""
    # Recursively fetch the entire git tree for the specified branch.
    tree = gh_json(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", timeout=60).get(
        "tree", []
    )
    # Filter the tree to include only files (blobs) and return their paths.
    return [t["path"] for t in tree if t.get("type") == "blob"]


def fetch_raw(owner, repo, branch, path, max_chars=300_000):
    """Fetch the raw text content of a file, with validation."""
    r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}", headers=H, timeout=60)
    # Return None if the file fetch fails.
    if r.status_code != 200:
        return None
    t = r.text
    # Skip empty, oversized, or non-text (binary) files.
    if not t.strip() or len(t) > max_chars or "\x00" in t:
        return None
    return t


def format_docs(docs):
    """Prepare retrieved documents for inclusion in the LLM prompt."""
    # Combine documents into a single string, separated by dividers.
    return (
        "\n\n-----\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs)
        if docs
        else "(no relevant snippets)"
    )


################################ Indexing & RAG Logic ################################
def index_repo(url, exts, chunk_size, overlap, k):
    """Fetch, parse, chunk, and embed repository files into a vector store."""
    owner, repo = parse(url)
    branch = default_branch(owner, repo)
    # Fetch and store the README content in the session state.
    st.session_state.readme = approx(fetch_readme(owner, repo))
    # Filter all file paths by the user-selected extensions.
    paths = [p for p in list_paths(owner, repo, branch) if any(p.lower().endswith(e.lower()) for e in exts)]
    # Configure the text splitter with the specified chunk size and overlap.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    # Fetch content for each path, split it, and create Document objects.
    for p in paths:
        # Fetch raw file content if available.
        if txt := fetch_raw(owner, repo, branch, p):
            # Split the text and create documents with rich metadata.
            docs.extend(
                Document(
                    page_content=s,
                    metadata={
                        "source": p,
                        "repo": f"{owner}/{repo}",
                        "branch": branch,
                        "chunk": j,
                        "url": f"https://github.com/{owner}/{repo}/blob/{branch}/{p}",
                    },
                )
                for j, s in enumerate(splitter.split_text(txt))
            )
    # Initialize the embedding model and the vector store.
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(embedding_function=embeddings)
    # Add the processed documents to the vector store if any exist.
    if docs:
        vs.add_documents(docs)
    # Create the retriever and store it in the session state for later use.
    st.session_state.retriever = vs.as_retriever(search_kwargs={"k": int(k)})
    st.success(f"Indexed {len(docs)} chunks from {len(paths)} files.")


def answer(question, model):
    """Retrieve documents, build a prompt, and query the LLM to get an answer."""
    # Ensure a retriever is available before proceeding.
    if not st.session_state.get("retriever"):
        return "Please index a repository first.", []
    # Find documents relevant to the user's question.
    docs = st.session_state.retriever.get_relevant_documents(question)
    # Format the retrieved documents and README for the prompt.
    ctx = format_docs(docs)
    readme = st.session_state.get("readme") or "(no README found)"
    # Construct the final prompt with context, README, and the user's question.
    prompt = (
        "You are a helpful code assistant. Use the repository README (below) and retrieved code/document snippets "
        "to answer questions. If you do not know, say so. Prefer citing file paths and lines when relevant.\n\n"
        f"Repository README (truncated):\n{readme}\n\n"
        f"Question: {question}\n\nRelevant snippets:\n{ctx}\n\n"
        "Answer:"
    )
    # Initialize the LLM with the specified model and temperature.
    llm = ChatOllama(model=model, temperature=0.2)
    # Send the prompt to the LLM and get the response.
    resp = llm.invoke(prompt)
    return resp.content, docs


################################ Streamlit UI ################################
def main():
    """Main function to run the Streamlit app."""

    st.title("💬 Chat with a GitHub Repository (LangChain + Chroma + Ollama)")
    st.write(
        "Ask questions about a repo’s README and code. README is always included; selected file types are embedded for retrieval."
    )

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Ollama Chat Model", value="llama3.2")
        k = st.slider("Top-K retrieved chunks", 2, 10, value=4)  # Number of chunks to retrieve for context
        chunk = st.slider("Chunk size (chars)", 500, 2000, value=1200, step=100)  # Max characters per text chunk
        overlap = st.slider("Chunk overlap (chars)", 0, 400, value=200, step=50)  # Character overlap between chunks

    # Main panel for repository input and indexing controls.
    default_exts = [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml", ".java", ".go", ".rs"]
    url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        value="https://github.com/JustCodeIt7/GitHub_Repo_Chat",
    )

    exts = st.multiselect(
        "File extensions to index (README always included):", default_exts, default=[".py", ".md", ".txt"]
    )
    # Handle the "Load & Index" button click event.
    if st.button("Load & Index Repository", type="primary"):
        # Run the indexing process within a try-except block to catch errors.
        try:
            # Validate that the URL and file extensions are provided.
            if not url.strip():
                st.error("Please enter a valid GitHub repository URL.")
            elif not exts:
                st.error("Please select at least one file extension.")
            else:
                # Trigger the repository indexing function.
                index_repo(url.strip(), exts, chunk, overlap, k)
        except Exception as e:
            st.exception(e)

    # Initialize session state variables if they don't already exist.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("readme", "")

    st.markdown("### Chat")
    # Display previous chat messages from the session state.
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Disable the chat input until a repository is successfully indexed.
    disabled = st.session_state.retriever is None or not st.session_state.readme
    user = st.chat_input("Ask about the repository's README or code...", disabled=disabled)
    st.sidebar.code("What is this repository about?")

    # Process user input if a message is entered.
    if user:
        # Add the user's message to the chat history.
        st.session_state.messages.append({"role": "user", "content": user})
        with st.chat_message("user"):
            st.markdown(user)
        # Generate and display the assistant's response.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get the answer from the RAG pipeline, handling potential errors.
                try:
                    ans, used = answer(user, model)
                except Exception as e:
                    ans, used = f"Error: {e}", []
            st.markdown(ans)
            # Add the assistant's response to the chat history.
            st.session_state.messages.append({"role": "assistant", "content": ans})
            # Display the source documents used to generate the answer.
            if used:
                with st.expander("Sources (retrieved snippets)"):
                    seen = set()
                    # Iterate through retrieved documents to show unique sources.
                    for d in used:
                        src, url = d.metadata.get("source"), d.metadata.get("url")
                        # Display each source file path only once.
                        if src and src not in seen:
                            seen.add(src)
                            st.write(f"- {src}")
                            # Display the URL to the source file on GitHub.
                            if url:
                                st.write(f"  {url}")


if __name__ == "__main__":
    main()
