import re, requests, streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="GitHub Repo Chat (RAG)", page_icon="💬", layout="wide")

# ---------------------------
# GitHub + utilities
# ---------------------------
H = {"Accept": "application/vnd.github+json", "User-Agent": "streamlit-rag-app"}


def approx(text, max_tokens=2000, ratio=4):
    mx = max_tokens * ratio
    return text if len(text) <= mx else text[:mx] + "\n\n[...truncated...]"


def parse(url):
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url or "")
    if not m:
        raise ValueError("Enter a valid GitHub URL like https://github.com/owner/repo")
    owner, repo = m.group(1), m.group(2)
    return owner, repo[:-4] if repo.endswith(".git") else repo


def gh_json(url, timeout=30):
    r = requests.get(url, headers=H, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:200]}")
    return r.json()


def default_branch(owner, repo):
    return gh_json(f"https://api.github.com/repos/{owner}/{repo}").get("default_branch", "main")


def fetch_readme(owner, repo):
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=H, timeout=30)
    if r.status_code == 200 and (u := r.json().get("download_url")):
        raw = requests.get(u, headers=H, timeout=30)
        if raw.status_code == 200:
            return raw.text
    branch = default_branch(owner, repo)
    for name in ["README.md", "README.MD", "Readme.md", "readme.md", "README", "README.txt", "README.rst"]:
        raw = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}", headers=H, timeout=30)
        if raw.status_code == 200 and raw.text.strip():
            return raw.text
    return "README not found or empty."


def list_paths(owner, repo, branch):
    tree = gh_json(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", timeout=60).get(
        "tree", []
    )
    return [t["path"] for t in tree if t.get("type") == "blob"]


def fetch_raw(owner, repo, branch, path, max_chars=300_000):
    r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}", headers=H, timeout=60)
    if r.status_code != 200:
        return None
    t = r.text
    if not t.strip() or len(t) > max_chars or "\x00" in t:
        return None
    return t


def format_docs(docs):
    return (
        "\n\n-----\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs)
        if docs
        else "(no relevant snippets)"
    )


# ---------------------------
# Indexing + RAG
# ---------------------------
def index_repo(url, exts, chunk_size, overlap, k):
    owner, repo = parse(url)
    branch = default_branch(owner, repo)
    st.session_state.readme = approx(fetch_readme(owner, repo))
    paths = [p for p in list_paths(owner, repo, branch) if any(p.lower().endswith(e.lower()) for e in exts)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    for p in paths:
        if txt := fetch_raw(owner, repo, branch, p):
            for j, s in enumerate(splitter.split_text(txt)):
                docs.append(
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
                )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(embedding_function=embeddings)
    if docs:
        vs.add_documents(docs)
    st.session_state.retriever = vs.as_retriever(search_kwargs={"k": int(k)})
    st.success(f"Indexed {len(docs)} chunks from {len(paths)} files.")


def answer(question, model):
    if not st.session_state.get("retriever"):
        return "Please index a repository first.", []
    docs = st.session_state.retriever.get_relevant_documents(question)
    ctx = format_docs(docs)
    readme = st.session_state.get("readme") or "(no README found)"
    prompt = (
        "You are a helpful code assistant. Use the repository README (below) and retrieved code/document snippets "
        "to answer questions. If you do not know, say so. Prefer citing file paths and lines when relevant.\n\n"
        f"Repository README (truncated):\n{readme}\n\n"
        f"Question: {question}\n\nRelevant snippets:\n{ctx}\n\n"
        "Answer:"
    )
    llm = ChatOllama(model=model, temperature=0.2)
    resp = llm.invoke(prompt)
    return resp.content, docs


# ---------------------------
# UI
# ---------------------------
st.title("💬 Chat with a GitHub Repository (LangChain + Chroma + Ollama)")
st.write(
    "Ask questions about a repo’s README and code. README is always included; selected file types are embedded for retrieval."
)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama Chat Model", value="llama3.2")
    k = st.slider("Top-K retrieved chunks", 2, 10, value=4)
    chunk = st.slider("Chunk size (chars)", 500, 2000, value=1200, step=100)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, value=200, step=50)

default_exts = [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml", ".java", ".go", ".rs"]
url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")
exts = st.multiselect(
    "File extensions to index (README always included):", default_exts, default=[".py", ".md", ".txt"]
)
if st.button("Load & Index Repository", type="primary"):
    try:
        if not url.strip():
            st.error("Please enter a valid GitHub repository URL.")
        elif not exts:
            st.error("Please select at least one file extension.")
        else:
            index_repo(url.strip(), exts, chunk, overlap, k)
    except Exception as e:
        st.exception(e)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("readme", "")

st.markdown("### Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

disabled = st.session_state.retriever is None or not st.session_state.readme
user = st.chat_input("Ask about the repository's README or code...", disabled=disabled)
if user:
    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ans, used = answer(user, model)
            except Exception as e:
                ans, used = f"Error: {e}", []
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        if used:
            with st.expander("Sources (retrieved snippets)"):
                seen = set()
                for d in used:
                    src, url = d.metadata.get("source"), d.metadata.get("url")
                    if src and src not in seen:
                        seen.add(src)
                        st.write(f"- {src}")
                        if url:
                            st.write(f"  {url}")
