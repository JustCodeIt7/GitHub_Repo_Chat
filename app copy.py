import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st

# LangChain / Chroma / Ollama imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# -------------------------------
# Minimal helper utilities
# -------------------------------

def parse_github_url(repo_url: str) -> Tuple[str, str]:
	"""Return (normalized_clone_url, repo_slug). Supports https and ssh URLs.

	repo_slug example: owner/repo
	"""
	url = repo_url.strip()
	if url.endswith(".git"):
		url = url[:-4]
	# Normalize SSH to HTTPS when possible for shallow clone without auth
	m = re.match(r"git@github.com:(?P<owner>[^/]+)/(?P<repo>[^/]+)$", url)
	if m:
		clone_url = f"https://github.com/{m.group('owner')}/{m.group('repo')}.git"
		slug = f"{m.group('owner')}/{m.group('repo')}"
		return clone_url, slug

	# HTTPS
	m = re.match(r"https?://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)", url)
	if m:
		slug = f"{m.group('owner')}/{m.group('repo')}"
		clone_url = f"https://github.com/{slug}.git"
		return clone_url, slug

	# Fallback - return as-is
	return repo_url, Path(url).name


def shallow_clone(repo_url: str, dest_dir: Path) -> Path:
	"""Shallow clone the repository to dest_dir. Returns the path to the repo root.
	Raises RuntimeError on failure.
	"""
	if dest_dir.exists():
		shutil.rmtree(dest_dir)
	dest_dir.mkdir(parents=True, exist_ok=True)

	try:
		subprocess.run(
			["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
			check=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Git clone failed: {e.stderr.decode(errors='ignore')}")
	return dest_dir


def read_text_file(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
	"""Read a file as UTF-8 text if it's not too large. Returns None if unreadable."""
	try:
		if path.stat().st_size > max_bytes:
			return None
		with path.open("r", encoding="utf-8", errors="ignore") as f:
			return f.read()
	except Exception:
		return None


def find_readme(repo_root: Path) -> Optional[Path]:
	candidates = [
		repo_root / "README.md",
		repo_root / "README.MD",
		repo_root / "README",
		repo_root / "readme.md",
		repo_root / "Readme.md",
	]
	for p in candidates:
		if p.exists() and p.is_file():
			return p
	# Fallback: search top-level for any readme-like file
	for p in repo_root.glob("*readme*" ):
		if p.is_file():
			return p
	return None


def approx_trim_tokens(text: str, max_tokens: int = 2000) -> str:
	"""Approximate token trimming assuming ~4 chars per token."""
	if not text:
		return ""
	max_chars = max_tokens * 4
	return text[:max_chars]


def collect_repo_documents(repo_root: Path, exts: List[str]) -> List[Document]:
	"""Walk repo and collect documents for given extensions, excluding README."""
	exts = [e.lower().strip() for e in exts]
	docs: List[Document] = []
	for path in repo_root.rglob("*"):
		if not path.is_file():
			continue
		if path.name.lower().startswith("readme"):
			# README is handled separately
			continue
		if exts and path.suffix.lower() not in exts:
			continue
		content = read_text_file(path)
		if not content:
			continue
		docs.append(Document(
			page_content=content,
			metadata={"source": str(path.relative_to(repo_root)), "repo_root": str(repo_root)}
		))
	return docs


def build_vectorstore(docs: List[Document], persist_dir: Path, embed_model: str = "nomic-embed-text") -> Chroma:
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
	splits = text_splitter.split_documents(docs)
	embeddings = OllamaEmbeddings(model=embed_model, base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
	vs = Chroma.from_documents(
		documents=splits,
		embedding=embeddings,
		persist_directory=str(persist_dir),
	)
	# Persist to disk so re-runs are faster within the same session
	vs.persist()
	return vs


@st.cache_resource(show_spinner=False)
def index_repo(repo_url: str, selected_exts: Tuple[str, ...], embed_model: str) -> Tuple[Chroma, str, str, Path]:
	"""Clone and index the repo. Returns (vectorstore, readme_text, repo_slug, repo_root).

	Cached by Streamlit per unique (url, exts, embed_model) across app session.
	"""
	clone_url, slug = parse_github_url(repo_url)
	base_tmp = Path(tempfile.gettempdir()) / "repo_chat_cache"
	base_tmp.mkdir(parents=True, exist_ok=True)
	# Unique folder per url hash+exts
	safe_slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
	key = f"{safe_slug}-{'-'.join(sorted(selected_exts))}-{embed_model}"
	repo_root = base_tmp / key

	# Clone fresh each time for simplicity/reliability
	shallow_clone(clone_url, repo_root)

	# README
	readme_path = find_readme(repo_root)
	readme_text = approx_trim_tokens(read_text_file(readme_path) or "") if readme_path else ""

	# Collect and index selected files
	persist_dir = repo_root / ".chroma"
	docs = collect_repo_documents(repo_root, list(selected_exts))

	if docs:
		vs = build_vectorstore(docs, persist_dir=persist_dir, embed_model=embed_model)
	else:
		# Create an empty store (Chroma requires embeddings; add tiny dummy document)
		embeddings = OllamaEmbeddings(model=embed_model, base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
		vs = Chroma.from_texts([""], embedding=embeddings, persist_directory=str(persist_dir))
		vs.persist()

	return vs, readme_text, slug, repo_root


def format_context_block(readme_text: str, retrieved_docs: List[Document], max_chars: int = 12000) -> str:
	parts = []
	if readme_text:
		parts.append("README (truncated):\n" + readme_text.strip())
	if retrieved_docs:
		parts.append("Relevant files:")
		for i, d in enumerate(retrieved_docs, 1):
			src = d.metadata.get("source", "")
			text = d.page_content[:2000]  # cap each to keep prompt small
			parts.append(f"[{i}] {src}\n{text}")
	ctx = "\n\n".join(parts)
	return ctx[:max_chars]


def answer_question(
	llm: ChatOllama,
	retriever,
	question: str,
	readme_text: str,
	chat_history: List[Tuple[str, str]],
	k: int = 5,
) -> str:
	retrieved = retriever.get_relevant_documents(question)[:k] if retriever else []
	context = format_context_block(readme_text, retrieved)

	# Build a simple prompt that always includes README content + retrieved docs
	system_instructions = (
		"You are a coding assistant answering questions about a GitHub repository. "
		"Use the README as the primary source of truth for high-level behavior. "
		"Augment with retrieved code snippets when helpful. Quote file paths when citing. "
		"If something is unclear or missing, say so and suggest where to look."
	)

	history_text = "\n\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history[-6:]])  # last 6 exchanges

	prompt = (
		f"<SYSTEM>\n{system_instructions}\n</SYSTEM>\n\n"
		f"<CONTEXT>\n{context}\n</CONTEXT>\n\n"
		f"<HISTORY>\n{history_text}\n</HISTORY>\n\n"
		f"<QUESTION>\n{question}\n</QUESTION>\n\n"
		"Provide a concise, helpful answer. If you cite details, include the file path in parentheses."
	)

	resp = llm.invoke(prompt)
	try:
		return resp.content if hasattr(resp, "content") else str(resp)
	except Exception:
		return str(resp)


# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="GitHub Repo Chat", page_icon="📚", layout="wide")

st.title("GitHub Repo Chat — Streamlit + LangChain + Chroma (Ollama embeddings)")

with st.sidebar:
	st.markdown("""
	Quick setup:
	1) Install and run Ollama (https://ollama.com/) and pull models:
	   - ollama pull nomic-embed-text
	   - ollama pull llama3.1
	2) pip install: streamlit langchain langchain-community chromadb
	""")
	gen_model = st.text_input("Generation model (Ollama)", value="llama3.1")
	top_k = st.slider("Top K documents", 2, 10, 5)
	temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")

ext_options = [".py", ".js", ".ts", ".tsx", ".md", ".txt", ".json", ".yaml", ".yml"]
selected_exts = st.multiselect(
	"File extensions to include (README is always included)",
	options=ext_options,
	default=[".py", ".md", ".txt"],
)

col1, col2 = st.columns([1, 3])
with col1:
	load_clicked = st.button("Load Repository", type="primary", use_container_width=True)

status_placeholder = st.empty()

if "chat_history" not in st.session_state:
	st.session_state.chat_history = []  # List[Tuple[user, assistant]]

if "retriever" not in st.session_state:
	st.session_state.retriever = None

if load_clicked:
	if not repo_url.strip():
		st.warning("Please enter a GitHub repository URL.")
	else:
		status_placeholder.info("Cloning and indexing repository… This may take a minute on first run.")
		try:
			vectorstore, readme_text, slug, repo_root = index_repo(repo_url.strip(), tuple(selected_exts), "nomic-embed-text")
			st.session_state.readme_text = readme_text
			st.session_state.repo_slug = slug
			st.session_state.repo_root = str(repo_root)
			st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
			status_placeholder.success(f"Loaded {slug}. You can start chatting below.")
		except Exception as e:
			status_placeholder.error(f"Failed to load repo: {e}")


if st.session_state.get("retriever") is not None:
	st.subheader(f"Chatting about: {st.session_state.get('repo_slug', '')}")

	# Show README preview
	with st.expander("README (always included in context) — preview"):
		st.write(st.session_state.get("readme_text", "(No README found)"))

	# Render chat history
	for user_msg, assistant_msg in st.session_state.chat_history:
		with st.chat_message("user"):
			st.markdown(user_msg)
		with st.chat_message("assistant"):
			st.markdown(assistant_msg)

	user_input = st.chat_input("Ask a question about this repository…")

	if user_input:
		with st.chat_message("user"):
			st.markdown(user_input)

		# Create LLM per request (fast) so sliders take effect
		llm = ChatOllama(model=gen_model, temperature=temperature, base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

		with st.chat_message("assistant"):
			with st.spinner("Thinking…"):
				try:
					answer = answer_question(
						llm=llm,
						retriever=st.session_state.retriever,
						question=user_input,
						readme_text=st.session_state.get("readme_text", ""),
						chat_history=st.session_state.chat_history,
						k=top_k,
					)
				except Exception as e:
					answer = f"Sorry, there was an error generating the answer: {e}"
			st.markdown(answer)

		# Save to history
		st.session_state.chat_history.append((user_input, answer))


else:
	st.info("Enter a GitHub repo URL, choose file types, and click Load Repository to begin.")

