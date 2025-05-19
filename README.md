# GitHub Repo Chatbot with PydanticAI & Ollama

A Streamlit application that lets you chat with the contents of any GitHub repository. It uses FAISS for vector storage, Ollama for embeddings and language model, and PydanticAI as the agent framework to handle retrieval and question-answering.

---

## Table of Contents

* [Features](#features)
* [Demo](#demo)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [How It Works](#how-it-works)
* [Customization](#customization)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **GitHub Repo Cloning**: Clone any public GitHub repository and extract text from code and documentation files.
* **Text Chunking**: Split large text into manageable chunks for efficient vector storage.
* **Vector Search**: Build a FAISS index of embeddings to enable semantic similarity search.
* **PydanticAI Agent**: Leverage PydanticAI’s agent-and-tool architecture to handle retrieval and LLM calls.
* **Ollama Integration**: Use Ollama’s local LLM and embedding models for both embeddings and chat completions.
* **Streamlit UI**: Interactive web interface for loading repos and chatting with contents.

---

## Demo

![Chat Demo](docs/chat-demo.gif)

---

## Prerequisites

* **Python 3.10+**
* **Ollama** installed and running locally (default HTTP endpoint `http://localhost:11434/v1`)
* Git installed on your system

---

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/YourUsername/github-repo-chatbot-pydanticai.git
   cd github-repo-chatbot-pydanticai
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *`requirements.txt` should include:*

   ```text
   streamlit
   pydantic-ai
   langchain-community
   langchain-ollama
   faiss-cpu            # or faiss-gpu if you have GPU support
   ```

---

## Configuration

1. **Run Ollama daemon**

   Ensure Ollama is up and running. By default it listens on `http://localhost:11434/v1`.

2. **Model names**

   * Embedding model: `all-minilm:33m`
   * LLM model: `llama3.2`

   You can change these in `app.py` when initializing `OllamaEmbeddings` and `OpenAIModel`.

---

## Usage

Launch the Streamlit app:

```bash
streamlit run app.py
```

1. Open the URL shown in the console (usually `http://localhost:8501`).
2. Enter a **GitHub repository URL** in the sidebar (e.g., `https://github.com/JustCodeIt7/GitHub_Repo_Chat`).
3. Select file extensions to include (default: `.py, .md, .txt, .js, .html, .css, .json`).
4. Click **Load Repository** to clone, process, and index the repo.
5. Once loaded, ask questions in the chat interface about the repository’s contents.

---

## Project Structure

```
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── README.md         # This documentation
├── docs/
│   └── chat-demo.gif # Demo GIF (optional)
└── .gitignore        # Excludes venv, __pycache__, etc.
```

---

## How It Works

1. **Clone & Extract**

   * `get_repo_text` clones the repository into a temporary directory.
   * Walks through files with allowed extensions and concatenates their contents.

2. **Chunking**

   * `split_text` splits the combined text into overlapping chunks of \~1000 characters.

3. **Vector Store**

   * `create_vectorstore` builds a FAISS index using OllamaEmbeddings on each chunk.

4. **Agent & Retrieval**

   * `initialize_agent` creates a PydanticAI `Agent` with an `retrieve` tool that returns the top-4 similar chunks.
   * When a user query comes in, PydanticAI handles tool invocation (retrieval) and crafts the prompt for Ollama to answer.

5. **Chat UI**

   * Streamlit displays previous messages and handles user input via `st.chat_input`.
   * Responses from the agent are streamed back into the chat interface.

---

## Customization

* **Chunk Size & Overlap**: Adjust `chunk_size` and `overlap` in `split_text`.
* **Number of Docs**: Change the `k` parameter in `vectorstore.similarity_search` within the `retrieve` tool.
* **Models**: Swap out `OllamaEmbeddings` or `OpenAIModel` parameters for different model sizes or temperatures.
* **Additional Tools**: Add more `@agent.tool` functions to perform extra tasks (e.g., code search, summary, translation).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
