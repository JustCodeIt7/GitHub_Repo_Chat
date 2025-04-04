import streamlit as st
import subprocess
import os
import tempfile

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
# Import Ollama wrappers for embeddings and chat models
from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_repo_text(repo_url: str) -> str:
    """
    Clone the given GitHub repository and extract text from allowed files.
    Allowed file extensions include common text/code formats.
    """
    # Define allowed file extensions (you can adjust this set as needed)
    allowed_extensions = {
        ".py",
        ".md",
        ".txt",
        ".js",
        ".html",
        ".css",
        ".json",
        ".yaml",
        ".yml",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".go",
        ".rs",
    }
    repo_texts = []

    # Create a temporary directory to clone the repository
    with tempfile.TemporaryDirectory() as tmpdirname:
        clone_command = ["git", "clone", repo_url, tmpdirname]
        try:
            subprocess.run(
                clone_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as e:
            st.error(f"Error cloning the repository: {e}")
            return ""

        # Walk through the cloned repository and process files
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in allowed_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content.strip():
                                # Add the file name as a marker along with its content
                                repo_texts.append(f"Filename: {file}\n{content}")
                    except Exception as e:
                        st.warning(f"Could not read {file_path}: {e}")

        if not repo_texts:
            st.error("No allowed text files found in the repository.")
            return ""
        # Return the combined text from all files
        return "\n".join(repo_texts)


def split_text(text: str):
    """
    Split the large text into smaller chunks to help the embedding model.
    """
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vectorstore(chunks):
    """
    Create a FAISS vector store from text chunks using Ollama embeddings.
    """
    # Instantiate OllamaEmbeddings with your chosen model.
    embeddings = OllamaEmbeddings(model="all-minilm:33m")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def answer_question(query: str, vectorstore) -> str:
    """
    Retrieve relevant text chunks via similarity search and use an Ollama LLM to generate an answer.
    """
    # Retrieve the top 4 most relevant chunks.
    docs = vectorstore.similarity_search(query, k=4)

    # Instantiate the ChatOllama model with your chosen model name.
    llm = ChatOllama(model="llama3.2", temperature=0)

    # Create a QA chain that stuffs the retrieved documents into the prompt.
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Use invoke() with a proper input dictionary instead of run()
    response = chain.invoke({
        "input_documents": docs,
        "question": query
    })
    
    # The response is a dictionary, get the answer text
    return response["output_text"]


# --- Streamlit App Layout ---
st.title("GitHub Repo Chatbot with LangChain and Ollama")

# Sidebar: Input for the GitHub repository URL.
repo_url = st.sidebar.text_input(
    "Enter the GitHub repository URL:", value="https://github.com/JustCodeIt7/GitHub_Repo_Chat"
)
if st.sidebar.button("Load Repository"):
    with st.spinner("Cloning repository and processing files..."):
        repo_text = get_repo_text(repo_url)
        if repo_text:
            chunks = split_text(repo_text)
            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            st.success("Repository loaded and processed successfully!")
        else:
            st.error("Failed to retrieve repository content.")

# Initialize chat history if not already set.
if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Chat with the Repository Content")

# Display chat messages from history
if "vectorstore" in st.session_state:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask a question about the repository..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(prompt, st.session_state.vectorstore)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info(
        "Please enter a GitHub repository URL in the sidebar and click 'Load Repository' to start."
    )