"""
GitHub Repo Chatbot with LangChain and Ollama
---------------------------------------------
This application allows you to chat with the contents of any GitHub repository.
It uses LangChain and Ollama to process and understand the code.
"""

# =========== SECTION 1: IMPORTS ===========
# Standard library imports
import os
import subprocess
import tempfile

# Streamlit for the web interface
import streamlit as st

# LangChain components
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

# =========== SECTION 2: REPOSITORY TEXT EXTRACTION ===========
def get_repo_text(repo_url: str, allowed_extensions: set = None) -> str:
    """
    Clone a GitHub repository and extract text from its files.
    
    Args:
        repo_url: URL of the GitHub repository
        allowed_extensions: Set of file extensions to include
        
    Returns:
        String containing the text content of allowed files
    """
    # Default file extensions if none provided
    if not allowed_extensions:
        allowed_extensions = {
            ".py", ".md", ".txt", ".js", ".html", ".css", ".json",
            ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".rb", ".go", ".rs",
        }
    
    repo_texts = []

    # Clone repository to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Step 1: Clone the repository
        clone_command = ["git", "clone", repo_url, tmpdirname]
        try:
            subprocess.run(
                clone_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as e:
            st.error(f"Error cloning the repository: {e}")
            return ""

        # Step 2: Process files in the repository
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                # Check if file extension is in allowed list
                ext = os.path.splitext(file)[1].lower()
                if ext in allowed_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content.strip():
                                # Add filename as a marker with the content
                                repo_texts.append(f"Filename: {file}\n{content}")
                    except Exception as e:
                        st.warning(f"Could not read {file_path}: {e}")

        # Check if we found any valid files
        if not repo_texts:
            st.error("No allowed text files found in the repository.")
            return ""
            
        # Combine all file contents into one string
        return "\n".join(repo_texts)

# =========== SECTION 3: TEXT PROCESSING ===========
def split_text(text: str):
    """
    Split large text into smaller chunks for processing.
    
    Args:
        text: The text to split
        
    Returns:
        List of text chunks
    """
    # Create a text splitter with specified chunk size and overlap
    text_splitter = CharacterTextSplitter(
        separator=" ", 
        chunk_size=1000,  # Each chunk will be ~1000 characters
        chunk_overlap=200 # With 200 character overlap to maintain context
    )
    
    # Split the text and return the chunks
    chunks = text_splitter.split_text(text)
    return chunks

# =========== SECTION 4: VECTOR STORE CREATION ===========
def create_vectorstore(chunks):
    """
    Create a FAISS vector store from text chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        FAISS vector store containing embeddings
    """
    # Create embeddings using Ollama's embedding model
    embeddings = OllamaEmbeddings(model="all-minilm:33m")
    
    # Create and return the vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# =========== SECTION 5: QUESTION ANSWERING ===========
def answer_question(query: str, vectorstore) -> str:
    """
    Answer a question about the repository content.
    
    Args:
        query: User's question
        vectorstore: FAISS vector store with repository content
        
    Returns:
        Answer to the user's question
    """
    # Step 1: Find relevant text chunks using similarity search
    docs = vectorstore.similarity_search(query, k=4)  # Get top 4 most relevant chunks

    # Step 2: Initialize the Ollama language model
    llm = ChatOllama(model="llama3.2", temperature=0)  # temperature=0 for deterministic answers

    # Step 3: Create a question-answering chain
    chain = load_qa_chain(llm, chain_type="stuff")  # "stuff" method puts all docs in one prompt
    
    # Step 4: Generate an answer based on the retrieved documents
    response = chain.invoke({
        "input_documents": docs,
        "question": query
    })
    
    # Return the generated answer
    return response["output_text"]

# =========== SECTION 6: STREAMLIT UI ===========
def main():
    """Main function to run the Streamlit app."""
    
    # Set up the page title
    st.title("GitHub Repo Chatbot with LangChain and Ollama")

    # === Sidebar Configuration ===
    # Input for GitHub repository URL
    repo_url = st.sidebar.text_input(
        "Enter the GitHub repository URL:", 
        value="https://github.com/JustCodeIt7/GitHub_Repo_Chat"
    )

    # Available file extensions to select
    all_file_extensions = [
        ".py", ".md", ".txt", ".js", ".html", ".css", ".json",
        ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".rb", ".go", ".rs"
    ]

    # Let users select which file types to include
    selected_extensions = st.sidebar.multiselect(
        "Select file types to include:",
        all_file_extensions,
        default=all_file_extensions[:7]  # Default to first 7 file types
    )

    # Button to load the repository
    if st.sidebar.button("Load Repository"):
        with st.spinner("Cloning repository and processing files..."):
            # Process the repository
            allowed_extensions_set = set(selected_extensions)
            repo_text = get_repo_text(repo_url, allowed_extensions_set)
            
            if repo_text:
                # Process the text and create the vector store
                chunks = split_text(repo_text)
                vectorstore = create_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Repository loaded and processed successfully!")
            else:
                st.error("Failed to retrieve repository content.")

    # === Chat Interface ===
    st.header("Chat with the Repository Content")

    # Initialize chat history if not already in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat interface if a repository is loaded
    if "vectorstore" in st.session_state:
        # Show previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Get user input
        if prompt := st.chat_input("Ask a question about the repository..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_question(prompt, st.session_state.vectorstore)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Prompt for repository loading if not already done
        st.info(
            "Please enter a GitHub repository URL in the sidebar and click 'Load Repository' to start."
        )

# Run the application
if __name__ == "__main__":
    main()