import streamlit as st
import os
import tempfile
import sys
from huggingface_hub import hf_hub_download
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- FIX for HuggingFace 401 Error ---
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "true"
# -------------------------------------

# --- CONFIGURATION ---

# 1. Embedding Model (runs on GPU)
#    UPDATED to the Qwen model you just tested
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" 

# 2. LLM Model (runs on CPU)
LLM_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLM_FILE_NAME = "Phi-3-mini-4k-instruct-q4.gguf" # The ~2.4GB file
LOCAL_LLM_PATH = f"./{LLM_FILE_NAME}"

# --- HELPER FUNCTIONS ---

def download_llm_model():
    """
    Downloads the Phi-3 GGUF model file if it doesn't exist.
    """
    if os.path.exists(LOCAL_LLM_PATH):
        return # Model already exists

    st.info(f"Downloading {LLM_FILE_NAME} (~2.4GB)... This may take a few minutes.")
    try:
        hf_hub_download(
            repo_id=LLM_REPO_ID,
            filename=LLM_FILE_NAME,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        st.success("LLM model downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading LLM model: {e}")
        st.stop()

@st.cache_resource
def get_embeddings_model():
    """Loads the embedding model from HuggingFace, configured for CUDA."""
    print("Loading embedding model (for CUDA)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda'}  # <-- Use CUDA
        )
        # Test query to make sure it's working
        embeddings.embed_query("Test embedding")
        print("Embedding model loaded to CUDA successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        st.error("Please ensure 'transformers' is up to date: pip install --upgrade transformers")
        st.stop()
        

@st.cache_resource
def get_llm():
    """Initializes the LlamaCpp LLM to run locally on CPU."""
    
    if not os.path.exists(LOCAL_LLM_PATH):
        st.error(f"LLM file not found: {LOCAL_LLM_PATH}")
        st.error("Please restart the app to trigger the download.")
        st.stop()
        
    print("Loading LlamaCpp model...")
    try:
        llm = LlamaCpp(
            model_path=LOCAL_LLM_PATH,
            n_gpu_layers=-1,  # <-- 0 = RUN ON CPU (SYSTEM MEMORY)
            n_batch=512,
            n_ctx=4096,
            temperature=0.5,
            verbose=True,
        )
        llm.invoke("Hello") # Simple check
        print("LlamaCpp model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading LlamaCpp model: {e}")
        st.stop()
    return llm

def format_docs(docs: list[Document]) -> str:
    """Helper function to format retrieved docs into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, llm):
    """Creates the complete RAG (Retrieval-Augmented Generation) chain."""
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3} # Retrieve top 3 most relevant chunks
    )

    # This is the core prompt, tailored for Phi-3
    template_str = """
<|system|>
You are an expert fraud analyst and policy advisor.
Your task is to meticulously analyze the provided policy context to find potential loopholes, ambiguities, or clauses that could be exploited.

Based *only* on the following context, answer the user's query.
If the context does not contain the answer, state that clearly.
Do not use any outside knowledge.<|end|>
<|user|>
CONTEXT:
{context}

QUERY:
{question}<|end|>
<|assistant|>
ANALYSIS:
"""
    prompt_template = PromptTemplate.from_template(template_str)


    # Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- STREAMLIT APP ---

st.set_page_config(page_title="Policy Analyzer", layout="wide")
st.title("📄 RAG Policy Document Analyzer (Hybrid)")
st.markdown(f"**LLM:** `Phi-3-Mini`| **Embeddings:** `{EMBEDDING_MODEL_NAME}` ")

# --- Download Model at Start ---
download_llm_model()

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar for Document Upload ---
with st.sidebar:
    st.header("1. Upload Documents")
    st.caption("Upload .txt, .md, or .pdf files. The vector store will be updated.")
    
    uploaded_files = st.file_uploader(
        "Upload your policy documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents... "):
                all_chunks = []
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        loader = None
                        if uploaded_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                        elif uploaded_file.name.endswith((".txt", ".md")):
                            loader = TextLoader(file_path, encoding="utf-8")
                        
                        if loader:
                            try:
                                docs = loader.load()
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=150
                                )
                                chunks = text_splitter.split_documents(docs)
                                all_chunks.extend(chunks)
                            except Exception as e:
                                st.error(f"Error loading {uploaded_file.name}: {e}")
                
                if all_chunks:
                    # Create vector store
                    embeddings = get_embeddings_model()
                    vector_store = FAISS.from_documents(all_chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    st.success(f"Processed {len(all_chunks)} chunks from {len(uploaded_files)} files. Ready to chat!")
                else:
                    st.error("No processable documents were found.")

    st.header("2. Chat")
    st.caption("Ask questions about your uploaded documents.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Find loopholes related to..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if documents are processed
    if st.session_state.vector_store is None:
        st.warning("Please upload and process your documents in the sidebar before chatting.")
    else:
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing... "):
                try:
                    llm = get_llm()
                    rag_chain = get_rag_chain(st.session_state.vector_store, llm)
                    
                    response = rag_chain.invoke(prompt)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")