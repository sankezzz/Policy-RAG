import streamlit as st
import os
import tempfile
import sys
import numpy as np # Needed for image processing
from PIL import Image # Needed to open images
import easyocr # The OCR library

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

# NEW: Import the translator
from googletrans import Translator

# --- FIX for HuggingFace 401 Error ---
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "true"

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" 
LLM_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLM_FILE_NAME = "Phi-3-mini-4k-instruct-q4.gguf" 
LOCAL_LLM_PATH = f"./{LLM_FILE_NAME}"

# --- HELPER FUNCTIONS ---

def download_llm_model():
    if os.path.exists(LOCAL_LLM_PATH):
        return 
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
    print("Loading embedding model (for CUDA)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda'} 
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()
        
@st.cache_resource
def get_llm():
    if not os.path.exists(LOCAL_LLM_PATH):
        st.error(f"LLM file not found: {LOCAL_LLM_PATH}")
        st.stop()
    try:
        llm = LlamaCpp(
            model_path=LOCAL_LLM_PATH,
            n_gpu_layers=0, 
            n_batch=512,
            n_ctx=4096,
            temperature=0.1,
            verbose=True,
        )
    except Exception as e:
        st.error(f"Error loading LlamaCpp model: {e}")
        st.stop()
    return llm

# --- NEW: OCR Helper Function ---
@st.cache_resource
def get_ocr_reader():
    """
    Initializes EasyOCR.
    We load English ('en'), Hindi ('hi'), and Marathi ('mr') 
    to match your app's language capabilities.
    gpu=True ensures it uses CUDA if available.
    """
    print("Loading EasyOCR Reader...")
    # Note: The first time this runs, it will download the OCR models (~20MB)
    return easyocr.Reader(['en', 'hi', 'mr'], gpu=True) 

def extract_text_from_image(image_path):
    """
    Reads an image file and returns the extracted text string.
    """
    reader = get_ocr_reader()
    # EasyOCR expects a numpy array or file path
    result = reader.readtext(image_path, detail=0, paragraph=True)
    # Result is a list of strings, join them
    return "\n\n".join(result)

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3} 
    )
    template_str = """
<|system|>
You are an expert fraud analyst and policy advisor.
Your task is to analyze the provided policy context (which may include OCR text from images) to find potential loopholes.
Based *only* on the following context, answer the user's query.
<|end|>
<|user|>
CONTEXT:
{context}

QUERY:
{question}<|end|>
<|assistant|>
ANALYSIS:
"""
    prompt_template = PromptTemplate.from_template(template_str)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- STREAMLIT APP ---

st.set_page_config(page_title="Policy Analyzer (Multilingual + OCR)", layout="wide")
st.title("📄 RAG Policy Analyzer")
st.markdown(f"**LLM:** `Phi-3-Mini` (CPU) | **Embeddings:** `{EMBEDDING_MODEL_NAME}` (GPU) | **OCR:** `EasyOCR`")
st.markdown("Supports PDF, TXT, MD, and Images (JPG/PNG). Questions in Marathi, Hindi, or English.")

download_llm_model()

try:
    translator = Translator()
except Exception as e:
    st.error(f"Translator Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar for Document Upload ---
with st.sidebar:
    st.header("1. Upload Documents")
    st.caption("Upload files. Images will be processed via OCR.")
    
    # ADDED: jpg, jpeg, png to the file types
    uploaded_files = st.file_uploader(
        "Upload policy documents",
        type=["pdf", "txt", "md", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing... (Reading PDFs & OCRing Images)"):
                all_chunks = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        # Write file to temp
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        documents = [] # List to hold LangChain Documents
                        
                        # --- Logic to handle different file types ---
                        try:
                            if uploaded_file.name.lower().endswith(".pdf"):
                                loader = PyPDFLoader(file_path)
                                documents = loader.load()
                                
                            elif uploaded_file.name.lower().endswith((".txt", ".md")):
                                loader = TextLoader(file_path, encoding="utf-8")
                                documents = loader.load()
                                
                            elif uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
                                # NEW: Handle Images
                                extracted_text = extract_text_from_image(file_path)
                                if extracted_text.strip():
                                    # Create a LangChain Document manually
                                    doc = Document(
                                        page_content=extracted_text,
                                        metadata={"source": uploaded_file.name}
                                    )
                                    documents = [doc]
                                else:
                                    st.warning(f"No text found in image: {uploaded_file.name}")
                            
                            # --- Split and Accumulate ---
                            if documents:
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=150
                                )
                                chunks = text_splitter.split_documents(documents)
                                all_chunks.extend(chunks)
                                
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")

                # --- Create Vector Store ---
                if all_chunks:
                    embeddings = get_embeddings_model()
                    vector_store = FAISS.from_documents(all_chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    st.success(f"Processed {len(all_chunks)} chunks from {len(uploaded_files)} files.")
                else:
                    st.error("No valid text could be extracted.")

    st.header("2. Chat")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface (Same as before) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    detected_lang = translator.detect(prompt).lang
                    if detected_lang not in ['en', 'en-US']:
                        translated_query = translator.translate(prompt, src=detected_lang, dest='en').text
                    else:
                        translated_query = prompt

                    llm = get_llm()
                    rag_chain = get_rag_chain(st.session_state.vector_store, llm)
                    response_en = rag_chain.invoke(translated_query)
                    
                    if detected_lang not in ['en', 'en-US']:
                        final_response = translator.translate(response_en, src='en', dest=detected_lang).text
                    else:
                        final_response = response_en
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                except Exception as e:
                    st.error(f"Error: {e}")