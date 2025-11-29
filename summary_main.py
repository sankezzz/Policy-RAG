import streamlit as st
import os
import tempfile
import sys
import requests
import mimetypes
import easyocr
from urllib.parse import urlparse, unquote

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

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en', 'hi', 'mr'], gpu=True) 

def extract_text_from_image(image_path):
    reader = get_ocr_reader()
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n\n".join(result)

def download_file_from_url(url, temp_dir):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        parsed_url = urlparse(url)
        filename = os.path.basename(unquote(parsed_url.path))
        if not filename:
            content_type = response.headers.get('content-type')
            ext = mimetypes.guess_extension(content_type) or ".txt"
            filename = f"downloaded_content{ext}"
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path, filename
    except Exception as e:
        st.error(f"Failed to download URL: {e}")
        return None, None

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, llm):
    # We increase 'k' slightly for summaries to get more context
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    
    template_str = """
<|system|>
You are an expert fraud analyst and policy advisor.
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

st.set_page_config(page_title=" Multi-Lingual Policy Analyzer", layout="wide")
st.title("📄 Multi-Lingual Policy Analyzer")
st.markdown(f"**LLM:** `Phi-3-Mini` | **Embeddings:** `{EMBEDDING_MODEL_NAME}` | **OCR:** `EasyOCR` | **Summarization:** `Bert-Large`")

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

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Documents in Marathi , Hindi , English")
    
    url_input = st.text_input("Paste PDF or Image URL here:")
    
    uploaded_files = st.file_uploader(
        "Select files (PDF, TXT, Images)",
        type=["pdf", "txt", "md", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        with st.spinner("Processing documents & links..."):
            all_chunks = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                files_to_process = []
                
                # A. Handle URL
                if url_input:
                    path, name = download_file_from_url(url_input, temp_dir)
                    if path:
                        files_to_process.append((path, name))
                
                # B. Handle Uploads
                if uploaded_files:
                    for up_file in uploaded_files:
                        path = os.path.join(temp_dir, up_file.name)
                        with open(path, "wb") as f:
                            f.write(up_file.getvalue())
                        files_to_process.append((path, up_file.name))

                # Process Files
                for file_path, file_name in files_to_process:
                    documents = []
                    try:
                        lower_name = file_name.lower()
                        if lower_name.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                            documents = loader.load()
                        elif lower_name.endswith((".txt", ".md")):
                            loader = TextLoader(file_path, encoding="utf-8")
                            documents = loader.load()
                        elif lower_name.endswith((".jpg", ".jpeg", ".png")):
                            extracted_text = extract_text_from_image(file_path)
                            if extracted_text.strip():
                                doc = Document(page_content=extracted_text, metadata={"source": file_name})
                                documents = [doc]
                        
                        if documents:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                            chunks = text_splitter.split_documents(documents)
                            all_chunks.extend(chunks)
                            st.write(f"✅ Loaded: {file_name}")
                            
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")

            # Save Vector Store
            if all_chunks:
                embeddings = get_embeddings_model()
                st.session_state.vector_store = FAISS.from_documents(all_chunks, embeddings)
                st.success(f"Processing Complete! {len(all_chunks)} chunks created.")
            else:
                st.warning("No content found to process.")

    st.divider()
    
    # --- NEW: LLM-Based Summary Button ---
    st.header("2. Tools")
    if st.button("📝 Generate Summary"):
        if st.session_state.vector_store:
            # We treat this exactly like a chat message
            summary_prompt = "Provide a detailed structured summary of this policy document. Include the Scope of Cover, Key Exclusions, and Claim Procedures."
            
            # Add to chat history so the user sees what happened
            st.session_state.messages.append({"role": "user", "content": summary_prompt})
            
            # Trigger the logic immediately
            with st.spinner("Generating Summary with Phi-3..."):
                try:
                    llm = get_llm()
                    rag_chain = get_rag_chain(st.session_state.vector_store, llm)
                    response = rag_chain.invoke(summary_prompt)
                    
                    # Append response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Rerun to update the chat window immediately
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
        else:
            st.warning("Please upload and process a document first.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the policy... - (Marathi,Hindi,English)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store is None:
        st.warning("Please upload documents or provide a URL first.")
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