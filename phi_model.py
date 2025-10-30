import os
import sys
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"

# THIS IS THE CORRECTED FILENAME (it's missing the "_K_M")
MODEL_FILE_NAME = "Phi-3-mini-4k-instruct-q4.gguf" # This is ~2.4GB
LOCAL_MODEL_PATH = f"./{MODEL_FILE_NAME}"
# ---------------------

def download_model():
    """
    Downloads the GGUF model file from Hugging Face if it doesn't exist.
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Model '{MODEL_FILE_NAME}' already exists locally.")
        return

    print(f"Downloading model to '{LOCAL_MODEL_PATH}'... This may take a few minutes.")
    try:
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILE_NAME,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check your internet connection and Hugging Face token (if needed).")
        sys.exit(1)

def create_llm():
    """
    Loads the LlamaCpp model, configured for CUDA.
    """
    print("Loading LlamaCpp model... (This may take a moment)")
    try:
        llm = LlamaCpp(
            model_path=LOCAL_MODEL_PATH,
            n_gpu_layers=-1,  # -1 = Offload ALL possible layers to GPU
            n_batch=512,
            n_ctx=4096,       # Context size for this model
            temperature=0.3,
            verbose=False,    # Set to True for more debug info
        )
        # Simple check
        llm.invoke("Hello")
        print("Model loaded successfully to CUDA.")
        return llm
    except Exception as e:
        print(f"Error loading LlamaCpp model: {e}")
        print("Please ensure 'llama-cpp-python' is installed correctly with CUDA support.")
        sys.exit(1)

def main():
    # 1. Get the model
    download_model()
    
    # 2. Load the model
    llm = create_llm()

    # 3. Define the Chat Template for Phi-3
    # This is the specific format Phi-3 requires
    template = """
<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""
    prompt = PromptTemplate.from_template(template)

    # 4. Create a simple chain
    llm_chain = prompt | llm | StrOutputParser()

    # 5. Start the chat loop
    print(f"\n--- Phi-3 Chat (Type 'exit' or 'quit' to end) ---")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break

            if not user_input.strip():
                continue

            print("Phi-3: ", end="", flush=True)
            
            # Use the chain to get a response
            response = llm_chain.invoke({"question": user_input})
            print(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()