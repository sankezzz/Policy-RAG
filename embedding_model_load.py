from langchain_huggingface import HuggingFaceEmbeddings

print("Attempting to load embedding model...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cuda'}  # Use CUDA
    )
    print("\n--- SUCCESS! ---")
    print("Embedding model loaded successfully.")
    
    # Optional: Test an embedding
    test_vector = embeddings.embed_query("This is a test.")
    print(f"Created a test vector of dimension: {len(test_vector)}")
    
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to load embedding model: {e}")