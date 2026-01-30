import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define persistence directory
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

def get_vector_store(chunks: List[Document] = None, reset: bool = False):
    """
    Initialize or get the Chroma Vector Store.
    If chunks are provided, add them to the store.
    """
    
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
       # Fallback or error - for this assignment we assume env is set or user needs to set it
       # We could use HuggingFaceEmbeddings if no key, but request asks for OpenAI/Anthropic support.
       pass
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if reset and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("Cleared existing vector store.")

    if chunks:
        print("Creating/Updating Vector Store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="policy_docs"
        )
        print("Vector Store updated.")
    else:
        # Load existing
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="policy_docs"
        )
    
    return vector_store

def retrieve_documents(query: str, k: int = 3):
    """
    Retrieve top-k relevant documents for a query.
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Test retrieval (Requires DB to be populated first)
    # This requires OPENAI_API_KEY in environment
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        results = retrieve_documents("What is the refund deadline?")
        for doc in results:
            print(f"\n[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:100]}...")
    except Exception as e:
        print(f"Retrieval failed (likely no API key or DB empty): {e}")

