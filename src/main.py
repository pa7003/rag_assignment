import os
import argparse
from dotenv import load_dotenv
from ingestion import load_documents, split_documents
from retrieval import get_vector_store, retrieve_documents
from generation import generate_answer

def setup_knowledge_base():
    """
    Loads data and rebuilds the vector store.
    """
    print("Initializing Knowledge Base...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    
    if not chunks:
        print("No data found to ingest!")
        return
        
    get_vector_store(chunks, reset=True)
    print("Knowledge Base ready!")

def query_system(question: str, use_improved: bool = True):
    print(f"\nQuestion: {question}")
    print("Retrieving docs...")
    docs = retrieve_documents(question)
    
    if not docs:
        print("No relevant documents found.")
        # Edge case handling: If no docs, we might skip generation or let the LLM handle "empty context"
        # Ideally, we pass empty context and let the prompt handle it (Probe V2 handles it).
    
    print(f"Checking {len(docs)} references...")
    answer = generate_answer(question, docs, use_improved_prompt=use_improved)
    
    print("\n--- Answer ---")
    print(answer)
    print("--------------\n")

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="RAG Policy Assistant")
    parser.add_argument("--setup", action="store_true", help="Re-ingest data and build vector store")
    parser.add_argument("--query", type=str, help="Ask a question to the assistant")
    parser.add_argument("--simple", action="store_true", help="Use the simple (V1) prompt instead of improved (V2)")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_knowledge_base()
    
    if args.query:
        query_system(args.query, use_improved=not args.simple)
        
    if not args.setup and not args.query:
        print("Please provide --setup to initialize or --query 'Your Question' to ask.")
