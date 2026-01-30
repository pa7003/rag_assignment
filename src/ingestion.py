import os
import glob
from typing import List
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(data_dir: str) -> List[Document]:
    """
    Load all Markdown documents from the specified directory.
    """
    documents = []
    # Find all .md files
    file_paths = glob.glob(os.path.join(data_dir, "*.md"))
    
    if not file_paths:
        print(f"No markdown files found in {data_dir}")
        return []

    for path in file_paths:
        try:
            # Using TextLoader as it's simple and reliable for basic text/md
            loader = TextLoader(path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return documents

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Rationale for Chunk Size (500):
    policy documents usually have distinct short sections. 
    500 chars is roughly 80-120 words, which is often enough to capture a full rule 
    or condition (e.g., "Refunds must be requested within 30 days...").
    Too large > might mix unrelated policies.
    Too small < might cut context (e.g., separating the condition from the consequence).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", " ", ""] # Try to split by headers first
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test execution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    if chunks:
        print(f"Example Chunk 0 content:\n{chunks[0].page_content}")
