import streamlit as st
import os
from dotenv import load_dotenv
from retrieval import retrieve_documents, get_vector_store
from generation import generate_answer
from ingestion import load_documents, split_documents

# Page Config
st.set_page_config(page_title="Policy RAG Assistant", page_icon="📝")

# Load Env
load_dotenv()

def initialize_knowledge_base():
    """Helper to rebuild DB from UI"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    with st.spinner("Loading and chunking documents..."):
        docs = load_documents(data_dir)
        chunks = split_documents(docs)
        if not chunks:
            st.error("No data found in data/ directory.")
            return False
            
    with st.spinner("Updating Vector Store..."):
        get_vector_store(chunks, reset=True)
        
    return True

# Sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.warning("Please enter your OpenAI API Key to proceed.")

if st.sidebar.button("Re-Initialize Knowledge Base"):
    if initialize_knowledge_base():
        st.sidebar.success("Knowledge Base Updated!")

# Main Interface
st.title("🤖 Company Policy Assistant")
st.markdown("Ask questions about Refund, Cancellation, or Shipping policies.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I return my order?"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please provide an API Key in the sidebar.")
        st.stop()
        
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # 1. Retrieve
            docs = retrieve_documents(prompt)
            
            # 2. Generate
            answer = generate_answer(prompt, docs, use_improved_prompt=True)
            
            # 3. Format Output
            full_response = answer
            
            # Append sources accordion
            if docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}**: {doc.metadata.get('source', 'Unknown')}")
                        st.caption(doc.page_content[:300] + "...")

            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
