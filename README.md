# RAG Mini Project: Company Policy Assistant

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system to answer questions based on company policy documents. It is designed to demonstrate prompt engineering, retrieval structure, and evaluation strategies.

## Architecture
1.  **Ingestion**: Loads Markdown policy documents and splits them into chunks (500 chars) using `RecursiveCharacterTextSplitter`.
2.  **Storage**: Uses **ChromaDB** with `OpenAIEmbeddings` to store vector representations of the text chunks.
3.  **Retrieval**: Fetches the top-3 most similar chunks for a given user query.
4.  **Generation**: Uses `gpt-3.5-turbo` (via LangChain) to generate answers.
    - **Prompt V1**: Basic direct question answering.
    - **Prompt V2 (Default)**: Structured "Policy Expert" persona with strict guidelines on hallucination avoidance and formatting.

## Setup Instructions

### Prerequisites
- Python 3.9+
- OpenAI API Key

### Installation
1.  Clone the repository (or navigate to directory).
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables:
    - Create a `.env` file in the root.
    - Add: `OPENAI_API_KEY=sk-your-key-here`

### Running the System

**1. Initialize the Knowledge Base** (First run only):
```bash
python src/main.py --setup
```
This loads policies from `data/` and builds the Chroma vector store.

**2. Ask a Question**:
```bash
python src/main.py --query "What is the refund policy?"
```

**3. Run Evaluation**:
```bash
python src/evaluate.py
```

### Web UI (Streamlit)
You can also interact with the system via a web interface.
```bash
streamlit run src/app.py
```
This will open `http://localhost:8501` in your browser.
This runs a test suite of answerable and unanswerable questions.

## Prompt Engineering
One of the key focuses was improving the prompt to handle edge cases.

**Initial Prompt (V1)**:
- Simple "Answer based on context".
- **Issue**: Often tries to be too helpful, potentially answering general knowledge questions even if not in the policy (e.g., "What is a refund?").

**Improved Prompt (V2)**:
- **Persona**: "Company Policy Expert".
- **Constraints**: Explicitly instructs to say "I cannot find specific information" if the context is missing.
- **Structure**: Requests bullet points and citations (e.g., "According to Section 2").
- **Result**: More professional, grounded answers with fewer hallucinations.

## Evaluation Strategy
A small evaluation script (`src/evaluate.py`) tests the system against:
1.  **Direct Facts**: "Refund deadline" -> Expected "30 days".
2.  **Inference**: "Cancel on Sunday" -> Checks 24h rule.
3.  **Out of Scope**: "Remote work policy" -> Expected "Cannot find info".

Current logic checks for keyword presence in the response as a proxy for accuracy.

## Future Improvements
- **Hybrid Search**: Combine keyword (BM25) with semantic search for better precision on specific terms (e.g., "Order #123").
- **Reranking**: Add a reranker (e.g., Cohere) after retrieval to optimize the top-k passed to the LLM.
- **Citations**: Implement strict programmed citation extraction rather than relying on LLM text generation.
