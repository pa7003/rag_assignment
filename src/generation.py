from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PROMPT V1: Basic ---
# Simple instruction, prone to hallucinations if context is missing.
PROMPT_V1 = """
You are a helpful assistant. Answer the user's question based on the context provided below.

Context:
{context}

Question: 
{question}

Answer:
"""

# --- PROMPT V2: Improved & Structured ---
# Improvements:
# 1. Persona: "Company Policy Expert" sets the tone.
# 2. Constraints: Explicitly forbids inventing info ("If the answer is not in the context...").
# 3. Structure: Asks for bullet points and source references.
# 4. Tone: Professional and clear.
PROMPT_V2 = """
You are a Company Policy Expert for TechGizmo Solutions. 
Your goal is to answer employee or customer questions accurately based ONLY on the provided policy documents.

Instructions:
1. Retrieval is Key: Use ONLY the context provided below to answer. Do not use outside knowledge.
2. Handling Unknowns: If the answer is not explicitly stated in the context, say "I cannot find specific information regarding this in the current policies." Do NOT make up an answer.
3. Structure: 
   - State the direct answer clearly.
   - Use bullet points for conditions or steps.
   - Mention which policy text you are referencing (e.g., "According to Section 2...").
4. Tone: Professional, concise, and helpful.

Context:
{context}

Question: 
{question}

Answer:
"""

def get_rag_chain(use_improved_prompt: bool = True):
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    template = PROMPT_V2 if use_improved_prompt else PROMPT_V1
    prompt = ChatPromptTemplate.from_template(template)
    
    # We assume 'context' is passed as a string (joined chunks)
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def generate_answer(question: str, context_docs: list, use_improved_prompt: bool = True):
    """
    Generate an answer given the question and retrieved documents.
    """
    # Simply join the page content for context
    context_text = "\n\n---\n\n".join([d.page_content for d in context_docs])
    
    chain = get_rag_chain(use_improved_prompt)
    
    # Invoke the chain
    # Note: The chain expects a dict input if we used itemgetter, but here I set up RunnablePassthrough slightly differently.
    # Let's fix the chain invoke signature matches the prompt variables.
    response = chain.invoke({"context": context_text, "question": question})
    return response
