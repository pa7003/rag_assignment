import pandas as pd
from retrieval import retrieve_documents
from generation import generate_answer

# Define Evaluation Dataset
EVAL_SET = [
    {
        "category": "Answerable",
        "question": "What is the deadline for requesting a refund?",
        "expected_key": "30 days"
    },
    {
        "category": "Answerable",
        "question": "How much does overnight shipping cost?",
        "expected_key": "25.00"
    },
    {
        "category": "Partially Answerable / Inference",
        "question": "Can I cancel my order on a Sunday if I placed it Saturday night?",
        "expected_key": "within 24 hours" # Context says 24 hours, so yes implicitly
    },
    {
        "category": "Unanswerable (Out of Scope)",
        "question": "What is the company's fierce remote work policy?",
        "expected_key": "cannot find specific information" # Should fallback
    },
    {
        "category": "Unanswerable (Fake Entity)",
        "question": "Do you offer refunds for 'MegaGizmo' products?",
        "expected_key": "cannot find" # Or generalized failure
    }
]

def run_evaluation():
    print("Running Evaluation Suite...\n")
    results = []
    
    for item in EVAL_SET:
        q = item["question"]
        print(f"Testing: {q}")
        
        # 1. Retrieval
        docs = retrieve_documents(q, k=3)
        retrieved_texts = [d.page_content for d in docs]
        retrieved_joined = "\n".join(retrieved_texts)
        
        # 2. Generation (Using V2 Improved Prompt)
        answer = generate_answer(q, docs, use_improved_prompt=True)
        
        # 3. Simple scoring
        # Check if expected keyword/concept is in the answer
        # Note: This is an auto-approximate; real eval needs human or LLM-as-judge
        passed = item["expected_key"].lower() in answer.lower()
        if item["category"] == "Unanswerable (Out of Scope)" and ("sorry" in answer.lower() or "no information" in answer.lower() or "cannot find" in answer.lower()):
            passed = True

        results.append({
            "Question": q,
            "Category": item["category"],
            "Passed": "✅" if passed else "❌",
            "Answer Preview": answer[:100].replace("\n", " ") + "..."
        })

    # Display Results
    df = pd.DataFrame(results)
    print("\n\n=== Evaluation Report ===")
    print(df.to_markdown(index=False))
    
    # Save to file
    with open("../evaluation_results.md", "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluation()
