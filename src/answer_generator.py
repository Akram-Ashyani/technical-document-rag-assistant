import requests

def build_context(retrieved_chunks:list[dict]) -> str:
    """
    Combine retrieved chunks into one context string.
    """
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(f"Source: page {chunk['page_number']}, chunk {chunk['chunk_id']}\n"
                             f"{chunk['text']}")
        
    return "\n\n---\n\n".join(context_parts)

def generate_basic_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate a simple answer using retrieved document chunks without LLM.
    """
    if not retrieved_chunks:
        return "No relevant chunks found."
    
    context = retrieved_chunks[0]

    answer = (
        f"Based on the retrieved chunks, the answer to the question '{question}' is:\n\n"
        f"{context['text'][:700]}\n\n"
        f"Source: page {context['page_number']}, "
        f"chunk {context['chunk_id']}"
    )

    return answer

def generate_ollama_answer(question: str, retrieved_chunks: list[dict],
                           model_name: str = "llama3.2") -> str:
    """
    Generate an answer using a local Ollama model.
     
    """
    if not retrieved_chunks:
        return "No relevant chunks found."
    
    context = build_context(retrieved_chunks)

    prompt = f"""
    You are a careful technical document question-answering assistant.

    You must answer ONLY using the provided context.
    Do not use outside knowledge.
    Do not guess.
    Do not invent numbers, units, page numbers, or chunk IDs.

    If the answer is not clearly present in the context, respond exactly:
    I could not find this information in the document.

    When you answer, use this format:

    Answer:
    <short answer>

    Sources:
    - page <page_number>, chunk <chunk_id>

    Question:
    {question}

    Context:
    {context}
    """

    response = requests.post("http://localhost:11434/api/generate",
                             json={"model": model_name, "prompt": prompt, "stream": False,},
                             timeout=120,)
    
    response.raise_for_status()

    return response.json()["response"]

def generate_answer(question: str, retrieved_chunks: list[dict], mode: str = "basic") -> str:
    """
    Generate an answer using the selected mode.
    Available modes:
    - basic: show retrieved text without an LLM
    - ollama: use a local Ollama model
    """
    if mode == "basic":
        return generate_basic_answer(question, retrieved_chunks)
    
    if mode == "ollama":
        return generate_ollama_answer(question, retrieved_chunks)
    
    raise ValueError(f"Invalid mode: {mode}")
    