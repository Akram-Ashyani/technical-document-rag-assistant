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
    Add retrieved source references using Python.
    """
    if not retrieved_chunks:
        return "No relevant chunks found."

    context = build_context(retrieved_chunks)
    print("\n========== CONTEXT SENT TO OLLAMA ==========")
    print(context)
    print("============================================\n")

    prompt = f"""
        You are a technical document question-answering assistant.

        Answer the user's question using only the provided document context.

        Instructions:
        - Identify the information in the context that is relevant to the question.
        - Base your answer on explicit evidence from the context.
        - Preserve numerical values, units, names, and technical specifications exactly.
        - You may combine information from multiple relevant excerpts.
        - Do not ignore relevant information simply because it describes
        a component, subsection, or feature of the subject.
        - If the question is ambiguous, explain the relevant distinctions
        supported by the context.
        - If the context partially answers the question, provide the available
        information and state what is missing.
        - If the context does not support an answer, say:
        I could not find this information in the provided context.
        - Do not use outside knowledge or invent missing information.
        - Do not include source references, page numbers, or chunk IDs.
        - Do not repeat the question.
        - Keep your answer concise.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

    response = requests.post("http://localhost:11434/api/generate",
                             json={"model": model_name, "prompt": prompt, "stream": False,
                                   "options": {"temperature": 0}}, timeout=120)

    response.raise_for_status()

    answer_text = response.json()["response"].strip()

    sources = "\n".join(
        f"- page {chunk['page_number']}, chunk {chunk['chunk_id']}"
        for chunk in retrieved_chunks
    )

    return f"{answer_text}\n\nRetrieved sources:\n{sources}"

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
    