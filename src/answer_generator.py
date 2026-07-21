def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate a simple answer using retrieved document chunks.
    """
    if not retrieved_chunks:
        return "No relevant chunks found."
    
    context = retrieved_chunks[0]["text"]

    answer = (
        f"Based on the retrieved chunks, the answer to the question '{question}' is:\n\n"
        f"{context[:700]}\n\n"
        f"Source: page {retrieved_chunks[0]['page_number']}, "
        f"chunk {retrieved_chunks[0]['chunk_id']}"
    )

    return answer