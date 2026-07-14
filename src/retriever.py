import numpy as np
from src.embeddings import create_embeddings

def cosine_similarity(query_embedding, document_embeddings):
    """
    Calculate cosine similarity between a query embedding and a list of document embeddings.
    Higher score means more similar meaning.
    """
    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(document_embeddings, axis=1)

    if query_norm == 0:
        raise ValueError("Query embedding has zero norm.")
    
    document_norms = np.where(document_norms == 0, 1e-10, document_norms)

    return np.dot(query_embedding, document_embeddings.T) / (query_norm * document_norms)

def retrieve_top_chunks(question: str, chunks: list[dict], model, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top K chunks that are most similar to the question.
    """
    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = create_embeddings(chunk_texts, model)
    question_embedding = model.encode(question, convert_to_numpy=True)

    similarities = cosine_similarity(question_embedding, chunk_embeddings)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []

    for index in top_indices:
        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "page_number": chunks[index]["page_number"],
            "text": chunks[index]["text"],
            "score": float(similarities[index])
        })

    return results