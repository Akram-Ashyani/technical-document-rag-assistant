from sentence_transformers import CrossEncoder

def load_reranker(model_name: str="cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Load a cross-encoder model for reranking.
    """
    return CrossEncoder(model_name)


def rerank_chunks(question: str, retrieved_chunks: list[dict],
                  reranker: CrossEncoder, top_k: int = 3) -> list[dict]:
    """
    Rerank retrieved chunks based on question relevance.
    """
    if not retrieved_chunks:
        return []
    
    pairs = [(question, chunk["text"]) for chunk in retrieved_chunks]

    scores = reranker.predict(pairs)

    reranked_chunks = []

    for chunk, score in zip(retrieved_chunks, scores):
        reranked_chunks.append({**chunk, "rerank_score": float(score)})

    reranked_chunks.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)

    return reranked_chunks[:top_k]