from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a sentence embedding model.
    It converts text into numerical vectors.
    """
    return SentenceTransformer(model_name)

def create_embeddings(texts: list[str], model: SentenceTransformer):
    """
    Create embedding vectors for a list of texts.
    """
    return model.encode(texts, convert_to_numpy=True)