import json
import numpy as np
from pathlib import Path
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

def save_embeddings(chunks: list[dict], embeddings, output_dir: str = "data/embedding_store"):
    
    """
    Save chunks and embeddings locally.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "chunk_embeddings.npy", embeddings)

    with open(output_path / "chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

def load_embeddings(input_dir: str = "data/embedding_store"):
    """
    Load chunks and embeddings from a local directory.
    """
    input_path = Path(input_dir)

    embeddings = np.load(input_path / "chunk_embeddings.npy")

    with open(input_path / "chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks, embeddings

def embeddings_exist(input_dir: str = "data/embedding_store") -> bool:
    """
    Check if embeddings exist in a local directory.
    """
    input_path = Path(input_dir)

    return ( (input_path / "chunk_embeddings.npy").exists() 
            and (input_path / "chunks.json").exists() )