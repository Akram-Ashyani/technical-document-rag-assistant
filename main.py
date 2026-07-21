from src.document_loader import load_document
from src.text_chunker import chunk_pdf_pages
from src.embeddings import (load_embedding_model, create_embeddings, 
                            save_embeddings, load_embeddings, embeddings_exist)
from src.retriever import retrieve_top_chunks

DOC_PATH = "data/sample_documents/raspberry_pi_pico_with_datasheet.pdf"
EMBEDDING_STORE_DIR = "data/embedding_store"

if __name__ == "__main__":

    model = load_embedding_model()

    if embeddings_exist(EMBEDDING_STORE_DIR):
        print("Embeddings already exist. Skipping embedding creation.")
        chunks, chunk_embeddings = load_embeddings(EMBEDDING_STORE_DIR)
    else:
        print("Creating embeddings...")
        pages = load_document(DOC_PATH)
        chunks = chunk_pdf_pages(pages)

        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = create_embeddings(chunk_texts, model)
        save_embeddings(chunks, chunk_embeddings, EMBEDDING_STORE_DIR)
        
        print("Number of pages:", len(pages))
    print("Number of chunks:", len(chunks))

    test_questions = [
        "What is the keep-out area for the antenna?",
        "What chip is used in Raspberry Pi Pico W?",
        "How can Raspberry Pi Pico W be powered?",
        "What are the recommended operating conditions?",
        "How much flash memory does Raspberry Pi Pico W have?"
    ]

    for question in test_questions:
        results = retrieve_top_chunks(question, chunks, chunk_embeddings, model, top_k=3)

        print("\n==================================================")
        print("\nQuestion:", question)

        for result in results:
            print("\n--------------------")
            print("Chunk ID:", result["chunk_id"])
            print("Page:", result["page_number"])
            print("Score:", result["score"])
            print(result["text"][:1000])