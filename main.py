from src.document_loader import load_document
from src.text_chunker import chunk_pdf_pages
from src.embeddings import (load_embedding_model, create_embeddings, 
                            save_embeddings, load_embeddings, embeddings_exist)
from src.retriever import retrieve_top_chunks
from src.answer_generator import generate_answer
from src.reranker import load_reranker, rerank_chunks

DOC_PATH = "data/sample_documents/raspberry_pi_pico_with_datasheet.pdf" # document to use
EMBEDDING_STORE_DIR = "data/embedding_store" # where to store chunks and embeddings
ANSWER_MODE = "ollama" # basic/ollama
# Similarity threshold temporarily disabled for retrieval evaluation.
# SIMILARITY_THRESHOLD = 0.4 # minimum cosine similarity score required to proceed

if __name__ == "__main__":

    model = load_embedding_model()
    reranker = load_reranker()

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

    while True:
        question = input("\nAsk a question about the document, or type 'exit' to quit: ")

        if question.lower() == "exit":
            print("Goodbye.")
            break

        print("\n==================================================")
        print("\nQuestion:", question)

        # Step 1: Retrieve 10 chunks using cosine similarity
        initial_results = retrieve_top_chunks(question, chunks, chunk_embeddings, model, top_k=10)

        if not initial_results:
            print("\nNo chunks retrieved.")
            continue

        # Step 2: Print initial retrieval results
        print("\nInitial retrieval results:")

        for result in initial_results[:5]:
            print(
                f"Page {result['page_number']}, "
                f"Chunk {result['chunk_id']}, "
                f"Cosine: {result['score']:.3f}"
            )

        # Step 3: Rerank the 10 candidates and keep the best 3
        results = rerank_chunks(question, initial_results, reranker, top_k=3)
        
        # Step 4: Print reranked results
        print("\nAfter reranking:")

        for result in results:
            print(f"- Page {result['page_number']}, Chunk {result['chunk_id']}, "
                  f"Cosine score: {result['score']:.3f}, "
                  f"Rerank score: {result['rerank_score']:.3f}")
            
        # Step 5: Generate an answer using the reranked chunks
        answer = generate_answer(question, results, mode=ANSWER_MODE)
        print(f"\nAnswer:\n{answer}")