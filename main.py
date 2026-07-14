from src.document_loader import load_document
from src.text_chunker import chunk_pdf_pages
from src.embeddings import load_embedding_model
from src.retriever import retrieve_top_chunks

DOC_PATH = "data/sample_documents/raspberry_pi_pico_with_datasheet.pdf"

if __name__ == "__main__":
    pages = load_document(DOC_PATH)
    chunks = chunk_pdf_pages(pages)

    print("Number of pages:", len(pages))
    print("Number of chunks:", len(chunks))

    model = load_embedding_model()

    test_questions = [
        "What is the keep-out area for the antenna?",
        "What chip is used in Raspberry Pi Pico W?",
        "How can Raspberry Pi Pico W be powered?",
        "What are the recommended operating conditions?",
        "How much flash memory does Raspberry Pi Pico W have?"
    ]

    for question in test_questions:
        results = retrieve_top_chunks(question, chunks, model, top_k=3)

        print("\n==================================================")
        print("\nQuestion:", question)

        for result in results:
            print("\n--------------------")
            print("Chunk ID:", result["chunk_id"])
            print("Page:", result["page_number"])
            print("Score:", result["score"])
            print(result["text"][:1000])