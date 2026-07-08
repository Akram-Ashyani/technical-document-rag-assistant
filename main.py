from src.document_loader import load_document
from src.text_chunker import chunk_pdf_pages

DOC_PATH = "data/sample_documents/raspberry_pi_pico_with_datasheet.pdf"

if __name__ == "__main__":
    pages = load_document(DOC_PATH)
    chunks = chunk_pdf_pages(pages)

    print("Number of pages:", len(pages))
    print("Number of chunks:", len(chunks))

    for chunk in chunks[:5]:
        print("\n--------------------")
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print(chunk["text"][:500])