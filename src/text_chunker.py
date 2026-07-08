def overlapping_chunking(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into chunks with overlapping characters.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    
    if overlap < 0:
        raise ValueError("Overlap must be greater than or equal to 0.")
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size.")
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def chunk_pdf_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """
    Split PDF pages into chunks with overlapping characters.
    """
    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = overlapping_chunking(page_text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            all_chunks.append({
                "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                "page_number": page_number,
                "text": chunk_text
            })

    return all_chunks