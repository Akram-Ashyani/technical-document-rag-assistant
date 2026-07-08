import pymupdf

def load_document(doc_path: str) -> list[dict]:
    """
    Load a PDF document and return a list of page dictionaries.
    """
    document = pymupdf.open(doc_path)
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text", sort=True)
        pages.append({"page_number": page_number, "text": text.strip()})

    return pages