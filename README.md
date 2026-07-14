# technical-document-rag-assistant

A RAG-powered Python assistant for technical PDFs, enabling document question answering with source references.

## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for technical PDF documents. 

The current version focuses on the retrieval part of RAG: loading a PDF, extracting text, splitting it into chunks, creating embeddings, and retrieving the most relevant chunks for a user question.

## Current Features

- Load PDF documents page by page
- Extract text using PyMuPDF
- Split page text into overlapping chunks
- Keep page numbers and chunk IDs for source tracking
- Create sentence embeddings using `sentence-transformers`
- Retrieve the most relevant chunks using cosine similarity
- Display chunk IDs, page numbers, similarity scores, and retrieved text

## Current Pipeline

```text
PDF document
→ page text extraction
→ overlapping text chunks
→ sentence embeddings
→ semantic similarity search
→ top relevant chunks with source metadata
```

## Sample Document

This project uses the public Raspberry Pi Pico W Datasheet as a sample technical PDF for testing the RAG pipeline. 
The PDF is **not included in this repository**. To test the project, download the datasheet from the official Raspberry Pi documentation website and place it here:

`data/sample_documents/raspberry_pi_pico_with_datasheet.pdf`

## Example Questions

```text
What is the keep-out area for the antenna?
What chip is used in Raspberry Pi Pico W?
How can Raspberry Pi Pico W be powered?
What are the recommended operating conditions?
How much flash memory does Raspberry Pi Pico W have?
```

## Example Retrieval Output

```text
Question: What is the keep-out area for the antenna?

Chunk ID: page_13_chunk_1
Page: 13
Score: 0.604

There is a cutout for the antenna (14mm × 9mm). If anything is placed close to the antenna, the effectiveness of the antenna is reduced.
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Note

The current version retrieves relevant document chunks only. Final natural-language answer generation will be added in a later version.