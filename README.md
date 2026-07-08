# technical-document-rag-assistant
A RAG-powered Python assistant for technical PDFs, enabling document question answering with source references.

## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for technical PDF documents. 

## Current Features

- Load PDF documents page by page
- Extract text using PyMuPDF
- Split page text into overlapping chunks
- Keep page numbers and chunk IDs for source tracking

## Sample Document

This project uses the public Raspberry Pi Pico W Datasheet as a sample technical PDF for testing the RAG pipeline. 
The PDF is **not included in this repository**. To test the project, download the datasheet from the official Raspberry Pi documentation website and place it here:

`data/sample_documents/raspberry_pi_pico_with_datasheet.pdf`