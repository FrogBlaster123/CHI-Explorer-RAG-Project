import os
import sys

# Hack path to allow internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.pdf_processor import process_pdf
from ingestion.chunker import chunk_segments
from retrieval.vector_store import RetrievalStorage
from utils.logger import logger

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

def run():
    logger.info("Starting Ingestion Pipeline...")
    storage = RetrievalStorage()
    all_chunks = []
    
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            logger.info(f"Processing {file}...")
            filepath = os.path.join(DATA_DIR, file)
            
            # Using max_pages to prevent massive mem overload on multi-gb PDFs
            segments = process_pdf(filepath, max_pages=15) 
            
            chunks = chunk_segments(segments)
            logger.info(f"Extracted {len(chunks)} chunks from {file}.")
            all_chunks.extend(chunks)
            
    if all_chunks:
        logger.info(f"Total chunks to ingest: {len(all_chunks)}. Sending to storage...")
        storage.ingest_chunks(all_chunks)
        logger.info("Ingestion Complete.")
    else:
        logger.info("No chunks found. Is the Data directory empty?")

if __name__ == "__main__":
    run()
