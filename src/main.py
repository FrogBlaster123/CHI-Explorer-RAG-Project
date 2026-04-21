import os
from config import DEBUG
from pdf_processor import get_all_pdf_paths, extract_text_page_by_page
from chunker import chunk_text
from embedder import Embedder
from retriever import BaselineRetriever
from llm_client import SimpleGeminiClient

def build_vector_store(input_path, embedder, retriever):
    pdf_paths = get_all_pdf_paths(input_path)
    if not pdf_paths:
        print("[ERROR] No PDFs found to process.")
        return
        
    print(f"\nFound {len(pdf_paths)} PDF(s). Processing incrementally...")
    
    for pdf_path in pdf_paths:
        # Step 1: Read page-by-page generator safely
        page_iter = extract_text_page_by_page(pdf_path)
        
        # Step 2: Chunk safely
        chunk_iter = chunk_text(page_iter, pdf_path)
        
        # Step 3: Embed & Store
        # To optimize speed slightly while remaining simple, we batch queries to the embedder 
        batch_texts = []
        batch_records = []
        
        for chunk in chunk_iter:
            batch_texts.append(chunk["text"])
            batch_records.append(chunk)
            
            if len(batch_texts) >= 32: # memory-safe batch limit
                embeddings = embedder.get_embeddings_batch(batch_texts)
                for rec, emb in zip(batch_records, embeddings):
                    rec["embedding"] = emb
                retriever.add_chunks(batch_records)
                batch_texts = []
                batch_records = []
                
        # Drain the rest of the file
        if batch_texts:
            embeddings = embedder.get_embeddings_batch(batch_texts)
            for rec, emb in zip(batch_records, embeddings):
                rec["embedding"] = emb
            retriever.add_chunks(batch_records)
            
    print(f"Index built! Total chunks housed safely in memory: {len(retriever.vector_store)}")

def run_query(query, retriever, llm_client):
    print(f"\n{'='*50}\n[QUERY]: {query}\n{'='*50}")
    
    # 1. Retrieve Phase
    chunks = retriever.retrieve(query, top_k=3)
    
    if DEBUG:
        print(f"\n[DEBUG] Pushing {len(chunks)} chunks to GEMINI...")
        
    # 2. Generation Phase
    answer = llm_client.generate_answer(query, chunks)
    
    print(f"\n[FINAL GROUNDED ANSWER]:\n{answer}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline RAG System")
    parser.add_argument("--query", type=str, default="What are the main interface challenges observed?", help="The academic question to ask")
    # Setting default to one of the known files for a quick test if executed blindly
    parser.add_argument("--data", type=str, default=os.path.join(os.path.dirname(__file__), "..", "Data", "2021_EyeTracking_Interface.pdf"), help="File or Folder path to parse")
    
    args = parser.parse_args()

    print("Initializing Baseline RAG System...")
    
    # Load core generic pieces
    embedder = Embedder()
    retriever = BaselineRetriever(embedder)
    llm = SimpleGeminiClient()
    
    # Run Orchestration
    build_vector_store(args.data, embedder, retriever)
    run_query(args.query, retriever, llm)
