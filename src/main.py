import os
import argparse
from config import DEBUG
from pdf_processor import get_all_pdf_paths, extract_text_page_by_page
from chunker import chunk_text
from embedder import Embedder
from retriever import BaselineRetriever
from llm_client import SimpleGeminiClient

def get_dir_paths(dataset_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", dataset_name)
    index_dir = os.path.join(base_dir, "data", "indexes", dataset_name)
    return data_dir, index_dir

def build_vector_store(dataset_name, embedder, retriever):
    data_dir, index_dir = get_dir_paths(dataset_name)
    
    pdf_paths = get_all_pdf_paths(data_dir)
    if not pdf_paths:
        print(f"[ERROR] No PDFs found to process in {data_dir}.")
        return False
        
    print(f"\nFound {len(pdf_paths)} PDF(s) in {dataset_name}. Indexing incrementally...")
    
    total_pages_processed = 0
    total_chunks_created = 0
    
    for pdf_path in pdf_paths:
        file_name = os.path.basename(pdf_path)
        page_iter = extract_text_page_by_page(pdf_path)
        chunk_iter = chunk_text(page_iter, pdf_path)
        
        batch_texts = []
        batch_records = []
        
        last_page_processed = 0
        file_chunks = 0
        
        for chunk in chunk_iter:
            batch_texts.append(chunk["text"])
            batch_records.append(chunk)
            
            file_chunks += 1
            last_page_processed = max(last_page_processed, chunk["metadata"]["page"])
            
            if len(batch_texts) >= 32:
                embeddings = embedder.get_embeddings_batch(batch_texts)
                for rec, emb in zip(batch_records, embeddings):
                    rec["embedding"] = emb
                retriever.add_chunks(batch_records)
                batch_texts = []
                batch_records = []
                
        if batch_texts:
            embeddings = embedder.get_embeddings_batch(batch_texts)
            for rec, emb in zip(batch_records, embeddings):
                rec["embedding"] = emb
            retriever.add_chunks(batch_records)
            
        total_chunks_created += file_chunks
        total_pages_processed += last_page_processed
        print(f"  -> '{file_name}': Processed up to page {last_page_processed}, generated {file_chunks} chunks.")
            
    print(f"\nIndex built! Metrics:")
    print(f"- Total Pages Processed: ~{total_pages_processed}")
    print(f"- Total Chunks Created: {total_chunks_created}")
    print(f"- FAISS Store Size: {retriever.vector_store.index.ntotal}")
    
    retriever.save(index_dir)
    return True

def run_query(query, dataset_name, retriever, llm_client):
    data_dir, index_dir = get_dir_paths(dataset_name)
    
    print(f"Loading index for '{dataset_name}'...")
    if not retriever.load(index_dir):
        print("[ERROR] Index could not be loaded. Please run --mode index first.")
        return
        
    print(f"\n{'='*50}\n[QUERY]: {query}\n{'='*50}")
    chunks = retriever.retrieve(query, top_k=3)
    
    if DEBUG:
        print(f"\n[DEBUG] Pushing {len(chunks)} chunks to GEMINI...")
        
    answer = llm_client.generate_answer(query, chunks)
    print(f"\n[FINAL GROUNDED ANSWER]:\n{answer}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portable Baseline RAG System")
    parser.add_argument("--mode", type=str, required=True, choices=["index", "query"], help="Mode of operation")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset folder e.g. 'chi_papers'")
    parser.add_argument("--query", type=str, default="What are the main interface challenges observed?", help="Query when in query mode")
    args = parser.parse_args()

    print(f"Initializing RAG System in {args.mode.upper()} mode for dataset: {args.dataset}")
    embedder = Embedder()
    retriever = BaselineRetriever(embedder)
    
    if args.mode == "index":
        build_vector_store(args.dataset, embedder, retriever)
    elif args.mode == "query":
        llm = SimpleGeminiClient()
        run_query(args.query, args.dataset, retriever, llm)
