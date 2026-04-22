from vector_store import FaissVectorStore

class BaselineRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vector_store = FaissVectorStore()
        
    def add_chunks(self, chunks_with_embeddings):
        """Adds chunks to the FAISS store."""
        self.vector_store.add_chunks(chunks_with_embeddings)
            
    def save(self, save_dir: str):
        """Saves current index."""
        self.vector_store.save(save_dir)
        
    def load(self, save_dir: str) -> bool:
        """Loads index into FAISS store."""
        return self.vector_store.load(save_dir)
        
    def retrieve(self, query: str, top_k: int = 3):
        """Retrieves using the FAISS vector store backing."""
        if self.vector_store.index.ntotal == 0:
            print("[WARNING] Vector store is empty. No chunks to retrieve.")
            return []
            
        query_emb = self.embedder.get_embedding(query)
        top_results = self.vector_store.retrieve_top_k(query_emb, top_k)
        
        # Obey DEBUG flag logging rules
        from config import DEBUG
        if DEBUG:
            print(f"\n[DEBUG] === Retrieval Results for: '{query}' ===")
            for i, res in enumerate(top_results):
                meta = res['metadata']
                yr = meta.get('year', 'Unknown')
                print(f"Rank {i+1} | L2 Dist: {res['score']:.4f} | Source: {meta['source']} ({yr}, Page {meta['page']})")
                print(f"  Snippet: {res['text'][:150]}...\n")
                
        return top_results
