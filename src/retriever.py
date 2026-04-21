import numpy as np
from config import DEBUG

class BaselineRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        # Vector storage is an in-memory list: [(chunk_text, embedding_vector, metadata_dict)]
        self.vector_store = []
        
    def add_chunks(self, chunks_with_embeddings):
        """
        Adds pre-computed embeddings to the in-memory list.
        Expects a list of dicts: {"text": str, "embedding": List[float], "metadata": dict}
        """
        for item in chunks_with_embeddings:
            self.vector_store.append((
                item["text"],
                np.array(item["embedding"]),
                item["metadata"]
            ))
            
    def cosine_similarity(self, vec_a, vec_b):
        """Standard cosine similarity between two numpy vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
        
    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieves the top-k chunks that most strongly match the query.
        """
        if not self.vector_store:
            print("[WARNING] Vector store is empty. No chunks to retrieve.")
            return []
            
        query_emb = np.array(self.embedder.get_embedding(query))
        
        # Calculate scores computationally against memory pool
        scored_chunks = []
        for text, emb, meta in self.vector_store:
            score = self.cosine_similarity(query_emb, emb)
            scored_chunks.append({
                "score": float(score),
                "text": text,
                "metadata": meta
            })
            
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_chunks[:top_k]
        
        # Obey DEBUG flag logging rules
        if DEBUG:
            print(f"\n[DEBUG] === Retrieval Results for: '{query}' ===")
            for i, res in enumerate(top_results):
                print(f"Rank {i+1} | Score: {res['score']:.4f} | Source: {res['metadata']['source']} (Page {res['metadata']['page']})")
                print(f"  Snippet: {res['text'][:150]}...\n")
                
        return top_results
