import os
import faiss
import pickle
import numpy as np

class FaissVectorStore:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        # Using IndexFlatL2 for simplicity. 
        # sentence-transformers output normalizes well, so L2 distance acts proportionally like Cosine Similarity.
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata_store = []
        
    def add_chunks(self, chunks_with_embeddings):
        if not chunks_with_embeddings:
            return
            
        embeddings = []
        for chunk in chunks_with_embeddings:
            embeddings.append(chunk["embedding"])
            self.metadata_store.append({
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            })
            
        # FAISS requires float32 contiguous arrays
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        faiss_path = os.path.join(save_dir, "faiss_index.bin")
        meta_path = os.path.join(save_dir, "metadata.pkl")
        
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
            
        print(f"[INFO] Vector store saved to {save_dir}")
        
    def load(self, save_dir: str):
        faiss_path = os.path.join(save_dir, "faiss_index.bin")
        meta_path = os.path.join(save_dir, "metadata.pkl")
        
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            print(f"[ERROR] Index files not found in {save_dir}")
            return False
            
        self.index = faiss.read_index(faiss_path)
        self.embedding_dim = self.index.d
        
        with open(meta_path, "rb") as f:
            self.metadata_store = pickle.load(f)
            
        print(f"[INFO] Loaded vector store from {save_dir}. Contains {self.index.ntotal} chunks.")
        return True
        
    def retrieve_top_k(self, query_embedding: list, top_k: int = 3):
        if self.index.ntotal == 0:
            return []
            
        q_emb = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: 
                continue # FAISS returns -1 if not enough results
            
            chunk_data = self.metadata_store[idx]
            results.append({
                "score": float(dist), # Note: For L2, smaller distance means more similar
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"]
            })
            
        return results
