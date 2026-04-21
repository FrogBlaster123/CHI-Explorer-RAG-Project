import os
import json
import pickle
from typing import List, Dict
import chromadb
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

persist_directory = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
bm25_path = os.path.join(persist_directory, "bm25_index.pkl")
chunks_path = os.path.join(persist_directory, "bm25_chunks.json")

class RetrievalStorage:
    def __init__(self):
        os.makedirs(persist_directory, exist_ok=True)
        
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True} 
        )
        
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="chi_papers",
            embedding_function=self.embeddings,
        )
        
        # BM25 Storage
        self.bm25 = None
        self.raw_chunks = []
        self._load_bm25()

    def _load_bm25(self):
        if os.path.exists(bm25_path) and os.path.exists(chunks_path):
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.raw_chunks = json.load(f)

    def _save_bm25(self):
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.raw_chunks, f)

    def ingest_chunks(self, chunks: List[Dict]):
        """Adds to Vector Store and Rebuilds BM25 index."""
        if not chunks: return
        
        ids = [f'{c["year"]}_{c["title"].replace(" ","_")}_{i}' for i, c in enumerate(chunks)]
        texts = [c["text"] for c in chunks]
        metadatas = [{"title": c["title"], "year": c["year"], "section": c["section"]} for c in chunks]

        # 1. Add to Chroma Vector Store
        print("Adding to ChromaDB...")
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        # 2. Build / Update BM25 Sparse Index
        print("Rebuilding BM25 Index...")
        self.raw_chunks.extend([{"id": _id, "text": t, "metadata": m} for _id, t, m in zip(ids, texts, metadatas)])
        
        tokenized_corpus = [doc["text"].lower().split(" ") for doc in self.raw_chunks] 
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25
        self._save_bm25()
        print(f"Successfully ingested {len(chunks)} chunks.")
