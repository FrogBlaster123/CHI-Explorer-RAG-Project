import os
import sys
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import CrossEncoder
from utils.logger import logger
from retrieval.vector_store import RetrievalStorage

class HybridRetriever:
    def __init__(self, storage: RetrievalStorage = None):
        if storage is None:
            self.storage = RetrievalStorage()
        else:
            self.storage = storage
            
        logger.info("Loading Cross-Encoder Reranker...")
        # BAAI/bge-reranker-base is highly performant
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        
    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        if not scores: return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s == 0:
            return [1.0 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _soft_boost(self, chunk: Dict, query: str, base_score: float) -> float:
        """Boosts chunk score if section matches query inferred intent."""
        query_lower = query.lower()
        section = chunk.get("section", "Unknown").lower()
        
        # Super naive intent matching
        if "method" in query_lower or "how did they" in query_lower:
            if section == "methodology": return base_score * 1.2
        if "result" in query_lower or "find" in query_lower:
            if section == "results": return base_score * 1.2
        if "limit" in query_lower or "future" in query_lower:
            if section == "discussion": return base_score * 1.2
            
        return base_score

    def retrieve(self, query: str, top_k: int = 5, hybrid_alpha: float = 0.7) -> List[Dict]:
        """
        Executes true hybrid retrieval combining Vector and Keyword, Min-Max normalizes,
        applies section soft-boosting, and finally cross-encodes via a ReRanker.
        """
        logger.info(f"Querying: '{query}'")
        
        # --- 1. Dense Retrieval (Semantic) ---
        dense_results_raw = self.storage.vectorstore.similarity_search_with_score(query, k=20)
        dense_chunks = []
        dense_scores = []
        for doc, dist in dense_results_raw:
            c = doc.metadata.copy()
            c["text"] = doc.page_content
            dense_chunks.append(c)
            dense_scores.append(1.0 / (1.0 + dist))
            
        logger.log_retrieval_stage("Vector Search", [{"id": idx, "score": s} for idx, s in enumerate(dense_scores)])

        # --- 2. Sparse Retrieval (Keyword) ---
        sparse_chunks = []
        sparse_scores = []
        if self.storage.bm25:
            tokenized_query = query.lower().split(" ")
            raw_bm25_scores = self.storage.bm25.get_scores(tokenized_query)
            # Get top 20
            top_bm25_indices = sorted(range(len(raw_bm25_scores)), key=lambda i: raw_bm25_scores[i], reverse=True)[:20]
            
            for i in top_bm25_indices:
                sparse_chunks.append({
                    "title": self.storage.raw_chunks[i]["metadata"]["title"],
                    "year": self.storage.raw_chunks[i]["metadata"]["year"],
                    "section": self.storage.raw_chunks[i]["metadata"]["section"],
                    "text": self.storage.raw_chunks[i]["text"]
                })
                sparse_scores.append(raw_bm25_scores[i])
                
        logger.log_retrieval_stage("Keyword Search", [{"id": idx, "score": s} for idx, s in enumerate(sparse_scores)])

        # --- 3. Normalize & Merge ---
        norm_dense = self._min_max_normalize(dense_scores)
        norm_sparse = self._min_max_normalize(sparse_scores)
        
        combined_map = {}
        for i, chunk in enumerate(dense_chunks):
            key = hash(chunk["text"])
            combined_map[key] = {"chunk": chunk, "dense": norm_dense[i], "sparse": 0.0}
            
        for i, chunk in enumerate(sparse_chunks):
            key = hash(chunk["text"])
            if key in combined_map:
                combined_map[key]["sparse"] = norm_sparse[i]
            else:
                combined_map[key] = {"chunk": chunk, "dense": 0.0, "sparse": norm_sparse[i]}
                
        hybrid_candidates = []
        for key, vals in combined_map.items():
            base_score = (hybrid_alpha * vals["dense"]) + ((1 - hybrid_alpha) * vals["sparse"])
            final_score = self._soft_boost(vals["chunk"], query, base_score)
            hybrid_candidates.append({
                "chunk": vals["chunk"],
                "score": final_score
            })
            
        hybrid_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = hybrid_candidates[:20]
        
        logger.log_retrieval_stage("Hybrid Merged", [{"id": idx, "score": c["score"]} for idx, c in enumerate(top_candidates)])

        # --- 4. Cross-Encoder Reranking ---
        if not top_candidates: return []
        
        pairs = [[query, c["chunk"]["text"]] for c in top_candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for i, score in enumerate(rerank_scores):
            top_candidates[i]["rerank_score"] = float(score)
            
        top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_top = top_candidates[:top_k]
        
        logger.log_retrieval_stage("Cross-Encoder Reranked", [{"id": idx, "score": c["rerank_score"]} for idx, c in enumerate(final_top)])
        
        return [c["chunk"] for c in final_top]
