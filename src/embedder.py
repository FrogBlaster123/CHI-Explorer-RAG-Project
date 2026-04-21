from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the baseline embedding model.
        We use 'all-MiniLM-L6-v2' as it is heavily optimized, lightweight, 
        and fast enough to execute entirely on CPU while extracting big PDFs.
        """
        self.model = SentenceTransformer(model_name)
        
    def get_embedding(self, text: str) -> List[float]:
        """Returns the embedding for a single text string as a float list."""
        return self.model.encode(text).tolist()
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Returns embeddings for a batch of texts to speed up memory execution."""
        return self.model.encode(texts, show_progress_bar=False).tolist()
