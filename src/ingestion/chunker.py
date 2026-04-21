from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

def chunk_segments(segments: List[Dict], chunk_size: int = 400, chunk_overlap: int = 50):
    """
    Applies Section-Aware Chunking: text is never split across two different sections.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    chunks = []
    for seg in segments:
        text = seg["text"]
        splits = splitter.split_text(text)
        for s in splits:
            if len(s.strip()) > 10: # Filter out extremely tiny chunks (artifacts usually)
                chunks.append({
                    "title": seg["title"],
                    "year": seg["year"],
                    "section": seg["section"],
                    "text": s.strip()
                })
    return chunks
