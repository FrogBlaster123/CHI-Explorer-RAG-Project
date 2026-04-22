"""
Configuration for the Baseline RAG System.
"""

DEBUG = True

# We use character-based chunking sizing since it doesn't require a tokenizer.
# ~400 tokens is typically around 2000-3000 characters.
CHUNK_SIZE_CHARS = 2500 
CHUNK_OVERLAP_CHARS = 300

# Limits how many pages we read from a single PDF.
# Set to None to process all 18000+ pages properly.
MAX_PAGES_PER_DOC = None
