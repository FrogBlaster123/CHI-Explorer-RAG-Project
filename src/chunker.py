import os
import re
from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

def chunk_text(page_iterator, source_file):
    """
    Takes a generator of (page_num, text) and yields chunks of text with metadata.
    Processes page-by-page strictly. A chunk never spans two pages to ensure 
    citations (source + page) are 100% accurate.
    """
    file_name = os.path.basename(source_file)
    
    # Try to infer year from filename
    year_match = re.search(r'(19|20)\d{2}', file_name)
    year = int(year_match.group()) if year_match else "Unknown"
    
    step_size = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    if step_size <= 0: 
        step_size = CHUNK_SIZE_CHARS # safety boundary
        
    for page_num, text in page_iterator:
        start = 0
        text_length = len(text)
        
        if text_length == 0:
            continue
            
        while start < text_length:
            end = start + CHUNK_SIZE_CHARS
            chunk_slice = text[start:end]
            
            # Standard output format structure
            yield {
                "text": chunk_slice,
                "metadata": {
                    "source": file_name,
                    "page": page_num,
                    "year": year
                }
            }
            
            start += step_size
