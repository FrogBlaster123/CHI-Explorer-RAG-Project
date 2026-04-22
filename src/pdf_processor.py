import os
import fitz # PyMuPDF
from config import DEBUG, MAX_PAGES_PER_DOC

def extract_text_page_by_page(pdf_path):
    """
    Generator that extracts text from a PDF page-by-page.
    Yields tuples of (page_num, text). 
    This generator pattern prevents loading entire massive PDFs into RAM at once.
    """
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    
    total_pages = len(doc)
    
    # Calculate bounds based on config
    pages_to_read = min(total_pages, MAX_PAGES_PER_DOC) if MAX_PAGES_PER_DOC else total_pages
    
    if DEBUG:
        print(f"[DEBUG] Processing '{file_name}': extracting {pages_to_read} of {total_pages} total pages.")
        
    for page_num in range(pages_to_read):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            # Simple whitespace normalization (replaces newlines and multiple spaces with a single space)
            text = " ".join(text.split())
            
            if text.strip():
                # Yield page_num + 1 so citations are 1-indexed like standard pages
                yield (page_num + 1, text)
                
            # Log progress for massively long documents without spamming
            if DEBUG and (page_num + 1) % 1000 == 0:
                print(f"  ...processed page {page_num + 1}/{total_pages} in {file_name}...")
                
        except Exception as e:
            if DEBUG:
                print(f"[WARNING] Skipping page {page_num + 1} due to error: {str(e)}")
            continue # Ensure we never terminate the iteration early
            
    doc.close()

def get_all_pdf_paths(input_path):
    """
    Determines if input is a single file or a folder representing the dataset.
    Returns a list of valid PDF absolute paths.
    """
    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        return [input_path]
    elif os.path.isdir(input_path):
        paths = []
        for file in os.listdir(input_path):
            if file.lower().endswith(".pdf"):
                paths.append(os.path.join(input_path, file))
        return paths
    else:
        print(f"[WARNING] Input path '{input_path}' is neither a PDF file nor a directory.")
        return []
