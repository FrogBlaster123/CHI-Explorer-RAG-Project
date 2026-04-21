import os
import re
import fitz  # PyMuPDF

def clean_text(text: str) -> str:
    """Aggressively clean extracted text for BM25 keyword matching."""
    text = re.sub(r'-\n', '', text) # joining hyphenated words
    text = re.sub(r'\n', ' ', text) # normalize line breaks
    text = re.sub(r'\s+', ' ', text) # squash multiple spaces
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    return text.strip()

def parse_filename(filename: str):
    """Parses standard format [Year]_[ShortTitle].pdf"""
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split('_', 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[0], parts[1].replace('_', ' ')
    return "Unknown", name_no_ext

def detect_section(block_text: str, font_size: float, is_bold: bool) -> str:
    """Uses Regex + Heuristics to detect sections. Fails gracefully to None."""
    text_upper = block_text.strip().upper()
    
    # 1. Regex/Exact Matching (Highest confidence)
    if re.match(r'^(1\.?\s*)?ABSTRACT', text_upper):
        return "Abstract"
    if re.match(r'^(1\.?\s*)?INTRODUCTION', text_upper):
        return "Introduction"
    if re.match(r'^([2-9]\.?\s*)?(METHODOLOGY|METHOD|METHODS|STUDY DESIGN)', text_upper):
        return "Methodology"
    if re.match(r'^([3-9]\.?\s*)?(RESULTS|FINDINGS)', text_upper):
        return "Results"
    if re.match(r'^([4-9]\.?\s*)?(DISCUSSION|CONCLUSION|LIMITATIONS)', text_upper):
        return "Discussion"
        
    # 2. Heuristics
    if len(block_text) < 100 and is_bold:
        if "METHOD" in text_upper: return "Methodology"
        if "RESULT" in text_upper: return "Results"
        if "DISCUSSION" in text_upper or "CONCLUSION" in text_upper: return "Discussion"
        
    return None

def process_pdf(pdf_path: str, max_pages: int = None):
    """Extracts text while preserving sections, bounding unmatched to 'Unknown'."""
    year, title = parse_filename(pdf_path)
    
    doc = fitz.open(pdf_path)
    current_section = "Unknown"  # Mandatory Default Guarantee
    segments = []
    current_chunk_text = ""
    
    num_pages = len(doc)
    if max_pages:
        num_pages = min(num_pages, max_pages)
        print(f"Limiting {title} to {num_pages} pages...")
        
    for page_num in range(num_pages):
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])
        
        for b in blocks:
            if b['type'] == 0:  # Text block
                block_text = ""
                is_bold = False
                max_font_size = 0.0
                
                for line in b["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                        if "bold" in span["font"].lower() or "black" in span["font"].lower():
                            is_bold = True
                
                clean_block = clean_text(block_text)
                if not clean_block: continue
                
                detected = detect_section(clean_block, max_font_size, is_bold)
                
                if detected and detected != current_section:
                    if current_chunk_text.strip():
                        segments.append({
                            "title": title,
                            "year": year,
                            "section": current_section,
                            "text": current_chunk_text.strip()
                        })
                    current_section = detected
                    current_chunk_text = clean_block + " " 
                else:
                    current_chunk_text += clean_block + " "
                    
    if current_chunk_text.strip():
        segments.append({
            "title": title,
            "year": year,
            "section": current_section, # This will fallback to 'Unknown' seamlessly if no sections were ever found.
            "text": current_chunk_text.strip()
        })
        
    return segments
