import os
import re
import fitz
import glob
from pathlib import Path

def sanitize_filename(name: str) -> str:
    """Removes special characters, keeps alphanumerics and spaces, converts to underscores."""
    # Remove "Papers:" or "Papers" prefix sometimes found in session titles
    name = re.sub(r'^Papers\s*:\s*', '', name, flags=re.IGNORECASE)
    
    # Keep only alphanumeric and space
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    # Replace spaces with underscores and strip trailing/leading
    name = re.sub(r'\s+', '_', name).strip('_')
    
    # Limit length just in case the extraction caught a paragraph
    if len(name) > 60:
        name = name[:60]
    return name

def save_session(doc, year, title, start_page, end_page, output_dir):
    """Extracts pages [start_page, end_page] from doc and saves to a new PDF."""
    if start_page > end_page:
        return
        
    # Create empty PDF
    out_pdf = fitz.Document()
    # Insert the page range exactly
    out_pdf.insert_pdf(doc, from_page=start_page, to_page=end_page)
    
    # Ensure it's not totally empty (0 pages)
    if len(out_pdf) > 0:
        out_filename = f"{year}_{title}.pdf"
        out_path = os.path.join(output_dir, out_filename)
        out_pdf.save(out_path)
        print(f"  -> Saved: {out_filename} ({len(out_pdf)} pages)")
    out_pdf.close()

def split_proceedings(pdf_path: str, output_base_dir: str):
    """Splits a large proceedings PDF into session-specific PDFs."""
    print(f"\nProcessing {pdf_path}...")
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    
    # Extract year from filename (assume format YYYY_...)
    year_match = re.match(r'^(\d{4})_', filename)
    year = year_match.group(1) if year_match else "UnknownYear"
    
    # Output structure: data/{year}/sessions
    output_dir = os.path.join(output_base_dir, year, "sessions")
    os.makedirs(output_dir, exist_ok=True)
    
    current_session_title = None
    session_start_page = -1
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Search for "SESSION:"
        match = re.search(r'SESSION:\s*(.*)', text, re.IGNORECASE)
        
        if match:
            new_title_raw = match.group(1).strip()
            # If the title is split across lines, grab the first line or up to next newline
            new_session_title = new_title_raw.split('\n')[0].strip()
            
            sanitized_title = sanitize_filename(new_session_title)
            if not sanitized_title:
                sanitized_title = f"Unknown_Session_{page_num}"
                
            # If we are already tracking a session, save it
            if current_session_title and session_start_page != -1:
                # Save previous session from its start up to the page *before* this new session
                save_session(doc, year, current_session_title, session_start_page, page_num - 1, output_dir)
                
            current_session_title = sanitized_title
            session_start_page = page_num
            print(f"Found session: '{current_session_title}' at page {page_num}")
            
    # Save the very last session
    if current_session_title and session_start_page != -1:
        save_session(doc, year, current_session_title, session_start_page, len(doc) - 1, output_dir)
        
    print(f"Finished processing {filename}. Sessions saved to: {output_dir}")
    doc.close()

if __name__ == "__main__":
    # Expecting to run from root or inside src/utils
    # We will compute data dir based on relative path
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(scripts_dir, "..", "..", "Data")
    output_base_dir = data_dir
    
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
    
    for pdf in pdf_files:
        split_proceedings(pdf, output_base_dir)
