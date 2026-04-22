# CHI-Explorer: Memory-Safe RAG Architecture

A decoupled, Retrieval-Augmented Generation (RAG) system built to parse, embed, and query massive 2GB+ documentation pipelines (like entire years of ACM CHI research proceedings) natively on standard hardware without Out-of-Memory (OOM) crashes.

## 🌟 Why This Project Exists

Traditional naïve RAG scripts load the entire PDF into RAM simultaneously to process chunking. This instantly crashes low-spec machines when you drop a 15,000+ page PDF archive into the directory. 

**CHI-Explorer** solves this by:
1. **Lazy Loading:** Iterates massive PDFs natively on a page-by-page generator cycle. It only ever holds a single page in RAM at a time.
2. **Local CPU Embedding:** Converts all text chunks into Dense Vectors safely on your CPU using PyTorch and `sentence-transformers/all-MiniLM-L6-v2`.
3. **Decoupled Architecture:** Employs persistent disk indexing (`FAISS`). You endure the indexing process just once, allowing you to instantly query the vector store infinity times later without ever loading the raw PDFs again.
4. **Resilient AI Generation:** Connects directly to Google's Gemini Models, enforcing strictly grounded, non-hallucinated citations natively embedded with the source file name and origin year.

---

## 🛠️ Environment Setup

1. **Install Requirements:**
Ensure your Python environment is set up, then install the required core packages:
```bash
pip install -r requirements.txt
```

2. **API Keys:**
Create a file named precisely `.env` in the root of the project directory. Mount your Gemini API key inside it so it can securely handle query generation.
```env
GEMINI_API_KEY=your_actual_api_key_here
```

---

## 🚀 How To Run The System

The system operates strictly on a two-phase architecture to isolate the massive computation load. 

### Phase 1: Indexing (Run it only once)
Whenever you start a new topic, create a folder in your `data/` directory (e.g. `data/chi_papers/`) and dump all your `.pdf` proceedings globally into it. 

Run the Index Mode to build the FAISS disk map:
```bash
python src/main.py --mode index --dataset chi_papers
```
*(Note: Because this reads and mathematically chunks thousands of pages, it may take 10-20 minutes locally on your CPU. However, it will cleanly export `faiss_index.bin` and save the metadata for instant rebooting).*

### Phase 2: Querying (Run it infinitely)
Now that the heavy lifting is done, you can instantly query your specific dataset. The system bypasses data extraction entirely, jumping straight to mathematical vector matching and AI generation.

```bash
python src/main.py --mode query --dataset chi_papers --query "Explain Growing Blip's Content"
```

### Phase 3: Visual Interface (Streamlit)
For a better experience, you can interact with the RAG system using the rich, graphical web interface rather than the CLI. The UI allows you to select your indexed datasets, adjust the top K retrieved results, and view formatted sources cleanly in expandable cards.

Simply run:
```bash
streamlit run app.py
```
This will automatically launch the web server and open the interface in your browser at `http://localhost:8501`.

## 📁 Repository Structure
* `app.py` - The complete graphical web application (Streamlit).
* `data/{dataset_name}/` - Drop your raw PDFs here.
* `data/indexes/{dataset_name}/` - The system generates your FAISS `.bin` and Metadata maps here.
* `src/main.py` - Core execution CLI orchestrator.
* `src/pdf_processor.py` - Handles the memory-safe lazy-loading PyMuPDF generator logic.
* `src/vector_store.py` - Serializes and manages the persistent FAISS local architecture.
* `src/config.py` - Global tuning variables.
