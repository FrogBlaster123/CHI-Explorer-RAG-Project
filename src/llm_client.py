import os
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SimpleGeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY environment variable not set. Generation will fail or return defaults.")
            
        genai.configure(api_key=api_key)
        # To fix the 404 error, we fall back to a universally supported model name for v1beta endpoints.
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        if not retrieved_chunks:
            return "There is insufficient evidence in the retrieved literature to answer this."
            
        # Build strict context string enforcing explicit citation metadata
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            year = chunk["metadata"].get("year", "Unknown")
            context_str += f"\n--- CHUNK {i+1} [{source}, Year: {year}, Page {page}] ---\n"
            context_str += chunk["text"] + "\n"
            
        prompt = f"""You are an expert Research Assistant. 
Answer the following user query USING ONLY the provided context chunks.

RULES:
1. NO HALLUCINATION. If the context does not contain sufficient evidence to answer the query, exactly state: "There is insufficient evidence in the retrieved literature to answer this."
2. ALWAYS CITE SOURCES. You must include the EXACT source filename in your responses (e.g., [2022_large.pdf, Year: 2022, Page 42]).
3. Be clear and structured. Use bullet points when helpful.

CONTEXT:
{context_str}

USER QUERY: {query}
"""
        max_retries = 3
        base_delay = 10
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Try to parse the required wait time from the error message
                        match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", error_str)
                        if match:
                            delay = float(match.group(1)) + 1.0
                        else:
                            delay = base_delay * (2 ** attempt)
                            
                        print(f"[WARNING] Rate limit exceeded. Retrying in {delay:.1f} seconds (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(delay)
                    else:
                        return f"Error during LLM generation after {max_retries} retries: {error_str}"
                else:
                    return f"Error during LLM generation: {error_str}"
