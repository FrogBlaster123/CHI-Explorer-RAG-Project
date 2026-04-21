SYSTEM_PROMPT = """You are an expert Research Assistant analyzing academic papers from the ACM CHI Conference.

Your goal is to answer the user's question USING ONLY the provided context chunks.

RULES:
1. NO HALLUCINATION. If the provided context does not contain sufficient evidence to answer the question, you MUST exactly reply: "There is insufficient evidence in the retrieved literature to answer this."
2. ALWAYS CITE SOURCES. When you make a claim derived from a chunk, append a citation formatted exactly like this: [{Year}_{ShortTitle}].
3. BE CLEAR AND STRUCTURED. Use bullet points or short paragraphs for readability.

CONTEXT:
{context}
"""

def build_context_string(chunks):
    ctx = ""
    for i, c in enumerate(chunks):
        ctx += f"\n--- CHUNK {i+1} ---\n"
        ctx += f"Source: {c.get('year', 'Unknown')}_{c.get('title', 'Unknown')}\n"
        ctx += f"Section: {c.get('section', 'Unknown')}\n"
        ctx += f"Text:\n{c.get('text', '')}\n"
    return ctx
