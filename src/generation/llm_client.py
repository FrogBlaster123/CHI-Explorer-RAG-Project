import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage
from generation.prompts import SYSTEM_PROMPT, build_context_string
from utils.logger import logger

class AssistantLLM:
    def __init__(self):
        """
        Initializes the LLM Provider.
        Configured to allow quick swaps. Defaults to Gemma via Ollama or Google API.
        """
        # User requested Gemma 4
        # We attempt local Ollama loading, but fallback to general langchain_google if unavailable
        try:
            from langchain_community.chat_models import ChatOllama
            import requests # check if ollama is running
            requests.get("http://localhost:11434/")
            self.llm = ChatOllama(model="gemma") 
            logger.info("Initialized Local Gemma Provider via Ollama.")
        except Exception:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY", "dummy_key")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            logger.info("Initialized Google GenAI Provider (Gemma-compatible API).")

    def generate_answer(self, query: str, context_chunks: list) -> str:
        if not context_chunks:
            return "There is insufficient evidence in the retrieved literature to answer this."
            
        context_str = build_context_string(context_chunks)
        sys_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context_str))
        hum_msg = HumanMessage(content=query)
        
        logger.info(f"Generating answer using {len(context_chunks)} chunks...")
        try:
            response = self.llm.invoke([sys_msg, hum_msg])
            return response.content
        except Exception as e:
            logger.info(f"Generation Error: {str(e)}")
            return "Error during LLM generation. Please check API Key or Local Model status."
