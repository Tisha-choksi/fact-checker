from retrieval import fetch_wikipedia_page, chunk_text, build_faiss_index
from generation import generate_answer
import numpy as np
import logging
from constants import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input(topic: str, query: str):
    if not topic.strip() or not query.strip():
        raise ValueError("Topic and query cannot be empty.")

def answer_question(topic: str, query: str):
    try:
        validate_input(topic, query)
        
        # Fetch and process text
        text = fetch_wikipedia_page(topic)
        chunks = chunk_text(text)
        
        # Build index and retrieve chunks
        index, model = build_faiss_index(chunks)
        query_embedding = model.encode(query)
        D, I = index.search(np.array([query_embedding]), k=3)
        
        # Generate answer
        context = " ".join([chunks[i] for i in I[0]])
        return generate_answer(query, context)
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print(answer_question("Quantum computing", "What is a qubit?"))