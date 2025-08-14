import wikipediaapi
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import time
import logging
from constants import *

# Add this below your imports
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_wikipedia_page(topic: str):
    try:
        wiki = wikipediaapi.Wikipedia(
            language=WIKIPEDIA_LANG,
            user_agent=USER_AGENT
        )
        page = wiki.page(topic)
        if not page.exists():
            raise ValueError(f"Page for '{topic}' not found on Wikipedia.")
        return page.text
    except Exception as e:
        logger.error(f"Wikipedia API error: {e}")
        time.sleep(2)  # Simple rate limit handling
        raise

def deduplicate_chunks(chunks):
    """Simple hash-based deduplication"""
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash(chunk)
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    return unique_chunks

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return splitter.split_text(text)

def build_faiss_index(chunks):
    chunks = deduplicate_chunks(chunks)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

if __name__ == "__main__":
    try:
        topic = "Quantum computing"
        text = fetch_wikipedia_page(topic)
        chunks = chunk_text(text)
        index, model = build_faiss_index(chunks)
        logger.info(f"Indexed {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error in main: {e}")