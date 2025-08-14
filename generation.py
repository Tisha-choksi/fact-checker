from transformers import pipeline
from constants import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_qa_pipeline = None

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        logger.info("Loading QA pipeline...")
        _qa_pipeline = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            device="cpu"  # Change to "cuda" if you have GPU
        )
    return _qa_pipeline

def generate_answer(query: str, context: str):
    try:
        truncated_context = context[:MAX_CONTEXT_CHARS]
        answer = get_qa_pipeline()(
            f"question: {query} context: {truncated_context}"
        )[0]['generated_text']
        return answer.strip().replace("\n", " ")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "Could not generate answer."