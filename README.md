
# Wikipedia Fact Checker

This project is a Wikipedia Fact Checker web application built with Streamlit. It allows users to ask questions about any Wikipedia topic and get AI-generated answers based on retrieved content from Wikipedia.

## About the Project
The app combines retrieval-based search and generative AI to provide fact-checked answers. Users input a Wikipedia topic and a question, and the app fetches relevant information, processes it, and generates a concise answer.

## Technologies Used
- **Streamlit**: For building the interactive web UI.
- **Wikipedia-API**: To fetch Wikipedia article content.
- **Sentence Transformers**: For semantic text embeddings and similarity search.
- **FAISS**: For fast vector similarity search and chunk retrieval.
- **LangChain**: For advanced text chunking and processing.
- **Transformers (Hugging Face)**: For generative question-answering using large language models.

## Technical Overview
- The app retrieves Wikipedia content using the Wikipedia-API.
- Text is split into manageable chunks using LangChain.
- Chunks are embedded with Sentence Transformers and indexed with FAISS for similarity search.
- The most relevant chunks are used as context for a Hugging Face transformer model to generate answers.

## File Structure
- `streamlit_app.py`: Main Streamlit UI
- `retrieval.py`: Wikipedia retrieval and chunking
- `generation.py`: AI answer generation
- `constants.py`: Configurations

---

Deployed and made by **Tisha Choksi**
