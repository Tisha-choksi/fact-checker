import streamlit as st
from retrieval import fetch_wikipedia_page, chunk_text, build_faiss_index
from generation import generate_answer
import numpy as np
from constants import *

# Streamlit UI Setup
st.set_page_config(page_title="Wikipedia Fact Checker", page_icon="🔍")
st.title("Wikipedia Fact Checker 🧠")
st.markdown("Ask questions about any Wikipedia topic!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    max_chunks = st.slider("Number of chunks to use", 1, 5, 3)

# Main input form
with st.form("question_form"):
    topic = st.text_input("Wikipedia Topic (e.g., 'Quantum computing')", 
                         placeholder="Enter a Wikipedia article title")
    question = st.text_input("Your Question", 
                           placeholder="What would you like to know?")
    submitted = st.form_submit_button("Get Answer")

# Processing logic
if submitted:
    if not topic.strip() or not question.strip():
        st.error("Please enter both a topic and a question!")
    else:
        with st.spinner("Searching Wikipedia and generating answer..."):
            try:
                # 1. Fetch and process content
                text = fetch_wikipedia_page(topic)
                chunks = chunk_text(text)
                
                # 2. Build index and retrieve chunks
                index, model = build_faiss_index(chunks)
                query_embedding = model.encode(question)
                D, I = index.search(np.array([query_embedding]), k=max_chunks)
                
                # 3. Generate answer
                context = " ".join([chunks[i] for i in I[0]])
                answer = generate_answer(question, context)
                
                # Display results
                st.success("Answer:")
                st.markdown(f"**{answer}**")
                
                with st.expander("See retrieved context"):
                    st.write(context)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try a different topic or rephrase your question.")