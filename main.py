# app.py
import streamlit as st
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# -------- Document Processing --------
def extract_text_from_pdf(file):
    text = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
    return "\n".join(text)

def extract_text_from_excel(file):
    try:
        df = pd.read_excel(file)
        return df.to_string(index=False)
    except Exception as e:
        st.error(f"Excel extraction failed: {e}")
        return ""

# -------- Text Chunking --------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# -------- Search Engine --------
def build_tfidf_search(chunks):
    vectorizer = TfidfVectorizer().fit(chunks)
    vectors = vectorizer.transform(chunks)
    return vectorizer, vectors

def search_chunks(query, vectorizer, vectors, chunks, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# -------- Ollama LLM --------
def generate_answer(context, question):
    prompt = f"Context: '''{context}'''\nQuestion: {question}\nAnswer as a financial expert:"
    try:
        response = ollama.chat(
            model="tinyllama",  #  "tinyllama"
            messages=[{"role": "user", "content": prompt}]
        )
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        elif "messages" in response:
            return response["messages"][-1].get("content", "No response.")
        else:
            return "No response from model."
    except Exception as e:
        return f"Error generating answer: {e}"

# -------- Streamlit App --------
# -------- Streamlit App --------
def main():
    st.set_page_config(page_title="Financial Document Q&A", layout="wide")
    st.title("📊 Financial Document Q&A Assistant")

    # Initialize chat history in session_state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload PDF or Excel financial document", type=["pdf", "xlsx"])

    if uploaded_file:
        with st.spinner("Extracting text from document..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                raw_text = extract_text_from_pdf(uploaded_file)
            else:
                raw_text = extract_text_from_excel(uploaded_file)

        if not raw_text.strip():
            st.error("No text could be extracted from the document.")
            return

        with st.spinner("Preparing document for Q&A..."):
            chunks = chunk_text(raw_text)
            vectorizer, vectors = build_tfidf_search(chunks)

        st.success("✅ Document processed. You can now ask financial questions.")

        # Question input
        user_question = st.text_input("Ask your question about the document:")

        if user_question:
            with st.spinner("Searching and generating answer..."):
                relevant_chunks = search_chunks(user_question, vectorizer, vectors, chunks)
                combined_context = "\n\n".join(relevant_chunks)
                answer = generate_answer(combined_context, user_question)
            # Store user question and answer in session_state chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        # Display chat history
        #----------
        st.markdown("### Conversation History")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

if __name__ == "__main__":
    main()
