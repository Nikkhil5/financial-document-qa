# 📊 Financial Document Q&A Assistant

This is a Streamlit web app that allows you to **ask questions about financial documents (PDF or Excel)** using a combination of:

- 🔍 Text extraction and chunking
- 📈 TF-IDF-based information retrieval
- 🤖 Local LLM (e.g., TinyLlama via Ollama) for intelligent answers

---

## 🚀 Features

- 📁 Upload `.pdf` or `.xlsx` financial documents
- 📚 Automatic text extraction and preprocessing
- 🔎 Semantic search using TF-IDF + Cosine Similarity
- 💬 Natural language question answering using a local LLM (TinyLlama by default)
- 🧠 Session-based chat history for interactive conversations

---

## 🛠️ Requirements

Install the following dependencies:

```bash
pip install -r requirements.txt
```
requirements.txt
```bash
streamlit
pandas
pdfplumber
scikit-learn
ollama
openpyxl
```
How to Run
1. ✅ Make sure Ollama is installed and running

Install Ollama from: https://ollama.com/

Then pull the desired model (e.g., TinyLlama):
```bash
ollama pull tinyllama
```
2. (Optional) Add more models

You can pull and use other models such as:
```bash
ollama pull llama2
ollama pull mistral
ollama pull gemma:2b
```

3. ▶️ Run the Streamlit app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 🧠 How It Works

1. Document Upload: Upload a .pdf or .xlsx file.

2. Text Extraction: The app uses pdfplumber or pandas to extract readable content.

3. Text Chunking: The text is split into overlapping chunks (default: 500 words with 50-word overlap).

4. TF-IDF Search: When you ask a question, the app retrieves the top 3 most relevant chunks.

5. LLM Answering: The retrieved context and your question are passed to the selected Ollama model for a response.
