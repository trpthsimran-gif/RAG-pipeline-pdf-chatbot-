# RAG-pipeline-pdf-chatbot-
AI-powered RAG Pipeline using LangChain, FAISS, HuggingFace Embeddings, and Ollama for PDF Question Answering.
# AI-Powered RAG Pipeline PDF Chatbot

## Overview
This project is an AI-powered Retrieval-Augmented Generation (RAG) pipeline built using LangChain, FAISS, HuggingFace Embeddings, and Ollama.

The system allows users to upload PDF documents and ask questions in natural language. The application retrieves relevant document chunks and generates context-aware answers using a Large Language Model (LLM).

---

## Features

- PDF document ingestion
- Text chunking
- Vector embeddings using Sentence Transformers
- FAISS vector database
- Semantic search
- Local LLM inference using Ollama
- Conversational question answering
- Scalable RAG architecture

---

## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama
- Jupyter Notebook

---

## Project Architecture

1. Load PDF
2. Split document into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Pass context to LLM
7. Generate final answer

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline-pdf-chatbot.git
cd rag-pipeline-pdf-chatbot
```

### Create Virtual Environment

```bash
python -m venv ragenv
```

### Activate Environment

```bash
ragenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Project

### Start Jupyter Notebook

```bash
jupyter notebook
```

OR

### Run Python App

```bash
python app/app.py
```

---

## Future Improvements

- Streamlit UI
- Multi-PDF support
- Chat history memory
- Hybrid search
- Cloud deployment
- API integration

---

## Author

Simran Tripathi
AI Engineer Aspirant
