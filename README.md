# 🤖 Agentic RAG PDF Chatbot

> **Upload any PDF → Ask anything → Get precise answers powered by LangGraph + OpenAI**

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline with a **LangGraph agentic layer** that intelligently routes every question to the best handler — document search, calculator, summariser, or general knowledge.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-PDF Upload** | Upload one or multiple PDFs at once via drag-and-drop |
| 🔍 **Hybrid Retrieval** | Combines semantic search (FAISS) + keyword search (BM25) for better recall |
| 🎯 **Cross-Encoder Re-ranking** | Re-ranks retrieved chunks using `ms-marco-MiniLM` for higher precision |
| 🔄 **Query Rewriting** | Automatically rewrites your question for better retrieval before searching |
| 🤖 **LangGraph Agent** | Smart routing — sends each question to the right node based on intent |
| 📊 **Confidence Score** | Shows 🟢 High / 🟡 Medium / 🔴 Low confidence for every answer |
| ⚡ **Streaming Mode** | Token-by-token streaming for fast, real-time answers |
| 📎 **Source Highlighting** | Every answer shows exactly which PDF page the answer came from |
| 🧠 **OpenAI Fallback** | If the answer isn't in your PDF, uses GPT-4o-mini's own knowledge |
| 📊 **Multi-doc Comparison** | Automatically detects comparison questions and answers across all loaded PDFs |

---

## 🗺️ LangGraph Agent Architecture

```
User Question
      ↓
[Node 1]  rewrite_query     → Rewrites question for better retrieval
      ↓
[Node 2]  route_question    → Decides which path to take
      ↓
  ┌──────────────────────────────────────────────────────────┐
  │  retrieve → rerank → generate    (document questions)   │
  │  calculate                       (math / % / premium)   │
  │  summarize                       ("summarize the doc")  │
  │  check_docs                      ("what files loaded?") │
  │  general                         (no docs loaded)       │
  └──────────────────────────────────────────────────────────┘
      ↓
  Final Answer + Confidence Badge + Source References
```

### Agent vs Stream Mode

| | 🤖 Agent Mode | ⚡ Stream Mode |
|---|---|---|
| Routing | ✅ Smart routing | Always retrieves |
| Confidence score | ✅ Shown | ❌ Not shown |
| Streaming | ❌ Full answer at once | ✅ Token-by-token |
| Best for | Complex / varied questions | Fast simple Q&A |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Re-ranker** | CrossEncoder — ms-marco-MiniLM-L-6-v2 |
| **Agent Framework** | LangGraph |
| **LLM Framework** | LangChain |
| **PDF Loading** | PyPDF |
| **UI** | Gradio |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline-pdf-chatbot.git
cd rag-pipeline-pdf-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv ragenv
# Windows
ragenv\Scripts\activate
# Mac / Linux
source ragenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Open `app.py` and replace the key on this line:
```python
OPENAI_API_KEY = "sk-..."   # 👈 paste your key here
```

Or use a `.env` file (recommended):
```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run the app
```bash
python app.py
```

Open your browser at **http://localhost:7860**

---

## 📖 How to Use

```
1. Drop your PDF(s) into the upload box (left panel)
2. Click ⚙️ Process Documents  — wait ~30 seconds
3. Toggle 🤖 Agent Mode  ON for smart routing
            OFF for fast streaming
4. Type your question and click Send 🚀
```

### Example questions to try
```
What is the pre-hospitalization period?
What are the disease categories defined?
What are the exclusions in this policy?
Is mental illness covered?
Summarize the document
What files are loaded?
Calculate 5% of 500000
Compare coverage across all uploaded documents
```

---

## 📁 Project Structure

```
rag-pipeline-pdf-chatbot/
│
├── app.py               # Complete pipeline + Gradio UI (single file)
├── requirements.txt     # All dependencies
├── .env.example         # API key template
├── .gitignore           # Excludes API keys, venv, __pycache__
└── README.md            # This file
```

---

## ⚙️ Configuration

All settings are in the `CONFIG` dict at the top of `app.py`:

```python
CONFIG = {
    "chunk_size"     : 1500,    # characters per chunk
    "chunk_overlap"  : 300,     # overlap between chunks
    "top_k"          : 10,      # chunks retrieved before re-ranking
    "rerank_top_k"   : 5,       # chunks kept after re-ranking
    "embedding_model": "text-embedding-3-small",
    "llm_model"      : "gpt-4o-mini",
    "reranker_model" : "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
```

---

## 💡 How RAG Works (Simple Explanation)

```
Traditional LLM:
  Question → GPT → Answer (may hallucinate)

RAG Pipeline:
  Question → Search your PDF → Find relevant chunks
           → Send chunks + question to GPT
           → Answer grounded in your actual document
```

```
This project adds on top:
  + Hybrid search  (semantic + keyword)
  + Re-ranking     (best chunks rise to top)
  + Query rewriting (better search terms)
  + LangGraph agent (smart routing per question type)
```

---

## 💰 Estimated API Cost

| Action | Cost |
|---|---|
| Embed a 50-page PDF | ~$0.002 |
| Ask 100 questions | ~$0.05 |
| Ask 1000 questions | ~$0.50 |

**$5 credit lasts months** for personal/demo usage with `gpt-4o-mini`.

---

## 🔮 Roadmap / Future Improvements

- [ ] Web search fallback (DuckDuckGo / Tavily)
- [ ] RAGAS evaluation tab for answer quality scoring
- [ ] Chat history persistence across sessions
- [ ] Support for DOCX, TXT, and URL ingestion
- [ ] Docker containerisation for easy deployment
- [ ] Authentication for multi-user support

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built by **[Simran Tripathi]**


---

> ⭐ If this project helped you, please give it a star — it helps others find it!
