# ============================================================
# RAG PIPELINE + LANGGRAPH AGENTIC RAG
# Run: python app.py  OR  paste in one Jupyter cell
# Features: Streaming · Query Rewriting · Re-ranking ·
#           Source Highlighting · LangGraph Agent ·
#           Confidence Score · Multi-doc Comparison
# ============================================================

import os, gc, math, warnings, logging
from collections import defaultdict
from typing import TypedDict, Annotated, List
import operator

# ── Suppress warnings ─────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)

# ── Install dependencies ──────────────────────────────────────────────────────
import subprocess, sys
pkgs = [
    "langchain", "langchain-openai", "langchain-community",
    "langchain-text-splitters", "faiss-cpu", "tiktoken",
    "pypdf", "sentence-transformers", "rank-bm25",
    "gradio", "openai", "langgraph", "langchain-core"
]
print("📦 Checking dependencies...")
for pkg in pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                   capture_output=True)
print("✅ Dependencies ready")

# ── Imports ───────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
import gradio as gr

print("✅ Imports done")

# ── Custom EnsembleRetriever ──────────────────────────────────────────────────
class EnsembleRetriever:
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights    = weights or [1/len(retrievers)] * len(retrievers)

    def invoke(self, query):
        scores, docs_map = defaultdict(float), {}
        for retriever, weight in zip(self.retrievers, self.weights):
            for i, doc in enumerate(retriever.invoke(query)):
                key = doc.page_content
                scores[key]  += weight * (1 / (i + 1))
                docs_map[key] = doc
        return [docs_map[k] for k in sorted(
            docs_map.keys(), key=lambda k: scores[k], reverse=True)]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit only this section
# ══════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY = "sk-proj-"      # 👈 paste your key here

CONFIG = {
    "chunk_size"     : 1500,
    "chunk_overlap"  : 300,
    "top_k"          : 10,
    "rerank_top_k"   : 5,
    "embedding_model": "text-embedding-3-small",
    "llm_model"      : "gpt-4o-mini",
    "reranker_model" : "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
# ══════════════════════════════════════════════════════════════════════════════

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── LLM & Embeddings ──────────────────────────────────────────────────────────
print("🤖 Initialising LLM & embeddings...")
llm = ChatOpenAI(
    model=CONFIG["llm_model"], temperature=0.3,
    api_key=OPENAI_API_KEY, streaming=True
)
llm_fast = ChatOpenAI(          # non-streaming, used inside graph nodes
    model=CONFIG["llm_model"], temperature=0, api_key=OPENAI_API_KEY
)
embeddings = OpenAIEmbeddings(
    model=CONFIG["embedding_model"], api_key=OPENAI_API_KEY
)

print("⏳ Loading re-ranker…")
reranker = CrossEncoder(CONFIG["reranker_model"])
print("✅ LLM + Embeddings + Re-ranker ready")

# ── Pipeline state (global) ───────────────────────────────────────────────────
pipeline = {
    "vectorstore"     : None,
    "hybrid_retriever": None,
    "chunks"          : [],
    "loaded_files"    : [],
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def rerank_docs(query: str, docs: list, top_k: int):
    if not docs:
        return docs, []
    try:
        scores = reranker.predict([(query, d.page_content) for d in docs])
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]], [float(s) for s, _ in scored[:top_k]]
    except Exception as e:
        print(f"⚠️ Re-rank failed: {e}")
        return docs[:top_k], []

def compute_confidence(scores: list):
    if not scores:
        return 0.0, "Unknown", "❓"
    avg = sum(sigmoid(s) for s in scores) / len(scores)
    pct = round(avg * 100, 1)
    if pct >= 75:   return pct, "High",   "🟢"
    elif pct >= 45: return pct, "Medium", "🟡"
    else:           return pct, "Low",    "🔴"

def format_context(docs):
    return "\n\n".join(
        f"--- Chunk {i} | {d.metadata.get('source_file','?')} "
        f"| page {d.metadata.get('page','?')} ---\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )

def format_sources(docs, scores=None) -> str:
    if not docs:
        return ""
    seen, lines = set(), ["\n\n---\n📎 **Sources:**"]
    for i, doc in enumerate(docs):
        src  = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        pno  = int(page)+1 if str(page).isdigit() else page
        key  = f"{src}::p{pno}"
        badge = ""
        if scores and i < len(scores):
            badge = f" · relevance `{round(sigmoid(scores[i])*100,1)}%`"
        if key not in seen:
            seen.add(key)
            preview = doc.page_content.strip()[:110].replace("\n", " ")
            lines.append(f"\n**{i+1}. 📄 `{src}` — Page {pno}**{badge}")
            lines.append(f"> _{preview}..._")
    return "\n".join(lines)

def is_comparison(q: str) -> bool:
    return any(w in q.lower() for w in
               ["compare","difference","vs","versus","contrast",
                "both","all documents","across","which is better"])

# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH — STATE + NODES
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    question        : str
    rewritten_query : str
    retrieved_docs  : List[Document]
    reranked_docs   : List[Document]
    rerank_scores   : List[float]
    context         : str
    answer          : str
    sources_text    : str
    confidence      : float
    conf_label      : str
    conf_emoji      : str
    route           : str
    messages        : Annotated[list, operator.add]

# ── Tools (used by agent nodes) ───────────────────────────────────────────────
@tool
def search_documents(query: str) -> str:
    """Search uploaded PDF documents for relevant information."""
    if pipeline["hybrid_retriever"] is None:
        return "No documents loaded."
    docs = pipeline["hybrid_retriever"].invoke(query)
    return format_context(docs[:5]) if docs else "No relevant chunks found."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression, e.g. '1000 * 0.05'."""
    try:
        result = eval(expression, {"__builtins__": {},
                                   "abs":abs,"round":round,"min":min,
                                   "max":max,"sum":sum,"pow":pow})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def summarize_document(filename: str) -> str:
    """Return a content sample from a specific uploaded file."""
    if not pipeline["chunks"]:
        return "No documents loaded."
    hits = [c for c in pipeline["chunks"]
            if filename.lower() in c.metadata.get("source_file","").lower()]
    if not hits:
        available = list({c.metadata.get("source_file","") for c in pipeline["chunks"]})
        return f"'{filename}' not found. Available: {available}"
    return "Sample:\n\n" + "\n\n".join(c.page_content for c in hits[:5])

@tool
def check_loaded_documents(_: str = "") -> str:
    """List currently loaded documents and their chunk counts."""
    if not pipeline["loaded_files"]:
        return "No documents loaded."
    ccount = defaultdict(int)
    for c in pipeline["chunks"]:
        ccount[c.metadata.get("source_file","")] += 1
    return "\n".join(f"• {f} ({ccount[f]} chunks)"
                     for f in pipeline["loaded_files"])

# ── Node functions ────────────────────────────────────────────────────────────
def node_rewrite(state: AgentState) -> AgentState:
    q = state["question"]
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite this question for better document retrieval. "
                       "Return ONLY the rewritten query."),
            ("human", "{q}")
        ])
        rw = (prompt | llm_fast | StrOutputParser()).invoke({"q": q}).strip()
        print(f"🔄 rewrite: '{q}' → '{rw}'")
    except:
        rw = q
    return {**state, "rewritten_query": rw}

def node_route(state: AgentState) -> AgentState:
    q = state["question"].lower()
    if any(w in q for w in ["summarize","summary","overview"]):
        route = "summarize"
    elif any(w in q for w in ["calculate","how much","total","premium",
                               "percent","%","multiply","divide"]):
        route = "calculate"
    elif any(w in q for w in ["what files","what documents","what pdf",
                               "which files","loaded"]):
        route = "check_docs"
    elif pipeline["hybrid_retriever"] is None:
        route = "general"
    else:
        route = "retrieve"
    print(f"🗺️  route: {route}")
    return {**state, "route": route}

def node_retrieve(state: AgentState) -> AgentState:
    query = state.get("rewritten_query", state["question"])
    docs  = pipeline["hybrid_retriever"].invoke(query)
    print(f"📄 retrieve: {len(docs)} chunks")
    return {**state, "retrieved_docs": docs}

def node_rerank(state: AgentState) -> AgentState:
    query    = state.get("rewritten_query", state["question"])
    docs     = state.get("retrieved_docs", [])
    ranked, scores = rerank_docs(query, docs, CONFIG["rerank_top_k"])
    conf, label, emoji = compute_confidence(scores)
    print(f"🎯 rerank: {len(ranked)} chunks | conf {conf}%")
    return {
        **state,
        "reranked_docs" : ranked,
        "rerank_scores" : scores,
        "context"       : format_context(ranked),
        "sources_text"  : format_sources(ranked, scores),
        "confidence"    : conf,
        "conf_label"    : label,
        "conf_emoji"    : emoji,
    }

def node_generate(state: AgentState) -> AgentState:
    q    = state["question"]
    ctx  = state.get("context", "")
    docs = state.get("reranked_docs", [])

    if not docs:
        p = ChatPromptTemplate.from_messages([
            ("system", "Answer from general knowledge. End with: "
                       "'⚠️ Answer based on general knowledge, not uploaded documents.'"),
            ("human", "{q}")
        ])
        answer = (p | llm_fast | StrOutputParser()).invoke({"q": q})

    elif is_comparison(q) and len(pipeline["loaded_files"]) > 1:
        p = ChatPromptTemplate.from_messages([
            ("system", """Compare across documents. Format:
## 📊 Comparison
### 📄 Document-by-Document:
### ⚖️ Key Differences:
### ✅ Similarities:
### 🏆 Summary:
Context: {ctx}"""),
            ("human", "{q}")
        ])
        answer = (p | llm_fast | StrOutputParser()).invoke({"ctx": ctx, "q": q})

    else:
        p = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst.
Read ALL chunks. Piece together answers across chunks.
State numbers/days/amounts exactly as written.

Format:
📄 From your documents:
[specific answer from PDF]

🧠 Additional context:
[general knowledge supplement, or 'Nothing to add.']

Context: {ctx}"""),
            ("human", "{q}")
        ])
        answer = (p | llm_fast | StrOutputParser()).invoke({"ctx": ctx, "q": q})

    print(f"✍️  generate: {len(answer)} chars")
    return {**state, "answer": answer}

def node_calculate(state: AgentState) -> AgentState:
    p = ChatPromptTemplate.from_messages([
        ("system", "Extract and solve the math step-by-step. "
                   "Explain results in insurance context if relevant."),
        ("human", "{q}")
    ])
    answer = (p | llm_fast | StrOutputParser()).invoke({"q": state["question"]})
    return {**state, "answer": answer, "sources_text": "",
            "confidence": 100, "conf_label": "High", "conf_emoji": "🟢"}

def node_summarize(state: AgentState) -> AgentState:
    loaded = pipeline["loaded_files"]
    target = next((f for f in loaded
                   if f.lower().replace(".pdf","") in state["question"].lower()),
                  loaded[0] if loaded else None)
    hits   = [c for c in pipeline["chunks"]
              if target and target.lower() in
              c.metadata.get("source_file","").lower()][:8]
    if not hits:
        return {**state, "answer": "Could not find content to summarise.",
                "sources_text": "", "confidence": 0,
                "conf_label": "Low", "conf_emoji": "🔴"}
    ctx = "\n\n".join(c.page_content for c in hits)
    p   = ChatPromptTemplate.from_messages([
        ("system", """Summarise this document as:
## 📋 Summary: {fn}
### 🎯 Main Purpose:
### 📌 Key Points:
### 💡 Important Details:
### ⚠️ Exclusions/Limitations:"""),
        ("human", f"Content:\n\n{ctx}")
    ])
    answer = (p | llm_fast | StrOutputParser()).invoke({"fn": target or "document"})
    return {**state, "answer": answer,
            "sources_text": format_sources(hits[:5]),
            "confidence": 90, "conf_label": "High", "conf_emoji": "🟢"}

def node_check_docs(state: AgentState) -> AgentState:
    answer = check_loaded_documents.invoke("")
    if pipeline["loaded_files"]:
        answer = "## 📂 Loaded Documents\n\n" + answer
    return {**state, "answer": answer, "sources_text": "",
            "confidence": 100, "conf_label": "High", "conf_emoji": "🟢"}

def node_general(state: AgentState) -> AgentState:
    p = ChatPromptTemplate.from_messages([
        ("system", "Answer from general knowledge. "
                   "End with: '⚠️ Answer based on general knowledge, "
                   "not uploaded documents.'"),
        ("human", "{q}")
    ])
    answer = (p | llm_fast | StrOutputParser()).invoke({"q": state["question"]})
    return {**state, "answer": answer, "sources_text": "",
            "confidence": 50, "conf_label": "Medium", "conf_emoji": "🟡"}

def route_edge(state: AgentState) -> str:
    return state.get("route", "retrieve")

# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("rewrite",   node_rewrite)
    g.add_node("route",     node_route)
    g.add_node("retrieve",  node_retrieve)
    g.add_node("rerank",    node_rerank)
    g.add_node("generate",  node_generate)
    g.add_node("calculate", node_calculate)
    g.add_node("summarize", node_summarize)
    g.add_node("check_docs",node_check_docs)
    g.add_node("general",   node_general)

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "route")
    g.add_conditional_edges("route", route_edge, {
        "retrieve"  : "retrieve",
        "calculate" : "calculate",
        "summarize" : "summarize",
        "check_docs": "check_docs",
        "general"   : "general",
    })
    g.add_edge("retrieve",  "rerank")
    g.add_edge("rerank",    "generate")
    for n in ["generate","calculate","summarize","check_docs","general"]:
        g.add_edge(n, END)
    return g.compile()

print("🔨 Building LangGraph agent…")
agent_graph = build_graph()
print("✅ LangGraph agent ready")

# ── Streaming chat prompt (for direct streaming path) ─────────────────────────
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert document analyst with broad general knowledge.
Read ALL chunks carefully. Piece together answers that span multiple chunks.
State numbers/days/amounts exactly as written.

Always structure as:
📄 From your documents:
[specific PDF answer]

🧠 Additional context:
[general knowledge supplement, or 'Nothing to add.']

Document Context:
{context}"""),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

general_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer from general knowledge. Be specific and helpful. "
               "End with: '⚠️ Answer based on general knowledge, not uploaded documents.'"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILD
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(pdf_paths: list, progress=None) -> str:
    global pipeline
    if not pdf_paths:
        return "❌ No PDF files provided."
    log = []
    try:
        if progress: progress(0.1, "📄 Loading PDFs…")
        raw_docs, loaded = [], []
        for path in pdf_paths:
            try:
                pages = PyPDFLoader(path).load()
                for p in pages:
                    p.metadata["source_file"] = os.path.basename(path)
                raw_docs.extend(pages)
                loaded.append(os.path.basename(path))
                log.append(f"✅ {os.path.basename(path)} — {len(pages)} pages")
            except Exception as e:
                log.append(f"❌ {os.path.basename(path)} — {e}")

        if not raw_docs:
            return "❌ Could not load any PDFs."

        if progress: progress(0.4, "✂️ Chunking…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        ccount = {}
        for c in chunks:
            src = c.metadata.get("source_file","unknown")
            ccount[src] = ccount.get(src, 0) + 1
            c.metadata["chunk_index"] = ccount[src]
        log.append(f"✂️ {len(chunks)} chunks created")

        if progress: progress(0.6, "🔢 Embedding… (~30s)")
        gc.collect()
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
        log.append(f"🔢 {len(vs.docstore._dict)} vectors embedded")

        if progress: progress(0.85, "🔍 Building retriever…")
        dense      = vs.as_retriever(search_type="similarity",
                                     search_kwargs={"k": CONFIG["top_k"]})
        bm25       = BM25Retriever.from_documents(chunks)
        bm25.k     = CONFIG["top_k"]
        hybrid     = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])

        pipeline.update({"vectorstore": vs, "hybrid_retriever": hybrid,
                         "chunks": chunks, "loaded_files": loaded})

        if progress: progress(1.0, "✅ Done!")
        return ("✅ Pipeline ready!\n\n" +
                "\n".join(f"  • {f}" for f in loaded) +
                "\n\n" + "\n".join(log) +
                "\n\n💬 Ask questions below.")
    except Exception as e:
        return f"❌ Pipeline error: {e}"

def upload_and_process(files, progress=gr.Progress()):
    if not files:
        return "⚠️ No files uploaded.", "No files loaded."
    paths  = [f.name for f in files] if hasattr(files[0], "name") else files
    status = build_pipeline(paths, progress=progress)
    label  = ("📂 Loaded: " + ", ".join(pipeline["loaded_files"])
              if pipeline["loaded_files"] else "Load failed.")
    return status, label

# ══════════════════════════════════════════════════════════════════════════════
# CHAT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Mode A: Agentic (LangGraph) ───────────────────────────────────────────────
def chat_agent(question, history):
    """LangGraph agent — routes, re-ranks, shows confidence + sources."""
    if not question.strip():
        yield history, ""; return

    if pipeline["hybrid_retriever"] is None:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant",
                        "content": "⚠️ Upload a PDF and click **⚙️ Process** first."})
        yield history, ""; return

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": "⏳ Agent thinking…"})
    yield history, ""

    try:
        init: AgentState = {
            "question": question, "rewritten_query": question,
            "retrieved_docs": [], "reranked_docs": [], "rerank_scores": [],
            "context": "", "answer": "", "sources_text": "",
            "confidence": 0, "conf_label": "Unknown", "conf_emoji": "❓",
            "route": "retrieve", "messages": [],
        }
        fs = agent_graph.invoke(init)

        answer       = fs.get("answer", "No answer generated.")
        sources_text = fs.get("sources_text", "")
        conf         = fs.get("confidence", 0)
        conf_label   = fs.get("conf_label", "Unknown")
        conf_emoji   = fs.get("conf_emoji", "❓")
        route        = fs.get("route", "retrieve")
        rewritten    = fs.get("rewritten_query", question)

        route_labels = {
            "retrieve"  : "📄 Document Search",
            "calculate" : "🔢 Calculator",
            "summarize" : "📋 Summariser",
            "check_docs": "📂 File Check",
            "general"   : "🧠 General Knowledge",
        }
        meta = (
            f"\n\n---\n"
            f"{conf_emoji} **Confidence:** `{conf}%` — {conf_label}  ·  "
            f"🤖 **Route:** {route_labels.get(route, route)}"
            + ("  ·  🔄 *query rewritten*" if rewritten != question else "")
        )
        history[-1] = {"role": "assistant",
                       "content": answer + meta + sources_text}
        yield history, ""

    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ Agent error: {e}"}
        yield history, ""


# ── Mode B: Streaming (original fast path) ────────────────────────────────────
def chat_stream(question, history):
    """Original streaming path — fast, token-by-token output."""
    if not question.strip():
        yield history, ""; return

    if pipeline["hybrid_retriever"] is None:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant",
                        "content": "⚠️ Upload a PDF and click **⚙️ Process** first."})
        yield history, ""; return

    # Rewrite query
    try:
        rw_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite for better retrieval. Return ONLY the rewritten query."),
            ("human", "{q}")
        ])
        rewritten = (rw_prompt | llm_fast | StrOutputParser()).invoke(
            {"q": question}).strip()
    except:
        rewritten = question

    # Retrieve + rerank
    raw_docs           = pipeline["hybrid_retriever"].invoke(rewritten)
    reranked, scores   = rerank_docs(rewritten, raw_docs, CONFIG["rerank_top_k"])
    context            = format_context(reranked)
    sources_text       = format_sources(reranked, scores)

    # LangChain history
    lc_history = []
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": "⏳ Thinking…"})
    yield history, ""

    try:
        chain = (chat_prompt if reranked else general_prompt) | llm | StrOutputParser()
        kwargs = ({"context": context, "question": question, "chat_history": lc_history}
                  if reranked else
                  {"question": question, "chat_history": lc_history})

        full = ""
        for token in chain.stream(kwargs):
            full += token
            history[-1] = {"role": "assistant", "content": full}
            yield history, ""

        history[-1] = {"role": "assistant",
                       "content": full + sources_text}
        yield history, ""

    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ Error: {e}"}
        yield history, ""


# ── Dispatcher — picks mode based on toggle ───────────────────────────────────
def chat_fn(question, history, use_agent):
    fn = chat_agent if use_agent else chat_stream
    yield from fn(question, history)

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
EXAMPLES = [
    "What is the pre hospitalization period?",
    "What are the disease categories defined?",
    "What is covered under day care treatment?",
    "What are the exclusions in this policy?",
    "What is the waiting period for pre existing diseases?",
    "What is post hospitalization coverage?",
    "Is mental illness covered?",
    "Summarize the document",
    "What files are loaded?",
    "Calculate 5% of 500000",
    "Compare coverage across all uploaded documents",
]

print("🚀 Building Gradio UI…")

with gr.Blocks(title="Agentic RAG") as demo:

    gr.Markdown("# 🤖 RAG Document Chat  +  LangGraph Agent")
    gr.Markdown(
        "> ✨ **Features:** Streaming · Query Rewriting · Re-ranking · "
        "Source Highlighting · LangGraph Agent · Confidence Score · "
        "Multi-doc Comparison"
    )

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row():

                # Left panel
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 📂 Upload Documents")
                    file_upload = gr.File(
                        label="Drop PDF(s) here",
                        file_types=[".pdf"], file_count="multiple"
                    )
                    upload_btn  = gr.Button("⚙️ Process Documents",
                                            variant="primary", size="lg")
                    file_status = gr.Textbox(
                        label="Status", value="No files loaded yet.",
                        lines=1, interactive=False
                    )
                    process_log = gr.Textbox(
                        label="Processing Log", lines=9, interactive=False
                    )

                    gr.Markdown("---")
                    gr.Markdown("### ⚙️ Mode")
                    use_agent = gr.Checkbox(
                        label="🤖 Use LangGraph Agent",
                        value=True,
                        info="Agent: routes + confidence + sources. "
                             "Stream: faster token-by-token output."
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 💡 Example Questions")
                    gr.Markdown("\n".join(f"- {q}" for q in EXAMPLES))

                # Right panel
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat with your Documents")
                    gr.Markdown(
                        "_**Agent mode:** routes to search / calculate / summarise / "
                        "general knowledge + shows confidence_  \n"
                        "_**Stream mode:** faster, direct answer with sources_"
                    )
                    chatbot = gr.Chatbot(
                        height=500, show_label=False, render_markdown=True
                    )
                    with gr.Row():
                        txt = gr.Textbox(
                            placeholder="Ask anything…",
                            scale=4, show_label=False, lines=1
                        )
                        send_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # ── Tab 2: Architecture ───────────────────────────────────────────────
        with gr.Tab("🗺️ Agent Architecture"):
            gr.Markdown("""
## 🤖 LangGraph Agent — How It Works

### Graph Flow
```
User Question
      ↓
[Node 1]  rewrite_query    → Improves question for better retrieval
      ↓
[Node 2]  route_question   → Decides which path to take
      ↓
  ┌──────────────────────────────────────────────────────┐
  │  retrieve → rerank → generate   (document questions) │
  │  calculate                      (math / numbers)     │
  │  summarize                      (summary requests)   │
  │  check_docs                     (what files loaded?) │
  │  general                        (no docs loaded)     │
  └──────────────────────────────────────────────────────┘
      ↓
  Final answer + Confidence badge + Sources
```

### 🛠️ Agent Tools

| Tool | Triggered by | Does |
|------|-------------|------|
| search_documents | Any doc question | Semantic + keyword hybrid search |
| calculate | Math / premium / % | Safe Python eval |
| summarize_document | "Summarize …" | Extracts & summarises a file |
| check_loaded_documents | "What files…" | Lists loaded files + chunk counts |

### 📊 Confidence Score

| Badge | Score | Meaning |
|-------|-------|---------|
| 🟢 High   | ≥ 75% | Strongly supported by retrieved chunks |
| 🟡 Medium | 45–74% | Partial match, some inference |
| 🔴 Low    | < 45% | Weak match — try rephrasing |

### 🔄 Agent vs Stream Mode

| | Agent Mode | Stream Mode |
|---|---|---|
| Speed | Slower (multi-node) | ⚡ Faster |
| Routing | ✅ Smart routing | Always retrieves |
| Confidence | ✅ Shown | ❌ Not shown |
| Streaming | ❌ Full answer at once | ✅ Token-by-token |
| Best for | Complex / varied questions | Simple Q&A |
""")

    # ── Event wiring ──────────────────────────────────────────────────────────
    upload_btn.click(
        fn=upload_and_process,
        inputs=[file_upload],
        outputs=[process_log, file_status]
    )
    send_btn.click(
        fn=chat_fn,
        inputs=[txt, chatbot, use_agent],
        outputs=[chatbot, txt],
        show_progress=False
    )
    txt.submit(
        fn=chat_fn,
        inputs=[txt, chatbot, use_agent],
        outputs=[chatbot, txt],
        show_progress=False
    )
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, txt])

# ── Launch ────────────────────────────────────────────────────────────────────
# ── Launch ────────────────────────────────────────────────────────────────────
import gradio as gr
gr.close_all()          # close any previous instance first

demo.launch(
    theme=gr.themes.Soft()   # no server_port — Gradio picks a free one automatically
)