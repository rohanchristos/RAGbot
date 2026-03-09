# 📄 RAG Agent — Chat with any PDF

A fully local, free-to-run RAG (Retrieval-Augmented Generation) agent that lets you chat with any PDF document.

**Stack:** Groq (LLM) + HuggingFace (Embeddings) + ChromaDB (Vector Store) + LangGraph (Agent)

---

## ⚡ Quickstart

### 1. Clone & install
```bash
git clone https://github.com/your-username/rag-agent.git
cd rag-agent
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Add your PDF and run
```bash
python rag_agent.py
```

---

## 🔑 API Keys

| Service | Required | Get it |
|---|---|---|
| Groq | ✅ Yes | [console.groq.com](https://console.groq.com) |
| HuggingFace | ❌ No | Runs locally |

---

## ⚙️ Configuration

All settings are controlled via `.env`:

| Variable | Description | Default |
|---|---|---|
| `PDF_PATH` | Path to your PDF | `your_document.pdf` |
| `PERSIST_DIRECTORY` | ChromaDB storage path | `./chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `my_collection` |
| `DOCUMENT_DESCRIPTION` | Describes your doc (used in prompts) | `the uploaded PDF document` |
| `GROQ_MODEL` | Groq model to use | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-small-en-v1.5` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `RETRIEVER_K` | Number of chunks to retrieve | `5` |

---

## 🏗️ Architecture

```
User Input
    ↓
LLM (Groq — llama-3.3-70b)
    ↓ (if tool call needed)
Retriever Tool → ChromaDB → HuggingFace Embeddings
    ↓
LLM generates final answer
    ↓
Output
```

---

## 📦 First Run Note

On first run, the embedding model (`~130MB`) will be downloaded and cached locally. Subsequent runs use the cache instantly.
