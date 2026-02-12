# Local RAG System (Pinecone + SentenceTransformers + Ollama)

## Architecture

PDF → Chunk → Local Embeddings (all-MiniLM-L6-v2) → Pinecone  
User Query → Local Embedding → Pinecone Search → Ollama (Mistral)

---

## Tech Stack

- SentenceTransformers (`all-MiniLM-L6-v2`)
- Pinecone (Vector Database)
- Ollama (Mistral LLM)
- LangChain Community (PDF Loader + Text Splitter)
- Python

---

## How It Works

1. Load PDF document
2. Split into semantic chunks
3. Generate embeddings locally (384-dimensional vectors)
4. Store vectors in Pinecone with namespace support
5. Embed user query
6. Retrieve top-k similar chunks
7. Send retrieved context to Mistral via Ollama
8. Generate context-aware answer

---

## Setup

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
