"""
PDF → Chunk → Local Embedding → Pinecone

This script:
1. Loads a PDF
2. Splits it into chunks
3. Creates local embeddings
4. Stores them inside Pinecone
"""

# ==============================
# 📦 Imports (Dependencies)
# ==============================

import os
import hashlib
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ==============================
# ⚙️ Environment Setup
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))  # Load API keys from .env


# ==============================
# 🧠 Local Embedding Model
# ==============================

# Load small fast embedding model (384-dimension output)
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# 🔑 Utility Functions
# ==============================

# Generate unique ID for each chunk (prevents duplicates)
def generate_id(text: str, index: int) -> str:
    return hashlib.md5((text + str(index)).encode("utf-8")).hexdigest()


# Convert list of texts → embedding vectors
def embed_texts(texts: List[str]):
    return model.encode(texts).tolist()


# ==============================
# 🚀 Main Indexing Logic
# ==============================

def index_document():

    print("🚀 Starting indexing process...\n")

    # 1️⃣ Load PDF file
    pdf_path = os.path.join(BASE_DIR, "data", "dsa.pdf")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    print("✅ PDF loaded")

    # 2️⃣ Split long text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,     # size of each chunk
        chunk_overlap=200    # overlapping context
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"✅ Chunking completed ({len(chunks)} chunks)")

    # Extract pure text from chunks
    texts = [chunk.page_content for chunk in chunks]

    # 3️⃣ Generate embeddings locally
    print("🔄 Generating local embeddings...")
    vectors = embed_texts(texts)
    print("✅ Embeddings created")

    # 4️⃣ Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    # 5️⃣ Prepare data for upsert
    upsert_payload = []

    for i, vector in enumerate(vectors):
        text = texts[i]

        upsert_payload.append({
            "id": generate_id(text, i),   # unique ID
            "values": vector,             # embedding vector
            "metadata": {                 # store text for retrieval
                "text": text[:1500],
                "source": "dsa.pdf"
            }
        })

    # 6️⃣ Store vectors in Pinecone
    pinecone_index.upsert(vectors=upsert_payload)

    print("\n🎉 Data stored successfully!")


# ==============================
# ▶️ Script Entry Point
# ==============================

if __name__ == "__main__":
    index_document()


# ============================================================
# 🔮 FUTURE: If You Want To Use Google Embeddings Again
# ============================================================

"""
Replace local embedding part with:

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectors = embeddings.embed_documents(texts)

⚠️ Make sure:
- You enable billing
- Avoid quota limits
"""
