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

import hashlib  # Generate unique IDs for chunks
import os  # File and path operations
from typing import List  # Type hints for functions

from dotenv import load_dotenv  # Load API keys from .env file
from langchain_community.document_loaders import PyPDFLoader  # Load PDF files
from langchain_text_splitters import \
    RecursiveCharacterTextSplitter  # Split PDF text into chunks
from pinecone import Pinecone  # Vector database client for cloud storage
from sentence_transformers import \
    SentenceTransformer  # Create text embeddings locally

# ==============================
# ⚙️ Environment Setup
# ==============================

# Determine base directory and load .env file for API keys
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

# Main function to load PDF, chunk, embed, and store in Pinecone
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

    # Create payload with unique ID, embedding vector, and metadata (original text)
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
# 🔮 FUTURE: Alternative Embedding Strategies
# ============================================================

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤗 OPTION 1: HuggingFace Embeddings (Current Approach)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: all-MiniLM-L6-v2 (currently in use)
- Lightweight: ~90MB, fast inference
- Dimension: 384 (compact representation)
- Free to use: No API keys or billing
- Best for: Local development, cost-effective production

Other HuggingFace Models:
- 'all-mpnet-base-v2': Higher quality (768-dim) but larger
- 'all-distilroberta-v1': Balanced speed/quality (768-dim)
- 'paraphrase-MiniLM-L6-v2': Better semantic similarity
- 'paraphrase-mpnet-base-v2': Highest quality but slowest

To use a different HuggingFace model:
    model = SentenceTransformer("all-mpnet-base-v2")

✅ Pros: Free, private, no rate limits
❌ Cons: Slightly lower quality than API-based models


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 OPTION 2: Google Generative AI Embeddings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectors = embeddings.embed_documents(texts)

Dimension: 768 (higher quality representations)
Cost: ~$0.02 per 1M tokens (adds up with large corpora)
Rate Limit: 15,000 req/min (usually sufficient)

⚠️ Requirements:
- Enable billing in Google Cloud Console
- Set GEMINI_API_KEY in .env
- Monitor usage to avoid unexpected costs

✅ Pros: Production-grade quality, API-based scalability
❌ Cons: Requires API key, adds latency, costs money


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👁️ OPTION 3: OpenAI Embeddings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectors = embeddings.embed_documents(texts)

Models: text-embedding-3-small or text-embedding-3-large
Cost: $0.02-$0.13 per 1M tokens (varies by model)
Dimension: 512-3072 (flexible dimension reduction)

✅ Pros: Industry standard, excellent quality
❌ Cons: Higher cost, requires OpenAI API key


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Comparison Chart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                Quality    Speed      Cost    Privacy
Local HF:       Good       Fast       Free    100%
Google:         Excellent  Medium     $       No
OpenAI:         Excellent  Medium     $$      No

Recommendation:
- Development: Use local HuggingFace (current setup)
- Production: Evaluate cost vs. quality trade-off
"""
