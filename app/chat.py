"""
Fully Local RAG System

Flow:
User Question → Local Embedding → Pinecone Search
Retrieved Context → Local LLM (Mistral) → Answer
"""

# ==============================
# 📦 Imports
# ==============================

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import ollama


# ==============================
# ⚙️ Environment Setup
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ==============================
# 🧠 Embedding Model (Local)
# ==============================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# 🗄 Pinecone Setup
# ==============================

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


# ==============================
# 💬 Chat Function
# ==============================

def chat(question: str):

    # 1️⃣ Convert question into embedding vector
    query_vector = embedding_model.encode(question).tolist()

    # 2️⃣ Search Pinecone for similar chunks
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    matches = results.get("matches", [])

    # 3️⃣ Build context string from retrieved chunks
    context = "\n\n---\n\n".join(
        match["metadata"]["text"]
        for match in matches
    )

    # 4️⃣ Create prompt for LLM
    prompt = f"""
You are a Data Structure and Algorithm expert.

Answer ONLY using the provided context.
If the answer is not present, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}
"""

    # 5️⃣ Ask local LLM (Mistral via Ollama)
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])


# ==============================
# ▶️ CLI Loop
# ==============================

def main():
    print("🚀 Fully Local RAG Ready\n")

    while True:
        q = input("Ask me anything --> ")
        if q.lower() in ["exit", "quit"]:
            break
        chat(q)


if __name__ == "__main__":
    main()


# ============================================================
# 🔮 FUTURE: If You Want To Use Google Gemini Again
# ============================================================

"""
1-- Install:
pip install google-genai langchain-google-genai


2-- Replace Ollama part with:

from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[{"role": "user", "parts": [{"text": prompt}]}]
)

print(response.text)

⚠️ Make sure:
- Billing enabled
- Quota limits handled
"""
