from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import uuid
import numpy as np

from pipeline.vector_store import VectorStore
from pipeline.embedder import embed_query
from utils.llm import ask_llm

router = APIRouter()

# ==============================
# 🔹 CONFIG
# ==============================
EMBEDDING_DIM = 768
DATA_ROOT = "embeddings"


# ==============================
# 🔹 REQUEST MODEL
# ==============================
class DocChatRequest(BaseModel):
    query: str
    doc: str
    session_id: str | None = None


# ==============================
# 🔹 DB FUNCTIONS
# ==============================
def get_chat_history(session_id, limit=5):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_query, response 
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (session_id, limit))

    rows = cursor.fetchall()
    conn.close()
    return rows[::-1]


def save_conversation(session_id, user_query, response, model):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO conversations (session_id, user_query, response, model)
        VALUES (?, ?, ?, ?)
    """, (session_id, user_query, response, model))

    conn.commit()
    conn.close()


# ==============================
# 🔥 SAFE EMBEDDING (FIXED)
# ==============================
def safe_embed(text):
    emb = embed_query(text)

    if emb is None:
        raise HTTPException(500, "Embedding failed")

    emb = np.array(emb).astype("float32")

    # ✅ MUST be 2D for FAISS
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    return emb


# ==============================
# 🔥 LOAD VECTOR STORE
# ==============================
def load_doc_vectorstore(doc_name: str):

    if not os.path.exists(DATA_ROOT):
        return None

    safe_name = os.path.splitext(doc_name)[0].replace(" ", "_")
    dataset_path = os.path.join(DATA_ROOT, safe_name)

    index_path = os.path.join(dataset_path, "vector.index")
    metadata_path = os.path.join(dataset_path, "chunks.json")

    if not os.path.exists(index_path):
        print(f"❌ No index found for {safe_name}")
        return None

    vs = VectorStore(dim=EMBEDDING_DIM)
    vs.load(index_path, metadata_path)

    print(f"✅ Loaded embeddings: {safe_name}")
    return vs


# ==============================
# 🔥 QUERY TYPE DETECTION
# ==============================
def is_generic_query(query: str):
    patterns = [
        "explain", "summary", "overview",
        "what is this", "describe",
        "tell me about", "what is this document"
    ]
    q = query.lower()
    return any(p in q for p in patterns)


# ==============================
# 🔥 RETRIEVE
# ==============================
def retrieve_doc(query, vs: VectorStore, k=4):
    emb = safe_embed(query)
    results = vs.search(emb, k)

    print("\n--- RETRIEVED CHUNKS ---")
    for i, c in enumerate(results):
        print(f"[{i+1}] {c[:200]}")
    print("------------------------\n")

    return results


# ==============================
# 🔹 CLEAN CHUNKS
# ==============================
def clean_chunks(chunks):
    cleaned = []
    for c in chunks:
        if len(c) > 100 and "license" not in c.lower():
            cleaned.append(c.strip())
    return cleaned


# ==============================
# 🔹 PROMPT
# ==============================
def build_prompt(query, context, history):

    history_text = ""
    for q, r in history:
        history_text += f"User: {q}\nAssistant: {r}\n"

    return f"""
You are a document assistant.

TASK:
Explain the document clearly using the provided context.

RULES:
- Use only the given context
- If information is partial, still summarize what is available
- Do NOT hallucinate outside context

Conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer:
- Give a clear explanation
- Include purpose, scope, and key ideas
"""


# ==============================
# 🔥 MAIN ENDPOINT
# ==============================
@router.post("/")
def chat_with_doc(req: DocChatRequest):

    session_id = req.session_id or str(uuid.uuid4())

    # 🔹 Load embeddings
    vs = load_doc_vectorstore(req.doc)
    if not vs:
        raise HTTPException(404, f"No embeddings found for {req.doc}")

    # 🔹 History
    history = get_chat_history(session_id)

    # 🔥 Smart retrieval
    if is_generic_query(req.query):
        print("⚡ Generic query detected → summary mode")

        chunks = retrieve_doc(
            f"overview scope purpose key concepts of {req.doc}",
            vs,
            k=8
        )
    else:
        retrieval_query = f"{req.query} in {req.doc}"
        chunks = retrieve_doc(retrieval_query, vs, k=4)

    # 🔹 Clean chunks
    clean = clean_chunks(chunks)

    # 🔹 Build context
    if not clean:
        context = "No relevant information found in document."
    else:
        context = "DOCUMENT INSIGHTS:\n"
        for c in clean[:6]:
            context += f"- {c}\n"

    print("\n--- FINAL CONTEXT ---")
    print(context[:1000])
    print("---------------------\n")

    # 🔹 Prompt
    prompt = build_prompt(req.query, context, history)

    # 🔥 Token safety
    if len(prompt) > 4000:
        prompt = prompt[:4000]

    # 🔥 LLM CALL (FIXED)
    llm_res = ask_llm(
        prompt,
        query=req.query,
        context=context
    )

    response = llm_res["response"]
    model_used = llm_res["model"]

    # 🔹 Save
    save_conversation(session_id, req.query, response, model_used)

    return {
        "session_id": session_id,
        "doc": req.doc,
        "response": response.strip(),
        "context_used": clean,
        "model": model_used
    }