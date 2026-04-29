from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import uuid
from dotenv import load_dotenv
from duckduckgo_search import DDGS

from pipeline.vector_store import VectorStore
from pipeline.embedder import embed_query
from utils.llm import ask_llm

load_dotenv()

router = APIRouter()

EMBEDDING_DIM = 768
DATA_ROOT = "embeddings"


# ==============================
# 🔹 DB HELPERS
# ==============================
def get_chat_history(session_id, limit=5):
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
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


def save_conversation(
    session_id,
    user_query,
    response,
    model,
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0
):
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO conversations 
    (session_id, user_query, response, model,
     prompt_tokens, completion_tokens, total_tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_query,
        response,
        model,
        prompt_tokens,
        completion_tokens,
        total_tokens
    ))

    conn.commit()
    conn.close()


# ==============================
# 🔹 LOAD VECTOR STORES
# ==============================
def load_all_vectorstores():
    stores = []

    if not os.path.exists(DATA_ROOT):
        return stores

    for dataset in os.listdir(DATA_ROOT):
        dataset_path = os.path.join(DATA_ROOT, dataset)

        if not os.path.isdir(dataset_path):
            continue

        index_path = os.path.join(dataset_path, "vector.index")
        metadata_path = os.path.join(dataset_path, "chunks.json")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            vs = VectorStore(dim=EMBEDDING_DIM)
            vs.load(index_path, metadata_path)

            stores.append({"name": dataset, "store": vs})
            print(f"✅ Loaded: {dataset}")

    return stores


stores = load_all_vectorstores()


# ==============================
# 🔹 RETRIEVE
# ==============================
def retrieve(query):
    if not stores:
        return []

    try:
        query_embedding = embed_query(query)
    except Exception as e:
        print("Embedding error:", e)
        return []

    results = []
    for s in stores:
        chunks = s["store"].search(query_embedding, 3)
        results.extend(chunks)

    return results[:3]


# ==============================
# 🔹 WEB SEARCH
# ==============================
def web_search(query, k=3):
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                snippet = r.get("body", "")
                link = r.get("href", "")
                title = r.get("title", "")

                if snippet and len(snippet) > 50:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "link": link
                    })
    except Exception as e:
        print("Web search error:", e)

    return results[:k]


# ==============================
# 🔹 PROMPT BUILDER
# ==============================
def build_prompt(query, context, history):
    history_text = ""
    for q, r in history:
        history_text += f"User: {q}\nAssistant: {r}\n"

    return f"""
You are an expert in IEC standards.
Use the provided context. If web info is present, prioritize recent facts.

Conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer clearly with bullet points where helpful.
"""


# ==============================
# 🔹 REQUEST MODEL
# ==============================
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    use_web: bool = False
    model: str | None = None


# ==============================
# 🔥 CHAT ENDPOINT
# ==============================
@router.post("/")
def chat(req: ChatRequest):

    session_id = req.session_id or str(uuid.uuid4())
    history = get_chat_history(session_id)

    context_chunks = []
    web_references = []

    # 🔹 Vector DB
    vector_chunks = retrieve(req.query)
    if vector_chunks:
        context_chunks.extend(vector_chunks)

    # 🔹 Web Search
    if req.use_web:
        web_results = web_search(req.query)

        for w in web_results:
            context_chunks.append(w["snippet"])
            web_references.append({
                "title": w["title"],
                "url": w["link"]
            })

    # 🔹 Context build
    if context_chunks:
        context = ""
        for i, chunk in enumerate(context_chunks):
            context += f"[Source {i+1}]: {chunk}\n\n"
    else:
        context = "No context available."

    prompt = build_prompt(req.query, context, history)

    # 🔥 trim safety
    if len(prompt) > 4000:
        prompt = prompt[:4000]

    # 🔥 LLM CALL
    llm_res = ask_llm(
        prompt,
        query=req.query,
        context=context,
        model=req.model
    )

    response = llm_res["response"]
    model_used = llm_res["model"]

    # 🔥 save
    save_conversation(
        session_id,
        req.query,
        response,
        model_used,
        llm_res["prompt_tokens"],
        llm_res["completion_tokens"],
        llm_res["total_tokens"]
    )

    return {
        "session_id": session_id,
        "query": req.query,
        "context_used": context_chunks,
        "response": response,
        "model": model_used,
        "web_references": web_references if req.use_web else []
    }