from fastapi import APIRouter
from pydantic import BaseModel
import sqlite3
import os
import uuid
from dotenv import load_dotenv
from duckduckgo_search import DDGS

from utils.pipeline import VectorStore, embed_query
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


def save_conversation(session_id, user_query, response, model):
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO conversations 
    (session_id, user_query, response, model)
    VALUES (?, ?, ?, ?)
    """, (session_id, user_query, response, model))

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

        index_path = os.path.join(dataset_path, "vector.index")
        metadata_path = os.path.join(dataset_path, "metadata.json")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            vs = VectorStore(dim=EMBEDDING_DIM)
            vs.load(index_path, metadata_path)

            stores.append(vs)
            print(f"✅ Loaded: {dataset}")

    return stores


stores = load_all_vectorstores()


# ==============================
# 🔹 SAFETY FILTER
# ==============================
def is_medical_risk(query: str) -> bool:
    risky_keywords = [
        "dose", "dosage", "insulin amount",
        "which medicine", "what drug",
        "treatment plan", "prescribe",
        "how much insulin"
    ]
    return any(k in query.lower() for k in risky_keywords)


# ==============================
# 🔹 CONTEXT QUALITY CHECK
# ==============================
def is_context_weak(chunks):
    if not chunks:
        return True

    avg_len = sum(len(c) for c in chunks) / len(chunks)
    return avg_len < 80


# ==============================
# 🔹 RETRIEVE
# ==============================
def retrieve(query):
    if not stores:
        return []

    if "diabetes" not in query.lower():
        query = f"diabetes {query}"

    try:
        query_embedding = embed_query(query)
    except Exception as e:
        print("Embedding error:", e)
        return []

    results = []
    for store in stores:
        results.extend(store.search(query_embedding, 3))

    return results[:5]


# ==============================
# 🔹 WEB SEARCH
# ==============================
def web_search(query, k=3):
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                snippet = r.get("body", "")
                if snippet and len(snippet) > 50:
                    results.append(snippet)
    except Exception as e:
        print("Web search error:", e)

    return results


# ==============================
# 🔹 PROMPT BUILDER
# ==============================
def build_prompt(query, context, history, use_context=True):

    history_text = ""
    for q, r in history:
        history_text += f"User: {q}\nAssistant: {r}\n"

    if use_context:
        return f"""
You are a medical assistant specialized in diabetes.

RULES:
- Use context if helpful
- If incomplete, use general knowledge
- Do NOT give dosage or prescriptions
- If treatment advice → say consult a doctor

Conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer clearly:
"""
    else:
        return f"""
You are a medical assistant.

Explain the following clearly and safely:

{query}

Do NOT give dosage or prescriptions.
Use simple explanation.
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

    # 🔥 SAFETY CHECK
    if is_medical_risk(req.query):
        return {
            "session_id": session_id,
            "query": req.query,
            "response": "Please consult a healthcare professional for medical advice.",
            "model": "safety-filter",
            "context_used": [],
            "web_references": []
        }

    # 🔹 Retrieve context
    context_chunks = retrieve(req.query)

    if req.use_web:
        context_chunks.extend(web_search(req.query))

    weak = is_context_weak(context_chunks)

    # 🔹 Build context text
    context = ""
    for i, chunk in enumerate(context_chunks):
        context += f"[Source {i+1}]\n{chunk}\n\n"

    # 🔹 Build prompt
    prompt = build_prompt(
        req.query,
        context,
        history,
        use_context=not weak
    )

    # 🔥 LLM CALL
    llm_res = ask_llm(prompt, model=req.model)

    response = llm_res["response"]
    model_used = llm_res["model"]

    # 🔹 Save
    save_conversation(session_id, req.query, response, model_used)

    return {
        "session_id": session_id,
        "query": req.query,
        "response": response,
        "model": model_used,
        "context_used": context_chunks,
        "web_references": []
    }









