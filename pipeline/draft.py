
import streamlit as st
import os
import sqlite3
import uuid
import io
import openpyxl
import requests

from pipeline.vector_store import VectorStore
from pipeline.embedder import embed_query
from llama_cpp import Llama
from fastapi import FastAPI
from routers.document_management import router 

app = FastAPI()

# ✅ Register router
app.include_router(router, prefix="/api", tags=["Upload"])
# ==============================
# 🔹 CONFIG
# ==============================
EMBEDDING_DIM = 768
DATA_ROOT = "embeddings"
MODEL_PATH = "llm/gemma-3-4b-it-Q4_K_M.gguf"



def save_conversation(session_id, user_query, response, model):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO conversations (session_id, user_query, response, model)
    VALUES (?, ?, ?, ?)
    """, (session_id, user_query, response, model))

    conn.commit()
    conn.close()


def get_sessions():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT session_id FROM conversations ORDER BY id DESC")
    sessions = [row[0] for row in cursor.fetchall()]

    conn.close()
    return sessions


def get_session_messages(session_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT user_query, response 
    FROM conversations 
    WHERE session_id = ?
    ORDER BY id
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()
    return rows


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

# ==============================
# 🔹 LOAD MODEL
# ==============================
@st.cache_resource
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=20,
        n_batch=256,
        f16_kv=True,
    )

# ==============================
# 🔹 OPENROUTER
# ==============================
def ask_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "Error in OpenRouter"

# ==============================
# 🔹 VECTOR STORE
# ==============================
@st.cache_resource
def load_all_vectorstores():
    stores = []

    for category in os.listdir(DATA_ROOT):
        category_path = os.path.join(DATA_ROOT, category)

        if not os.path.isdir(category_path):
            continue

        for dataset in os.listdir(category_path):
            dataset_path = os.path.join(category_path, dataset)

            index_path = os.path.join(dataset_path, "vector.index")
            metadata_path = os.path.join(dataset_path, "chunks.json")

            if os.path.exists(index_path):
                vs = VectorStore(dim=EMBEDDING_DIM)
                vs.load(index_path, metadata_path)

                stores.append({"name": dataset, "store": vs})

    return stores

# ==============================
# 🔹 RAG
# ==============================
def retrieve(query, stores):
    query_embedding = embed_query(query)
    results = []

    for s in stores:
        chunks = s["store"].search(query_embedding, 3)
        results.extend(chunks)

    return results[:3]


def build_prompt(query, context, history):
    history_text = ""
    for q, r in history:
        history_text += f"User: {q}\nAssistant: {r}\n"

    return f"""
You are an expert in IEC standards.

Conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""


def ask_llm(query, stores, llm, session_id, use_openrouter=False):
    context_chunks = retrieve(query, stores)
    context = "\n\n".join(context_chunks)

    history = get_chat_history(session_id)

    prompt = build_prompt(query, context, history)

    if use_openrouter:
        response = ask_openrouter(prompt)
    else:
        output = llm(prompt, max_tokens=300)
        response = output["choices"][0]["text"]

    return response, context_chunks

# ==============================
# 🔹 EXCEL
# ==============================
def create_excel_from_text(text):
    wb = openpyxl.Workbook()
    ws = wb.active

    for line in text.split("\n"):
        row = [c.strip() for c in line.split("|") if c.strip()]
        if row:
            ws.append(row)

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

# ==============================
# 🔹 UI
# ==============================
# st.set_page_config(page_title="RAG Assistant", layout="wide")

# st.title("📄 Engineering Standards RAG Assistant")

# llm = load_llm()
# stores = load_all_vectorstores()

# # SESSION
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())

# if st.sidebar.button("➕ New Chat"):
#     st.session_state.session_id = str(uuid.uuid4())

# # SESSION LIST
# st.sidebar.markdown("## 💬 Sessions")
# sessions = get_sessions()

# for s in sessions:
#     if st.sidebar.button(s[:8]):
#         st.session_state.session_id = s

# # LLM choice
# llm_choice = st.sidebar.selectbox("LLM", ["Gemma", "OpenRouter"])

# # SHOW CHAT
# messages = get_session_messages(st.session_state.session_id)

# for q, r in messages:
#     with st.chat_message("user"):
#         st.markdown(q)
#     with st.chat_message("assistant"):
#         st.markdown(r)

# # INPUT
# query = st.chat_input("Ask...")

# if query:
#     with st.chat_message("user"):
#         st.markdown(query)

#     use_openrouter = llm_choice == "OpenRouter"

#     response, context = ask_llm(
#         query,
#         stores,
#         llm,
#         st.session_state.session_id,
#         use_openrouter
#     )

#     with st.chat_message("assistant"):
#         st.markdown(response)

#         if "excel" in query.lower():
#             file = create_excel_from_text(response)
#             st.download_button("Download Excel", file, "data.xlsx")

#     save_conversation(
#         st.session_state.session_id,
#         query,
#         response,
#         llm_choice
#     )

#     st.rerun()