from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import fitz  # PyMuPDF
import re
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from utils.pipeline import DATA_DIR, VectorStore, chunk_text, clean_text, embed_query, generate_embeddings, load_pdf
from pydantic import BaseModel
import sqlite3
import uuid
from utils.llm import ask_llm

router = APIRouter()

# ==============================
# 🔹 CONFIG
# ==============================
EMBEDDING_DIM = 768
DATA_ROOT = "embeddings"

# ==============================
# 🔥 UPLOAD + EMBED API
# ==============================
@router.post("/upload-and-embed/")
async def upload_and_embed(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF allowed")

    original_name = os.path.splitext(file.filename)[0]
    safe_name = original_name.replace(" ", "_")

    file_path = f"temp_{safe_name}.pdf"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # 🔹 Extract text
        raw_text = load_pdf(file_path)

        if not raw_text.strip():
            raise HTTPException(400, "Empty PDF")

        # 🔹 Clean + chunk
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        if not chunks:
            raise HTTPException(400, "No valid chunks created")

        print("📦 Total chunks:", len(chunks))

        # 🔹 Generate embeddings
        embeddings = generate_embeddings(chunks)

        # 🔹 Save vector store
        dim = embeddings.shape[1]
        vs = VectorStore(dim)

        vs.add(embeddings, chunks)

        save_path = os.path.join(DATA_DIR, safe_name)
        os.makedirs(save_path, exist_ok=True)

        vs.save(
            index_path=os.path.join(save_path, "vector.index"),
            metadata_path=os.path.join(save_path, "metadata.json")
        )

        return {
            "message": "Document embedded successfully 🚀",
            "document": safe_name,
            "chunks": len(chunks)
        }

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(500, str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

