from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import fitz
import re
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

router = APIRouter()

# ==============================
# 🔹 CONFIG
# ==============================
DATA_DIR = "embeddings"
os.makedirs(DATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

model = SentenceTransformer("BAAI/bge-base-en", device=device)


# ==============================
# 🔹 PDF LOADER
# ==============================
def load_pdf(file_path):
    doc = fitz.open(file_path)
    return " ".join([page.get_text() for page in doc])


# ==============================
# 🔹 CLEAN TEXT
# ==============================
def clean_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:\-\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==============================
# 🔥 BETTER CHUNKING (WORD BASED)
# ==============================
def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk.strip()) > 50:  # 🔥 filter tiny chunks
            chunks.append(chunk)

    return chunks[:500]  # 🔥 safety limit


# ==============================
# 🔹 VECTOR STORE
# ==============================
class VectorStore:
    def __init__(self, dim=768):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings).astype("float32")

        if len(vectors) == 0:
            raise ValueError("No embeddings to add")

        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        query_vector = np.array(query_embedding).astype("float32")

        # 🔥 FIX: ensure 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)

        return [
            self.texts[i]
            for i in indices[0]
            if i < len(self.texts)
        ]

    def save(self, path):
        faiss.write_index(self.index, os.path.join(path, "vector.index"))

        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(self.texts, f)

    def load(self, path):
        self.index = faiss.read_index(os.path.join(path, "vector.index"))

        with open(os.path.join(path, "chunks.json"), "r") as f:
            self.texts = json.load(f)


# ==============================
# 🔥 SAFE EMBEDDING
# ==============================
def generate_embeddings(chunks):

    try:
        print(f"⚡ Embedding {len(chunks)} chunks on {device}")

        embeddings = model.encode(
            chunks,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings

    except RuntimeError as e:
        if "CUDA" in str(e):
            print("⚠️ CUDA OOM → switching to CPU")

            cpu_model = SentenceTransformer("BAAI/bge-base-en", device="cpu")

            embeddings = cpu_model.encode(
                chunks,
                batch_size=4,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            return embeddings
        else:
            raise e

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================
# 🔥 MAIN API
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
        # 🔹 TEXT PIPELINE
        raw_text = load_pdf(file_path)

        if not raw_text.strip():
            raise HTTPException(400, "Empty PDF")

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        if not chunks:
            raise HTTPException(400, "No valid chunks created")

        # 🔥 EMBEDDINGS
        embeddings = generate_embeddings(chunks)

        # 🔹 SAVE
        save_path = os.path.join(DATA_DIR, safe_name)
        os.makedirs(save_path, exist_ok=True)

        vs = VectorStore()
        vs.add(embeddings, chunks)
        vs.save(save_path)

        print(f"✅ Saved embeddings → {save_path}")

        return {
            "message": "Document embedded successfully 🚀",
            "document": safe_name,
            "chunks": len(chunks)
        }

    except Exception as e:
        print("❌ Embedding error:", str(e))
        raise HTTPException(500, str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)