from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import fitz  # PyMuPDF
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
cpu_model = SentenceTransformer("BAAI/bge-base-en", device="cpu")


# ==============================
# 🔥 PDF LOADER
# ==============================
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text += b[4] + "\n"

    return text


# ==============================
# 🔹 CLEAN TEXT
# ==============================
def clean_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # safer
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==============================
# 🔥 SMART CHUNKING (sentence-based)
# ==============================
def chunk_text(text, chunk_size=300, overlap=50):
    sentences = text.split(". ")
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "

    if current:
        chunks.append(current.strip())

    return chunks


# ==============================
# 🔹 VECTOR STORE
# ==============================
class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.texts = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array(query_embedding).astype("float32")
        distances, indices = self.index.search(query_vector, k)

        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

    def save(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, indent=2)

    def load(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)


# ==============================
# 🔥 EMBEDDING FUNCTION
# ==============================
def generate_embeddings(chunks):
    try:
        print(f"⚡ Embedding {len(chunks)} chunks on {device}")

        embeddings = model.encode(
            chunks,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings

    except RuntimeError as e:
        if "CUDA" in str(e):
            print("⚠️ CUDA OOM → switching to CPU")

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
def embed_query(query):
    return model.encode(
        [query],
        normalize_embeddings=True
    )