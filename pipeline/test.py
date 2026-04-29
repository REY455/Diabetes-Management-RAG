
import fitz

def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    
    for page in doc:
        text += page.get_text()
    
    return text
text = load_pdf("data/Engneering_standards/IEC 62443-2-1.pdf")



def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

import re

def clean_text(text):
    # Fix broken spacing like "haveahighincidence"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Fix hyphen words
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Remove garbage characters
    text = re.sub(r'[^a-zA-Z0-9.,;:\-\s]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


from sentence_transformers import SentenceTransformer
import torch

# auto device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("BAAI/bge-base-en", device=device)

def embed_text(chunks):
    return model.encode(
        chunks,
        batch_size=32,                 # 🔥 speed boost
        show_progress_bar=True,
        normalize_embeddings=True      # 🔥 improves similarity search
    )

def embed_query(query):
    return model.encode(
        [query],
        normalize_embeddings=True
    )

def retrieve(query, embedder, vector_store, k=5):
    query_vec = embedder.embed_query(query)
    results = vector_store.search(query_vec, k)
    return results

import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        query_vector = np.array(query_embedding).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return [self.texts[i] for i in indices[0]]

    # 🔥 Save index + metadata
    def save(self, index_path="vector.index", metadata_path="metadata.json"):
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, indent=2)

    # 🔥 Load index + metadata
    def load(self, index_path="vector.index", metadata_path="metadata.json"):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)