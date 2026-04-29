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