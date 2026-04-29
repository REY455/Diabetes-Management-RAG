
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