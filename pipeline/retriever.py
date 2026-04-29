

def retrieve(query, embedder, vector_store, k=5):
    query_vec = embedder.embed_query(query)
    results = vector_store.search(query_vec, k)
    return results