# Retrieval

#Given a user question, embed it with the same model used at ingestion time,
#then fetch the top-k most similar chunks from the Chroma vector store.


import chromadb
from google import genai

from config import (
    GEMINI_API_KEY, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K,
)


def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_collection(COLLECTION_NAME)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Compares the user query to the vector database and collects the top k hits

    Args:
        query (str):
        top_k (int, optional): Defaults to TOP_K.
    Returns:
        list[dict]: 
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    #Takes in the user's question and converts into numbers(vectors) to search the vector database
    query_embedding = client.models.embed_content(
        model=EMBEDDING_MODEL, contents=[query]
    ).embeddings[0].values

    collection = get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "pages": meta.get("pages", "unknown"),
            "distance": dist,
        })
    return hits


if __name__ == "__main__":
    #Quick sanity-check for retrieval quality
    import sys
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    for hit in retrieve(question):
        print(f"[{hit['source']} p.{hit['pages']} #{hit['chunk_index']}] (dist={hit['distance']:.3f})")
        print(hit["text"][:200], "...\n")
