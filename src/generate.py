# Generation
#LLM's instructions of operation and execution framework

from google import genai

from config import GEMINI_API_KEY, GENERATION_MODEL
from retrieve import retrieve


#Sets the context and operational constraints the system will follow when approaching answering
SYSTEM_PROMPT = """You are a clinical documentation assistant. Answer the user's question
using ONLY the provided context excerpts below. Follow these rules strictly:

1. If the context does not contain enough information to answer, say so clearly —
   do not use outside knowledge or make anything up.
2. After each claim, cite the source it came from using the format [source_name, p.PAGE].
3. Be concise and precise. Do not repeat the question back.
4. If excerpts conflict, note the disagreement rather than picking one silently.
"""


def build_context_block(hits: list[dict]) -> str:
    """
    Args:
        hits (list[dict]):
    Returns:
        str: 
    """
    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit['source']}, p.{hit['pages']} #{hit['chunk_index']}]\n{hit['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, top_k: int = 5) -> dict:
    """
    Args:
        question (str):
        top_k (int, optional): Defaults to 5.
    Returns:
        dict: 
    """
    hits = retrieve(question, top_k=top_k)

    if not hits:
        return {
            "answer": "No relevant documents found. Ensure you have run ingestion.",
            "sources": [],
        }

    context_block = build_context_block(hits)

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_block}\n\nQUESTION: {question}\n\nANSWER:"

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return {"answer": response.text, "sources": hits}

#Sanity-check
if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    result = answer_question(question)
    print(result["answer"])
    print("\n--- Sources used ---")
    for s in result["sources"]:
        print(f"  {s['source']} p.{s['pages']} #{s['chunk_index']}")
