
# Ingestion pipeline.

# Loads every PDF in DATA_DIR, splits it into overlapping text chunks,
# embeds each chunk with Gemini's embedding model, and writes the
# vectors + text + source metadata into a local Chroma collection.

# ** Run this once whenever you add/change documents **


import sys
import time
import re
from collections import deque
from pathlib import Path
# Import vector store database for chunks
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import errors as genai_errors

# importing variables from configurations file
from config import (
    GEMINI_API_KEY, DATA_DIR, CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)
# Handling the rate-limits
BATCH_SIZE = 10 # Smaller to control the number of chunks lost in each failed request
MAX_RETRIES = 5
INITIAL_BACKOFF = 10.0

# Dynamic rate limiter for embedding process.
# Limiter automatically adjusts pace to stay under the rest limit
# instead of a set delay
REQUESTS_PM = 90


class AdaptiveRateLimiter:
    """
    Watches the recent request timestamps within a minute and
    sleeps just long enough to remain under max RPM limit before performing next request.
    """

    def __init__(self, requests_per_minute: int):
        """
        Args:
            requests_per_minute (int):
        """
        self.min_interval = 60/ requests_per_minute
        self.last_request_time: float | None = None

    def wait(self):
        now = time.monotonic()
        if self.last_request_time is not None:
            elapsed = now - self.last_request_time
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_time = time.monotonic()

def _extract_retry_after(error: genai_errors.ClientError) -> float | None:
    """
    Attempt at reading the API's own suggested wait time instead of guessing one.
    If not present, return None. Caller will use exponential backoff rather.

    Args:
        error (genai_errors.ClientError):
    Returns:
        float | None: 
    """

    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) if response else None
    if headers and "retry_after" in {k.lower() for k in headers.keys()}:
        for k,v in headers.items():
            if k.lower() == "retry-after":
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    #Checks first for a millisecond time match for wait times in that order of magnitude
    #before switching to seconds match for wait times of that order of magnitude
    ms_match = re.search(r"retry.{0,20}?(\d+(?:\.\d+)?)\s*ms\b", str(error), re.IGNORECASE)
    if ms_match:
        return float(ms_match.group(1)) / 1000.0

    s_match = re.search(r"retry.{0,20}?(\d+(?:\.\d+)?)\s*s\b", str(error), re.IGNORECASE)
    if s_match:
        return float(s_match.group(1))

    return None

LONG_WAIT_THRESHOLD_SECONDS = 60

def _format_duration(seconds: float) -> str:
    """
    Args:
        seconds (float):
    Returns:
        str: 
    """
    if seconds < 60:
        return f"{seconds:,0f}s"
    minutes = seconds /60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes:.1f}h"


# Function to take in text from a PDF
def load_pdf_pages(pdf_path: Path) -> str:
    """Extract raw text from a single PDF per page.

    Args:
        pdf_path (Path):
    Returns:
        str: 
    """
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return pages

def build_text(pages: list[str]) -> tuple[str, list[dict]]:
    """
    Joins per-page text while noting each page's character offset range.
    Further on chunking will ensure each character span is mapped to the page(s)
    it is situated in.

    Args:
        pages (list[str]):
    Returns:
        tuple[str | list[dict]]: 
    """
    parts = []
    offsets = []
    cursor = 0
    for i,page_text in enumerate(pages):
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        offsets.append({"page": i + 1, "start": start, "end": cursor})
        if i < len(pages)-1:
            parts.append("\n")
            cursor += 1
    return "".join(parts), offsets


def pages_for_range(range_begin: int, range_end: int, page_offsets: list[dict]) -> list[int]:
    """
    Maps a chunk's character range back to the page number(s) it overlaps

    Args:
        range_begin (int):
        range_end (int):
        page_offsets (list[dict]):
    Returns:
        list[int]: 
    """
    return [
        po["page"] for po in page_offsets
        if po["start"] < range_end and po["end"] > range_begin
    ]


def format_page_label(pages: list[int]) -> str:
    """
    Formats the page presentation in citation.

    Args:
        pages (list[int]):
    Returns:
        str: 
    """
    if not pages:
        return "unknown"
    unique_sorted = sorted(set(pages))
    if len(unique_sorted) == 1:
        return str(unique_sorted[0])
    if unique_sorted == list(range(unique_sorted[0], unique_sorted[-1] + 1)):
        return f"{unique_sorted[0]}-{unique_sorted[-1]}"
    return ",".join(str(p) for p in unique_sorted)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Args:
        text (str):
        chunk_size (int):
        overlap (int):
    Returns:
        list[str]: 
    """
    chunks = []
    start = 0 # Tracks index point in text characters
    while start < len(text): #continues as long as the index point is less than the entire text
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk: # Checking if chunk is empty from being stripped of whitespace, appends chunks only if not the case
            chunks.append({"text": chunk, "start": start, "end": end})
        start += chunk_size - overlap # Advances the window in the current chunk
    return chunks # chunks returns a list of strings with each being as chunk

class LongQuotaWait(Exception):
    """
    Will be raised when API sends a retry-after time that is too long.
    """
# Takes text from the chunks and converts to vector embeddings
def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Embeds a batch of texts using the Gemini embedding model.
    A short wait window from a 429 error will be associated with a rpm limit contravention.
    Run will be paused for that amount of time before resumption.

    A long wait window (>= LONG_WAIT_THRESHOLD) from a 429 error will be associated with a daily limit contravention.
    Run will bbe stopped.

    Fallback is the exponential backoff.

    Args:
        client (genai.Client):
        texts (list[str]):
    Returns:
        list[list[float]]: 
    Raises:
        LongQuotaWait: If raw_wait is not None and raw_wait > LONG_WAIT_THRESHOLD_SECONDS
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES+1):
        try:
            result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
            return [e.values for e in result.embeddings]
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if not is_rate_limit:
                raise

            #Shows the real error
            raw_message = str(e)
            print(f"  [429 raw error] {raw_message[:500]}")

            raw_wait = _extract_retry_after(e)

            if raw_wait is not None and raw_wait > LONG_WAIT_THRESHOLD_SECONDS:
                raise LongQuotaWait(
                    f"API requires a wait time of {_format_duration(raw_wait)} seconds "
                    f"to proceed with operation, inline with - likely - a daily limit exhaustion."
                    f"Not worth retrying within the current run."
                ) from e

            if attempt >= MAX_RETRIES:
                raise

            wait_time = (raw_wait + 0.5) if raw_wait is not None else backoff
            print(f" Rate limited. Waiting {wait_time:.0f} seconds before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait_time)
            backoff *= 2



def main():
    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No PDFs found in {DATA_DIR}. Add some source documents and re-run.")

    print(f"Found {len(pdf_files)} PDF(s) in {DATA_DIR}")

    client = genai.Client(api_key=GEMINI_API_KEY)
    limiter = AdaptiveRateLimiter(REQUESTS_PM)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing_names = [c.name for c in chroma_client.list_collections()]
    collection= (
        chroma_client.get_collection(COLLECTION_NAME)
        if COLLECTION_NAME in existing_names
        else chroma_client.create_collection(COLLECTION_NAME)
    )

# Allows for resumability in the case a run has failed due to rate limits.
# Gives a checkpoint for the next run in the process maintaining the chunks
# that have already been collected by skipping already embedded IDs.
    already_embedded_ids = set(collection.get(include=[])["ids"])
    previously_done_count = len(already_embedded_ids)
    if previously_done_count:
        print(f"Found {previously_done_count} chunk(s) already embedded from a previous run. Resuming...")

    chunks_to_embed, ids_to_embed, metadatas_to_embed = [], [], []
    total_corpus_chunks = 0

    for pdf_path in pdf_files:
        print(f"  Processing {pdf_path.name}...")
        pages = load_pdf_pages(pdf_path)
        text, page_offsets = build_text(pages)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total_corpus_chunks += len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_path.stem}_{i}"
            if chunk_id in already_embedded_ids:
                continue
            chunk_pages = pages_for_range(chunk["start"], chunk["end"], page_offsets)
            chunks_to_embed.append(chunk["text"])
            ids_to_embed.append(chunk_id)
            metadatas_to_embed.append({
                "source": pdf_path.name,
                "chunk_index": i,
                "pages": format_page_label(chunk_pages)
            })  
        print(f"    -> {len(chunks)} chunks")

        chunks_left = len(chunks_to_embed)

    if chunks_left == 0:
        print(f"Nothing new to embed, all chunks already embedded.")
        print(f"Done. Vector store saved to {CHROMA_DIR}")
        return

    print(f"Embedding {len(chunks_to_embed)} total chunk(s) (skipped {len(already_embedded_ids)} already done)...")

    for start in range(0, chunks_left, BATCH_SIZE):
        end = min(start + BATCH_SIZE, chunks_left)
        batch_chunks = chunks_to_embed[start:end]
        batch_ids = ids_to_embed[start:end]
        batch_metadatas = metadatas_to_embed[start:end]

        limiter.wait()
        embeddings = embed_texts(client, batch_chunks)

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_metadatas,
        )
        pct = end / chunks_left *100
        print(f"  Embedded {end}/{chunks_left} in this run ({pct:.0f}%)")

    print(f"Done. Vector store saved to {CHROMA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except LongQuotaWait as e:
        print(f"\n{e}")
        print("Progress is saved in vector store, re-run script after quota resets to resume.")
        sys.exit(1)
