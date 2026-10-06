import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() #allows me to retrieve the gemini api key from the .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#the paths variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" # PDFs
CHROMA_DIR = PROJECT_ROOT / ".chroma_store" # local vector store, automatically created when running
COLLECTION_NAME = "clinical_docs"

# models that will be used
# Go to https://ai.google.dev/gemini-api/docs/models for current model names
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash-lite"

# Chunking and retrieval variables. Will play with the values as I iterate
CHUNK_SIZE = 1000 # characters per chunk
CHUNK_OVERLAP = 150 # overlap between consecutive chunks, preserves context across splits

TOP_K = 5 # number of chunks retrieved per query
