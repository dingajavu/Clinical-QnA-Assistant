# Streamlit UI build for RAG deployment

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import GEMINI_API_KEY, CHROMA_DIR, COLLECTION_NAME  # noqa: E402

st.set_page_config(page_title="Clinical Guidelines Assistant", page_icon="🩺", layout="centered")

st.title("🩺 Clinical Guidelines Assistant")
st.caption("A retrieval-augmented generation assistant where answers are grounded in and cited to your "
           "source documents.")

# Checks whether the setup steps are in order before proceeding
if not GEMINI_API_KEY:
    st.error("No API key found. Copy Add your key to .env file.")
    st.stop()

if not CHROMA_DIR.exists():
    st.warning(
        "No vector store found. Add PDFs to dataset and run "
        "ingestion before using the app."
    )
    st.stop()

from generate import answer_question  # noqa: E402

# Q&A interface
question = st.text_input("Ask a question about medical guidelines:", placeholder="e.g. What is standard practice in "
                                                                             "medical device design?")

top_k = st.slider("Number of source chunks to retrieve", min_value=2, max_value=10, value=5)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving relevant context and generating answer..."):
        try:
            result = answer_question(question, top_k=top_k)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.markdown("### Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.markdown("### Sources used")
        for hit in result["sources"]:
            with st.expander(
                    f"{hit['source']} — p. {hit['pages']}"
                    f"chunk #{hit['chunk_index']} (distance: {hit['distance']:.3f})"
            ):
                st.write(hit["text"])

st.divider()
st.caption(
    "Tip: lower distance = more semantically similar chunk. "
    "If answers seem ungrounded, try adjusting chunk size or overlap in configuration and rerun ingestion."
)
