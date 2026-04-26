"""
Simple RAG in Python (Beginner Friendly)

Requirements covered:
1) Load a text file
2) Split text into chunks
3) Convert text into embeddings (using Gemini API)
4) Store embeddings in FAISS
5) Take user input
6) Retrieve relevant chunks
7) Send context to Gemini
8) Print answer

Usage:
    1. Set the GEMINI_API_KEY environment variable.
    2. Run:
       python rag_simple.py sample.txt
"""

import os
import sys

import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv


# Simple settings you can tweak
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MIN_SIMILARITY = 0.40
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-2.5-flash"


def setup_gemini(api_key: str = None) -> None:
    """Configure the Gemini API key."""
    load_dotenv()
    if api_key:
        genai.configure(api_key=api_key)
    elif os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    else:
        raise ValueError("No Gemini API Key provided. Please set the GEMINI_API_KEY environment variable or create an .env file.")


def load_text(file_path: str) -> str:
    """Read a UTF-8 text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError("The file is empty. Please provide a text file with content.")

    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Get embeddings from Gemini in batches to avoid 429 errors."""
    # The API can handle up to 100 texts per batch
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch  # Passing the list here triggers batch processing
        )
        all_embeddings.extend(response['embedding'])
        
    return np.array(all_embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Get embeddings for a search query from Gemini."""
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    return np.array([response['embedding']], dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Create and fill a FAISS cosine-similarity index."""
    if len(vectors.shape) != 2:
        raise ValueError("Embeddings must be a 2D array.")

    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def retrieve_chunks(
    index: faiss.IndexFlatIP,
    chunks: list[str],
    question: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> list[str]:
    """Embed user question, search FAISS, return relevant chunks only."""
    query_vector = embed_query(question)
    faiss.normalize_L2(query_vector)
    k = min(top_k, len(chunks))
    similarities, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(similarities[0], indices[0]):
        if score < min_similarity:
            continue
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


def ask_llm(question: str, context_chunks: list[str]) -> str:
    """Ask Gemini using retrieved context."""
    if not context_chunks:
        return "I don't know based on the provided document."

    context = "\n\n---\n\n".join(context_chunks)

    prompt = (
        "You are a helpful assistant. "
        "Answer using only the provided context. "
        "If the answer is not in the context, say: "
        "'I don't know based on the provided document.'\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)

    return response.text or "No answer returned."


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python rag_simple.py <path_to_text_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        text = load_text(file_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    chunks = chunk_text(text)
    if not chunks:
        print("Error: Could not create chunks from the file.")
        sys.exit(1)

    try:
        setup_gemini()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks from: {file_path}")
    print("Creating embeddings and FAISS index...")

    try:
        chunk_vectors = embed_texts(chunks)
    except Exception as exc:
        print("Error: Could not create embeddings with Gemini.")
        print(f"Details: {exc}")
        print("Make sure your API key is correct and you have an internet connection.")
        sys.exit(1)
    index = build_faiss_index(chunk_vectors)

    question = input("\nAsk a question: ").strip()
    if not question:
        print("No question entered. Exiting.")
        sys.exit(0)

    context_chunks = retrieve_chunks(index, chunks, question, top_k=TOP_K)
    print(f"Retrieved {len(context_chunks)} relevant chunks.")

    try:
        answer = ask_llm(question, context_chunks)
    except Exception as exc:
        print("Error: Could not generate answer with Gemini.")
        print(f"Details: {exc}")
        sys.exit(1)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()