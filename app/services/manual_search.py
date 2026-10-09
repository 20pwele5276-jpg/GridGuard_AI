from functools import lru_cache
from io import BytesIO

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_manual_chunks(
    pdf_bytes,
    source_name,
    chunk_size=1000,
    overlap=150,
):
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "source": source_name,
                    "page": page_number,
                    "text": chunk_text,
                })

            start += chunk_size - overlap

    if not chunks:
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or contain no extractable text."
        )

    return chunks


def build_manual_index(chunks):
    if not chunks:
        raise ValueError("No manual passages were provided.")

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_manual(chunks, index, question, top_k=4):
    if not question.strip():
        raise ValueError("Please enter a question.")

    if not chunks or index.ntotal == 0:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position < 0:
            continue

        result = chunks[int(position)].copy()
        result["similarity"] = float(
            np.clip(score, 0.0, 1.0)
        )
        results.append(result)

    return results