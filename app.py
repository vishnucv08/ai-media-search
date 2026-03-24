import streamlit as st
import numpy as np
import faiss
import os


client = OpenAI()

from sentence_transformers import SentenceTransformer


# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Read files
# -----------------------------
def load_files(folder):

    documents = []
    paths = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if file.endswith(".txt"):

            with open(path, "r", encoding="utf-8") as f:

                text = f.read()

                documents.append(text)

                paths.append(path)

    return documents, paths


# -----------------------------
# Split text into chunks
# -----------------------------
def split_chunks(text, size=200):

    words = text.split()

    chunks = []

    for i in range(0, len(words), size):

        chunk = " ".join(words[i:i+size])

        chunks.append(chunk)

    return chunks


# -----------------------------
# Build search index
# -----------------------------
def build_index(folder):

    docs, paths = load_files(folder)

    embeddings = []
    chunk_paths = []
    chunks = []

    for doc, path in zip(docs, paths):

        pieces = split_chunks(doc)

        for p in pieces:

            emb = model.encode(p)

            embeddings.append(emb)

            chunk_paths.append(path)

            chunks.append(p)

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index, chunk_paths, chunks


# -----------------------------
# Build index once
# -----------------------------
index, paths, chunks = build_index("test_folder")


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI File Search")

query = st.text_input("Search your files")


if query:

    q_emb = model.encode([query]).astype("float32")

    distances, indices = index.search(q_emb, 5)

    st.subheader("Most relevant results")

    for rank, i in enumerate(indices[0]):

        st.write("File:", paths[i])

        st.write("Text:", chunks[i])

        st.write("---")