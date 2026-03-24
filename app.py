import streamlit as st
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

st.title("🔍 AI Media Search System")

model = SentenceTransformer("all-MiniLM-L6-v2")

uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    names = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        images.append(img)
        names.append(file.name)

    embeddings = model.encode(names)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = st.text_input("Enter search query")

    if query:
        query_vec = model.encode([query]).astype("float32")
        distances, indices = index.search(query_vec, min(6, len(names)))

        st.subheader("Results")

        cols = st.columns(3)
        for i, idx in enumerate(indices[0]):
            with cols[i % 3]:
                st.image(images[idx], caption=names[idx], use_container_width=True)
