import streamlit as st
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.title("🔍 AI Media Search System")
st.write("Upload images and search using natural language")

# -------------------------------
# Upload Images
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:

    images = []
    paths = []
    descriptions = []

    # -------------------------------
    # Convert images to descriptions
    # -------------------------------
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        images.append(img)
        paths.append(file.name)

        # Simple description (can upgrade later)
        descriptions.append(file.name)

    # -------------------------------
    # Create embeddings
    # -------------------------------
    embeddings = model.encode(descriptions)
    embeddings = np.array(embeddings).astype("float32")

    # -------------------------------
    # Build FAISS index
    # -------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # -------------------------------
    # Search Query
    # -------------------------------
    query = st.text_input("Enter search query")

    if query:
        query_vector = model.encode(query)
        query_vector = np.array([query_vector]).astype("float32")

        k = min(6, len(paths))
        distances, indices = index.search(query_vector, k)

        st.subheader("🔎 Results")

        cols = st.columns(3)

        for idx, i in enumerate(indices[0]):
            with cols[idx % 3]:
                st.image(images[i], caption=paths[i], use_container_width=True)

        
