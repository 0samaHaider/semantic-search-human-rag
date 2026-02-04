import streamlit as st
import pandas as pd
import uuid

from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="Semantic Search (Mini RAG)", layout="wide")

st.title("🔎 Semantic Search App (Mini RAG)")
st.write("Upload text or CSV, store embeddings, and ask questions.")

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="documents"
)

# -------------------------------
# Helper Functions
# -------------------------------
def add_texts_to_db(texts):
    embeddings = model.encode(texts).tolist()
    ids = [str(uuid.uuid4()) for _ in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids
    )

def search_db(query, n_results=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results["documents"][0]

def generate_human_answer(query, top_texts, similarity_threshold=0.6):
    """
    Converts top search results into a GPT-like human-friendly answer.
    If no relevant results, returns a polite message.
    """
    if not top_texts:
        return "I’m sorry, I don’t have information about that."

    # Simple similarity check: if the query is totally unrelated
    # Using cosine similarity between query and first text
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    query_emb = model.encode([query])
    text_emb = model.encode(top_texts)
    sim_score = cosine_similarity(query_emb, text_emb).max()

    if sim_score < similarity_threshold:
        return "I’m sorry, I don’t have information about that."

    # Otherwise, combine results into a friendly paragraph
    answer = "Based on what I found, here’s what might help:\n\n"
    for i, t in enumerate(top_texts, 1):
        answer += f"- {t}\n"
    return answer

# -------------------------------
# Upload Section
# -------------------------------
st.header("📤 Upload Data")

uploaded_file = st.file_uploader("Upload TXT or CSV", type=["txt", "csv"])

if uploaded_file:
    texts = []

    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.split("\n") if line.strip()]

    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        texts = df.astype(str).agg(" ".join, axis=1).tolist()

    if st.button("📥 Store in Vector DB"):
        add_texts_to_db(texts)
        st.success(f"Stored {len(texts)} documents successfully!")

# -------------------------------
# Query Section
# -------------------------------
st.header("💬 Ask a Question")

query = st.text_input("Enter your question")

if st.button("🔍 Search") and query:
    top_texts = search_db(query)
    human_answer = generate_human_answer(query, top_texts)
    st.markdown(human_answer)
